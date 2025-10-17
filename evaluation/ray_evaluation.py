import random
import sys
import os
import re
import json
import ray
from typing import List, Dict, Any, Tuple
import argparse
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch
import sys

import load_datasets

# Set random seeds for reproducible results


@dataclass
class EvaluationConfig:
    model_name: str
    datasets: List[str]
    num_gpus_per_node: int = 8
    batch_size: int = 8
    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 1024
    gpu_memory_utilization: float = 0.8
    enable_prefix_caching: bool = False
    limit_mm_per_prompt: Dict[str, int] = None
    num_generations_per_sample: int = (
        8  # Number of generations per sample for averaging
    )
    save_dir: str = None
    seed: int = 1001
    worker_num_gpus: int = 1
    worker_tensor_parallel_size: int = 1
    judge_num_gpus: int = 2
    judge_tensor_parallel_size: int = 2
    judge_gpu_memory_utilization: float = 0.35
    judge_max_model_len: int = 2048

    def __post_init__(self):
        if self.limit_mm_per_prompt is None:
            self.limit_mm_per_prompt = {"image": 10}

        self.worker_num_gpus = max(1, self.worker_num_gpus)
        self.worker_tensor_parallel_size = max(
            1, min(self.worker_tensor_parallel_size, self.worker_num_gpus)
        )
        self.judge_num_gpus = max(1, self.judge_num_gpus)
        self.judge_tensor_parallel_size = max(
            1, min(self.judge_tensor_parallel_size, self.judge_num_gpus)
        )
        self.judge_gpu_memory_utilization = min(
            max(self.judge_gpu_memory_utilization, 0.05), 0.95
        )
        self.judge_max_model_len = max(512, min(self.judge_max_model_len, 32768))


@ray.remote
class EvaluationWorker:
    def __init__(self, model_name: str, config: EvaluationConfig):
        # Set up paths for Ray worker
        import sys
        import os

        self.model_name = model_name
        self.config = config
        self.processor = None
        self.llm = None
        self.sampling_params = None
        self.judge_llm = None  # For mathverse judge evaluation
        self.judge_processor = None

        # Track the GPUs assigned by Ray to this worker
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.gpu_device_ids = [
            dev.strip() for dev in visible_devices.split(",") if dev.strip()
        ]
        if not self.gpu_device_ids:
            self.gpu_device_ids = ["0"]
        self.available_gpu_count = max(1, len(self.gpu_device_ids))
        self.worker_tp_size = min(
            self.config.worker_tensor_parallel_size, self.available_gpu_count
        )
        self.judge_tp_size = min(
            self.config.judge_tensor_parallel_size, self.available_gpu_count
        )

        if self.worker_tp_size < self.config.worker_tensor_parallel_size:
            print(
                f"[Worker Init] Adjusted worker tensor parallel size to {self.worker_tp_size} "
                f"due to {self.available_gpu_count} assigned GPU(s).",
                flush=True,
            )
        if self.judge_tp_size < self.config.judge_tensor_parallel_size:
            print(
                f"[Worker Init] Adjusted judge tensor parallel size to {self.judge_tp_size} "
                f"due to {self.available_gpu_count} assigned GPU(s).",
                flush=True,
            )

        print(
            f"Worker assigned to GPUs: {self.gpu_device_ids} "
            f"(CUDA_VISIBLE_DEVICES: {visible_devices or 'Not set'})",
            flush=True,
        )

        self._load_model()

    def _extract_boxed_content(self, text: str) -> str:
        """
        Extract content from \boxed{} command, properly handling nested braces.
        Returns None if no \boxed{} command is found.
        """
        if r"\boxed{" not in text:
            return None

        # Find the start of \boxed{
        start_idx = text.find(r"\boxed{")
        if start_idx == -1:
            return None

        # Start after "\boxed{"
        pos = start_idx + 7
        depth = 1
        content_start = pos

        # Track brace depth to find matching closing brace
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            # Found matching closing brace
            return text[content_start : pos - 1].strip()
        else:
            # No matching closing brace found
            return None

    def _load_judge_model(self):
        """Load the judge model for mathverse evaluation."""
        if self.judge_llm is not None:
            return  # Already loaded

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        judge_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        print(f"Loading judge model: {judge_model_name}")

        # Unload main model first to free memory
        if self.llm is not None:
            del self.llm
            self.llm = None
            import torch

            torch.cuda.empty_cache()
            print("Unloaded main model to free memory for judge model")

        self.judge_processor = AutoTokenizer.from_pretrained(judge_model_name)
        judge_tp = max(1, self.judge_tp_size)
        base_util = self.config.judge_gpu_memory_utilization
        base_max_len = self.config.judge_max_model_len

        def _create_judge_llm(tensor_parallel_size, gpu_util, max_len):
            print(
                f"Loading judge model on GPUs {self.gpu_device_ids} with tensor_parallel_size={tensor_parallel_size}, "
                f"gpu_memory_utilization={gpu_util}, max_model_len={max_len}",
                flush=True,
            )
            return LLM(
                model=judge_model_name,
                gpu_memory_utilization=gpu_util,
                trust_remote_code=True,
                max_model_len=max_len,
                tensor_parallel_size=tensor_parallel_size,
            )

        try:
            self.judge_llm = _create_judge_llm(judge_tp, base_util, base_max_len)
        except ValueError as exc:
            if "No available memory for the cache blocks" not in str(exc):
                raise

            fallback_util = min(0.95, max(base_util, 0.85))
            fallback_len = min(base_max_len, 1024)
            fallback_tp = max(
                judge_tp, min(self.available_gpu_count, self.config.judge_num_gpus)
            )

            print(
                "[Judge Loader] Initial load failed due to insufficient KV cache memory. "
                f"Retrying with tensor_parallel_size={fallback_tp}, "
                f"gpu_memory_utilization={fallback_util}, max_model_len={fallback_len}.",
                flush=True,
            )

            self.judge_llm = _create_judge_llm(fallback_tp, fallback_util, fallback_len)
            self.judge_tp_size = fallback_tp

    def _unload_judge_model(self):
        """Unload the judge model and reload main model."""
        if self.judge_llm is not None:
            del self.judge_llm
            self.judge_llm = None
            del self.judge_processor
            self.judge_processor = None
            import torch

            torch.cuda.empty_cache()
            print("Unloaded judge model")

            # Reload main model
            self._load_model()
            print("Reloaded main model")

    def _judge_with_llm(self, question, response: str, ground_truth) -> float:
        """Use LLM to judge if the response matches the ground truth."""
        # Ensure judge model is loaded (should already be loaded in process_samples)
        if self.judge_llm is None:
            print("Warning: Judge model not loaded, loading now...")
            self._load_judge_model()

        # Convert ground_truth to string if it's a list
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else ""

        # Create judge prompt
        judge_prompt = f"""You are a strict evaluator assessing answer correctness. You must output {{positive}} for fully correct answers and {{negative}} for any other case.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{ground_truth}
```
Model Prediction:
```
{response}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score {{positive}} if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score {{positive}} if the prediction matches the answer semantically, it can be in different format.
  * Score {{negative}} for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision

# Strict Output format
{{positive}} or {{negative}}"""

        messages = [
            {"role": "system", "content": "You are a precise mathematical evaluator."},
            {"role": "user", "content": judge_prompt},
        ]

        # Apply chat template
        prompt = self.judge_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate judgment
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=5,
        )

        outputs = self.judge_llm.generate(
            [prompt], sampling_params=sampling_params, use_tqdm=False
        )
        judgment = outputs[0].outputs[0].text.strip()
        # Extract score from judgment
        if "{positive}" in judgment:
            return 1.0
        elif "{negative}" in judgment:
            return 0.0
        else:
            # Default to 0 if no clear judgment
            print(f"Warning: Judge output unclear: {judgment}")
            return 0.0

    def _load_model(self):
        # Import required modules after path setup
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        print(
            f"Loading main model on GPUs {self.gpu_device_ids} from {self.model_name} "
            f"(tensor_parallel_size={self.worker_tp_size})",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Use full GPU memory for main model since judge model is text-only and small
        gpu_util = self.config.gpu_memory_utilization
        if "mathverse" in self.config.datasets:
            # Keep higher GPU utilization since judge model is small (4B) and text-only
            gpu_util = max(0.7, gpu_util)  # Changed from min(0.5) to max(0.7)
            print(
                f"Using GPU memory utilization of {gpu_util} for main model (judge model is text-only)"
            )

        max_len = 20000

        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=gpu_util,
            enable_prefix_caching=self.config.enable_prefix_caching,
            limit_mm_per_prompt=self.config.limit_mm_per_prompt,
            max_model_len=max_len,
            tensor_parallel_size=self.worker_tp_size,
        )

        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )

    def infer_batch(
        self, image_paths: List, queries: List[str], num_generations: int = 1
    ) -> List[List[str]]:
        # Import required modules
        from qwen_vl_utils import process_vision_info

        instruction_following = (
            r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            r"The reasoning process MUST BE enclosed within <think> </think> tags. "
            r"The final answer MUST BE put in \boxed{}."
        )

        prompts = []
        for bid in range(len(queries)):
            query = queries[bid]
            image_path = image_paths[bid]
            if isinstance(image_path, list):
                image_content = [
                    {
                        "type": "image",
                        "image": one_image,
                    }
                    for one_image in image_path
                ]
            else:
                image_content = [
                    {
                        "type": "image",
                        "image": image_path,
                    }
                ]

            prompts.append(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": image_content
                        + [
                            {
                                "type": "text",
                                "text": f"{query}\n{instruction_following}",
                            }
                        ],
                    },
                ]
            )

        inputs_vllm = []
        for idx, messages in enumerate(prompts):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_data, _ = process_vision_info(messages)

            inputs_vllm.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                }
            )

        # Create sampling params with n parameter for multiple generations
        from vllm import SamplingParams

        # Set seed for reproducibility
        # Use a fixed seed or config seed if available
        if hasattr(self.config, "seed") and self.config.seed is not None:
            generation_seed = self.config.seed
        else:
            # Use a fixed default seed for reproducibility
            generation_seed = 1001

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            n=num_generations,  # Generate multiple samples at once
            seed=generation_seed,
        )

        # Only log seed info once per worker
        if not hasattr(self, "_seed_logged") and num_generations > 1:
            print(
                f"Using seed {generation_seed} for {num_generations} generations",
                flush=True,
            )
            self._seed_logged = True

        # Generate all responses in a single call
        outputs = self.llm.generate(
            inputs_vllm, sampling_params=sampling_params, use_tqdm=False
        )

        # Process outputs: each input has num_generations outputs
        responses_per_sample = []
        for output in outputs:
            sample_responses = []
            for completion in output.outputs:
                # Decode each completion
                text = self.processor.decode(
                    completion.token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                sample_responses.append(text)
            responses_per_sample.append(sample_responses)

        return responses_per_sample

    def process_samples(
        self, samples: List[Dict[str, Any]], dataset_name: str
    ) -> Tuple[List[float], List[Dict]]:
        all_scores = []
        all_results = []

        batch_size = self.config.batch_size
        num_generations = self.config.num_generations_per_sample

        # Check if we need judge evaluation for mathverse
        needs_judge = dataset_name == "mathverse" and any(
            sample.get("eval_type") == "judge" for sample in samples
        )

        # Collect all responses first if we need judge evaluation
        all_responses_data = []

        # Process samples in batches
        for i in tqdm(
            range(0, len(samples), batch_size),
            desc=f"Processing {dataset_name} batches",
        ):
            sample_batch = samples[i : i + batch_size]

            image_paths = [sample["image"] for sample in sample_batch]
            prompts = [sample["query"] for sample in sample_batch]
            pids = [sample["pid"] for sample in sample_batch]
            answers = [sample["gt"] for sample in sample_batch]
            eval_types = [sample["eval_type"] for sample in sample_batch]
            choices_list = [sample.get("choices", None) for sample in sample_batch]

            # Get multiple model responses for each sample
            responses_per_sample = self.infer_batch(
                image_paths, prompts, num_generations
            )

            # Store responses data for later evaluation
            for idx, sample_responses in enumerate(responses_per_sample):
                all_responses_data.append(
                    {
                        "responses": sample_responses,
                        "pid": pids[idx],
                        "prompt": prompts[idx],
                        "answer": answers[idx],
                        "eval_type": eval_types[idx],
                        "choices": choices_list[idx],
                    }
                )

        # If we have judge evaluations, load judge model once
        if needs_judge:
            print("Loading judge model for mathverse evaluation...")
            self._load_judge_model()

        # Now evaluate all collected responses
        for response_data in all_responses_data:
            sample_scores = []
            sample_formatted_responses = []
            sample_extracted_responses = []

            for response in response_data["responses"]:
                formatted_response = self._extract_boxed_content(response)
                if formatted_response is None:
                    formatted_response = response.strip()

                score, extracted = self._eval_one_sample(
                    formatted_response,
                    response_data["answer"],
                    response_data["eval_type"],
                    response_data["choices"],
                    dataset_name,
                    question=response_data["prompt"],
                )

                sample_scores.append(score)
                sample_formatted_responses.append(formatted_response)
                sample_extracted_responses.append(extracted)

            # Calculate average score for this sample
            avg_score = sum(sample_scores) / len(sample_scores)

            # Calculate pass@1 metric
            pass_at_1 = sample_scores[0] if len(sample_scores) >= 1 else 0

            choices_for_written = (
                response_data["choices"].tolist()
                if response_data["choices"] is not None
                and isinstance(response_data["choices"], np.ndarray)
                else None
            )

            cur_res_dic = {
                "pid": response_data["pid"],
                "prompt": response_data["prompt"],
                "responses": response_data["responses"],  # All responses
                "formatted_responses": sample_formatted_responses,  # All formatted responses
                "extracted_responses": sample_extracted_responses,  # All extracted responses
                "individual_scores": sample_scores,  # Scores for each generation
                "answer": response_data["answer"],
                "score": avg_score,  # Average score
                "pass_at_1": pass_at_1,
                "eval_type": response_data["eval_type"],
                "num_generations": num_generations,
                "choices": choices_for_written,
            }
            all_scores.append(avg_score)
            all_results.append(cur_res_dic)

        # Unload judge model if it was loaded
        if needs_judge:
            print("Unloading judge model and reloading main model...")
            self._unload_judge_model()

        return all_scores, all_results

    def _eval_one_sample(
        self,
        response: str,
        answers,
        eval_type: str,
        options=None,
        dataset_name: str = None,
        question=None,
    ) -> Tuple[float, str]:
        # Import evaluation utilities
        from utils_eval import (
            parse_multi_choice_response,
            eval_multi_choice,
            parse_open_response,
            eval_open,
            calculate_anls,
            eval_rouge,
            symbols,
            eval_number,
        )

        if eval_type == "multiple-choice":
            chosen = parse_multi_choice_response(
                response,
                symbols[: len(options)],
                {s: o for s, o in zip(symbols[: len(options)], options)},
            )
            correct = eval_multi_choice(answers, chosen)
        elif eval_type == "open-ended" and dataset_name in [
            "mathverse",
            "mathvision",
            "mmmu",
            "mathvista",
        ]:
            # Use LLM judge for mathverse open-ended questions
            chosen = response  # Keep the full response as chosen
            # chosen = parse_open_response(response)
            correct = self._judge_with_llm(question, response, answers)
        elif eval_type == "open-ended":
            if not isinstance(answers, list):
                answers = [answers]
            chosen = parse_open_response(response)
            correct = eval_open(answers, chosen)
        elif eval_type == "number":
            chosen = parse_open_response(response)
            correct = eval_number(answers, chosen)
        elif eval_type == "anls":
            formated_response = response.split("Answer: ")[-1].lower()
            chosen = formated_response
            answers = [answer.lower() for answer in answers]
            correct = calculate_anls(gt=answers, pred=formated_response)
        elif eval_type == "captioning":
            chosen = response
            correct = eval_rouge([answers[0].lower()], [response.lower()])
        else:
            raise ValueError(f"Invalid question type: {eval_type}")

        return correct, chosen


@ray.remote
class ResultAggregator:
    def __init__(self, model_name: str, save_dir: str = None):
        self.model_name = model_name
        self.results = {}
        self.save_dir = save_dir

    def save_results(
        self,
        dataset_name: str,
        scores: List[float],
        results: List[Dict],
        worker_id: int,
    ):
        if dataset_name not in self.results:
            self.results[dataset_name] = {"scores": [], "results": []}

        self.results[dataset_name]["scores"].extend(scores)
        self.results[dataset_name]["results"].extend(results)

        # Save intermediate results
        if "checkpoint" in self.model_name:
            model_save_name = self.model_name.split("/")[-2]
        else:
            model_save_name = self.model_name.split("/")[-1]

        if self.save_dir is None:
            out_file_dir = os.path.join(self.model_name, f"eval_results/{dataset_name}")
        else:
            out_file_dir = os.path.join(self.save_dir, f"eval_results/{dataset_name}")
        os.makedirs(out_file_dir, exist_ok=True)

        # Save worker-specific results
        jsonl_details = f"{out_file_dir}/worker_{worker_id}_details.jsonl"
        with open(jsonl_details, "w") as jsonl_w:
            for result in results:
                jsonl_w.write(json.dumps(result) + "\n")

    def finalize_results(self, dataset_name: str):
        if dataset_name not in self.results:
            return

        all_scores = self.results[dataset_name]["scores"]
        all_results = self.results[dataset_name]["results"]

        # Calculate final accuracy
        accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
        total = len(all_scores)
        correct = sum(all_scores)

        print(f"Dataset {dataset_name}: Accuracy = {accuracy:.4f} ({correct}/{total})")

        # Save final aggregated results
        if "checkpoint" in self.model_name:
            model_save_name = self.model_name.split("/")[-2]
        else:
            model_save_name = self.model_name.split("/")[-1]

        if self.save_dir is None:
            out_file_dir = os.path.join(self.model_name, f"eval_results/{dataset_name}")
        else:
            out_file_dir = os.path.join(self.save_dir, f"eval_results/{dataset_name}")
        os.makedirs(out_file_dir, exist_ok=True)

        # Sort results by pid to ensure consistent ordering
        try:
            all_results.sort(key=lambda x: int(x["pid"]))
        except:
            all_results.sort(key=lambda x: x["pid"])

        # Save all results in final_details.jsonl
        final_details_file = f"{out_file_dir}/final_details.jsonl"
        with open(final_details_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

        # Clean up worker detail files after creating final_details.jsonl
        import glob

        for worker_file in glob.glob(f"{out_file_dir}/worker_*_details.jsonl"):
            try:
                os.remove(worker_file)
                print(f"Removed worker detail file: {worker_file}")
            except Exception as e:
                print(f"Failed to remove {worker_file}: {e}")

        # Calculate eval_type statistics
        eval_type_stats = {}
        pass_at_stats = {"total": {"pass_at_1": 0, "count": 0}}

        for result in all_results:
            eval_type = result.get("eval_type", "unknown")
            if eval_type not in eval_type_stats:
                eval_type_stats[eval_type] = {"correct": 0, "total": 0}
                pass_at_stats[eval_type] = {"pass_at_1": 0, "count": 0}

            eval_type_stats[eval_type]["total"] += 1
            eval_type_stats[eval_type]["correct"] += result["score"]

            # Accumulate pass@1 statistics
            pass_at_stats[eval_type]["pass_at_1"] += result.get("pass_at_1", 0)
            pass_at_stats[eval_type]["count"] += 1

            pass_at_stats["total"]["pass_at_1"] += result.get("pass_at_1", 0)
            pass_at_stats["total"]["count"] += 1

        # Calculate accuracy for each eval_type
        for eval_type in eval_type_stats:
            eval_type_stats[eval_type]["acc"] = (
                eval_type_stats[eval_type]["correct"]
                / eval_type_stats[eval_type]["total"]
            )
            # Calculate pass@1 percentage for each eval_type
            if eval_type in pass_at_stats and pass_at_stats[eval_type]["count"] > 0:
                eval_type_stats[eval_type]["pass_at_1"] = (
                    pass_at_stats[eval_type]["pass_at_1"]
                    / pass_at_stats[eval_type]["count"]
                )

        # Calculate total pass@1 metric
        total_pass_at_1 = (
            pass_at_stats["total"]["pass_at_1"] / pass_at_stats["total"]["count"]
            if pass_at_stats["total"]["count"] > 0
            else 0
        )

        # Print detailed statistics
        print(
            f"*** {dataset_name} *** \n\n *** acc: {accuracy:.2%} correct: {correct:.2f} total: {total} *** \n\n"
        )
        print(f"*** Total Pass@1 Metric ***\n" f"    Pass@1: {total_pass_at_1:.2%}\n")
        print(
            f"Note: Scores are averaged over {self.results[dataset_name]['results'][0].get('num_generations', 1)} generations per sample"
        )
        for eval_type, stats in eval_type_stats.items():
            print(
                f"*** {eval_type} *** acc: {stats['acc']:.2%} correct: {stats['correct']:.2f} total: {stats['total']} ***"
            )
            if "pass_at_1" in stats:
                print(f"    Pass@1: {stats['pass_at_1']:.2%}")

        # Save enhanced summary with eval_type breakdown
        final_acc_data = {
            "acc": f"{accuracy:.2%}",
            "correct": correct,
            "total": total,
            "pass_at_1": f"{total_pass_at_1:.2%}",
            "num_generations_per_sample": self.results[dataset_name]["results"][0].get(
                "num_generations", 1
            ),
            "eval_type_stats": {
                eval_type: {
                    "acc": f"{stats['acc']:.2%}",
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "pass_at_1": f"{stats.get('pass_at_1', 0):.2%}",
                }
                for eval_type, stats in eval_type_stats.items()
            },
        }

        final_acc_file = f"{out_file_dir}/final_acc.json"
        with open(final_acc_file, "w") as f:
            json.dump(final_acc_data, f, indent=2)

        # Keep the original summary.json for backward compatibility
        summary = {
            "dataset": dataset_name,
            "model": self.model_name,
            "total_samples": total,
            "correct_samples": correct,
            "accuracy": accuracy,
        }

        summary_file = f"{out_file_dir}/summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return accuracy


def load_dataset_samples(dataset_name: str) -> List[Dict[str, Any]]:
    """Load samples for a given dataset."""
    # Import dataset loading functions
    if dataset_name == "mathvista":
        return load_datasets.load_data_mathvista()
    elif dataset_name == "mathvision":
        return load_datasets.load_data_mathvision()
    elif dataset_name == "mathverse":
        return load_datasets.load_data_mathverse()
    elif dataset_name == "wemath":
        return load_datasets.load_data_wemath()
    elif dataset_name == "mmmu":
        return load_datasets.load_data_mmmu()
    elif dataset_name == "mmmu_pro":
        return load_datasets.load_data_mmmu_pro()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")


def run_evaluation(config: EvaluationConfig):
    """Main evaluation function using Ray."""
    if not ray.is_initialized():
        # Set up runtime environment for Ray workers
        runtime_env = {
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "TOKENIZERS_PARALLELISM": "true",
            }
        }
        ray.init(runtime_env=runtime_env)

    print(f"Starting evaluation with model: {config.model_name}")
    print(f"Datasets: {config.datasets}")
    print(f"Number of GPUs per node: {config.num_gpus_per_node}")
    print(
        f"Worker GPU allocation: {config.worker_num_gpus} "
        f"(tensor_parallel_size={config.worker_tensor_parallel_size})"
    )
    print(
        f"Judge GPU allocation: {config.judge_num_gpus} "
        f"(tensor_parallel_size={config.judge_tensor_parallel_size}, "
        f"gpu_memory_utilization={config.judge_gpu_memory_utilization}, "
        f"max_model_len={config.judge_max_model_len})"
    )

    # Create result aggregator
    aggregator = ResultAggregator.remote(config.model_name, config.save_dir)

    dataset_results = {}

    # Process each dataset
    for dataset_idx, dataset_name in enumerate(
        tqdm(config.datasets, desc="Processing datasets")
    ):
        print(f"\n{'='*60}")
        print(
            f"Processing dataset {dataset_idx + 1}/{len(config.datasets)}: {dataset_name}"
        )
        print(f"{'='*60}")

        # Load dataset samples
        samples = load_dataset_samples(dataset_name)
        print(f"Loaded {len(samples)} samples from {dataset_name} dataset.")

        # Get detailed resource information
        resources = ray.available_resources()
        total_gpus = ray.cluster_resources().get("GPU", 0)
        available_gpus = resources.get("GPU", 0)

        print(f"Ray cluster info:")
        print(f"  Total GPUs in cluster: {total_gpus}")
        print(f"  Available GPUs: {available_gpus}")
        print(f"  All available resources: {resources}")

        needs_judge = dataset_name == "mathverse" and any(
            sample.get("eval_type") == "judge" for sample in samples
        )
        gpus_per_worker = (
            max(config.worker_num_gpus, config.judge_num_gpus, 1)
            if needs_judge
            else max(config.worker_num_gpus, 1)
        )
        if config.num_gpus_per_node < gpus_per_worker:
            print(
                f"Warning: worker requires {gpus_per_worker} GPU(s) but config.num_gpus_per_node="
                f"{config.num_gpus_per_node}. Adjusting to allow at least one worker.",
                flush=True,
            )

        max_gpus_per_worker = gpus_per_worker
        available_gpu_slots = int(available_gpus // max_gpus_per_worker)
        config_gpu_slots = max(1, config.num_gpus_per_node // max_gpus_per_worker)
        max_possible_workers = min(config_gpu_slots, available_gpu_slots)

        if max_possible_workers == 0:
            raise RuntimeError(
                f"Insufficient available GPUs ({available_gpus}) to schedule workers that require "
                f"{max_gpus_per_worker} GPU(s) each. Consider reducing judge/work tensor parallel size."
            )

        num_workers = min(max_possible_workers, len(samples))
        if num_workers == 0:
            print(f"No samples found for {dataset_name}, skipping worker creation.")
            continue

        print(
            f"Using {num_workers} worker(s) for this dataset "
            f"(GPUs per worker: {gpus_per_worker}, needs_judge={needs_judge})"
        )

        # Distribute samples across workers
        samples_per_worker = len(samples) // num_workers
        worker_futures = []
        workers = []

        # Create workers and distribute work
        for worker_id in range(num_workers):
            start_idx = worker_id * samples_per_worker
            end_idx = (
                (worker_id + 1) * samples_per_worker
                if worker_id < num_workers - 1
                else len(samples)
            )
            worker_samples = samples[start_idx:end_idx]

            if len(worker_samples) > 0:  # Only create worker if there are samples
                worker = EvaluationWorker.options(num_gpus=gpus_per_worker).remote(
                    config.model_name, config
                )
                future = worker.process_samples.remote(worker_samples, dataset_name)
                worker_futures.append((future, worker_id))
                workers.append(worker)

        # Collect results from all workers
        for future, worker_id in worker_futures:
            scores, results = ray.get(future)
            aggregator.save_results.remote(dataset_name, scores, results, worker_id)

        # Clean up workers explicitly to free GPUs
        print(f"Cleaning up {len(workers)} workers...")
        for worker in workers:
            ray.kill(worker)

        # Wait a moment for GPU cleanup and resource registration
        import time

        time.sleep(3)
        print(
            f"After cleanup - Available GPUs: {ray.available_resources().get('GPU', 0)}"
        )

        # Finalize results for this dataset
        accuracy = ray.get(aggregator.finalize_results.remote(dataset_name))
        dataset_results[dataset_name] = accuracy

    # Print final summary
    print("\n" + "=" * 50)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 50)
    for dataset_name, accuracy in dataset_results.items():
        print(f"{dataset_name}: {accuracy:.4f}")

    # Run overall summary
    try:
        if config.save_dir:
            os.system(f"python evaluation/sum_all_dataset_res.py -s {config.save_dir}")
        else:
            os.system(
                f"python evaluation/sum_all_dataset_res.py -s {config.model_name}"
            )
    except Exception as e:
        print(f"Warning: Could not run overall summary script: {e}")

    ray.shutdown()


def get_args():
    parser = argparse.ArgumentParser(description="Ray-based evaluation script")
    parser.add_argument(
        "-s",
        "--model_name",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated list of datasets or 'all' for all datasets",
    )
    parser.add_argument(
        "--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--worker-num-gpus",
        type=int,
        default=1,
        dest="worker_num_gpus",
        help="Number of GPUs reserved per evaluation worker for the main model",
    )
    parser.add_argument(
        "--worker-tensor-parallel-size",
        type=int,
        default=1,
        dest="worker_tensor_parallel_size",
        help="Tensor parallel size used when loading the main model",
    )
    parser.add_argument(
        "--judge-num-gpus",
        type=int,
        default=8,
        dest="judge_num_gpus",
        help="Number of GPUs reserved per worker when loading the judge model",
    )
    parser.add_argument(
        "--judge-tensor-parallel-size",
        type=int,
        default=8,
        dest="judge_tensor_parallel_size",
        help="Tensor parallel size used for the judge model",
    )
    parser.add_argument(
        "--judge-gpu-memory-utilization",
        type=float,
        default=0.9,
        dest="judge_gpu_memory_utilization",
        help="GPU memory utilization fraction for the judge model",
    )
    parser.add_argument(
        "--judge-max-model-len",
        type=int,
        default=2000,
        dest="judge_max_model_len",
        help="Maximum context length used when loading the judge model",
    )
    parser.add_argument(
        "--num_generations_per_sample",
        type=int,
        default=1,
        help="Number of generations per sample for averaging (default: 8)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
        help="Random seed for reproducible results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Parse datasets argument
    ALL_DATASETS = [
        "mathvista",
        "mathverse",
        "mathvision",
        "wemath",
        "mmmu",
        "mmmu_pro",
    ]

    if args.datasets == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
        # Validate dataset names
        for dataset in datasets:
            if dataset not in ALL_DATASETS:
                raise ValueError(
                    f"Invalid dataset: {dataset}. Available: {ALL_DATASETS}"
                )

    # Create configuration
    config = EvaluationConfig(
        model_name=args.model_name,
        datasets=datasets,
        num_gpus_per_node=args.num_gpus_per_node,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_generations_per_sample=args.num_generations_per_sample,
        save_dir=args.save_dir,
        seed=args.seed,
        worker_num_gpus=args.worker_num_gpus,
        worker_tensor_parallel_size=args.worker_tensor_parallel_size,
        judge_num_gpus=args.judge_num_gpus,
        judge_tensor_parallel_size=args.judge_tensor_parallel_size,
        judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
        judge_max_model_len=args.judge_max_model_len,
    )

    # Run evaluation
    run_evaluation(config)
