#!/usr/bin/env python3
"""
Two-stage rubric pipeline for geometry3k_new:

Stage 1:
    - Load multimodal questions (with images) from Parquet
    - Use Qwen2.5-VL-7B via vLLM with Ray data-parallel workers to sample eight responses per question
    - Score each response against the provided answer
    - Save (problem, answer, responses, scores, problem_id) to an intermediate Parquet

Stage 2:
    - Read the intermediate Parquet (no images involved)
    - Filter correct responses via the stored scores
    - Query an OSS GPT model (through vLLM) to derive calculation-focused rubrics
    - Save (problem, answer, responses, scores, rubric, problem_id) to a final Parquet
"""

from __future__ import annotations

import argparse
import json
import random
import re
import os
from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
import ray
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from mathruler.grader import extract_boxed_content, grade_answer

# Reduce vLLM logging noise if supported.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


@dataclass
class ResponseGenerationConfig:
    """Configuration for Qwen2.5-VL-7B response sampling."""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    num_samples: int = 8
    temperature: float = 1
    top_p: float = 0.9
    max_new_tokens: int = 512
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    seed: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."


@dataclass
class RubricGenerationConfig:
    """Configuration for OSS GPT rubric generation via vLLM."""

    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9


class QwenResponder:
    """Utility wrapper to sample responses from Qwen2.5-VL-7B."""

    def __init__(self, config: ResponseGenerationConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        tensor_parallel = max(1, config.tensor_parallel_size)
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sample_counter = 0

    def generate_responses(
        self, problem: str, pil_images: Sequence[Image.Image]
    ) -> List[str]:
        """Generate `num_samples` responses for the given problem and images."""
        responses: List[str] = []

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in pil_images],
                    {
                        "type": "text",
                        "text": (
                            r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.\n"
                            f"{problem}"
                        ),
                    },
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        multi_modal: Dict[str, Any] = {}
        if pil_images:
            multi_modal["image"] = list(pil_images)

        generate_inputs: Dict[str, Any] = {"prompt": prompt_text}
        if multi_modal:
            generate_inputs["multi_modal_data"] = multi_modal

        for _ in range(self.config.num_samples):
            if self.config.seed is not None:
                sampling_seed = (self.config.seed + self.sample_counter) % (2**31 - 1)
            else:
                sampling_seed = random.randint(0, 2**31 - 1)
            self.sample_counter += 1

            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_new_tokens,
                seed=sampling_seed,
            )

            outputs = self.llm.generate(
                [generate_inputs],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            if not outputs:
                continue

            request_output = outputs[0]
            for completion in request_output.outputs:
                responses.append(completion.text.strip())

        return responses


@ray.remote(num_gpus=1)
class ResponseGenerationWorker:
    """Ray worker that generates responses on a single GPU using vLLM."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = ResponseGenerationConfig(**config_dict)
        self.responder = QwenResponder(self.config)

    def generate(
        self,
        sample_index: int,
        problem_id: str,
        data_source: str,
        problem: str,
        answer: str,
        images: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pil_images = bytes_to_pil_images(images)
        responses = self.responder.generate_responses(problem, pil_images)
        extracted_answers = [extract_final_answer(resp) for resp in responses]
        scores = [score_response(resp or "", answer) for resp in responses]

        return {
            "sample_index": sample_index,
            "problem_id": problem_id,
            "data_source": data_source,
            "problem": problem,
            "answer": answer,
            "responses": responses,
            "extracted_answers": extracted_answers,
            "scores": scores,
        }


class OssRubricGenerator:
    """Generate rubrics by comparing correct reasoning trajectories."""

    def __init__(self, config: RubricGenerationConfig):
        self.config = config
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=None,
        )

    def generate_rubric(
        self, problem: str, answer: str, correct_responses: Sequence[str]
    ) -> Dict[str, Any]:
        if not correct_responses:
            return {
                "error": "No correct responses available to derive rubric",
                "checkpoints": {},
            }

        processes_text = "\n\n".join(
            [
                f"Process {idx + 1}:\n{response}"
                for idx, response in enumerate(correct_responses)
            ]
        )

        prompt = f"""You are given the TEXT DESCRIPTION of a multimodal (image + text) geometry question and several reasoning processes produced by another model. 
Only the textual question statement is provided here—assume that all visual details from the original images are already encoded inside the reasoning traces. Your task is to derive calculation-focused scoring checkpoints that a grader could use to verify a solver's work.

CRITICAL TASK: Carefully compare the reasoning processes to determine which quantitative steps are genuinely correct and indispensable.
1. CROSS-VALIDATE: When multiple processes perform the same answer using different intermediate steps, determine which is correct. Only include the correct steps in the rubric.
2. VERIFY CALCULATIONS: Confirm arithmetic, algebraic, and geometric computations. When different processes compute the same quantity, keep only the numerically correct expression or result.
3. IGNORE PARROTING: Do not include checkpoints that merely restate givens, diagrams, or qualitative descriptions of the problem. Focus on actionable calculations or numeric relationships that must be established.
4. FORMULATE SCORING CRITERIA: Phrase each rubric item as a grading requirement (e.g., “Computes the area of triangle ABC as 24 by applying ...”) rather than copying a solver’s sentence. Each checkpoint should describe what must be computed or demonstrated to earn credit.
5. ENSURE NECESSITY: Include only steps that are necessary to reach the correct final answer. If reliable checkpoints cannot be extracted, return an empty JSON object.

Provide the output strictly as JSON:
{{
    "Rubric 1": "... first essential and correct checkpoint ...",
    "Rubric 2": "... second essential and correct checkpoint ...",
    ...
}}

Question (text only):
{problem}

Correct Answer: {answer}

Reasoning Processes to Analyze:
{processes_text}

Return only the JSON with the validated checkpoints."""

        result = self.llm.generate(
            [prompt],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )[0]

        if not result.outputs:
            return {"error": "Model returned no outputs", "checkpoints": {}}

        response_text = result.outputs[0].text.strip()
        rubric = parse_json_response(response_text)
        if rubric is not None:
            return rubric

        return {
            "error": "Failed to parse JSON from OSS model response",
            "raw_response": response_text,
        }


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
BOX_PATTERN = re.compile(r"\\boxed\s*\{([^{}]+)\}")


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON object from model output."""
    if not text:
        return None

    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate).strip()

    try:
        return json.loads(candidate)
    except Exception:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        snippet = match.group().strip()
        try:
            return json.loads(snippet)
        except Exception:
            pass

    # Scan for balanced top-level JSON objects.
    segments: List[str] = []
    stack: List[int] = []
    in_string = False
    escape = False
    for idx, ch in enumerate(candidate):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                segments.append(candidate[start : idx + 1])

    for snippet in reversed(segments):
        try:
            return json.loads(snippet)
        except Exception:
            continue

    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer heuristic from model response."""
    if not text:
        return None

    answer_match = ANSWER_PATTERN.search(text)
    if answer_match:
        return answer_match.group(1).strip()

    box_matches = BOX_PATTERN.findall(text)
    if box_matches:
        return box_matches[-1].strip()

    think_match = THINK_PATTERN.search(text)
    if think_match:
        think_content = think_match.group(1)
        box_in_think = BOX_PATTERN.findall(think_content)
        if box_in_think:
            return box_in_think[-1].strip()

    tail_match = re.search(
        r"(?:the answer is|answer:|final answer:)\s*([^\n]+)", text, re.IGNORECASE
    )
    if tail_match:
        return tail_match.group(1).strip()

    return text.strip().splitlines()[-1].strip()


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def score_response(predicted: str, answer: str) -> float:
    """Score response using mathruler grading."""
    if not predicted:
        return 0.0
    return acc_reward(predicted.strip(), answer.strip(), use_boxed=True)


def load_dataset(parquet_path: str) -> List[Dict[str, Any]]:
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def save_dataset(records: List[Dict[str, Any]], output_path: str) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


def bytes_to_pil_images(images: Sequence[Dict[str, Any]]) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for image_entry in images:
        image_bytes = image_entry.get("bytes")
        if image_bytes is None:
            continue
        with BytesIO(image_bytes) as buffer:
            pil_image = Image.open(buffer).convert("RGB")
            pil_images.append(pil_image)
    return pil_images


def process_generate_responses(
    data: List[Dict[str, Any]],
    response_config: ResponseGenerationConfig,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Stage 1: produce responses and scores for every sample using Ray data parallelism."""

    if max_samples is not None:
        data = data[:max_samples]

    if not data:
        return []

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    worker_config = asdict(response_config)
    workers = [
        ResponseGenerationWorker.options(num_gpus=1).remote(worker_config)
        for _ in range(num_workers)
    ]

    tasks = []
    for sample_index, entry in enumerate(data):
        worker = workers[sample_index % num_workers]
        task = worker.generate.remote(
            sample_index=sample_index,
            problem_id=entry["problem_id"],
            data_source=entry["data_source"],
            problem=entry["extra_info"]["question"],
            answer=entry["extra_info"]["answer"],
            images=entry.get("images", []),
        )
        tasks.append(task)

    results: List[Dict[str, Any]] = []
    pending = list(tasks)
    with tqdm(total=len(tasks), desc="Generating responses") as pbar:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            ready_results = ray.get(ready)
            results.extend(ready_results)
            pbar.update(len(ready_results))

    results.sort(key=lambda item: item["sample_index"])
    return results


def process_generate_rubrics(
    records: List[Dict[str, Any]],
    rubric_generator: OssRubricGenerator,
    score_threshold: float = 0.5,
    min_correct_processes: int = 1,
) -> List[Dict[str, Any]]:
    """Stage 2: derive rubrics from previously generated responses."""

    enriched_records: List[Dict[str, Any]] = []
    for entry in tqdm(records, desc="Generating rubrics"):
        problem = entry.get("problem", "")
        answer = entry.get("answer", "")
        responses = entry.get("responses", []) or []
        scores = entry.get("scores", []) or []
        extracted_answers = entry.get("extracted_answers")
        if extracted_answers is None:
            extracted_answers = [extract_final_answer(resp) for resp in responses]

        correct_responses = [
            resp
            for resp, score in zip(responses, scores)
            if score is not None and score > score_threshold
        ]

        if len(correct_responses) >= min_correct_processes:
            rubric = rubric_generator.generate_rubric(
                problem=problem,
                answer=answer,
                correct_responses=correct_responses,
            )
        else:
            rubric = {
                "error": (
                    f"Not enough correct responses to derive rubric "
                    f"(found {len(correct_responses)}, require ≥{min_correct_processes})"
                )
            }

        enriched_records.append(
            {
                "sample_index": entry.get("sample_index"),
                "problem_id": entry.get("problem_id"),
                "data_source": entry.get("data_source"),
                "problem": problem,
                "answer": answer,
                "responses": responses,
                "extracted_answers": extracted_answers,
                "scores": scores,
                "rubric": json.dumps(rubric, ensure_ascii=False),
            }
        )

    return enriched_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses and rubrics for geometry3k_new Parquet data."
    )
    parser.add_argument(
        "--stage",
        choices=["both", "responses", "rubrics"],
        default="both",
        help="Which stage(s) to run: generate responses, generate rubrics, or both.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Parquet file containing problem, answer, and images columns.",
    )
    parser.add_argument(
        "--responses-output",
        type=str,
        default=None,
        help="Parquet path for generated responses and scores (read when stage includes rubrics).",
    )
    parser.add_argument(
        "--response-tensor-parallel-size",
        type=int,
        default=-1,
        help="Tensor parallel size for Qwen response generation (default: 1, no sharding).",
    )
    parser.add_argument(
        "--response-num-workers",
        type=int,
        default=-1,
        help="Number of data-parallel Qwen workers (default: one per visible GPU).",
    )
    parser.add_argument(
        "--response-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for Qwen response generation (default: 0.9).",
    )
    parser.add_argument(
        "--rubric-output",
        type=str,
        default=None,
        help="Parquet path to store rubrics alongside responses and scores.",
    )
    parser.add_argument(
        "--oss-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Path or identifier for the GPT OSS model served by vLLM.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of examples to process (useful for testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=-1,
        help="Tensor parallel size for the OSS model (default: use all visible GPUs).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for the OSS model (default: 0.9).",
    )
    parser.add_argument(
        "--oss-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for the OSS rubric generation (default: 2048).",
    )
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.responses_output:
        args.responses_output = args.input.replace(
            ".parquet", f"_responses_{timestamp}.parquet"
        )
    if not args.rubric_output:
        args.rubric_output = args.input.replace(
            ".parquet", f"_rubric_{timestamp}.parquet"
        )

    return args


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stage = args.stage

    if stage in {"both", "responses"}:
        print("Loading dataset for response generation...")
        data = load_dataset(args.input)
        print(f"Loaded {len(data)} examples from {args.input}")

        response_tensor_parallel = (
            1
            if args.response_tensor_parallel_size == -1
            else args.response_tensor_parallel_size
        )
        response_tensor_parallel = max(1, response_tensor_parallel)

        num_response_workers = (
            torch.cuda.device_count()
            if args.response_num_workers == -1
            else args.response_num_workers
        )
        num_response_workers = max(1, num_response_workers)

        responder_cfg = ResponseGenerationConfig(
            tensor_parallel_size=response_tensor_parallel,
            gpu_memory_utilization=args.response_gpu_memory_utilization,
            seed=args.seed,
        )

        print("Stage 1: Generating responses with Qwen2.5-VL-7B (data parallel)...")
        response_records = process_generate_responses(
            data=data,
            response_config=responder_cfg,
            num_workers=num_response_workers,
            max_samples=args.max_samples,
        )
        print(
            f"Saving {len(response_records)} response entries to {args.responses_output} ..."
        )
        save_dataset(response_records, args.responses_output)

        ray.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Skipping response generation stage.")

    if stage in {"both", "rubrics"}:
        if stage == "rubrics":
            print("Loading responses dataset for rubric generation...")
        else:
            print("Reloading responses dataset for rubric generation...")

        response_data = load_dataset(args.responses_output)

        tensor_parallel = (
            torch.cuda.device_count()
            if args.tensor_parallel_size == -1
            else args.tensor_parallel_size
        )
        tensor_parallel = max(1, tensor_parallel)

        rubric_cfg = RubricGenerationConfig(
            model_path=args.oss_model,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.oss_max_tokens,
        )
        rubric_generator = OssRubricGenerator(rubric_cfg)

        print("Stage 2: Generating rubrics with OSS model...")
        rubric_records = process_generate_rubrics(
            records=response_data,
            rubric_generator=rubric_generator,
        )
        print(
            f"Saving {len(rubric_records)} rubric entries to {args.rubric_output} ..."
        )
        save_dataset(rubric_records, args.rubric_output)
    else:
        print("Skipping rubric generation stage.")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
