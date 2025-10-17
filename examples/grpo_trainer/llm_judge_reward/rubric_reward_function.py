#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rubric-based Reward Function for Geometry3K dataset with LLM-as-Judge
Standalone implementation with rubric-based scoring
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import List, Optional, Dict
import requests
from mathruler.grader import extract_boxed_content, grade_answer
import threading
import json

# Configuration for load balancing
if "LLM_JUDGE_URL" in os.environ:
    BASE_URLS = os.environ["LLM_JUDGE_URL"].split(",")
else:
    judge_mode = os.environ.get("LLM_JUDGE_MODE", "")

    if not judge_mode:
        model_name = os.environ.get("LLM_JUDGE_MODEL", "")
        if "4B" in model_name or "8B" in model_name:
            judge_mode = "multi-4gpu-4model"
        elif "30B" in model_name:
            judge_mode = "multi-4gpu-2model"
        else:
            judge_mode = "multi-4gpu-2model"
            print(
                f"Warning: Could not determine judge mode, defaulting to {judge_mode}"
            )

    if judge_mode == "multi-4gpu-4model":
        BASE_URLS = [
            "http://localhost:8000",
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003",
        ]
    elif judge_mode == "multi-4gpu-2model":
        BASE_URLS = ["http://localhost:8000", "http://localhost:8001"]
    else:
        BASE_URLS = ["http://localhost:8000"]

    print(f"Using judge mode: {judge_mode} with {len(BASE_URLS)} instance(s)")

BASE_URL = BASE_URLS[0]

# Round-robin load balancer
_url_cycle = None
_url_lock = None
_request_counter = 0


def _get_next_url():
    """Get next URL using round-robin, thread-safe."""
    global _url_cycle, _url_lock, _request_counter

    if _url_lock is None:
        _url_lock = threading.Lock()

    with _url_lock:
        url = BASE_URLS[_request_counter % len(BASE_URLS)]
        _request_counter += 1
        return url


# API configuration
API_KEY = os.environ.get("LLM_JUDGE_API_KEY", "EMPTY")
MAX_RETRIES = 3
BASE_DELAY = 0
MAX_WORKERS = int(os.environ.get("LLM_JUDGE_MAX_WORKERS", "32"))
MODEL_NAME = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# Weights for combining different score components
RULE_WEIGHT = float(os.environ.get("RULE_REWARD_WEIGHT", "1"))
LLM_WEIGHT = float(os.environ.get("LLM_REWARD_WEIGHT", "1"))
RUBRIC_WEIGHT = float(os.environ.get("RUBRIC_REWARD_WEIGHT", "0.5"))
# ENABLE_RUBRIC_SCORING = (
#     os.environ.get("ENABLE_RUBRIC_SCORING", "true").lower() == "true"
# )

# Dynamic weight scheduling configuration
ENABLE_DYNAMIC_WEIGHTS = (
    os.environ.get("ENABLE_DYNAMIC_WEIGHTS", "true").lower() == "true"
)
WEIGHT_SCHEDULE = os.environ.get("WEIGHT_SCHEDULE", "linear")
RULE_WEIGHT_START = float(os.environ.get("RULE_WEIGHT_START", "0.0"))
RULE_WEIGHT_END = float(os.environ.get("RULE_WEIGHT_END", "1.0"))
LLM_WEIGHT_START = float(os.environ.get("LLM_WEIGHT_START", "1.0"))
LLM_WEIGHT_END = float(os.environ.get("LLM_WEIGHT_END", "1.0"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "0"))


RUBRIC_SCORING_PROMPT_V1 = """You are an evaluator. Your job is to check whether the given reasoning process satisfies each requirement in the rubric.

# Inputs

## Problem:
{problem}

## Proposed Reasoning Process:
{solution}

## Evaluation Rubric:
{rubric_checkpoints}


# Evaluation Procedure

For each rubric criterion:

Read the rubric criterion carefully and understand what requirement it sets. 
- Recognize mathematical equivalence. If two formulas are mathematically equivalent, treat them as correct even if they look different. (E.g., "x - y - z" is equivalent to "x - (y + z)") 
- Be strict about the names of geometric objects. For example, ∠OAB = 90° is NOT the same as ∠OAC = 90°.

Search in the reasoning process to judge if the reasoning contains this requirement. Expressions may be implied, but any calculation must be explicitly mentioned. If yes, mark Satisfied (1 point). If no, mark Not satisfied (0 points).”
Continue until all criteria are covered.

Calculate the total score as the fraction of rubrics satisfied.


# Output Rules

For each rubric, first output a short analyse, then output the final justification in the following format:

Rubric X: [the analyse referencing the reasoning and the rubric]. [Satisfied / Not satisfied].

After all rubric criteria, output the final score as the following decimal format: Final Rubric Score: \\boxed{{score}}
"""

RUBRIC_SCORING_PROMPT_V2 = """
You are an evaluator. Your job is to check whether the given reasoning process satisfies each requirement in the rubric, with a calculation-centric focus.

# Inputs

## Problem:
{problem}

## Proposed Reasoning Process:
{solution}

## Rubric Checkpoints (Problem-Specific):
{rubric_checkpoints}

## Claimed Final Answer:
{final_answer}


# Evaluation Procedure

## Key Principle — Calculation-Centric and Consistency-Focused Judging
- For each rubric criterion, focus on the **key computational expressions** (equations, substitutions, transformations, or numeric evaluations) that directly satisfy the requirement.  
- Narrative or descriptive text is **not required**. As long as the reasoning provides the **necessary and consistent calculations**, the criterion may be considered satisfied.  
- If a rubric criterion is declarative (a statement or claim without explicit computation), its fulfillment can be inferred from **consistent calculations** elsewhere in the reasoning.  
- Recognize **mathematical equivalence** (e.g., `x - y - z` ≡ `x - (y + z)`), and require strict correspondence for named geometric or algebraic objects (e.g., ∠OAB ≠ ∠OAC unless justified).

### Revised Decision Rule — Focus on Contradictions, Not Alternatives
When judging whether a rubric item is satisfied:

1. Identify the relevant parts of the reasoning that address the criterion.  
2. If the reasoning **reaches an equivalent result** or **uses an alternative but valid computational path**, treat it as **Satisfied**.  
3. Mark **Not satisfied** **only if** there is a **clear contradiction or inconsistency** — for example:
   - The same quantity or condition is computed to **different numerical or algebraic values** within the reasoning and rubric.  
   - The reasoning’s calculation directly **conflicts** with the rubric’s stated condition (e.g., wrong substitution, reversed inequality, invalid transformation).  
   - The claimed step or value is **logically incompatible** with the problem context or with prior correct steps.

If no such contradiction exists, even if intermediate forms differ or steps are expressed differently, the rubric should be considered **Satisfied**.

## Additional General Rubric — Answer Consistency
After evaluating all problem-specific rubric items, add one more criterion:

- **General — Answer Consistency**:  
  Compare the final result derived in the reasoning with {{final_answer}}.  
  - Accept mathematically equivalent forms.  
  - For numeric results, allow minor rounding differences (small absolute/relative tolerance) unless exact forms are explicitly required.  
  - Mark **Satisfied** if consistent; otherwise **Not satisfied**.

## Scoring
- Let **N** be the number of problem-specific rubric items in {{rubric_checkpoints}}.  
- Total points = (number of Satisfied items among those N) + (1 if General — Answer Consistency is satisfied).  
- Maximum points = **N + 1**.  
- Final score = Total points / (N + 1).  
- Output the score strictly as a **decimal number**, never as a fraction or with approximation symbols.

# Output Rules
For each rubric, first output a short analysis that references the reasoning and the rubric item, then output the final judgment:

Rubric X: [brief analysis referencing the reasoning and rubric]. [Satisfied / Not satisfied].

(Use the given order for the problem-specific rubrics, where X is their index. Then append:)

Rubric (General — Answer Consistency): [analysis comparing the computed result in the reasoning with {{final_answer}}]. [Satisfied / Not satisfied].

Finally, print the total:

Final Rubric Score: \\boxed{{score}}
"""


def format_rubric_for_prompt(rubric: Dict[str, str]) -> str:
    """Format rubric checkpoints for inclusion in the prompt."""
    if not rubric or not isinstance(rubric, dict):
        return "No rubric available"

    formatted_checkpoints = []
    for i, (key, value) in enumerate(rubric.items(), 1):
        checkpoint_num = (
            key.replace("Checkpoint ", "").strip() if "Checkpoint" in key else str(i)
        )
        formatted_checkpoints.append(f"Rubric {checkpoint_num}: {value}")

    return "\n".join(formatted_checkpoints)


def extract_score_from_boxed(raw_content):
    raw_content = raw_content.split("Final Rubric Score:")[-1]
    # Remove log prefixes if present (but preserve \( and \) math delimiters)
    content = re.sub(r"\((?!\\)[^)]+\)\s*", "", raw_content)

    # Try different patterns - handle various escape formats
    patterns = [
        # Math mode patterns with \( \) delimiters
        (r"\\\(\\?boxed\{([0-9]*\.?[0-9]+)\}\\\)", lambda m: m.group(1)),
        (r"\\\(.*?\\?boxed\{([0-9]*\.?[0-9]+)\}\\\)", lambda m: m.group(1)),
        # Pattern 1a: Direct decimal with regular braces \boxed{0.00}
        (r"\\?boxed\{([0-9]*\.?[0-9]+)\}", lambda m: m.group(1)),
        # Pattern 1b: Direct decimal with escaped braces \boxed\{0.00\}
        (r"\\?boxed\\\{([0-9]*\.?[0-9]+)\\\}", lambda m: m.group(1)),
        # Pattern 1c: Direct decimal with double backslash and escaped braces \\boxed\\{0.00\\}
        (r"\\\\boxed\\\\\{([0-9]*\.?[0-9]+)\\\\\}", lambda m: m.group(1)),
        # Pattern 2a: Fraction/equation = decimal with regular braces
        (r"\\?boxed\{[^=]+=([0-9]*\.?[0-9]+)\}", lambda m: m.group(1)),
        # Pattern 2b: Fraction/equation = decimal with escaped braces
        (r"\\?boxed\\\{[^=]+=([0-9]*\.?[0-9]+)\\\}", lambda m: m.group(1)),
        # Pattern 2c: Fraction/equation with double backslash and escaped braces
        (r"\\\\boxed\\\\\{[^=]+=([0-9]*\.?[0-9]+)\\\\\}", lambda m: m.group(1)),
        # Pattern 3a: Simple fraction like \frac{1}{5} with regular braces
        (
            r"\\?boxed\{\\?frac\{(\d+)\}\{(\d+)\}\}",
            lambda m: str(float(m.group(1)) / float(m.group(2))),
        ),
        # Pattern 3b: Simple fraction with escaped outer braces
        (
            r"\\?boxed\\\{\\?frac\{(\d+)\}\{(\d+)\}\\\}",
            lambda m: str(float(m.group(1)) / float(m.group(2))),
        ),
        # Pattern 3c: Simple fraction with double backslash
        (
            r"\\\\boxed\\\\\{\\\\?frac\{(\d+)\}\{(\d+)\}\\\\\}",
            lambda m: str(float(m.group(1)) / float(m.group(2))),
        ),
        # Pattern 4a: Fraction with t prefix like \tfrac{1}{5}
        (
            r"\\?boxed\{\\?tfrac\{(\d+)\}\{(\d+)\}\}",
            lambda m: str(float(m.group(1)) / float(m.group(2))),
        ),
        # Pattern 4b: Fraction with t prefix and escaped braces
        (
            r"\\?boxed\\\{\\?tfrac\{(\d+)\}\{(\d+)\}\\\}",
            lambda m: str(float(m.group(1)) / float(m.group(2))),
        ),
        # Pattern 4c: Fraction with t prefix and double backslash
        (
            r"\\\\boxed\\\\\{\\\\?tfrac\{(\d+)\}\{(\d+)\}\\\\\}",
            lambda m: str(float(m.group(1)) / float(m.group(2))),
        ),
        # Pattern 5a: Fraction with equals and decimal
        (
            r"\\?boxed\{\\?t?frac\{(\d+)\}\{(\d+)\}=([0-9]*\.?[0-9]+)\}",
            lambda m: m.group(3),
        ),
        # Pattern 5b: Fraction with equals and decimal with escaped braces
        (
            r"\\?boxed\\\{\\?t?frac\{(\d+)\}\{(\d+)\}=([0-9]*\.?[0-9]+)\\\}",
            lambda m: m.group(3),
        ),
        # Pattern 5c: Fraction with equals and double backslash
        (
            r"\\\\boxed\\\\\{\\\\?t?frac\{(\d+)\}\{(\d+)\}=([0-9]*\.?[0-9]+)\\\\\}",
            lambda m: m.group(3),
        ),
    ]

    for pattern, extractor in patterns:
        match = re.search(pattern, content)
        if match:
            return extractor(match)

    return None


def calculate_dynamic_weight(
    start_weight: float,
    end_weight: float,
    current_step: int,
    total_steps: int,
    schedule: str = "linear",
    warmup_steps: int = 0,
) -> float:
    """Calculate dynamic weight based on training progress."""
    import math

    if current_step <= warmup_steps:
        return start_weight

    effective_step = current_step - warmup_steps
    effective_total = max(1, total_steps - warmup_steps)
    progress = min(1.0, effective_step / effective_total)

    if schedule == "linear":
        weight = start_weight + (end_weight - start_weight) * progress
    elif schedule == "cosine":
        cosine_progress = (1 - math.cos(math.pi * progress)) / 2
        weight = start_weight + (end_weight - start_weight) * cosine_progress
    elif schedule == "exponential":
        if end_weight > start_weight:
            weight = start_weight * ((end_weight / start_weight) ** progress)
        else:
            weight = start_weight * ((end_weight / start_weight) ** progress)
    elif schedule == "step":
        if progress < 0.5:
            weight = start_weight
        else:
            weight = end_weight
    else:
        weight = start_weight + (end_weight - start_weight) * progress

    return weight


def compute_rule_based_score(solution_str: str, ground_truth: str) -> float:
    """Compute the rule-based score for Geometry3K."""
    format_pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    has_good_format = bool(re.fullmatch(format_pattern, solution_str))

    answer = extract_boxed_content(solution_str)
    is_correct = grade_answer(answer, ground_truth) if answer else False

    format_score = 0.1
    accuracy_score = 1.0 if is_correct else 0.0
    format_bonus = format_score if has_good_format else 0.0

    return min(accuracy_score * (1.0 - format_score) + format_bonus, 1.0)


def call_rubric_judge(
    problem: str, solution: str, rubric: Dict[str, str], answer
) -> tuple[Optional[float], Optional[str], Optional[Dict[str, bool]]]:
    """
    Call the LLM judge API to evaluate solution against rubric.

    Returns:
        A tuple of (rubric_score, judge_response, checkpoint_results)
    """

    if not rubric or not isinstance(rubric, dict):
        return None, None, None

    rubric_checkpoints_str = format_rubric_for_prompt(rubric)
    prompt = RUBRIC_SCORING_PROMPT_V2.format(
        problem=problem,
        solution=solution,
        rubric_checkpoints=rubric_checkpoints_str,
        final_answer=answer,
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            if API_KEY != "EMPTY":
                headers["Authorization"] = f"Bearer {API_KEY}"

            base_url = _get_next_url()
            chat_url = f"{base_url}/v1/chat/completions"

            data = {
                "model": "llm-judge",
                "messages": messages,
                "max_tokens": 5000,
            }

            response = requests.post(chat_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            raw_content = response.json()["choices"][0]["message"]["content"]

            checkpoint_results = {}
            for i in range(1, len(rubric) + 1):
                pattern = rf"Criterion {i}:\s*(Satisfied|Not Satisfied)"
                match = re.search(pattern, raw_content, re.IGNORECASE)
                if match:
                    checkpoint_results[f"Criterion {i}"] = (
                        match.group(1).lower() == "satisfied"
                    )

            # score_match = re.search(r"\boxed\{([0-9]*\.?[0-9]+)\}", raw_content)
            score_str = extract_score_from_boxed(raw_content)
            if score_str:
                try:
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))
                    return score, raw_content, checkpoint_results
                except ValueError:
                    print(
                        f"Warning: Could not convert extracted score to float: {score_str}"
                    )

            if checkpoint_results:
                satisfied_count = sum(1 for v in checkpoint_results.values() if v)
                score = satisfied_count / len(rubric)
                return score, raw_content, checkpoint_results

            print(f"Warning: Could not extract rubric score from LLM response")
            return 0.5, raw_content, checkpoint_results

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                # delay = BASE_DELAY * (2**attempt)
                # print(
                #     f"Rubric Judge API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                # )
                pass
            else:
                print(f"Failed to call Rubric Judge after {MAX_RETRIES} attempts: {e}")
                return None, None, None

        except Exception as e:
            print(f"Unexpected error in rubric judge: {e}")
            return None, None, None

    return None, None, None


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict:
    """
    Compute combined score using rule-based and rubric evaluation.

    Returns:
        Dictionary containing all score components
    """

    # Get rule-based score
    rule_score = compute_rule_based_score(solution_str, ground_truth)

    # Initialize result dictionary
    result = {
        "score": rule_score,
        "rule_score": rule_score,
        "rubric_score": 0.0,
        "rule_weight": RULE_WEIGHT,
        "rubric_weight": RUBRIC_WEIGHT,
        "weighted_rule_reward": rule_score * RULE_WEIGHT,
        "weighted_rubric_reward": 0.0,
        "judge_response": None,
        "rubric_response": None,
        "checkpoint_results": None,
        "rubric_details": None,  # Add rubric details
        "global_step": extra_info.get("global_step") if extra_info else None,
        "total_steps": extra_info.get("total_steps") if extra_info else None,
    }

    # Get problem text
    problem_text = extra_info.get("question", "") or extra_info.get("problem", "")

    if not problem_text:
        print("Warning: No problem text available, using rule-based score only")
        return result

    # Check if rubric scoring is enabled and rubric is available
    rubric_ori = extra_info.get("rubric") if extra_info else None
    if rubric_ori and isinstance(rubric_ori, str):
        rubric = json.loads(rubric_ori)
    elif rubric_ori and isinstance(rubric_ori, dict):
        rubric = rubric_ori
    else:
        rubric = {}
    has_rubric = (
        rubric_ori
        and isinstance(rubric, dict)
        and len(rubric) > 0
        and "No correct processes found with score > 0.1" not in rubric_ori
    )

    if has_rubric:
        answer = extract_boxed_content(solution_str)
        rubric_score, rubric_response, checkpoint_results = call_rubric_judge(
            problem_text, solution_str, rubric, answer
        )

        if rubric_score is not None:
            result["rubric_score"] = rubric_score
            result["rubric_response"] = rubric_response
            result["checkpoint_results"] = checkpoint_results
            result["rubric_details"] = rubric  # Store the rubric details
            result["weighted_rubric_reward"] = rubric_score * RUBRIC_WEIGHT

            if os.environ.get("DEBUG_SCORING", "").lower() == "true":
                print(
                    f"Rubric scoring: {rubric_score:.3f} "
                    f"({sum(1 for v in checkpoint_results.values() if v)}/{len(rubric)} checkpoints)"
                )

    # Determine weights to use - initialize with defaults first
    rule_weight = RULE_WEIGHT
    rubric_weight = RUBRIC_WEIGHT

    if ENABLE_DYNAMIC_WEIGHTS and extra_info:
        global_step = extra_info.get("global_step", None)
        total_steps = extra_info.get("total_steps", None)

        if global_step is not None and total_steps is not None:
            rule_weight = calculate_dynamic_weight(
                RULE_WEIGHT_START,
                RULE_WEIGHT_END,
                global_step,
                total_steps,
                schedule=WEIGHT_SCHEDULE,
                warmup_steps=WARMUP_STEPS,
            )

            rubric_weight = RUBRIC_WEIGHT

            result["rule_weight"] = rule_weight
            result["rubric_weight"] = rubric_weight

    combined_score = rule_score + result["rubric_score"]

    result["score"] = combined_score

    return result


def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[dict],
) -> List[dict]:
    """
    Batch compute scores for multiple solutions in parallel.

    Returns:
        List of score dictionaries
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            future = executor.submit(
                compute_score, data_source, solution_str, ground_truth, extra_info
            )
            futures.append(future)

        results = [future.result() for future in futures]

    return results


# For backward compatibility
def compute_score_default(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """Default interface matching VERL's expected signature."""
    if extra_info is None:
        extra_info = {}
    return compute_score(data_source, solution_str, ground_truth, extra_info)
