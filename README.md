# AutoRubric-R1V

The official repository for "[AutoRubric-R1V: Rubric-Based Generative Rewards for Faithful Multimodal Reasoning](https://arxiv.org/abs/2510.14738)".



## Introduction
We propose AutoRubric-R1V,
a framework that integrates RLVR with process-level supervision through automatically collected rubric-based generative rewards. Our key innovation lies in
a scalable self-aggregation method that distills consistent reasoning checkpoints
from successful trajectories, enabling problem-specific rubric construction without
human annotation or stronger teacher MLLMs.

![AutoRubric-R1V method overview](assets/method.png)

## Usage
### Environment Setup
Our training framework is built upon [Verl](https://github.com/volcengine/verl), please install Verl first.

We use the GPT-OSS served by vllm as rubric generation and judge model, please also install the environment following [here](https://cookbook.openai.com/articles/gpt-oss/run-vllm).


### Generating Rubrics

```bash
# First generate responses, then use those with correct answer to compose rubrics.
python infer_gen_rubric/infer_then_gen_rubrics.py \
--input  [original_parquet] 

# merge generated rubrics to original training samples.
python infer_gen_rubric/merge_rubrics_parquet.py \
--original [original_parquet] \
--rubrics [rubric_parquet] \
--output [target_path]
```

### Training with Rubric-based Judge

```bash
# Using vllm to serve the judge model
bash examples/grpo_trainer/llm_judge_reward/start_vllm_judge.sh

# Training with judge
bash examples/grpo_trainer/llm_judge_reward/run_qwen2_5_vl_7b_rubric.sh
```

### Evaluation
```bash
bash evaluation/merge_eval.sh [saved_checkpoint_dir]
```