#!/usr/bin/env bash
# Training script with LLM-as-judge reward for Geometry3K dataset
# This script uses both rule-based (answer correctness) and LLM-based (reasoning consistency) rewards
# The LLM judge focuses on detecting inconsistencies between reasoning process and final answers

set -x
# Configuration
ENGINE=${1:-vllm}
export WANDB_API_KEY=45d3ceb6a15fed92a1ca9fd03f5d5833b77b5c9f
SAVE_DIR=/mnt/s3/training_saves_aftsub
# PROJECT_NAME=verl_grpo_7b_aftsub
PROJECT_NAME=debug

DATA_PATH=/mnt/s3/datasets/verl_training_datasets/ViRL39K/train_w_rubrics.parquet
# DATA_PATH=$DATA_DIR/geometry3k_new/train_w_rubrics.parquet
# VAR_DATA=$DATA_DIR/val6dataset/validation_data.parquet
VAR_DATA_PATH=$DATA_PATH
EXPERIMENT_NAME=debug
LOCAL_SAVE_PATH=$SAVE_DIR/$PROJECT_NAME/$EXPERIMENT_NAME

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# LLM Judge configuration
export LLM_JUDGE_URL=${LLM_JUDGE_URL:-"http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003"}
# export LLM_JUDGE_URL=${LLM_JUDGE_URL:-"http://localhost:8000,http://localhost:8001"}
export LLM_JUDGE_MODEL="llm-judge"
export LLM_JUDGE_MAX_WORKERS=32

# Reward weight configuration (can be adjusted)
# Static weights (used as fallback if dynamic weights are disabled)
export RULE_REWARD_WEIGHT=1  # Weight for rule-based reward (answer correctness)
export LLM_REWARD_WEIGHT=1   # Weight for LLM-based reward (reasoning consistency)

# Dynamic weight scheduling configuration
export ENABLE_DYNAMIC_WEIGHTS=true  # Enable dynamic weight scheduling
export WEIGHT_SCHEDULE=linear       # Options: linear, cosine, exponential, step

# Rule reward weight: starts from 0, gradually increases to 1 over training
export RULE_WEIGHT_START=1.0        # Starting weight for rule-based reward
export RULE_WEIGHT_END=1.0          # Ending weight for rule-based reward

# LLM reward weight: remains constant at 1.0 throughout training
export LLM_WEIGHT_START=1         # Starting weight for LLM-based reward
export LLM_WEIGHT_END=1           # Ending weight for LLM-based reward

# Optional warmup steps (default: 0)
export WARMUP_STEPS=0               # Number of warmup steps before starting dynamic weighting

# Optional: Enable debug logging
# export DEBUG_SCORING=true

# Check if vLLM judge service is running
# Split URLs and check each one
IFS=',' read -ra URLS <<< "$LLM_JUDGE_URL"
all_services_running=true
for url in "${URLS[@]}"; do
    if ! curl -s "${url}/health" >/dev/null 2>&1; then
        echo "Warning: LLM Judge service is not running at ${url}"
        all_services_running=false
    else
        echo "✓ LLM Judge service is running at ${url}"
    fi
done

if [ "$all_services_running" = false ]; then
    echo ""
    echo "Please start the vLLM judge service(s) first:"
    echo "  cd llm_judge_reward && ./start_vllm_judge.sh"
    echo ""
    echo "Or to run without LLM judge (rule-based only), set:"
    echo "  export LLM_REWARD_WEIGHT=0"
    echo "  export RULE_REWARD_WEIGHT=1.0"
    exit 1
fi

echo "All LLM Judge services are running"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run training with custom reward function
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$VAR_DATA_PATH \
    data.train_batch_size=8 \
    data.max_prompt_length=5000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=${SCRIPT_DIR}/rubric_reward_function.py \
    custom_reward_function.name=compute_score_batch \
    reward_model.launch_reward_fn_async=True \
    trainer.critic_warmup=0 \
    trainer.logger="wandb" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$LOCAL_SAVE_PATH \
    trainer.rollout_data_dir=$LOCAL_SAVE_PATH \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10000 \
    trainer.total_epochs=30 $@

echo "Training completed!"
