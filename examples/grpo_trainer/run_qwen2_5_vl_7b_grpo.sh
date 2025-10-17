set -x
ENGINE=${1:-vllm}
export WANDB_API_KEY=45d3ceb6a15fed92a1ca9fd03f5d5833b77b5c9f

# DATA_DIR=/mnt/s3/virl_dataset/verl_training_datasets
SAVE_DIR=/mnt/s3/training_saves_aftsub
# PROJECT_NAME=verl_grpo_7b_aftsub
PROJECT_NAME=debug

# DATA_PATH=$DATA_DIR/ViRL39K/train.parquet

DATA_PATH=/mnt/s3/datasets/verl_training_datasets/ViRL39K/train.parquet
# VAR_DATA=$DATA_DIR/val6dataset/validation_data.parquet
VAR_DATA_PATH=$DATA_PATH
EXPERIMENT_NAME=debug
LOCAL_SAVE_PATH=$SAVE_DIR/$PROJECT_NAME/$EXPERIMENT_NAME
# INIT_MODEL_PATH=/mnt/s3/training_saves/verl_checkpoints/verl_grpo_7b_geo3k/qwen2_5_vl_7b_geo_20k_verl_open_kl0/global_step_74/actor/huggingface
INIT_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$VAR_DATA_PATH \
    data.train_batch_size=64 \
    data.max_prompt_length=5200 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$INIT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="wandb" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$LOCAL_SAVE_PATH \
    trainer.rollout_data_dir=$LOCAL_SAVE_PATH \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=100000 \
    trainer.total_epochs=30 $@

    # trainer.validation_data_dir=$LOCAL_SAVE_PATH \

# aws s3 sync /mnt/s3/training_saves/verl_checkpoints/verl_grpo_7b_geo3k  s3://orby-osu/mengzhao/training_saves/verl_checkpoints/verl_grpo_7b_geo3k/  --exclude "**/debug/*" --exclude "**/__pycache__/*" --exclude "**/open_r1.egg-info/*" --exclude "**/*.pt"

# eval all checkpoints
# cd /mnt/s3/Codes/R1_Multimodal/verl/evaluation

# # Find all global_step_xxx directories and sort by step number in descending order
# echo "Finding all checkpoints in: $LOCAL_SAVE_PATH"
# for checkpoint_dir in $(find "$LOCAL_SAVE_PATH" -maxdepth 1 -type d -name "global_step_*" | \
#     sed 's/.*global_step_//' | sort -rn | sed "s|^|$LOCAL_SAVE_PATH/global_step_|"); do
    
#     if [ -d "$checkpoint_dir/actor" ]; then
#         CHECKPOINT_PATH="$checkpoint_dir/actor"
#         STEP_NUM=$(basename "$checkpoint_dir" | sed 's/global_step_//')
#         echo "========================================"
#         echo "Evaluating checkpoint at step: $STEP_NUM"
#         echo "Path: $CHECKPOINT_PATH"
#         echo "========================================"
        
#         bash merge_eval.sh "$CHECKPOINT_PATH"
        
#         echo "Finished evaluating step $STEP_NUM"
#         echo ""
#     else
#         echo "Warning: actor directory not found in $checkpoint_dir, skipping..."
#     fi
# done

# echo "All checkpoints evaluation completed!"

# aws s3 sync /mnt/s3/training_saves/verl_checkpoints/verl_grpo_7b_geo3k  s3://orby-osu/mengzhao/trzde "**/open_r1.egg-info/*" --exclude "**/*.pt"



