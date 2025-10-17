CHECKPOINT_DIR=$1

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_DIR \
    --target_dir $CHECKPOINT_DIR/huggingface

# mv $CHECKPOINT_DIR/huggingface $CHECKPOINT_DIR/../../

python evaluation/ray_evaluation.py --model_name $CHECKPOINT_DIR/huggingface --datasets mathvista