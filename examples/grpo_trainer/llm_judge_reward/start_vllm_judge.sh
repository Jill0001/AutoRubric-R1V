#!/usr/bin/env bash
# Start vLLM service for Qwen3 judge model
# Automatically selects mode based on model size:
#   - 4B/8B models: 4 instances, 1 GPU each
#   - 30B models: 2 instances, 2 GPUs each
#
# Usage:
#   ./start_vllm_judge.sh                          # Auto-detect mode based on MODEL_NAME
#   MODE=single ./start_vllm_judge.sh              # Force single instance on all GPUs
#   MODE=multi-4gpu-2model ./start_vllm_judge.sh   # Force 2 instances, 2 GPUs each
#   MODE=multi-4gpu-4model ./start_vllm_judge.sh   # Force 4 instances, 1 GPU each
source myenv/bin/activate
set -e

# Configuration
# MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME="openai/gpt-oss-20b"

# Auto-detect mode based on model name if not specified
if [ -z "$MODE" ]; then
    if echo "$MODEL_NAME" | grep -qE "4B|8B|20b"; then
        MODE="multi-4gpu-4model"
        echo "Auto-detected 4B/8B model, using multi-4gpu-4model mode"
    elif echo "$MODEL_NAME" | grep -qE "30B"; then
        MODE="multi-4gpu-2model"
        echo "Auto-detected 30B model, using multi-4gpu-2model mode"
    else
        # Default to multi-4gpu-2model for unknown models
        MODE="multi-4gpu-2model"
        echo "Unknown model size, defaulting to multi-4gpu-2model mode"
    fi
fi

# Mode selection: "single", "multi-4gpu-2model", "multi-4gpu-4model"
# single: single instance on all GPUs
# multi-4gpu-2model: 2 instances, 2 GPUs each (for large models like 30B)
# multi-4gpu-4model: 4 instances, 1 GPU each (for small models like 4B/8B)
PORT=${VLLM_PORT:-8000}
HOST=${VLLM_HOST:-0.0.0.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10000}
XDG_CACHE_HOME="/tmp/xdg_cache_${PORT}" \
TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${PORT}" \
TRITON_CACHE_DIR="/tmp/triton_${PORT}" \
# Single mode configuration
GPU_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}  # Use 4 GPUs for the judge model
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}

# Multi mode configuration - will be set based on MODE
NUM_INSTANCES=0
GPU_ALLOCATIONS=()

# Function to start single instance
start_single_instance() {
    # Check if service is already running
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $PORT is already in use. Please stop the existing service or use a different port."
        exit 1
    fi

    echo "Starting vLLM service for LLM judge (single instance mode)..."
    echo "Model: $MODEL_NAME"
    echo "Port: $PORT"
    echo "GPUs: $GPU_DEVICES"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"

    # Start vLLM server
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES vllm serve $MODEL_NAME \
        --served-model-name llm-judge \
        --host $HOST \
        --port $PORT \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-model-len $MAX_MODEL_LEN \
        --enable-prefix-caching \
        --disable-log-requests \
        --trust-remote-code \
        2>&1 | tee vllm_judge.log &

    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID"

    # Save PID to file for later cleanup
    echo $VLLM_PID > vllm_judge.pid
}

# Function to start multiple instances
start_multi_instances() {
    echo "Starting vLLM service for LLM judge (multi instance mode)..."
    echo "Number of instances: $NUM_INSTANCES"
    
    # Check port availability for all instances
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        port=$((PORT + i))
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "Error: Port $port is already in use"
            exit 1
        fi
    done
    
    # Start all instances
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        port=$((PORT + i))
        gpu_devices="${GPU_ALLOCATIONS[$i]}"
        
        if [ -z "$gpu_devices" ]; then
            echo "Error: No GPU allocation defined for instance $i"
            exit 1
        fi
        
        # Calculate tensor parallel size
        gpu_count=$(echo "$gpu_devices" | tr ',' '\n' | wc -l)
        
        echo "Starting instance $i on GPUs: $gpu_devices, Port: $port"
        
        CUDA_VISIBLE_DEVICES=$gpu_devices vllm serve $MODEL_NAME \
            --served-model-name llm-judge \
            --host $HOST \
            --port $port \
            --tensor-parallel-size $gpu_count \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --max-model-len $MAX_MODEL_LEN \
            --enable-prefix-caching \
            --disable-log-requests \
            --trust-remote-code \
            2>&1 | tee "vllm_judge_instance_${i}.log" &
        
        pid=$!
        echo $pid > "vllm_judge_instance_${i}.pid"
        echo "Instance $i started with PID: $pid"
    done
}

# Configure based on mode
case "$MODE" in
    "multi-4gpu-2model")
        NUM_INSTANCES=2
        GPU_ALLOCATIONS=("0,1" "2,3")
        echo "Mode: 4 GPUs, 2 models (2 GPUs each) - for large models"
        # Set environment variable for reward_function.py
        export LLM_JUDGE_MODE="multi-4gpu-2model"
        ;;
    "multi-4gpu-4model")
        NUM_INSTANCES=4
        GPU_ALLOCATIONS=("0" "1" "2" "3")
        echo "Mode: 4 GPUs, 4 models (1 GPU each) - for small models"
        # Set environment variable for reward_function.py
        export LLM_JUDGE_MODE="multi-4gpu-4model"
        ;;
    "single")
        echo "Mode: Single instance on all GPUs"
        export LLM_JUDGE_MODE="single"
        ;;
    *)
        echo "Error: Invalid MODE. Use: single, multi-4gpu-2model, or multi-4gpu-4model"
        exit 1
        ;;
esac

# Main execution based on mode
if [ "$MODE" != "single" ]; then
    start_multi_instances
    # For multi mode, we need to track multiple PIDs
    WAIT_FOR_MULTI=true
else
    start_single_instance
    WAIT_FOR_MULTI=false
fi

# Function to check if server is ready
wait_for_server() {
    local port=${1:-$PORT}
    local instance_name=${2:-""}
    local max_attempts=60
    local attempt=0
    local sleep_time=5
    
    if [ -n "$instance_name" ]; then
        echo "Waiting for vLLM $instance_name to be ready..."
    else
        echo "Waiting for vLLM server to be ready..."
    fi
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "vLLM server is ready on port $port!"
            return 0
        fi
        echo "Waiting... (attempt $((attempt+1))/$max_attempts)"
        sleep $sleep_time
        ((attempt++))
    done
    
    echo "Error: vLLM server failed to start after $max_attempts attempts"
    return 1
}

# Wait for servers to be ready and show final status
if [ "$WAIT_FOR_MULTI" = true ]; then
    # Multi instance mode
    echo ""
    echo "Waiting for all instances to be ready..."
    all_ready=true
    
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        port=$((PORT + i))
        if ! wait_for_server $port "instance $i"; then
            all_ready=false
        fi
    done
    
    if [ "$all_ready" = true ]; then
        echo ""
        echo "=========================================="
        echo "All vLLM Instances Started Successfully!"
        echo "=========================================="
        echo ""
        echo "Active instances:"
        for i in $(seq 0 $((NUM_INSTANCES - 1))); do
            port=$((PORT + i))
            pid=$(cat "vllm_judge_instance_${i}.pid" 2>/dev/null || echo "unknown")
            echo "  Instance $i: http://localhost:$port (PID: $pid, GPUs: ${GPU_ALLOCATIONS[$i]})"
        done
        echo ""
        echo "To stop all instances, run:"
        echo "  ./stop_vllm_judge.sh"
        echo ""
        echo "To test an instance:"
        echo "  curl http://localhost:$PORT/v1/models"
    else
        echo "Error: Some instances failed to start"
        exit 1
    fi
else
    # Single instance mode
    if wait_for_server $PORT; then
        echo ""
        echo "=========================================="
        echo "vLLM Judge Service Started Successfully!"
        echo "=========================================="
        echo "Service URL: http://localhost:$PORT"
        echo "PID: $VLLM_PID (saved in vllm_judge.pid)"
        echo ""
        echo "To stop the service, run:"
        echo "  ./stop_vllm_judge.sh"
        echo "Or manually:"
        echo "  kill \$(cat vllm_judge.pid)"
        echo ""
        echo "To test the service:"
        echo "  curl http://localhost:$PORT/v1/models"
    else
        kill $VLLM_PID 2>/dev/null || true
        rm -f vllm_judge.pid
        exit 1
    fi
fi