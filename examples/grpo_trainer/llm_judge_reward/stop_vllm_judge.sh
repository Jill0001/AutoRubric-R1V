#!/usr/bin/env bash
# Stop vLLM judge service (supports both single and multi instance modes)

set -e

# Function to kill process and all its children
kill_process_tree() {
    local pid=$1
    local children=$(pgrep -P $pid 2>/dev/null || true)
    
    # Kill children first
    for child in $children; do
        kill_process_tree $child
    done
    
    # Then kill the parent
    if kill -0 $pid 2>/dev/null; then
        kill $pid 2>/dev/null || true
    fi
}

# Function to find and kill all vLLM processes
kill_all_vllm_processes() {
    # Find all vLLM processes (including those started with vllm serve)
    local vllm_pids=$(pgrep -f "vllm serve|ray::RayWorkerWrapper" | grep -v grep | grep -v $$ || true)
    
    if [ -n "$vllm_pids" ]; then
        echo "Found vLLM processes: $vllm_pids"
        for pid in $vllm_pids; do
            if kill -0 $pid 2>/dev/null; then
                echo "Killing process $pid..."
                kill $pid 2>/dev/null || true
            fi
        done
        
        # Wait a moment for processes to terminate
        sleep 2
        
        # Force kill any remaining processes
        local remaining=$(pgrep -f "vllm serve|ray::RayWorkerWrapper" | grep -v grep | grep -v $$ || true)
        if [ -n "$remaining" ]; then
            echo "Force killing remaining processes: $remaining"
            for pid in $remaining; do
                kill -9 $pid 2>/dev/null || true
            done
        fi
        
        echo "All vLLM processes stopped"
        return 0
    else
        echo "No vLLM processes found"
        return 1
    fi
}

# Check for multi-instance PID files first
multi_instance_found=false
for pid_file in vllm_judge_instance_*.pid; do
    if [ -f "$pid_file" ]; then
        multi_instance_found=true
        break
    fi
done

if [ "$multi_instance_found" = true ]; then
    # Multi instance mode
    echo "Detected multi-instance mode, stopping all instances..."
    
    for pid_file in vllm_judge_instance_*.pid; do
        if [ -f "$pid_file" ]; then
            instance_id=$(echo "$pid_file" | sed 's/vllm_judge_instance_\(.*\)\.pid/\1/')
            pid=$(cat "$pid_file" 2>/dev/null)
            
            if [ -n "$pid" ]; then
                if kill -0 $pid 2>/dev/null; then
                    echo "Stopping instance $instance_id (PID: $pid)..."
                    kill_process_tree $pid
                else
                    echo "Instance $instance_id (PID: $pid) is not running"
                fi
            fi
            
            # Remove PID file
            rm -f "$pid_file"
        fi
    done
    
    # Also check for any remaining vLLM processes
    echo "Checking for remaining vLLM processes..."
    kill_all_vllm_processes || true
    
    # Clean up multi-instance log files
    if ls vllm_judge_instance_*.log 1> /dev/null 2>&1; then
        echo "Removing instance log files..."
        rm -f vllm_judge_instance_*.log
        echo "Log files removed"
    fi
    
elif [ -f vllm_judge.pid ]; then
    # Single instance mode
    echo "Detected single instance mode..."
    PID=$(cat vllm_judge.pid)
    
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping vLLM judge service (main PID: $PID)..."
        
        # Kill the process tree (main process and all children)
        kill_process_tree $PID
        
        # Also check for any other vLLM processes that might have been spawned
        echo "Checking for additional vLLM processes..."
        kill_all_vllm_processes || true
        
        echo "vLLM judge service stopped"
    else
        echo "Process $PID is not running"
        echo "Checking for other vLLM processes..."
        kill_all_vllm_processes || true
    fi
    
    # Clean up PID file
    rm -f vllm_judge.pid
    
    # Clean up log file
    if [ -f vllm_judge.log ]; then
        echo "Removing log file..."
        rm -f vllm_judge.log
        echo "Log file removed"
    fi
else
    echo "No PID files found. Service may not be running."
    echo "Checking for vLLM processes..."
    
    if kill_all_vllm_processes; then
        exit 0
    fi
fi

echo "Cleanup complete"