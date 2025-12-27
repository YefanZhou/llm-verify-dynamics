#!/bin/bash
set -Eeuo pipefail

# Display usage information
show_usage() {
    cat << EOF
Usage: $0 JUDGE_MODELS [START_GPU] [END_GPU] [BASE_PORT] [EVAL_PROTOCOL] [TEMP] [TOP_P] [NUM_SEQ] [BASE_PATH] [DATASETS] [MODEL_LEN] [NUM_PROC] [GPU_MEM] [TP] [PROVIDER] [PROMPT_STRATEGY]

REQUIRED:
  JUDGE_MODELS          Comma-separated list of judge models to evaluate
                        Example: "Qwen/Qwen3-4B" or "model1,model2,model3"

OPTIONAL (with defaults):
  START_GPU             Starting GPU ID (default: 0)
  END_GPU               Ending GPU ID (default: 7)
  BASE_PORT             Base port for vLLM servers (default: 8011)
  EVAL_PROTOCOL         Evaluation protocol: "single_verification" or "pairwise" (default: pairwise)
  TEMP                  Temperature for sampling (default: 0.7)
  TOP_P                 Top-p for sampling (default: 1.0)
  NUM_SEQ               Number of sequences to generate (default: 8)
  BASE_PATH             Base path for project (default: current directory)
  DATASETS              Comma-separated dataset names (default: "math_test")
  MODEL_LEN             Maximum model length (default: 24000)
  NUM_PROC              Number of processes for parallel processing (default: 128)
  GPU_MEM               GPU memory utilization (0.0-1.0, default: 0.85)
  TP                    Tensor parallel size (default: 1)
  PROVIDER              API provider: "local" or API name (default: local)
  PROMPT_STRATEGY       Prompting strategy (default: vanilla)

EXAMPLES:
  # Single model, single GPU, single verification
  $0 "Qwen/Qwen3-4B" 0 0 8011 single_verification 0.0 1.0 1 /path/to/project dataset_name

  # Multiple models across 8 GPUs, pairwise evaluation
  $0 "model1,model2,model3" 0 7 8011 pairwise 0.7 1.0 8 /path/to/project ""

  # Using API provider instead of local vLLM
  $0 "gpt-4" 0 0 8011 single_verification 0.0 1.0 1 /path/to/project dataset_name 24000 128 0.85 1 openai

NOTES:
  - Number of GPUs (END_GPU - START_GPU + 1) must be divisible by TP
  - Each model instance will use TP GPU(s)
EOF
}

# Check arguments
if [ $# -lt 1 ] || [ $# -gt 16 ]; then
    echo "❌ ERROR: Invalid number of arguments (got $#, need 1-16)"
    echo ""
    show_usage
    exit 1
fi


judge_models_input=$1
start_gpu=${2:-0}
end_gpu=${3:-7}
base_port=${4:-8011}
evaluation_protocol=${5:-pairwise}
temperature=${6:-0.7}
top_p=${7:-1.0}
num_sequences=${8:-8}
base_path=${9:-''}
datasets_arg=${10:-"math_test"}
model_len=${11:-24000}
num_proc=${12:-128}
gpu_memory=${13:-0.85}
tp=${14:-1}  # Single GPU per instance for judge models
provider=${15:-'local'} 
prompt_strategy=${16:-'vanilla'}

echo "=== INPUT VERIFICATION (15 args) ==="
echo "1-4: Models='$judge_models_input' | GPUs=$start_gpu-$end_gpu | Port=$base_port"
echo "5-8: Protocol='$evaluation_protocol' | temp=$temperature | top_p=$top_p | seqs=$num_sequences"
echo "9-12: base_path='$base_path' | datasets='$datasets_arg' | model_len=$model_len | proc=$num_proc"
echo "13-14: gpu_mem=$gpu_memory | tp=$tp | provider=$provider prompt_strategy=$prompt_strategy"
echo "====================================="


log_dir="${base_path}/pipeline_logs_judge"
mkdir -p "$log_dir"

run_id=$(date +%Y%m%d_%H%M%S)_$$
log_file="${log_dir}/multi_judge_eval_${run_id}.log"

echo "=== MULTI-MODEL JUDGE EVALUATION PIPELINE ==="
echo "Run ID: ${run_id}"
echo "Log file: ${log_file}"
echo "Starting at: $(date)"
echo "=================================================="

# Start logging to file while showing output in console
exec > >(tee -a "${log_file}") 2>&1

echo "Logging started. All output will be saved to: ${log_file}"
echo "Run ID: ${run_id}"
echo "Command: $0 $*"
echo "Started at: $(date)"
echo ""


# Parse models and datasets into arrays
IFS=',' read -ra MODEL_ARRAY <<< "$judge_models_input"
IFS=',' read -ra DATASET_ARRAY <<< "$datasets_arg"
num_models=${#MODEL_ARRAY[@]}
num_datasets=${#DATASET_ARRAY[@]}


# Environment setup


export PYTHONUNBUFFERED=1
export SRC_PATH=${base_path}

# Set paths
JUDGE_SRC_PATH=${base_path}

# Create ports string
num_gpus=$((end_gpu - start_gpu + 1))
if [ $((num_gpus % tp)) -ne 0 ]; then
    echo "❌ ERROR: Number of GPUs ($num_gpus) must be divisible by tensor parallel size ($tp)"
    echo "Each model instance needs $tp GPU(s)"
    exit 1
fi

num_instances=$((num_gpus / tp))

ports_array=()
for ((i=0; i<num_instances; i++)); do
    ports_array+=($((base_port + i)))
done
JUDGE_PORTS=$(IFS=','; echo "${ports_array[*]}")

# Create sampling parameters string
sampling_params="${evaluation_protocol}/temp${temperature}_topp${top_p}_seqs${num_sequences}"

echo "=== MULTI-MODEL JUDGE EVALUATION PIPELINE ==="
echo "Judge Models to evaluate: $num_models"
for ((i=0; i<num_models; i++)); do
    echo "  $((i+1)). ${MODEL_ARRAY[i]}"
done
echo "GPUs: $start_gpu to $end_gpu ($num_gpus GPUs)"
echo "Base port: $base_port"
echo "Ports: $JUDGE_PORTS"
echo "Evaluation protocol: $evaluation_protocol"
echo "Sampling: temperature=$temperature, top_p=$top_p, num_sequences=$num_sequences"
echo "Sampling folder: $sampling_params"
echo "Base path: $base_path"
echo "Judge source path: $JUDGE_SRC_PATH"
echo "Model length: $model_len"
echo "Datasets: ${DATASET_ARRAY[*]} (total: $num_datasets)"
echo "=================================================="

# Array to store process IDs for current model
declare -a PIDS=()

# Cleanup function for current model servers
cleanup_current_model() {
    if [ "$provider" = "local" ] && [ ${#PIDS[@]} -gt 0 ]; then
        echo "  Stopping ${#PIDS[@]} vLLM processes..."
        
        # Send TERM signal first (graceful)
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "    Stopping PID $pid (graceful)..."
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        
        # Wait a bit for graceful shutdown
        sleep 3
        
        # Send KILL signal for stubborn processes
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "    Force killing PID $pid..."
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        
        # Wait for all processes to finish
        for pid in "${PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        
        echo "  ✓ All current model servers stopped"
    fi
    
    # Also kill any remaining vllm processes (safety net)
    if [ "$provider" = "local" ]; then
        pkill -f "vllm serve" 2>/dev/null || true
    fi
    
    # Clear PID array
    PIDS=()
}

# Global cleanup function
cleanup_all() {
    echo ""
    if [ "$provider" = "local" ]; then
        echo "[CLEANUP] Stopping all vLLM servers..."
        cleanup_current_model
    fi
    echo "[CLEANUP] Complete"
}

# Set up trap to cleanup on script exit
trap cleanup_all EXIT INT TERM

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout=1200  # 5 minutes
    local elapsed=0
    
    echo "    Waiting for server on port $port..."
    
    while [ $elapsed -lt $timeout ]; do
        if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1 || \
           curl -s -f "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "    ✓ Server on port $port is ready! (${elapsed}s)"
            return 0
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
        
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "      Still waiting for port $port... (${elapsed}s elapsed)"
        fi
    done
    
    echo "    ✗ Timeout waiting for server on port $port"
    return 1
}

# Function to launch vLLM servers for judge model
launch_judge_servers() {
    local model_name=$1
    
    echo "  Starting vLLM servers for judge model: $model_name"
    
    # Set model-specific max_model_len
    local current_model_len=$model_len
    if [[ "$model_name" == "google/gemma-2-9b-it" ]] || [[ "$model_name" == "google/gemma-2-27b-it" ]] || [[ "$model_name" == "google/gemma-2-2b-it" ]]; then
        echo "    Detected google/gemma-2/9/27b-it, setting max_model_len to 8192"
        current_model_len=8192
    fi
    
    # Check for model-specific configurations
    local additional_args=""
    if [[ "$model_name" == *"mistral"* ]] || [[ "$model_name" == *"Mistral"* ]]; then
        echo "    Detected Mistral model, adding --tokenizer-mode mistral"
        additional_args="--tokenizer-mode mistral"
    fi

    if [[ "$model_name" == *"gemma-3"* ]] || [[ "$model_name" == *"Gemma-3"* ]]; then
        echo "    Detected Gemma-3 model, adding specific configuration flags"
        additional_args="$additional_args --chat-template-content-format openai"
        additional_args="$additional_args --limit-mm-per-prompt image=0"
    fi
    
    for ((instance=0; instance<num_instances; instance++)); do
        port=$((base_port + instance))
        start_gpu_for_instance=$((start_gpu + instance * tp))
        end_gpu_for_instance=$((start_gpu_for_instance + tp - 1))
        
        gpu_list=""
        for ((gpu=start_gpu_for_instance; gpu<=end_gpu_for_instance; gpu++)); do
            if [ -z "$gpu_list" ]; then
                gpu_list="$gpu"
            else
                gpu_list="$gpu_list,$gpu"
            fi
        done

        echo "    Launching instance $((instance + 1)) on GPUs $gpu_list (port $port)..."
        
        # Start vLLM server
        CUDA_VISIBLE_DEVICES=$gpu_list vllm serve "$model_name" \
            --tensor_parallel_size=$tp \
            --max_model_len=$current_model_len \
            --gpu_memory_utilization=$gpu_memory \
            --disable-log-stats \
            --disable-log-requests \
            --port $port \
            --trust-remote-code \
            --dtype bfloat16 \
            $additional_args &
        
        # Store PID
        PIDS+=($!)
        echo "      Started instance $((instance + 1)) with PID ${PIDS[-1]} (port $port, GPUs: $gpu_list)"
        
        # Small delay to prevent resource conflicts
        sleep 20
    done
    
    echo "  All judge servers launched! PIDs: ${PIDS[*]}"
}

# Function to wait for all servers to be ready
wait_for_all_servers() {
    echo "  Waiting for all judge servers to be ready..."
    local all_ready=true
    
    for ((instance=0; instance<num_instances; instance++)); do
        port=$((base_port + instance))
        if ! wait_for_server $port; then
            all_ready=false
            echo "    ❌ Server instance $((instance + 1)) (port $port) failed to start"
        fi
    done
    
    if [ "$all_ready" = true ]; then
        echo "  🎉 All judge vLLM servers are ready!"
        return 0
    else
        echo "  ❌ Some judge servers failed to start"
        return 1
    fi
}

# Function to run judge evaluation
run_judge_evaluation() {
    local model_name=$1
    
    echo "  === RUNNING JUDGE EVALUATION ==="
    echo "  Judge Model: $model_name"
    echo "  Ports: $JUDGE_PORTS"
    echo "  Evaluation Protocol: $evaluation_protocol"
    echo "  Datasets: ${DATASET_ARRAY[*]}"
    
    # Create model-safe name
    local model_safe=$model_name
    
    cd "${JUDGE_SRC_PATH}"
    
    for dataset_name in "${DATASET_ARRAY[@]}"; do
        echo ""
        echo "  Processing dataset: ${dataset_name} at $(date)"
        
        if [ "$provider" = "local" ]; then
            # Check if servers are still alive
            echo "  Checking server health..."
            for pid in "${PIDS[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "❌ vLLM server PID $pid has died. Cannot continue."
                    return 1
                fi
            done
            echo "  ✓ All servers are alive"
        fi

        echo "  Processing dataset: ${dataset_name} at $(date)"
        
        # Create output directory with sampling parameters
        save_dir="${JUDGE_SRC_PATH}/verifier_data/${model_safe}/${sampling_params}/${prompt_strategy}/${dataset_name}"
        mkdir -p "${save_dir}"
        
        # Create log file
        log_file="${save_dir}/execution_log_$(date +%Y%m%d_%H%M%S).log"
        
        echo "  Output directory: ${save_dir}"
        echo "  Logging to: ${log_file}"
        echo "  Starting ${dataset_name} at $(date)" >> "${log_file}"
        
        echo "  Running judge evaluation for ${dataset_name}..."
        echo "  This may take a while..."
        
        # Run judge evaluation with multiple ports
        if [ "$provider" = "local" ]; then
            if python -u verification/main.py \
                --output_dir "${JUDGE_SRC_PATH}/verifier_data" \
                --save_dir "${save_dir}" \
                --judge_model "${model_name}" \
                --evaluation_protocol "${evaluation_protocol}" \
                --prompt_strategy ${prompt_strategy} \
                --dataset_name "${dataset_name}" \
                --base_url "http://localhost:${base_port}/v1" \
                --temperature "${temperature}" \
                --top_p "${top_p}" \
                --ports "${JUDGE_PORTS}" \
                --num_sequences "${num_sequences}" \
                --num_proc ${num_proc} \
                --force_rerun 2>&1 | tee -a "${log_file}"; then
                echo "  ✓ Successfully completed ${dataset_name} at $(date)" | tee -a "${log_file}"
            else
                echo "  ✗ Failed to complete ${dataset_name} at $(date)" | tee -a "${log_file}"
                echo "  Check log file for details: ${log_file}"
                return 1
            fi
        else
            echo "  Using API provider: $provider"
            if python -u verification/main_api.py \
                --output_dir "${JUDGE_SRC_PATH}/verifier_data" \
                --save_dir "${save_dir}" \
                --judge_model "${model_name}" \
                --evaluation_protocol "${evaluation_protocol}" \
                --prompt_strategy ${prompt_strategy} \
                --dataset_name "${dataset_name}" \
                --temperature "${temperature}" \
                --top_p "${top_p}" \
                --num_sequences "${num_sequences}" \
                --num_proc ${num_proc} \
                --provider "${provider}" \
                --force_rerun 2>&1 | tee -a "${log_file}"; then
                echo "  ✓ Successfully completed ${dataset_name} at $(date)" | tee -a "${log_file}"
            else
                echo "  ✗ Failed to complete ${dataset_name} at $(date)" | tee -a "${log_file}"
                echo "  Check log file for details: ${log_file}"
                return 1
            fi
        fi
        
        echo "  Finished ${dataset_name} at $(date)" >> "${log_file}"
        echo "  =================================" >> "${log_file}"
    done
    
    return 0
}

# Main execution loop
echo ""
echo "Starting judge evaluation pipeline..."

# Track results
declare -a SUCCESS_MODELS=()
declare -a FAILED_MODELS=()

for ((model_idx=0; model_idx<num_models; model_idx++)); do
    current_model="${MODEL_ARRAY[model_idx]}"
    
    echo ""
    echo "=================================================="
    echo "EVALUATING JUDGE MODEL $((model_idx + 1))/$num_models: $current_model"
    echo "Provider: $provider"
    echo "=================================================="
    
    if [ "$provider" = "local" ]; then
        # Step 1: Launch vLLM servers
        echo ""
        echo "STEP 1: Launching vLLM servers..."
        if ! launch_judge_servers "$current_model"; then
            echo "❌ Failed to launch servers for $current_model"
            FAILED_MODELS+=("$current_model")
            cleanup_current_model
            continue
        fi
        
        # Step 2: Wait for servers to be ready
        echo ""
        echo "STEP 2: Waiting for servers to be ready..."
        if ! wait_for_all_servers; then
            echo "❌ Servers failed to start for $current_model"
            FAILED_MODELS+=("$current_model")
            cleanup_current_model
            continue
        fi
    else
        echo ""
        echo "STEP 1-2: Skipping vLLM server setup (using $provider API)"
    fi
    
    # Step 3: Run judge evaluation
    echo ""
    echo "STEP 3: Running judge evaluation..."
    if run_judge_evaluation "$current_model"; then
        echo "✅ Successfully completed judge evaluation for $current_model"
        SUCCESS_MODELS+=("$current_model")
    else
        echo "❌ Judge evaluation failed for $current_model"
        FAILED_MODELS+=("$current_model")
    fi
    
    # Step 4: Cleanup current model servers (unless it's the last model)
    if [ $((model_idx + 1)) -lt $num_models ] && [ "$provider" = "local" ]; then
        echo ""
        echo "STEP 4: Stopping current model servers..."
        cleanup_current_model
        echo "✓ Cleanup complete, ready for next model"
        
        # Brief pause before next model
        echo "Waiting 5 seconds before next model..."
        sleep 5
    fi
done

# Final summary
echo ""
echo "=================================================="
echo "JUDGE EVALUATION PIPELINE COMPLETE"
echo "=================================================="
echo "Total models: $num_models"
echo "Successful: ${#SUCCESS_MODELS[@]}"
echo "Failed: ${#FAILED_MODELS[@]}"

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "✅ Successfully evaluated judge models:"
    for model in "${SUCCESS_MODELS[@]}"; do
        echo "  - $model"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed judge models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model"
    done
fi

echo ""
# Final cleanup will be handled by the trap
echo "Pipeline finished!"