#!/bin/bash
# Example: Single Model Local Evaluation
# This example shows how to evaluate a single verifier model using local vLLM inference


# Run evaluation
bash serve_eval_verifier.sh \
    "Qwen/Qwen2.5-7B-Instruct" \
    0 0 \
    8011 \
    single_verification \
    0.0 1.0 1 \
    "$BASE_PATH" \
    "Qwen_Qwen2.5_3B_Instruct_math_subsample" \
    24000 128 0.85 1 \
    local vanilla

# Parameters explained:
# - Verifier Model: Qwen/Qwen2.5-7B-Instruct (single model)
# - GPUs: 0-7 (using 8 GPUs)
# - Port: 8011 (using 8011-8018 for the 8 GPU instances)
# - Protocol: single_verification (binary correct/incorrect)
# - Sampling: temp=0.0, top_p=1.0, num_seq=1 (deterministic/greedy)
# - Dataset: generator model name + domain name
# - Model length: 24000 tokens
# - Workers: 128 parallel processes
# - GPU memory: 85%
# - Tensor parallel: 1 (single GPU)
# - Provider: local (vLLM)
# - Strategy: vanilla prompting

