#!/bin/bash
# Example: Using API Provider (OpenAI/Anthropic/Together)
# This example shows how to evaluate models using API providers instead of local inference


# Set your API key as environment variable
# export OPENAI_API_KEY="your-api-key-here"
# export ANTHROPIC_API_KEY="your-api-key-here"
# export TOGETHER_API_KEY="your-api-key-here"

# Run evaluation using API provider
bash serve_eval_verifier.sh \
    "gpt-4o" \
    0 0 \
    8011 \
    single_verification \
    0.0 1.0 1 \
    "$BASE_PATH" \
    "Qwen_Qwen2.5_3B_Instruct_math_subsample" \
    24000 128 0.85 1 \
    openai vanilla 

# Parameters explained:
# - Verifier Model: gpt-4o (OpenAI model name)
# - GPUs: 0-0 (ignored for API providers, but required parameter)
# - Port: 8011 (ignored for API providers)
# - Protocol: single_verification (binary correct/incorrect)
# - Sampling: temp=0.0, top_p=1.0, num_seq=1 (deterministic)
# - Dataset: generator model name + domain name
# - Model length: 24000 tokens
# - Workers: 128 (ignored for API providers)
# - GPU memory: 0.85 (ignored for API providers)
# - Tensor parallel: 1 (ignored for API providers)
# - Provider: openai (can also use "anthropic" or "together")
# - Strategy: vanilla prompting

# Note: When using API providers, vLLM servers are NOT started.
# The script will use the API directly instead.

