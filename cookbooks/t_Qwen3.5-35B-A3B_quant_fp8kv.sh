#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODELS=(
    "/models/lkk/cri_models/Qwen3.5-35B-A3B-MXFP4-Mixed-FP8KV/Qwen3.5-35B-A3B-mxfp-w8g32/"
    "/models/lkk/cri_models/Qwen3.5-35B-A3B-MXFP4-Mixed-FP8KV-FP8Attn/Qwen3.5-35B-A3B-mxfp-w8g32/"
)

run_eval() {
    MODEL=$1
    MODEL_NAME=$(basename "$MODEL")

    echo "======================================"
    echo "Evaluating ${MODEL_NAME}"
    echo "======================================"

    MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.8,dtype=bfloat16,trust_remote_code=True,add_bos_token=True,enable_prefix_caching=False,enable_thinking=False,reasoning_parser=qwen3,max_gen_toks=2048,kv_cache_dtype=fp8"

    OUTPUT_DIR="lm_eval_results/${MODEL_NAME}"

    # GSM8K
    lm_eval \
      --model vllm \
      --model_args "$MODEL_ARGS" \
      --tasks gsm8k \
      --batch_size 64 \
      --seed 42 \
      --apply_chat_template \
      --fewshot_as_multiturn \
      --output_path "${OUTPUT_DIR}"

    # Commonsense + Knowledge
    lm_eval \
      --model vllm \
      --model_args "$MODEL_ARGS" \
      --tasks piqa,mmlu,hellaswag \
      --batch_size 64 \
      --seed 42 \
      --output_path "${OUTPUT_DIR}"
}

for MODEL in "${MODELS[@]}"; do
    run_eval "$MODEL"
done
