export CUDA_VISIBLE_DEVICES=5
export VLLM_WORKER_MULTIPROC_METHOD=spawn



NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}


MODEL=./Qwen3.5-9b_autoround_rtn_mxfp8/Qwen3.5-9B-mxfp-w8g32/
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=qwen3,enable_thinking=False"


lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,hellaswag \
  --batch_size 64 \
  --seed 42 \
  --output_path lm_eval_results


lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k \
  --batch_size 64 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --seed 42 \
  --output_path lm_eval_results
