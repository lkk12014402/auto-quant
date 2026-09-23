export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn


MODEL=/models/lkk/cri_models/Qwen3.5-35B-A3B
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.8,dtype=bfloat16,trust_remote_code=True,add_bos_token=True,enable_prefix_caching=False,enable_thinking=False,reasoning_parser=qwen3,max_gen_toks=2048"


lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k \
  --batch_size 64 \
  --seed 42 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path lm_eval_results


lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,hellaswag \
  --batch_size 64 \
  --seed 42 \
  --output_path lm_eval_results
