export CUDA_VISIBLE_DEVICES=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn


MODEL=/models/lkk/cri_models/qwen3.8-27B-MXFP8-model-free
NUM_GPUS=1
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.85,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=qwen3,language_model_only=True,enable_thinking=False"


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
