
export CUDA_VISIBLE_DEVICES=0


AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round \
  /models/lkk/cri_models/Qwen3.5-35B-A3B \
  --iters 0 \
  --static_kv_dtype fp8 \
  --scheme MXFP8 \
  --ignore_layers "lm_head,embed_tokens,visual,in_proj_a,in_proj_b,shared_expert_gate" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./Qwen3.5-35B-A3B-MXFP4-Mixed-FP8KV



AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round \
  /models/lkk/cri_models/Qwen3.5-35B-A3B \
  --iters 0 \
  --static_kv_dtype fp8 \
  --static_attention_dtype fp8 \
  --scheme MXFP8 \
  --ignore_layers "lm_head,embed_tokens,visual,in_proj_a,in_proj_b,shared_expert_gate" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./Qwen3.5-35B-A3B-MXFP4-Mixed-FP8KV-FP8Attn


exit

AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round \
  /models/lkk/cri_models/Qwen3.5-35B-A3B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "lm_head,embed_tokens,visual,in_proj_a,in_proj_b" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./Qwen3.5-35B-A3B-MXFP4-Mixed
