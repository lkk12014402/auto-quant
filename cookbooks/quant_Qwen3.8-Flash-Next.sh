
export CUDA_VISIBLE_DEVICES=1


AR_MODEL_FREE_SHARD_PARALLELISM=1 auto_round \
  --model_name ./Qwen3.8-Flash-Next \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers \
    "visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection" \
  --format llm_compressor \
  --output_dir ./Qwen3.8-Flash-Next-MXFP8

