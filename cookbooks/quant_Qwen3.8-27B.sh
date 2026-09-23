
export CUDA_VISIBLE_DEVICES=1


AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round ./Qwen3.8-27B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "visual,lm_head,embed_tokens" \
  --format llm_compressor \
  --device_map auto \
  --output_dir "./qwen3.8-27B-MXFP8-model-free"

