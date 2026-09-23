
export CUDA_VISIBLE_DEVICES=1

AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round \
  /models/lkk/cri_models/gemma-4-31B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "lm_head,embed_tokens,vision_tower,embed_vision,multi_modal_projector" \
  --format llm_compressor \
  --output_dir ./gemma-4-31B-MXFP8

