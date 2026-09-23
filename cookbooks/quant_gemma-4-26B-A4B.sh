
export CUDA_VISIBLE_DEVICES=2

AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round \
  --model_name /models/lkk/cri_models/gemma-4-26B-A4B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "router,vision_tower,embed_vision,multi_modal_projector" \
  --format llm_compressor \
  --output_dir ./gemma-4-26B-A4B-MXFP8

exit



AR_MODEL_FREE_SHARD_PARALLELISM=1 auto-round \
  --model_name /models/lkk/cri_models/gemma-4-26B-A4B \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers "router,vision_tower,embed_vision,multi_modal_projector" \
  --layer_config "{experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir ./gemma-4-26B-A4B-MXFP4


