export CUDA_VISIBLE_DEVICES=7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
#MODEL=llm-compressor/examples/transform/data-quip-w4a16/
MODEL=./Qwen3-0.6B_autoround_rtn_mxfp4_quarot/Qwen3-0.6B-mxfp-w4g32/
MODEL=./Qwen3-0.6B_autoround_rtn_mxfp4/Qwen3-0.6B-mxfp-w4g32/
MODEL=./Qwen3.6-27b_autoround_rtn_mxfp4/Qwen3.6-27B-mxfp-w4g32
MODEL=./Qwen3.6-27b_autoround_rtn_mxfp4_ht/Qwen3.6-27B-mxfp-w4g32
MODEL=./Qwen3.6-27b_autoround_rtn_mxfp4_ht_iters200/Qwen3.6-27B-mxfp-w4g32
MODEL=./Qwen3.8-27b_autoround_rtn_mxfp4_ht/Qwen3.8-27B-mxfp-w4g32/
MODEL=/home/kaokaolv/my_commit/Qwen3.8-27b_autoround_iters200_mxfp4_ht/Qwen3.8-27B-mxfp-w4g32
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,enable_thinking=False"


lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks gsm8k \
  --batch_size 64 \
  --gen_kwargs "max_gen_toks=2048" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path lm_eval_results


lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks piqa,hellaswag,mmlu \
  --batch_size 64 \
  --seed 42 \
  --output_path lm_eval_results
