#!/usr/bin/env bash
# Qwen3.8-27B 量化产物验证 + 基准评测
#
#   ./eval_Qwen3.8-27B-quant.sh <ckpt>                    # 正式跑：只跑基准，不冒烟（默认）
#   RUN_SMOKE=1 ./eval_Qwen3.8-27B-quant.sh <ckpt>        # 先冒烟再基准（会多完整加载一遍权重）
#   SMOKE_ONLY=1 ./eval_Qwen3.8-27B-quant.sh <ckpt>       # 只冒烟，验加载/生成，不跑基准
#   RUN_GSM8K=1 ./eval_Qwen3.8-27B-quant.sh <ckpt>        # 顺带跑 gsm8k（需 chat template 多轮口径）
#   TASKS="piqa" LIMIT=16 ./eval_Qwen3.8-27B-quant.sh <ckpt>
#   DRY_RUN=1 ./eval_Qwen3.8-27B-quant.sh <ckpt>          # 只打印参数组合，不动 GPU
#
# 依据: 现有 eval_Qwen3.8-27B-mxfp8.sh 的 vLLM 参数（language_model_only=True 走纯文本路径，
# reasoning_parser=qwen3 + enable_thinking=False 与 baseline 口径一致）
set -euo pipefail

CKPT="${1:?usage: $0 <quantized_ckpt_dir>}"
# 相对路径/尾斜杠都接得住：vLLM 的 spawn 子进程按 cwd 解析，统一转绝对路径最稳
CKPT="$(realpath "$CKPT")"
ENVBIN="${ENVBIN:-/models/miniforge3/envs/muse_verify/bin}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS="${NUM_GPUS:-1}"
TASKS="${TASKS:-piqa,mmlu,hellaswag}"
LIMIT="${LIMIT:-}"
OUTDIR="${OUTDIR:-/models/lkk/cri_models/lm_eval_results}"
# ---- 开关（默认值 = 正式跑的形态：不冒烟、不跑 gsm8k）----
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_GSM8K="${RUN_GSM8K:-0}"
[ "${SMOKE_ONLY:-0}" = 1 ] && RUN_SMOKE=1

COMMON="pretrained=${CKPT},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=${GPU_MEM:-0.85},dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=qwen3,language_model_only=True,enable_thinking=False"

echo "### 环境: $ENVBIN | CKPT: $CKPT | GPUS=$CUDA_VISIBLE_DEVICES TP=$NUM_GPUS"
echo "### 开关: RUN_SMOKE=$RUN_SMOKE SMOKE_ONLY=${SMOKE_ONLY:-0} RUN_GSM8K=$RUN_GSM8K | TASKS=$TASKS${LIMIT:+ LIMIT=$LIMIT}"
echo "### model_args: $COMMON"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "### DRY_RUN=1，不启动 vLLM/lm_eval"; exit 0; fi

if [ "$RUN_SMOKE" = 1 ]; then
  # 必须用独立脚本文件：vLLM 的 spawn 子进程要重新 exec __main__，
  # `python - <<PY` 会让子进程去找 <stdin> 而失败（与被测产物无关，坑过一次）
  echo "### 冒烟：vLLM 加载 + 3 条确定性生成（profile_run 会过一遍全部 Linear，kernel 维度不兼容在这崩）"
  GPU_MEM="${GPU_MEM:-0.85}" MAX_MODEL_LEN=4096 NUM_GPUS="$NUM_GPUS" \
    python "$HERE/smoke_vllm_ckpt.py" "$CKPT" "$NUM_GPUS"
  if [ "${SMOKE_ONLY:-0}" = 1 ]; then echo "### SMOKE_ONLY=1，跳过基准评测"; exit 0; fi
fi

if [ "$RUN_GSM8K" = 1 ]; then
  echo "### gsm8k（chat template + fewshot 多轮，口径与 baseline 一致）"
  lm_eval --model vllm --model_args "$COMMON" --tasks gsm8k --batch_size 64 --seed 42 \
    --apply_chat_template --fewshot_as_multiturn ${LIMIT:+--limit $LIMIT} --output_path "$OUTDIR"
fi

echo "### piqa / mmlu / hellaswag"
lm_eval --model vllm --model_args "$COMMON" --tasks "$TASKS" --batch_size 64 --seed 42 \
  ${LIMIT:+--limit $LIMIT} --output_path "$OUTDIR"

echo "### done -> $OUTDIR"
