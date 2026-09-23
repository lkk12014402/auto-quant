#!/usr/bin/env bash
# Qwen3.8-27B — AutoRound MXFP8 / MXFP4 量化（方案与依据见 Qwen3.8-27B-quantization-recipe.md）
#
#   ./quant_Qwen3.8-27B-mxfp.sh mxfp8              # 方案1 全 MXFP8, model_free RTN（默认，建下限）
#   ./quant_Qwen3.8-27B-mxfp.sh mxfp4_mixed        # 方案2 MLP=MXFP4 + attention=MXFP8, model_free
#   ./quant_Qwen3.8-27B-mxfp.sh mxfp4              # 方案3 全 MXFP4, model_free RTN（激进）
#   ./quant_Qwen3.8-27B-mxfp.sh mxfp4_mixed tune   # 方案4 同上 + SignRound 调优(iters=200)
#   DRY_RUN=1 ./quant_Qwen3.8-27B-mxfp.sh mxfp8    # 只跑前置审计 + 打印命令，不动 GPU
#
# 环境: /models/miniforge3/envs/muse_verify（auto-round 0.16.0 editable @ /models/lkk/auto-round，
#       transformers 5.16.1 认 qwen3_5，校准集 NeelNanda/pile-10k 已在 HF_HOME 缓存）
set -euo pipefail

MODE="${1:-mxfp8}"
KIND="${2:-rtn}"

ENVBIN=/models/miniforge3/envs/muse_verify/bin
MODEL=/models/lkk/cri_models/Qwen3.8-27B
OUTROOT=/models/lkk/cri_models
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"   # 启动前自行 nvidia-smi 确认该卡空闲
export AR_MODEL_FREE_SHARD_PARALLELISM=1                   # 控制 model_free 的 host RAM 峰值

# 必保 BF16 的层（依据见文档 §2.3/§3）：
#   visual                 视觉塔（文本评测不走），量化后 vLLM 侧无对应 kernel
#   lm_head embed_tokens   各 1.27B 的头/表，常规保高精度
#   in_proj_a in_proj_b    GDN 小投影 N=48；vLLM 融合成 in_proj_ba N=96，
#                          FlashInferCutedslMxfp8 断言 "mm_mxfp8 requires N >= 128" 直接崩
#   mtp                    1 层 MTP 投机解码模块，主路径不受益，保明文
# 注意：这是 dense 模型，绝不能写 mlp.gate —— 子串匹配会误伤 mlp.gate_proj。
IGNORE="visual,lm_head,embed_tokens,mtp,in_proj_a,in_proj_b"
LC_ATTENTION_MXFP8='{"self_attn":{"bits":8,"data_type":"mx_fp","group_size":32,"act_bits":8,"act_data_type":"mx_fp","act_group_size":32},"linear_attn":{"bits":8,"data_type":"mx_fp","group_size":32,"act_bits":8,"act_data_type":"mx_fp","act_group_size":32}}'

case "$MODE" in
  mxfp8)       SCHEME=MXFP8; LAYER_CONFIG="" ;;
  mxfp4)       SCHEME=MXFP4; LAYER_CONFIG="" ;;
  mxfp4_mixed) SCHEME=MXFP4; LAYER_CONFIG="$LC_ATTENTION_MXFP8" ;;
  *) echo "unknown MODE=$MODE (mxfp8|mxfp4_mixed|mxfp4)"; exit 1 ;;
esac
case "$KIND" in rtn|tune) : ;; *) echo "unknown KIND=$KIND (rtn|tune)"; exit 1 ;; esac

# 注意：不要用 `X="$( [ cond ] && echo -Foo )"` 之类的写法——命令替换返回非 0 会让
# set -e 直接终止脚本（本脚本 v1 就是这么静默挂掉的）。用显式 if。
SUFFIX=""
if [ "$MODE" = mxfp4_mixed ]; then SUFFIX="-Mixed"; fi
if [ "$KIND" = tune ]; then SUFFIX="${SUFFIX}-Tuned"; fi

OUT="$OUTROOT/Qwen3.8-27B-${SCHEME}${SUFFIX}-AutoRound"
LOG="$OUTROOT/quant_$(basename "$OUT").log"
AUDIT="$OUTROOT/audit_$(basename "$OUT").log"

# 第一个元素必须是程序名——否则 "${CMD[@]}" 会把 --model_name 当命令执行
CMD=(auto-round --model_name "$MODEL" --scheme "$SCHEME" --ignore_layers "$IGNORE"
     --format llm_compressor --output_dir "$OUT")
if [ -n "$LAYER_CONFIG" ]; then CMD+=(--layer_config "$LAYER_CONFIG"); fi
if [ "$KIND" = tune ]; then
  CMD+=(--iters 200 --nsamples 128 --seqlen 2048 --enable_alg_ext
        --gradient_accumulate_steps 8 --batch_size 1 --low_gpu_mem_usage)
else
  CMD+=(--model_free)
fi
CMD+=(--device_map auto)

command -v auto-round >/dev/null 2>&1 || { echo "!! PATH 里找不到 auto-round（env: $ENVBIN）"; exit 1; }

echo "### 量化前审计：逐张量精度分配 + vLLM MX kernel 约束  -> $AUDIT"
audit_args=(--model "$MODEL" --scheme "$SCHEME" --ignore_layers "$IGNORE")
if [ -n "$LAYER_CONFIG" ]; then audit_args+=(--layer_config "$LAYER_CONFIG"); fi
# 审计不过（例如 in_proj_a/b 漏保护）就不许启动量化
if ! python "$HERE/audit_precision_plan.py" "${audit_args[@]}" > "$AUDIT" 2>&1; then
  cat "$AUDIT"
  echo "!! 前置审计失败，见 $AUDIT（检查 ignore_layers / layer_config）"; exit 2
fi
tail -n 30 "$AUDIT"

echo
disp=""
for a in "${CMD[@]}"; do case "$a" in *\{*) a="'$a'";; esac; disp="$disp $a"; done
echo "### 命令:$disp"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "### DRY_RUN=1，未启动量化"
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "$LOG"
AR_RC="${PIPESTATUS[0]}"
if [ "$AR_RC" -ne 0 ]; then echo "!! auto-round 退出码 $AR_RC，见 $LOG"; exit "$AR_RC"; fi

echo "### 导出后核对"
# 1) model_free 必须真免加载：静默回退会在 day-0 架构上炸，也说明配置不被支持
if [ "$KIND" != tune ] && grep -qi "falling back to the regular AutoRound flow" "$LOG"; then
  echo "!! 日志出现 model_free 静默回退，产物不是免加载路径，请检查 scheme/格式组合"; exit 3
fi
# 2) 导出的 quantization_config 必须与实际落盘权重一致，否则 vLLM 按量化层构建明文权重 -> KeyError
python "$HERE/check_quant_export.py" "$OUT"
echo "### done -> $OUT  (log: $LOG)"
echo "### 下一步: ./eval_Qwen3.8-27B-quant.sh $OUT"
