#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4_exp, 180B MoE) — AutoRound MXFP8 / MXFP4-Mixed 量化
# 方案与依据见 Qwen3.8-Flash-Next-quantization-recipe.md
#
#   ./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp4_mixed      # 主推：experts MXFP4 + 其余 MXFP8（复现 HF 参考产物）
#   ./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp8           # 全 MXFP8（保守档）
#   ./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp4           # 全 MXFP4（不推荐，见文档 §5）
#   DRY_RUN=1 ./quant_Qwen3.8-Flash-Next-mxfp.sh mxfp4_mixed
#
# 与 Qwen3.8-27B 脚本的差别：这是 MoE + 一堆引擎侧结构模块，ignore 清单长得多，
# 且 experts 以 fused 3-D 存储（auto-round 会拆成 per-expert 再量化）。
set -euo pipefail

MODE="${1:-mxfp4_mixed}"

ENVBIN=/models/miniforge3/envs/muse_verify/bin
MODEL=/models/lkk/cri_models/Qwen3.8-Flash-Next
OUTROOT=/models/lkk/cri_models
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export AR_MODEL_FREE_SHARD_PARALLELISM=1        # 360GB 权重流式重写，必须限并发控 RAM

# BF16 保留清单（逐项依据见文档 §4）：
#   visual / lm_head / embed_tokens        常规模块
#   mlp.gate                               router（512 experts，softmax 对量化极敏感）
#   mlp.shared_expert_gate                 [1,2560] 的门控向量，N=1 不可能量化
#   in_proj_a in_proj_b                    GDN 小投影 N=48 -> vLLM 融合成 in_proj_ba N=96，
#                                          mm_mxfp8 断言 N>=128（27B 上实测崩过）
#   block_inject_weight                    hyper-connection 注入权重 [4,10240]，N=4
#   hyper_connection                       输入混合投影，引擎侧结构模块
#   ple                                    PLE + n-gram 表（51.2B / 102GB），是 embedding 不是 Linear
#   mtp                                    MTP 投机解码层（自带 512 experts），主路径不受益
# 注意：MoE 模型上 `mlp.gate` 只命中 router；`mlp.experts.N.gate_proj` 不含 "mlp.gate" 子串，安全。
IGNORE="visual,lm_head,embed_tokens,mlp.gate,mlp.shared_expert_gate,in_proj_a,in_proj_b,block_inject_weight,ple,mtp,hyper_connection"

# llm_compressor 按有效 weight bits 重建整组 preset：bits=4 + mx_fp 出来是 W4A4（不是 W4A8），
# 与 HF 参考产物 INCModel3/Qwen3.8-Flash-Next-MXFP4-Mixed-CT-AutoRound 一致。
LC_EXPERTS_MXFP4='{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}'

case "$MODE" in
  mxfp8)       SCHEME=MXFP8; LAYER_CONFIG="" ;;
  mxfp4)       SCHEME=MXFP4; LAYER_CONFIG="" ;;
  mxfp4_mixed) SCHEME=MXFP8; LAYER_CONFIG="$LC_EXPERTS_MXFP4" ;;
  *) echo "unknown MODE=$MODE (mxfp8|mxfp4_mixed|mxfp4)"; exit 1 ;;
esac

SUFFIX=""
ART_LABEL="$SCHEME"
if [ "$MODE" = mxfp4_mixed ]; then
  SUFFIX="-Mixed"
  ART_LABEL=MXFP4      # 命名规范：Mixed 产物前导 scheme 取最低比特（对齐 aqa artifact 命名）
fi
OUT="$OUTROOT/Qwen3.8-Flash-Next-${ART_LABEL}${SUFFIX}-AutoRound"
LOG="$OUTROOT/quant_$(basename "$OUT").log"
AUDIT="$OUTROOT/audit_$(basename "$OUT").log"

CMD=(auto-round --model_name "$MODEL" --scheme "$SCHEME" --ignore_layers "$IGNORE"
     --format llm_compressor --output_dir "$OUT" --model_free --device_map auto)
if [ -n "$LAYER_CONFIG" ]; then CMD+=(--layer_config "$LAYER_CONFIG"); fi
command -v auto-round >/dev/null 2>&1 || { echo "!! PATH 里找不到 auto-round（env: $ENVBIN）"; exit 1; }

echo "### 量化前审计：逐张量精度分配 + vLLM MX kernel 约束 -> $AUDIT"
audit_args=(--model "$MODEL" --scheme "$SCHEME" --ignore_layers "$IGNORE")
if [ -n "$LAYER_CONFIG" ]; then audit_args+=(--layer_config "$LAYER_CONFIG"); fi
if ! python "$HERE/audit_precision_plan.py" "${audit_args[@]}" > "$AUDIT" 2>&1; then
  cat "$AUDIT"
  echo "!! 前置审计失败，见 $AUDIT"; exit 2
fi
tail -n 12 "$AUDIT"

echo
disp=""
for a in "${CMD[@]}"; do case "$a" in *\{*) a="'$a'";; esac; disp="$disp $a"; done
echo "### 命令:$disp"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "### DRY_RUN=1，未启动量化"; exit 0; fi

echo "### 开始量化（360GB 源，I/O 受限；产物预计见上面审计表）"
"${CMD[@]}" 2>&1 | tee "$LOG"
AR_RC="${PIPESTATUS[0]}"
if [ "$AR_RC" -ne 0 ]; then echo "!! auto-round 退出码 $AR_RC，见 $LOG"; exit "$AR_RC"; fi

echo "### 导出后核对"
if grep -qi "falling back to the regular AutoRound flow" "$LOG"; then
  echo "!! 日志出现 model_free 静默回退（=真加载了 360GB 模型），检查 scheme/格式组合"; exit 3
fi
python "$HERE/check_quant_export.py" "$OUT"
echo "### done -> $OUT  (log: $LOG)"
echo "### 下一步: 先 SMOKE_ONLY=1 ./eval_Qwen3.8-Flash-Next-quant.sh $OUT"
