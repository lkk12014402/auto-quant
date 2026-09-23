#!/usr/bin/env bash
# GLM-5.3-Flash (glm5_next, 321B MoE) — AutoRound MXFP4-Mixed / MXFP8 量化
# 方案与依据见 GLM-5.3-Flash-quantization-recipe.md
#
#   ./quant_GLM-5.3-Flash-mxfp.sh                      # G0  MXFP4-Mixed（复现 HF 参考产物）
#   MODE=mxfp4_mixed_plus ./quant_...                  # G0+ 再多量化 48 个稀疏MLA投影（预期 vLLM 加载失败，见下）
#   MODE=mxfp8 ./quant_...                             # G1  纯 MXFP8（不推荐，见下）
#   MODE=mxfp8_plus ./quant_...                        # G1+ 同上 + 稀疏MLA
#   DRY_RUN=1 ./quant_GLM-5.3-Flash-mxfp.sh            # 只跑前置审计 + 打印命令
#   SOURCE=fp8 ./quant_GLM-5.3-Flash-mxfp.sh           # 从本地原生 FP8 源重量化（次优，需显式确认）
#
# ignore 清单不写在脚本里，由 gen_glm_ignore.py 按产物 config 现算（plus 口径要按层号锚定
# KDA 的 o_proj —— KDA 与稀疏MLA 的输出投影同名 self_attn.o_proj，靠子串分不开）。
#
# ★ 源的选择（默认 BF16，与 HF 参考产物一致）：
#   SOURCE=bf16  zai-org/GLM-5.3-Flash-BF16（~642 GB / 120 分片）—— 官方推荐的量化输入
#   SOURCE=fp8   /models/lkk/cri_models/GLM-5.3-Flash（306 GB，原生 block-FP8 128×128）
#                auto-round 的 model_free 会把 FP8 反量化再重量化（代码路径存在），但这是
#                **二次量化**：FP8(128×128 scale) → dequant → MXFP4(1×32 E8M0)，多一层误差，
#                且参考产物的分数不能直接对标。除非明确接受，别用。
set -euo pipefail

MODE="${MODE:-mxfp4_mixed}"
SOURCE="${SOURCE:-bf16}"

ENVBIN="${ENVBIN:-/models/miniforge3/envs/muse_verify/bin}"
OUTROOT=/models/lkk/cri_models
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BF16_DIR="${BF16_DIR:-/models/lkk/cri_models/GLM-5.3-Flash-BF16}"
FP8_DIR="${FP8_DIR:-/models/lkk/cri_models/GLM-5.3-Flash}"
export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export AR_MODEL_FREE_SHARD_PARALLELISM=1        # 642 GB 流式重写，必须限并发控 host RAM

# BF16 保留清单（与 HF 参考产物逐条一致；依据见文档 §4）
#   lm_head embed_tokens visual norm A_log dt_bias conv1d   常规模块 / 非 Linear
#   mlp.gate.             router（**尾点锚定**：裸 mlp.gate 会误伤 dense 层的 mlp.gate_proj）
#   self_attn             GLM 里 KDA(34层) 与稀疏 MLA(11层) 的投影**都叫 self_attn.\* —— 一刀全保
#   indexer               DSA 打分件（wk/weights_proj）
#   hc_                   hyper-connection（hc_attn_*/hc_ffn_*）
#   eh_proj enorm hnorm shared_head   MTP 结构件
# 与 Qwen3.8-Flash-Next 的清单**不能互抄**：那边 linear_attn.in_proj_a/b 是 N=48 小投影，
# 这边小投影叫 self_attn.{b,f_a,f_b,g_a,g_b}_proj，已被 self_attn 覆盖。
# llm_compressor 按有效 weight bits 重建整组 preset：bits=4+mx_fp => W4A4（激活也 4bit），
# 与参考产物 group_0 的 W4A4 一致；别补 act_bits:8。
LC_EXPERTS_MXFP4='{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}'

# MODE 决定两件事：位宽组合（mxfp4_mixed / mxfp8）+ 注意力口径（aligned / plus）
#   aligned = HF 参考产物口径（注意力全 BF16）
#   plus    = 把官方 block-FP8 也量了的 48 个稀疏MLA投影（q_a/q_b/kv_a/o_proj ×12）纳入 MXFP8
#             审计预测：8bit 138→186 层，产物 183.46→182.29 GB（省 1.17 GB）
PROFILE=aligned
case "$MODE" in
  mxfp4_mixed)      SCHEME=MXFP8; LAYER_CONFIG="$LC_EXPERTS_MXFP4"; LABEL=MXFP4-Mixed ;;
  mxfp4_mixed_plus) SCHEME=MXFP8; LAYER_CONFIG="$LC_EXPERTS_MXFP4"; LABEL=MXFP4-Mixed-Plus; PROFILE=plus ;;
  mxfp8)            SCHEME=MXFP8; LAYER_CONFIG="";                  LABEL=MXFP8 ;;
  mxfp8_plus)       SCHEME=MXFP8; LAYER_CONFIG="";                  LABEL=MXFP8-Plus;     PROFILE=plus ;;
  *) echo "unknown MODE=$MODE (mxfp4_mixed|mxfp4_mixed_plus|mxfp8|mxfp8_plus)"; exit 1 ;;
esac

if [ "$SOURCE" = bf16 ]; then MODEL="$BF16_DIR"; else MODEL="$FP8_DIR"; fi
if [ ! -f "$MODEL/model.safetensors.index.json" ]; then
  echo "!! 找不到量化源 $MODEL"
  cat <<'EOF'
   BF16 源需要单独下载（~642 GB；先 df -h /models 确认空间）：
     hf download zai-org/GLM-5.3-Flash-BF16 --local-dir /models/GLM-5.3-Flash-BF16
   或显式改用本地原生 FP8 源（次优，二次量化）：SOURCE=fp8 $0
EOF
  exit 1
fi
# ignore 清单由生成器按 config 现算（plus 口径要按层号锚定 KDA 的 o_proj，KDA/MLA 同名）
IGNORE="$(python "$(dirname "$0")/gen_glm_ignore.py" "$MODEL" --profile "$PROFILE")"
echo "### ignore profile = $PROFILE（生成器：gen_glm_ignore.py，层号取自 config.linear_attn_config.kda_layers）"

if [ "$SOURCE" = fp8 ]; then
  echo "!! SOURCE=fp8：从原生 block-FP8 二次量化，精度不可对标参考产物（文档 §3）。5 秒内 Ctrl-C 可取消。"
  sleep 5
fi
case "$MODE" in
  mxfp8|mxfp8_plus)
    echo "!! 提醒：纯 MXFP8 产物 ≈338-339 GB，比官方 FP8(306 GB) 还大（MX 的 1/32 E8M0 scale 开销 > 128×128 block scale）。"
    echo "   这个模型做 MX 量化的唯一理由是 experts 下 4 bit。5 秒内 Ctrl-C 可取消。"; sleep 5 ;;
esac
if [ "$PROFILE" = plus ]; then
  cat <<'EOF'
!! plus 口径的已知风险（按你 venv 里那份 vllm 源码核实，非推测）：
   - vllm/models/glm5next/common/model.py:330  Glm5NextMLAAttention(... quant_config=None,
       # MLA projections are BF16 in checkpoint)
     -> 稀疏 MLA 的投影在建模时**根本不接 quant_config**，compressed-tensors 不会给它们量化方法；
   - common/model.py:1239 _FP8_ATTN_PROJS 的兼容层只认 block-FP8 的 `weight + weight_scale_inv`，
     而 CT 的 MXFP8 导出的是 `weight(fp8) + weight_scale(uint8 E8M0)` -> 兼容层不会触发；
   - common/kda.py:189-194 KDA 层构造期把 vllm_config.quant_config 置 None（同理）。
   所以 plus 产物**预计加载即报"unexpected tensor / 参数缺失"类错误**。真值以实跑为准：
   跑完先 SMOKE_ONLY=1 验加载，失败就回退 aligned（把 self_attn 加回 ignore 即可）。
EOF
fi

OUT="$OUTROOT/GLM-5.3-Flash-${LABEL}-AutoRound-aligned"
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
  cat "$AUDIT"; echo "!! 前置审计失败，见 $AUDIT"; exit 2
fi
tail -n 12 "$AUDIT"

echo
disp=""
for a in "${CMD[@]}"; do case "$a" in *\{*) a="'$a'";; esac; disp="$disp $a"; done
echo "### 命令:$disp"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "### DRY_RUN=1，未启动量化"; exit 0; fi

echo "### 开始量化（源 $(du -sh "$MODEL" | cut -f1)，I/O 受限；G0 产物预计 ≈183 GB）"
"${CMD[@]}" 2>&1 | tee "$LOG"
AR_RC="${PIPESTATUS[0]}"
if [ "$AR_RC" -ne 0 ]; then echo "!! auto-round 退出码 $AR_RC，见 $LOG"; exit "$AR_RC"; fi

echo "### 导出后核对"
if grep -qi "falling back to the regular AutoRound flow" "$LOG"; then
  echo "!! 出现 model_free 静默回退（=真加载了 642 GB），检查 scheme/格式组合"; exit 3
fi
python "$HERE/check_quant_export.py" "$OUT"
echo "### done -> $OUT  (log: $LOG)"
echo "### 注意：本机 vLLM 不认 glm5_next，评测前要先按 §6/§7 建可用环境"
