#!/usr/bin/env bash
# Qwen3.8-Flash-Next 量化产物验证 + 基准评测（180B MoE）
#
#   ./eval_Qwen3.8-Flash-Next-quant.sh <ckpt>                       # 正式跑：只跑基准（默认不冒烟）
#   RUN_SMOKE=1 ./eval_Qwen3.8-Flash-Next-quant.sh <ckpt>           # 先冒烟再基准（多加载一遍）
#   SMOKE_ONLY=1 ./eval_Qwen3.8-Flash-Next-quant.sh <ckpt>          # 只冒烟（排查加载类问题用）
#   RUN_GSM8K=0 ./eval_Qwen3.8-Flash-Next-quant.sh <ckpt>           # 跳过 gsm8k（省一次完整加载）
#   TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 TASKS=piqa LIMIT=16 ./eval_Qwen3.8-Flash-Next-quant.sh <ckpt>
#   DRY_RUN=1 ./eval_Qwen3.8-Flash-Next-quant.sh <ckpt>             # 只看判定结果与参数组合
#   ENVBIN=/models/lkk/venv_qwen38fn/bin ...                        # 用 §7 的干净环境
#
# 硬约束（本机 vLLM 代码级依据见 Qwen3.8-Flash-Next-quantization-recipe.md §6.4/§6.6）：
#   * dtype 必须 bfloat16；**KV 不许量化**（QSA: NotImplementedError）→ 不接受 KV_CACHE_DTYPE
#   * **不开 expert parallel**：CutlassExpertsMxfp4 要求 ep_size==1，EP>1 无干净回退
#   * PP 必须=1（PLE 要原始 input_ids）；不开 spec decode（CT 的 ignore 不会被重映射到 draft）
#   * TP 必须整除 linear_num_value_heads(48) 与 num_attention_heads(24)
# 显存账（B300 单卡 275040 MiB = 268.6 GiB）：
#   bf16 源 360 GB → TP4；MXFP8 权重 240.4 GB=223.9 GiB → **TP2**；MXFP4-Mixed ≈180 GB → TP2 宽松
#   **TP=1 不可行（实测两次）**：不是因为权重装不下（0.95 下 219 GiB 权重能进），而是 PLE n-gram 表
#   （51.2B×2B = **95.37 GiB**）在 TP=1 下需要整份第二份拷贝 → OOM；TP≥2 该表按 vocab 切分，峰值减半。
#   且它与 max_num_batched_tokens **无关**（16384 与 2048 两次都是同一笔 95.37 GiB）——别指望调 batch 解决。
#   需要强行 TP=1 时：FORCE_TP1=1（会打印警告）。
set -euo pipefail

CKPT="${1:?usage: $0 <quantized_ckpt_dir>}"
CKPT="$(realpath "$CKPT")"

ENVBIN="${ENVBIN:-/models/miniforge3/envs/muse_verify/bin}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 不显式设 VLLM_GDN_DECODE_KERNEL=cuda：让 vLLM 自行在 cuda/triton 间回落

GPUS="${CUDA_VISIBLE_DEVICES:-1,2}"
TP="${TP:-2}"
TASKS="${TASKS:-piqa,mmlu,hellaswag}"
LIMIT="${LIMIT:-}"
OUTDIR="${OUTDIR:-/models/lkk/cri_models/lm_eval_results}"
GPU_MEM_USER_SET=0
[ -n "${GPU_MEM:-}" ] && GPU_MEM_USER_SET=1
GPU_MEM="${GPU_MEM:-0.85}"      # TP=1 的抬高在拿到产物位宽判定后再做（见下面 TP 规则块）
MAX_LEN="${MAX_MODEL_LEN:-8192}"
RUN_SMOKE="${RUN_SMOKE:-0}"      # 正式跑默认不冒烟（要冒烟：RUN_SMOKE=1；只冒烟：SMOKE_ONLY=1）
RUN_GSM8K="${RUN_GSM8K:-1}"      # gsm8k 会多一次完整加载；只想跑 TASKS 时给 RUN_GSM8K=0
[ "${SMOKE_ONLY:-0}" = 1 ] && RUN_SMOKE=1

if [ "${ENABLE_EXPERT_PARALLEL:-0}" = 1 ]; then
  echo "!! 本模型 MXFP4 MoE 走 CutlassExpertsMxfp4，要求 ep_size==1（EP>1 无干净回退）—— refusing"; exit 1
fi
if [ -n "${KV_CACHE_DTYPE:-}" ]; then
  echo "!! Qwen4Exp 的稀疏注意力(QSA)明确不支持 KV 量化，别传 KV_CACHE_DTYPE"; exit 1
fi
if [ $((48 % TP)) -ne 0 ] || [ $((24 % TP)) -ne 0 ]; then
  echo "!! TP=$TP 必须整除 linear_num_value_heads(48) 与 num_attention_heads(24)"; exit 1
fi
command -v lm_eval >/dev/null 2>&1 || { echo "!! $ENVBIN 里没有 lm_eval"; exit 1; }
IFS=',' read -r -a _g <<< "$GPUS"
if [ "${#_g[@]}" -lt "$TP" ]; then
  echo "!! CUDA_VISIBLE_DEVICES=$GPUS 只有 ${#_g[@]} 张卡，但 TP=$TP"; exit 1
fi

# model_args 统一在下面按 **JSON** 生成（见 MODEL_ARGS 处的说明），这里只留 REASONING_PARSER 默认值
REASONING_PARSER="${REASONING_PARSER-qwen3}"

# ---- MoE 后端自动选择（判据是"每卡 scale 列数"，不是 TP 本身） ---------------------
# 背景 bug：_shuffle_mxfp8_moe_weights
# （vllm/model_executor/layers/quantization/utils/flashinfer_utils.py:459-460）对 MXFP8 MoE 的
# w2(down_proj) scale 做 FlashInfer TRTLLM 的 128x4 swizzle —— **列数向上补到 4 的倍数** ——
# 之后却 `.reshape(w2_scale.shape)` 按**补前**形状还原 → 元素数不匹配：
#     RuntimeError: shape '[512, 2560, 10]' is invalid for input of size 15728640
# 真正判据：每卡列数 cols = moe_intermediate_size / TP / 32
#     cols % 4 == 0  → 不 pad → 原生后端安全（FlashInfer TRTLLM，最快）→ 不覆盖，让 vLLM 自动选型
#     cols % 4 != 0  → 必 pad → 必须换后端
# 本模型 moe_intermediate=640 → cols = 20/TP：TP=1→20 ✓ | TP=2→10 ✗ | TP=4→5 ✗
# CPU 实测佐证：shuffle_matrix_sf_a((2560,10))->30720(=2560*12)、((2560,20))->51200(不补)
# 仅当 **experts 落在 8-bit(MXFP8) 组** 时才适用；experts 是 4-bit(MXFP4) 时走
#   CutlassExpertsMxfp4，其 swizzle 显式按 padded_N 还原，无此 bug（M0 走自动选型即可）。
# 绕开后端的可选性与本机实测（oracle/mxfp8.py:_BACKEND_NAME_MAP）：
#   deep_gemm  -> 需装 deep_gemm（本机未装）
#   marlin     -> 在 wheel 内，本机实测可用（已过权重后处理 + warmup + CUDA graph 捕获）
#   triton     -> 本机实测被 CUDA 拒绝：No supported MXFP8 expert class for TRITON_MXFP8
#   humming    -> 需装 humming_kernels（本机未装）
read -r COLS EBITS NEED BYPASS_LIST HAS_PLE <<<"$(python - "$CKPT" "$TP" <<'PY'
import importlib.util, json, os, sys
d, tp = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(os.path.join(d, "config.json")))
tc = cfg.get("text_config", cfg)
mi = tc.get("moe_intermediate_size")
qcfg = cfg.get("quantization_config", {})
groups = qcfg.get("config_groups", {})
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
has_moe = any(".mlp.experts." in k for k in idx)
# experts 所在组的位宽：优先看显式列出 expert 目标的组；否则兜底目标 "Linear"/"RoutedExperts"
ebits = None
for v in groups.values():
    tg = v["targets"]
    if any("experts" in t for t in tg):
        ebits = v["weights"]["num_bits"]; break
if ebits is None and any("Linear" in t or "RoutedExperts" in t for v in groups.values() for t in v["targets"]):
    ebits = max(v["weights"]["num_bits"] for v in groups.values())
cols = (mi / tp / 32) if mi else None
ok_cols = cols is not None and float(cols).is_integer() and int(cols) % 4 == 0
need = 1 if (has_moe and ebits == 8 and not ok_cols) else 0
usable = [b for b, mod in (("deep_gemm", "deep_gemm"), ("marlin", None), ("humming", "humming_kernels"))
          if mod is None or importlib.util.find_spec(mod)]
has_ple = 1 if any("ngram_embedding" in k for k in idx) else 0
print(int(cols) if cols is not None else -1, ebits or -1, need, ",".join(usable) or "none", has_ple)
PY
)"
if [ -z "${COLS:-}" ] || [ -z "${NEED:-}" ]; then
  echo "!! 后端判定失败（读不到 $CKPT 的 config/index，或 moe_intermediate_size 缺失）"; exit 1
fi
if [ -z "${MOE_BACKEND:-}" ]; then
  if [ "$NEED" = 1 ]; then
    # 按优先级挑一个本机真装得上的绕开后端
    MOE_BACKEND=none
    for b in deep_gemm marlin humming; do
      case ",$BYPASS_LIST," in *",$b,"*) MOE_BACKEND=$b; break;; esac
    done
    echo "### MoE 后端判定: experts=${EBITS}bit, moe_intermediate/TP/32 = ${COLS} 列（非 4 的倍数）"
    echo "###   -> 会触发 vLLM FlashInfer scale-swizzle 形状 bug，自动选 MOE_BACKEND=$MOE_BACKEND（本机可用绕开后端：$BYPASS_LIST）"
  else
    MOE_BACKEND=none
    if [ "$COLS" = "-1" ]; then
      echo "### MoE 后端判定: 该产物没有 moe_intermediate_size（dense 或无量化 MoE）-> 不涉及 MXFP8 MoE bug，用自动选型"
    elif [ "$EBITS" = 8 ]; then
      echo "### MoE 后端判定: experts=8bit 且每卡 ${COLS} 列（是 4 的倍数）-> 不触发 pad，用自动选型（原生 FlashInfer TRTLLM 最快）"
    else
      echo "### MoE 后端判定: experts=${EBITS}bit（非 8bit）-> 不走 _shuffle_mxfp8_moe_weights，用自动选型"
    fi
  fi
fi

# ---- TP 规则（实测，不是拍脑袋） -------------------------------------------------
# TP=1：cols=20 是 4 的倍数，能规避 swizzle bug，但 PLE n-gram 表要整份第二份拷贝
#       （51.2B×2B = 95.37 GiB）→ 0.95 下也 OOM；且与 max_num_batched_tokens 无关
#       （实测 16384 与 2048 两次都申请同一笔 95.37 GiB）。所以默认拒绝。
# TP>=2：该表按 vocab 切分（PLEVocabParallelEmbedding），峰值减半 → 可跑；
#        此时若 cols 不是 4 的倍数，上面已自动换成可用的绕开后端。
if [ "$TP" = 1 ] && [ "$HAS_PLE" = 1 ] && [ "${FORCE_TP1:-0}" != 1 ]; then
  echo "!! TP=1 不可行：PLE n-gram 表需整份额外拷贝 95.37 GiB（=51.2B×2B），0.95 也 OOM"
  echo "   （与 max_num_batched_tokens 无关，实测 16384/2048 申请的是同一笔）→ 用 TP=2 起；强行试：FORCE_TP1=1"
  exit 1
fi
if [ "$TP" = 1 ] && [ "${GPU_MEM_USER_SET:-0}" = 0 ]; then
  GPU_MEM=0.95   # 强行 TP=1 时至少把预算拉到最大，便于复现/对比
  echo "### TP=1 + FORCE_TP1 -> GPU_MEM=$GPU_MEM"
fi
MAX_TOKENS="${MAX_TOKENS:-16384}"
if [ "$MOE_BACKEND" != "none" ]; then
  export MOE_BACKEND          # 只在这条分支导出，冒烟脚本才会带上 kernel_config
else
  MOE_BACKEND=none            # 不导出：smoke 读不到就走 vLLM 自动选型
fi

# ---- --model_args 用 JSON，不用逗号串 ------------------------------------------
# lm_eval(MergeDictAction, _cli/utils.py:160-176) 对这个参数只认两种形态：
#   ① 单个 JSON 对象；② 空格分隔的一串 k=v。
# 它会先对**第一个 token** 试 json.loads，只要串里含 "{" 就放弃回退并直接报
#   ValueError: Invalid JSON: ...kernel_config={"moe_backend":"marlin"}
# —— 所以逗号串里混进任何 dict 值都会炸（27B 脚本没有 dict 值，逗号串仍可用）。
# JSON 形态顺带解决类型：数字/布尔不加引号，嵌套 kernel_config 天然合法。
MODEL_ARGS="$(python - "$CKPT" "$TP" "$MAX_LEN" "$MAX_TOKENS" "$GPU_MEM" "$MOE_BACKEND" \
                "$REASONING_PARSER" "${ENABLE_THINKING:-0}" <<'PY'
import json, sys
ckpt, tp, max_len, max_tok, gpu_mem, moe_backend, rparser, thinking = sys.argv[1:9]
d = {
    "pretrained": ckpt,
    "tensor_parallel_size": int(tp),
    "max_model_len": int(max_len),
    "max_num_batched_tokens": int(max_tok),
    "max_num_seqs": 64,
    "add_bos_token": True,
    "gpu_memory_utilization": float(gpu_mem),
    "dtype": "bfloat16",
    "max_gen_toks": 2048,
    "enable_prefix_caching": False,
    "language_model_only": True,
    "trust_remote_code": True,
    "enable_thinking": thinking == "1",
}
if rparser:
    d["reasoning_parser"] = rparser
if moe_backend != "none":
    d["kernel_config"] = {"moe_backend": moe_backend}
print(json.dumps(d))
PY
)"

echo "### 环境: $ENVBIN | CKPT: $CKPT | TP=$TP | GPUS=$GPUS | MOE_BACKEND=$MOE_BACKEND | GPU_MEM=$GPU_MEM | tokens=$MAX_TOKENS"
echo "### 开关: RUN_SMOKE=$RUN_SMOKE SMOKE_ONLY=${SMOKE_ONLY:-0} RUN_GSM8K=$RUN_GSM8K | TASKS=$TASKS${LIMIT:+ LIMIT=$LIMIT}"
python -c "import vllm;print('    vllm', vllm.__version__)"
echo "### model_args: $MODEL_ARGS"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "### DRY_RUN=1，不启动 vLLM/lm_eval"; exit 0; fi

if [ "$RUN_SMOKE" = 1 ]; then
  echo "### 冒烟：vLLM 加载 + 3 条确定性生成（会完整加载一遍权重；之后基准还会各自再加载一次）"
  CUDA_VISIBLE_DEVICES="$GPUS" NUM_GPUS="$TP" LANGUAGE_MODEL_ONLY=1 \
    GPU_MEM="$GPU_MEM" MAX_MODEL_LEN="$MAX_LEN" MAX_NUM_BATCHED_TOKENS="$MAX_TOKENS" \
    REASONING_PARSER="${REASONING_PARSER:-qwen3}" python "$HERE/smoke_vllm_ckpt.py" "$CKPT" "$TP"
  if [ "${SMOKE_ONLY:-0}" = 1 ]; then echo "### SMOKE_ONLY=1，跳过基准"; exit 0; fi
fi

if [ "$RUN_GSM8K" = 1 ]; then
  echo "### gsm8k（chat template + fewshot 多轮，口径与 baseline 一致）"
  CUDA_VISIBLE_DEVICES="$GPUS" lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks gsm8k \
    --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn \
    ${LIMIT:+--limit $LIMIT} --output_path "$OUTDIR"
fi

echo "### piqa / mmlu / hellaswag"
CUDA_VISIBLE_DEVICES="$GPUS" lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks "$TASKS" \
  --batch_size 32 --seed 42 ${LIMIT:+--limit $LIMIT} --output_path "$OUTDIR"

echo "### done -> $OUTDIR"
