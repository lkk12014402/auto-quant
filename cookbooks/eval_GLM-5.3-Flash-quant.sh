#!/usr/bin/env bash
# GLM-5.3-Flash 量化产物验证 + 基准评测（321B MoE / 34 KDA + 11 稀疏MLA / hc / MTP）
#
#   ./eval_GLM-5.3-Flash-quant.sh <ckpt>                       # 正式跑：gsm8k + TASKS（不冒烟）
#   RUN_SMOKE=1 ./eval_GLM-5.3-Flash-quant.sh <ckpt>           # 先冒烟（多完整加载一遍）
#   SMOKE_ONLY=1 ./eval_GLM-5.3-Flash-quant.sh <ckpt>          # 只冒烟，排查加载类问题
#   RUN_GSM8K=0 TASKS=piqa LIMIT=16 ./eval_GLM-5.3-Flash-quant.sh <ckpt>
#   DRY_RUN=1 ./eval_GLM-5.3-Flash-quant.sh <ckpt>             # 看预检判定与参数组合，不动 GPU
#
# 上游 vLLM（v0.29.1rc0+/main 的 vllm.models.glm5next）源码级硬约束，全部带 file:line：
#   * **block_size 必须是 index_kpool×32 的倍数**（本模型 kpool=4 → 128）
#       vllm/platforms/cuda.py:430-447 _get_indexer_block_alignment；
#       vllm/models/glm5next/nvidia/attention.py:116,135-143（报错原文 "must be a multiple of index_kpool * 32"）
#   * **deep_gemm 必需**：layers/sparse_attn_indexer_kpool.py:918-921 直接 RuntimeError，
#       且它不在 vllm wheel 的 requires_dist 里（要自己装，见 setup_vllm_env_glm53.sh）
#   * TP 必须整除 num_attention_heads 与 KDA linear heads（都是 64）
#       nvidia/model.py:693、nvidia/kda.py:199、nvidia/attention.py:436
#   * **PP 实质禁用**（nvidia/model.py:735-737；无 make_empty_intermediate_tensors）→ 本脚本不暴露 PP
#   * KDA 状态页必须塞得进 MLA 页，否则 kv_cache_utils.py:1249-1254 抛
#       "the mamba state page ... does not fit the MLA page; increase tensor parallelism..."
#       → max_num_seqs 与 KV 宽度耦合，默认给保守的 256
#   * **indexer decode logits 是显存地雷**：sparse_attn_indexer_kpool.py:310-325 按
#       max_num_seqs×max_model_len×4B 预留；1M 上下文下轻松上 GiB，且 9/19 的修复 #57701 不在 v0.30.0
#       → max_model_len 必须显式封顶（默认 8192）
#   * dtype 必须 bf16（layers/mhc.py:667,702 硬编码 bf16 scratch）
#   * KV dtype：sm103 会选 FLASHINFER_MLA_SPARSE，允许 auto/fp8/fp8_e4m3；
#       **只有 fp8_ds_mla / nvfp4_ds_mla 打包布局会炸**（cache_kernels.cu:937,948 断言 pe_dim==64，
#       而 GLM 是 NoPE；open issue #57578/#55773）→ 只禁这两种
#   * 不开 expert parallel：CT 的 W4A4 MXFP4 MoE 只在 CUTLASS/Marlin 间选，
#       CUTLASS 要求 ep_size==1（fused_moe/experts/cutlass_moe.py:1041-1044）
#   * 不开 spec decode：MTP 的 shared_head 在量化 ckpt 上有加载期误报（open #57532），
#       且 eh_proj 是普通 nn.Linear 必须明文（nvidia/mtp.py:49）
set -euo pipefail

CKPT="${1:?usage: $0 <quantized_ckpt_dir>}"
CKPT="$(realpath "$CKPT")"

ENVBIN="${ENVBIN:-/models/lkk/venv_glm53/bin}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP="${TP:-4}"
TASKS="${TASKS:-piqa,mmlu,hellaswag}"
LIMIT="${LIMIT:-}"
OUTDIR="${OUTDIR:-/models/lkk/cri_models/lm_eval_results}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"        # 必须封顶，见头部"indexer decode logits"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
MAXSEQ="${MAX_NUM_SEQS:-256}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_GSM8K="${RUN_GSM8K:-1}"
# glm5next 支持 language_model_only（nvidia/model.py:1093 视觉塔 quant_config=None，
# 且官方镜像实测配方带 --language-model-only）；纯文本评测默认开，可置 0 关
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
[ "${SMOKE_ONLY:-0}" = 1 ] && RUN_SMOKE=1

[ -x "$ENVBIN/python" ] || { echo "!! $ENVBIN 里没有 python；先 ./setup_vllm_env_glm53.sh 建环境（文档 §7）"; exit 1; }
command -v lm_eval >/dev/null 2>&1 || { echo "!! $ENVBIN 里没有 lm_eval"; exit 1; }
IFS=',' read -r -a _g <<< "$GPUS"
if [ "${#_g[@]}" -lt "$TP" ]; then echo "!! CUDA_VISIBLE_DEVICES=$GPUS 只有 ${#_g[@]} 张卡，但 TP=$TP"; exit 1; fi
case "${KV_CACHE_DTYPE:-}" in
  *ds_mla*) echo "!! KV_CACHE_DTYPE=$KV_CACHE_DTYPE 走 *_ds_mla 打包布局，GLM 是 NoPE → cache_kernels.cu:937 断言 pe_dim==64 必炸（open #57578）"; exit 1 ;;
esac
if [ "${ENABLE_EXPERT_PARALLEL:-0}" = 1 ]; then
  echo "!! MXFP4 MoE 的 CUTLASS 后端要求 ep_size==1（cutlass_moe.py:1041-1044）——拒绝"; exit 1
fi
if [ "${PIPELINE_PARALLEL:-1}" != "1" ]; then
  echo "!! glm5next 的 PP 实质禁用（nvidia/model.py:735-737）——拒绝"; exit 1
fi

# ---- 预检①：从产物本身读硬约束参数（不靠记忆） ---------------------------------
read -r COLS EBITS NEED BYPASS_LIST NA_HEADS KDA_HEADS KPOOL <<<"$(python - "$CKPT" "$TP" <<'PY'
import importlib.util, json, os, sys
d, tp = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(os.path.join(d, "config.json")))
tc = cfg.get("text_config", cfg)
mi = tc.get("moe_intermediate_size")
na = tc.get("num_attention_heads") or (tc.get("linear_attn_config") or {}).get("num_heads")
kda = (tc.get("linear_attn_config") or {}).get("num_heads")
kpool = tc.get("index_kpool") or 1
groups = cfg.get("quantization_config", {}).get("config_groups", {})
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
has_moe = any(".mlp.experts." in k for k in idx)
ebits = None
for v in groups.values():
    if any("experts" in t for t in v["targets"]):
        ebits = v["weights"]["num_bits"]; break
if ebits is None and groups and any("Linear" in t or "RoutedExperts" in t
                                    for v in groups.values() for t in v["targets"]):
    ebits = max(v["weights"]["num_bits"] for v in groups.values())
cols = (mi / tp / 32) if mi else None
ok = cols is not None and float(cols).is_integer() and int(cols) % 4 == 0
need = 1 if (has_moe and ebits == 8 and not ok) else 0
usable = [b for b, mod in (("deep_gemm", "deep_gemm"), ("marlin", None), ("humming", "humming_kernels"))
          if mod is None or importlib.util.find_spec(mod)]
print(int(cols) if cols is not None else -1, ebits if ebits is not None else -1, need,
      ",".join(usable) or "none", na or 64, kda or 64, kpool)
PY
)"
[ "${COLS:-}" ] || { echo "!! 预检①取不到产物参数"; exit 1; }
if [ $(( NA_HEADS % TP )) -ne 0 ] || [ $(( KDA_HEADS % TP )) -ne 0 ]; then
  echo "!! TP=$TP 必须整除 num_attention_heads($NA_HEADS) 与 KDA heads($KDA_HEADS)"; exit 1
fi
if [ $(( BLOCK_SIZE % (KPOOL * 32) )) -ne 0 ]; then
  echo "!! BLOCK_SIZE=$BLOCK_SIZE 必须是 index_kpool($KPOOL)×32 = $((KPOOL*32)) 的倍数"; exit 1
fi
if [ -z "${MOE_BACKEND:-}" ]; then
  if [ "$NEED" = 1 ]; then
    MOE_BACKEND=none
    for b in deep_gemm marlin humming; do case ",$BYPASS_LIST," in *",$b,"*) MOE_BACKEND=$b; break;; esac; done
    echo "### MoE 后端: experts=${EBITS}bit，每卡 ${COLS} 列（非 4 的倍数）-> 自动 MOE_BACKEND=$MOE_BACKEND（绕开 FI swizzle bug）"
  else
    MOE_BACKEND=none
    echo "### MoE 后端: experts=${EBITS}bit，每卡 ${COLS} 列 -> vLLM 自动选型（本模型不受 MXFP8-MoE swizzle bug 影响）"
  fi
fi

# ---- 预检②：vLLM 认不认 glm5_next + deep_gemm 在不在 ----------------------------
if ! python - <<'PY'
import importlib.util, sys
from vllm.model_executor.models.registry import ModelRegistry as R
need = "Glm5NextForConditionalGeneration"
if need not in R.get_supported_archs():
    sys.exit(f"{need} NOT registered")
print(f"    vllm OK: {need} registered")
if not importlib.util.find_spec("deep_gemm"):
    sys.exit("deep_gemm 未安装：DSA indexer 会 RuntimeError（sparse_attn_indexer_kpool.py:918-921）")
print("    deep_gemm OK")
PY
then
  echo "!! 环境不满足 → 按文档 §6.3/§7 用 nightly 镜像或 setup_vllm_env_glm53.sh"
  exit 1
fi

# ---- --model_args 用 JSON（这版 lm_eval 对含 "{" 的逗号 k=v 串会直接 Invalid JSON） ----
MODEL_ARGS="$(python - "$CKPT" "$TP" "$MAX_LEN" "$MAXSEQ" "$BLOCK_SIZE" "${GPU_MEM:-0.85}" \
                "$MOE_BACKEND" "${REASONING_PARSER:-glm45}" "${ENABLE_THINKING:-0}" \
                "$LANGUAGE_MODEL_ONLY" "${KV_CACHE_DTYPE:-}" <<'PY'
import json, sys
a = sys.argv[1:]
d = {
    "pretrained": a[0],
    "tensor_parallel_size": int(a[1]),
    "max_model_len": int(a[2]),
    "max_num_seqs": int(a[3]),
    "block_size": int(a[4]),
    "gpu_memory_utilization": float(a[5]),
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "add_bos_token": True,
    "enable_prefix_caching": False,
    "max_gen_toks": 2048,
    "enable_thinking": a[8] == "1",
}
if a[9] == "1":
    d["language_model_only"] = True
if a[7]:
    d["reasoning_parser"] = a[7]
if a[10]:
    d["kv_cache_dtype"] = a[10]
if a[6] != "none":
    d["kernel_config"] = {"moe_backend": a[6]}
print(json.dumps(d))
PY
)"

echo "### 环境: $ENVBIN | CKPT: $CKPT | TP=$TP | GPUS=$GPUS"
python -c "import vllm;print('    vllm', vllm.__version__)"
echo "### 约束: block_size=$BLOCK_SIZE(=kpool $KPOOL×32) max_num_seqs=$MAXSEQ max_model_len=$MAX_LEN language_model_only=$LANGUAGE_MODEL_ONLY MOE_BACKEND=$MOE_BACKEND"
echo "### 开关: RUN_SMOKE=$RUN_SMOKE SMOKE_ONLY=${SMOKE_ONLY:-0} RUN_GSM8K=$RUN_GSM8K | TASKS=$TASKS${LIMIT:+ LIMIT=$LIMIT}"
echo "### model_args: $MODEL_ARGS"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "### DRY_RUN=1，不启动 vLLM/lm_eval"; exit 0; fi

if [ "$RUN_SMOKE" = 1 ]; then
  echo "### 冒烟：vLLM 加载 + 3 条确定性生成（会完整加载一遍权重）"
  CUDA_VISIBLE_DEVICES="$GPUS" NUM_GPUS="$TP" LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
    GPU_MEM="${GPU_MEM:-0.85}" MAX_MODEL_LEN="$MAX_LEN" REASONING_PARSER="${REASONING_PARSER:-glm45}" \
    python "$HERE/smoke_vllm_ckpt.py" "$CKPT" "$TP"
  if [ "${SMOKE_ONLY:-0}" = 1 ]; then echo "### SMOKE_ONLY=1，跳过基准"; exit 0; fi
fi

if [ "$RUN_GSM8K" = 1 ]; then
  echo "### gsm8k（chat template + fewshot 多轮）"
  CUDA_VISIBLE_DEVICES="$GPUS" lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks gsm8k \
    --batch_size 32 --seed 42 --apply_chat_template --fewshot_as_multiturn \
    ${LIMIT:+--limit $LIMIT} --output_path "$OUTDIR"
fi

echo "### piqa / mmlu / hellaswag"
CUDA_VISIBLE_DEVICES="$GPUS" lm_eval --model vllm --model_args "$MODEL_ARGS" --tasks "$TASKS" \
  --batch_size 32 --seed 42 ${LIMIT:+--limit $LIMIT} --output_path "$OUTDIR"

echo "### done -> $OUTDIR"
