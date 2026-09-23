#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4_exp) 评测环境构建：系统 python3 venv + uv 装依赖
#
#   ./setup_vllm_env_qwen38fn.sh              # 建 venv 并装栈
#   DRY_RUN=1 ./setup_vllm_env_qwen38fn.sh    # 只打印将要执行的命令
#   VENV=/some/path ./setup_vllm_env_qwen38fn.sh
#   CHANNEL=nightly ./setup_vllm_env_qwen38fn.sh   # 装 nightly（要 QSA FP8 KV / 9 月中旬改动时才用）
#
# 依据（2026-09-21 核实，细节见 Qwen3.8-Flash-Next-quantization-recipe.md §6）：
#   * qwen4_exp 支持由 vllm PR #53896 提供，2026-08-31 合入 main，
#     首个包含它的 stable release = **v0.29.0**（PyPI 当前最新，2026-09-09 上传）→ 纯 pip 即可，
#     不需要源码编译，也不该再用 8/26 那个 PR 分支临时镜像 :qwen38-flash-next。
#   * 本机 GPU 是 B300 (sm_103)，torch 必须 cu130 构建 → 用 uv 的 --torch-backend=cu130。
#   * 现有 muse_verify 里的定制 vLLM (0.28.1rc1.dev312+g41848caa6) 已注册 Qwen4Exp*，
#     所以"要不要新建环境"取决于你是否想脱离那棵 vendor 树；本脚本给的是干净可复现路径。
set -euo pipefail

VENV="${VENV:-/models/lkk/venv_qwen38fn}"
PYBIN="${PYBIN:-/usr/bin/python3}"          # 系统 python（3.12.3），不用 conda
CHANNEL="${CHANNEL:-release}"               # release | nightly
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
UV="${UV:-$(command -v uv || true)}"
DRY_RUN="${DRY_RUN:-0}"

run() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }

[ -x "$PYBIN" ] || { echo "!! 找不到 $PYBIN（可用 PYBIN=... 指定）"; exit 1; }
"$PYBIN" -c 'import sys,venv,ensurepip;assert sys.version_info[:2]>=(3,10),sys.version;print("python",sys.version.split()[0],"venv OK")'

if [ -z "$UV" ]; then
  echo "!! PATH 里没有 uv；先装一个（装进 venv 之外，避免污染）："
  echo "   curl -LsSf https://astral.sh/uv/install.sh | sh    # 或 pipx install uv"
  exit 1
fi
echo "### uv: $($UV --version)"

# ---- 1) 用系统 python 建 venv（不用 conda）
if [ ! -x "$VENV/bin/python" ]; then
  run "$PYBIN" -m venv "$VENV"
fi
export VIRTUAL_ENV="$VENV"          # 让 uv 装进这个 venv 而不是系统 python
echo "### venv -> $VENV"

# ---- 2) 装推理栈
run "$UV" pip install -U pip setuptools wheel
if [ "$CHANNEL" = nightly ]; then
  # PyPI 还没有 0.30.0；需要 QSA FP8 KV(#55557) 等 9 月中旬改动时才走 nightly
  run "$UV" pip install --torch-backend="$TORCH_BACKEND" --pre \
      --extra-index-url https://wheels.vllm.ai/nightly "vllm>=0.29.0"
else
  run "$UV" pip install --torch-backend="$TORCH_BACKEND" "vllm==0.29.0"
fi

# ---- 3) flashinfer 的 cubin 不在 PyPI（SM100+ 的 MXFP4/MXFP8 快路径要用）
run "$UV" pip install --extra-index-url https://flashinfer.ai/whl/ "flashinfer-cubin==0.6.18"

# ---- 4) 量化 + 评测侧
#   auto-round：本机源码树在 /models/lkk/auto-round（0.16.0）；离线环境用 PyPI 版即可
run "$UV" pip install "auto-round>=0.15.0" "lm-eval" "transformers>=5.16.1" "accelerate"
# triton_kernels 不是硬依赖：只有 MXFP4 MoE 选中 TRITON 后端才需要（CUDA 上默认走 flashinfer/DeepGEMM/Marlin）
# run "$UV" pip install triton_kernels

# ---- 5) 自检
echo "### 自检"
run "$VENV/bin/python" - <<'PY'
import importlib, json
import torch, vllm, transformers, compressed_tensors
print("torch      ", torch.__version__, "| cuda", torch.version.cuda)
print("vllm       ", vllm.__version__)
print("transformers", transformers.__version__)
print("compressed_tensors", compressed_tensors.__version__)
from vllm.model_executor.models.registry import ModelRegistry as R
archs = R.get_supported_archs()
need = ["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM", "Qwen4ExpMTP"]
print("qwen4_exp registered:", {a: (a in archs) for a in need})
# MXFP4 / MXFP8 后端可选性（不同版本导出符号名不同，逐个 try）
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as QM
except Exception:  # pragma: no cover
    try:
        from vllm.model_executor.layers.quantization import QuantizationMethods
        from typing import get_args
        QM = list(get_args(QuantizationMethods))
    except Exception as e:
        QM = [f"<无法枚举: {e}>"]
print("mxfp/compressed 相关方法:", sorted(m for m in QM if isinstance(m, str) and ("mxfp" in m or m in ("compressed-tensors", "inc", "fp8"))))
if torch.cuda.is_available():
    print("device cc:", torch.cuda.get_device_capability(0), "(sm100+ 才有 MXFP4/MXFP8 快路径)")
else:
    print("（无 GPU 上下文，跳过 device capability 检查）")
PY

cat <<EOF

### 完成。用法：
  export PATH=$VENV/bin:\$PATH
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  # 别再显式设 VLLM_GDN_DECODE_KERNEL=cuda（会让不满足条件的配置直接抛错而不是回落 triton）
  ./eval_Qwen3.8-Flash-Next-quant.sh <产物目录>     # ENVBIN=$VENV/bin 走这个新环境
EOF
