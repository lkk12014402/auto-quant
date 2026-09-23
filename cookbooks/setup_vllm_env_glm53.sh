#!/usr/bin/env bash
# GLM-5.3-Flash (glm5_next) 评测环境构建：系统 python venv + uv 装依赖
#
#   ./setup_vllm_env_glm53.sh                  # 建/补齐环境（默认锁到已验证的那套版本）
#   CHECK_ONLY=1 ./setup_vllm_env_glm53.sh     # 只体检现有环境，不装任何东西（秒级）
#   PIN=0 ./setup_vllm_env_glm53.sh            # 不锁版本，跟随 moving nightly
#   CHANNEL=release ./setup_vllm_env_glm53.sh  # 等 v0.30.0 上 PyPI 后用（仍要自备 DeepGEMM）
#   SKIP_DEEPGEMM=1 ./setup_vllm_env_glm53.sh  # 只装 vllm 本体（之后加载必然 RuntimeError）
#   DRY_RUN=1 ./setup_vllm_env_glm53.sh        # 只打印将执行的命令
#
# ============================ 本机实测可用的组合（2026-09-21） ============================
# 下面这套是**在 4xB300(sm103) 这台机器上真装成并验过**的，不是从文档抄来的：
#   venv              /models/lkk/venv_glm53        （由 /usr/bin/python3 3.12.3 建，不用 conda）
#   vllm              0.29.1rc1.dev471+g82daf9f57   （wheels.vllm.ai/nightly/cu130）
#   torch             2.13.0+cu130   （torch.version.cuda == 13.0）
#   deep_gemm         2.8.0+e1f418c  （vLLM 官方 tools/install_deepgemm.sh 源码编）
#   flashinfer / flashinfer_cubin / tilelang / compressed_tensors / lm_eval   import OK
#   registry          Glm5NextForConditionalGeneration = True  ← 硬门槛
#   系统前置          libdw-dev 0.190-1.1ubuntu0.1（Ubuntu 24.04 默认没有，缺了 DeepGEMM 编译必挂）
#   已知副作用        装 lm-eval 会把 transformers 从 5.16.1 顶到 5.17.0（vllm 要求 >=5.10.4，无碍）
# 体检会拿这些值比对，漂移会打印 DRIFT 提醒（不阻断，除非 registry 不含 glm5next）。
# ==========================================================================================
#
# 为什么不能 `pip install vllm`（2026-09-21 实测/源码核实）：
#   * 支持由 PR #53906 提供，已于 2026-09-03 合并（merge commit 98ed0856，base main）。
#   * 该 PR 改了 `csrc/libtorch_stable/cache_kernels.cu`：dequant-cache 的 head_dim 白名单
#     从 {320,576} 放宽到 {320,512,576}（512 就是 GLM 的 MLA 隐头宽）→ **已装好的 0.29.0 wheel
#     覆盖 Python 文件也绕不过去**，必须换二进制。
#   * 含支持的最低 tag = v0.29.1rc0；稳定 tag = v0.30.0。但 **PyPI 最新仍只有 0.29.0**，
#     `:v0.30.0` 镜像也未发布 → release 路线暂时不通。
#   * v0.30.0 比 main 少 10 条 glm5next 提交（含 9/19 indexer workspace 修复 #57701、
#     9/20 kpool tail 寻址修复 #57477）→ 要跑就跟 nightly(=main)，别锁 v0.30.0。
#
# 两个必须手动补的依赖（vllm wheel 的 requires_dist 里没有）：
#   1) deep_gemm：`vllm/models/glm5next/nvidia/sparse_indexer.py:662` 会
#      RuntimeError("Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.")
#      → ref 由官方 tools/install_deepgemm.sh 自带（与 cmake/external_projects/deepgemm.cmake 同步），
#        **不要在调用方硬钉 sha**（本脚本曾抄错一位导致 uv 报 failed to find branch, tag, or commit）。
#   2) flashinfer-cubin：v0.30.0 的 requirements/cuda.txt 明确把它从 wheel 依赖里排除了。
#
# 不想折腾系统包的等价方案：
#   docker run --gpus all --network host -v /models:/models -e HF_HOME=/models/huggingface -it \
#     vllm/vllm-openai:nightly bash           # 含 glm5next + cu130 + DeepGEMM + cubin
#   注：专用 tag `glm53-flash-x86_64-cu130` 停在 09-09，缺 9/10-9/20 的 10 条修复，已非首选。
set -euo pipefail

VENV="${VENV:-/models/lkk/venv_glm53}"
PYBIN="${PYBIN:-/usr/bin/python3}"              # 系统 python（3.12.3），不用 conda
CHANNEL="${CHANNEL:-nightly}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"         # 本机 B300 = sm103
PIN="${PIN:-1}"                                 # 1=先试已验证的那个 dev 版本，失败再跟随 nightly
VERIFIED_VLLM="${VERIFIED_VLLM:-0.29.1rc1.dev471+g82daf9f57}"
VERIFIED_TORCH="${VERIFIED_TORCH:-2.13.0+cu130}"
VERIFIED_DEEPGEMM="${VERIFIED_DEEPGEMM:-2.8.0+e1f418c}"
VLLM_REF="${VLLM_REF:-main}"                    # 官方 install_deepgemm.sh 的来源 ref
DEEPGEMM_REF_FALLBACK="${DEEPGEMM_REF:-e1f418c2a4f20818221f6b0e578b4c2f634d4c3f}"   # 已校验 HTTP 200
CUDA_VER="${CUDA_VER:-13.0}"
FLASHINFER_VER="${FLASHINFER_VER:-0.6.18.post1}"
UV="${UV:-$(command -v uv || true)}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }

# ---------------- 体检（可独立运行：CHECK_ONLY=1） ----------------
selfcheck() {
  "$VENV/bin/python" - "$VERIFIED_VLLM" "$VERIFIED_TORCH" "$VERIFIED_DEEPGEMM" <<'PY'
import importlib.metadata as md, importlib.util, sys, torch, vllm
from vllm.model_executor.models.registry import ModelRegistry as R
want_v, want_t, want_dg = sys.argv[1:4]

def _has(dist):
    try:
        md.version(dist)
        return True
    except Exception:
        return False

print("vllm        ", vllm.__version__, "  DRIFT(!=已验证)" if vllm.__version__ != want_v else "  == verified")
print("torch       ", torch.__version__, "cuda", torch.version.cuda,
      "  DRIFT" if torch.__version__ != want_t else "  == verified")
reg = R.get_supported_archs()
hit = {n: n in reg for n in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM", "Glm5NextMTPModel")}
print("glm5next    ", hit)
# (import 名, 发行名候选) —— flashinfer 的 import 名与 dist 名不一致，逐个试
PROBES = (("deep_gemm", ("deep-gemm", "deep_gemm")),
          ("flashinfer", ("flashinfer-python", "flashinfer")),
          ("flashinfer_cubin", ("flashinfer-cubin",)),
          ("tilelang", ("tilelang",)),
          ("compressed_tensors", ("compressed-tensors",)),
          ("lm_eval", ("lm-eval",)),
          ("transformers", ("transformers",)))
for mod, dists in PROBES:
    if importlib.util.find_spec(mod) is None:
        tail = "  <- glm5next 的 DSA indexer 硬依赖 deep_gemm" if mod == "deep_gemm" else ""
        print(f"{mod:18s} **缺失**{tail}")
        continue
    ver = next((md.version(d) for d in dists if _has(d)), "?")
    mark = ""
    if mod == "deep_gemm":
        mark = "  == verified" if ver == want_dg else "  DRIFT(!=已验证)"
    print(f"{mod:18s} {ver}{mark}")
if not any(hit.values()):
    raise SystemExit("!! FAIL: 这个 vllm 不含 glm5_next —— 用 CHANNEL=nightly 或官方 nightly 镜像")
cv = torch.version.cuda or ""
if not cv.startswith(("13.", "12.9")):
    print(f"!! WARN: torch 是 {cv} 构建，B300(sm103) 需要 cu12.9+/cu130")
if torch.cuda.is_available():
    print("device cc   ", torch.cuda.get_device_capability(0), "(sm103 满足 MXFP4 experts 的 100<=cap<120)")
print("PASS: registry 含 glm5next")
PY
}

if [ "$CHECK_ONLY" = 1 ]; then
  [ -x "$VENV/bin/python" ] || { echo "!! $VENV 不存在；先不带 CHECK_ONLY 跑一次"; exit 1; }
  echo "### 体检 $VENV（不安装任何东西）"
  selfcheck; exit $?
fi

# ---------------- 前置检查 ----------------
[ -x "$PYBIN" ] || { echo "!! 找不到 $PYBIN（可用 PYBIN=... 指定）"; exit 1; }
"$PYBIN" -c 'import sys,venv,ensurepip;assert sys.version_info[:2]>=(3,10),sys.version;print("python",sys.version.split()[0],"venv OK")'
[ -n "$UV" ] || { echo "!! PATH 里没有 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
df -h "$(dirname "$VENV")" | tail -1 | awk '{print "### 目标盘剩余空间: "$4"（venv 约 25-30 GB）"}'

# 已存在且体检通过 -> 只做增量补齐，不重复大动作
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "
from vllm.model_executor.models.registry import ModelRegistry as R
import importlib.util, sys
sys.exit(0 if 'Glm5NextForConditionalGeneration' in R.get_supported_archs() and importlib.util.find_spec('deep_gemm') else 1)" 2>/dev/null; then
  echo "### 环境已就绪（registry + deep_gemm 均通过）—— 直接进体检，不重复安装"
  selfcheck; exit 0
fi

# 1) 系统 python 建 venv（不用 conda）
[ -x "$VENV/bin/python" ] || RUN "$PYBIN" -m venv "$VENV"
export VIRTUAL_ENV="$VENV"
echo "### venv -> $VENV | uv: $($UV --version)"

# 2) vLLM 本体
RUN "$UV" pip install -U pip setuptools wheel
case "$CHANNEL" in
  nightly)
    SPEC="vllm>=0.29.1rc1.dev0"
    [ "$PIN" = 1 ] && SPEC="vllm==$VERIFIED_VLLM"
    # cu130 专用索引；锁到已验证版本失败（nightly 轮子会滚动淘汰）就退回跟随 nightly
    if ! RUN "$UV" pip install --torch-backend="$TORCH_BACKEND" --pre \
            --extra-index-url https://wheels.vllm.ai/nightly/cu130 "$SPEC"; then
      echo "!! $SPEC 在该索引已取不到，退回 vllm>=0.29.1rc1.dev0（跟随 moving nightly）"
      RUN "$UV" pip install --torch-backend="$TORCH_BACKEND" --pre \
          --extra-index-url https://wheels.vllm.ai/nightly/cu130 "vllm>=0.29.1rc1.dev0"
    fi
    ;;
  release)
    RUN "$UV" pip install --torch-backend="$TORCH_BACKEND" "vllm>=0.30.0"
    ;;
  *) echo "unknown CHANNEL=$CHANNEL (nightly|release)"; exit 1 ;;
esac

# 3) vllm wheel 不带、但 glm5next 必需的依赖
# DeepGEMM 是源码编译（setup.py bdist_wheel），先做构建依赖预检——缺哪个报哪个，
# 别等编译到一半死在某个 header 上（实测踩过：缺 elfutils/libdwfl.h）。
missing=()
[ -f /usr/include/elfutils/libdwfl.h ] || missing+=("libdw-dev(提供 elfutils/libdwfl.h)")
command -v nvcc   >/dev/null || missing+=("nvcc(CUDA>=12.8)")
command -v cmake  >/dev/null || missing+=("cmake")
command -v ninja  >/dev/null || missing+=("ninja")
if [ "${#missing[@]}" -gt 0 ] && [ "${SKIP_DEEPGEMM:-0}" != 1 ]; then
  echo "!! DeepGEMM 构建依赖缺失: ${missing[*]}"
  echo "   Debian/Ubuntu: apt-get install -y libdw-dev   （系统级改动，请你自己确认后执行）"
  echo "   或跳过本 venv 路线，直接用已含 DeepGEMM 的镜像: docker run ... vllm/vllm-openai:nightly"
  echo "   只想先把 vllm 装上（之后加载会 RuntimeError 要 DeepGEMM）：SKIP_DEEPGEMM=1 $0"
  exit 1
fi

RUN "$UV" pip install --extra-index-url https://flashinfer.ai/whl/ "flashinfer-cubin==$FLASHINFER_VER"
if [ "${SKIP_DEEPGEMM:-0}" != 1 ]; then
  # 用 vLLM 自己的 tools/install_deepgemm.sh：内含权威 DEEPGEMM_GIT_REF，做
  # clone --recursive + checkout + setup.py bdist_wheel + uv pip install。
  DGS=/tmp/install_deepgemm_${VLLM_REF}.sh
  echo "### 取官方 install_deepgemm.sh ($VLLM_REF)"
  if [ "$DRY_RUN" = 1 ]; then
    echo "+ curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/$VLLM_REF/tools/install_deepgemm.sh -o $DGS"
    echo "+ VIRTUAL_ENV=$VENV PATH=$VENV/bin:\$PATH bash $DGS --cuda-version $CUDA_VER"
  elif curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/$VLLM_REF/tools/install_deepgemm.sh" -o "$DGS"; then
    grep -q '^DEEPGEMM_GIT_REF=' "$DGS" && echo "    官方 pin: $(grep '^DEEPGEMM_GIT_REF=' "$DGS" | cut -d'"' -f2)"
    # PATH 前置 venv/bin：官方脚本内部用 `python3 setup.py bdist_wheel`，
    # 不前置就会拿系统 python/系统 torch 的头文件编（本机恰好同版本才侥幸能用）。
    VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" bash "$DGS" --cuda-version "$CUDA_VER"
  else
    echo "!! 下载官方脚本失败，退回手工 clone+checkout ($DEEPGEMM_REF_FALLBACK)"
    T=$(mktemp -d)
    git clone --recursive --shallow-submodules https://github.com/vllm-project/DeepGEMM.git "$T/deepgemm" \
      && (cd "$T/deepgemm" && git checkout "$DEEPGEMM_REF_FALLBACK" \
          && "$VENV/bin/python" setup.py bdist_wheel \
          && VIRTUAL_ENV="$VENV" "$UV" pip install dist/*.whl)
    rc=$?; rm -rf "$T"; [ $rc -eq 0 ] || { echo "!! DeepGEMM 兜底安装也失败"; exit 1; }
  fi
else
  echo "!! SKIP_DEEPGEMM=1：加载阶段会撞 RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM"
fi

# 4) 量化 / 评测侧（版本交给 resolver；注意 lm-eval 会连带升 transformers，体检会报 DRIFT）
RUN "$UV" pip install "auto-round>=0.15.0" "lm-eval" "transformers>=5.16.1" accelerate

# 5) 体检
echo "### 自检"
[ "$DRY_RUN" = 1 ] || selfcheck

cat <<EOF

### 完成。评测：
  CHECK_ONLY=1 $0                              # 随时体检，不装东西
  ENVBIN=$VENV/bin ./eval_GLM-5.3-Flash-quant.sh <量化产物目录>
### 量化不需要这个环境（本机 muse_verify 的 auto-round 就够）：./quant_GLM-5.3-Flash-mxfp.sh
EOF
