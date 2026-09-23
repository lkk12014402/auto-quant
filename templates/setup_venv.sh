#!/usr/bin/env bash
# <Model> 量化/评测环境构建（第②步产物）：venv + uv 安装 + 自检
#
# 模板使用：替换 <...>；按本模型实际需要裁剪 PINS 与 PROBES；头部注释块必须填——
# 它是"为什么是这个版本组合"的唯一记录。实例参照：cookbooks/setup_vllm_env_glm53.sh。
#
#   ./setup_venv.sh                  # 建/补齐环境（幂等：已就绪则只自检）
#   CHECK_ONLY=1 ./setup_venv.sh     # 只体检，不装任何东西（秒级）
#   DRY_RUN=1 ./setup_venv.sh        # 只打印将执行的命令
#
# ============================ 本机实测可用的组合（<YYYY-MM-DD>） ============================
# 下面这套必须是**真装成并验过**的，不是从文档抄来的：
#   venv              <路径>（由 <python 来源与版本> 建；conda 优先，无 conda 用系统 python -m venv）
#   <关键包>          <版本>（<来源：PyPI / nightly 索引 / 源码 commit>）
#   registry          <架构名> = True   ← 硬门槛
#   系统前置          <libdw-dev / nvcc / ...>
#   已知副作用        <如"装 lm-eval 会把 transformers 顶到 x.y.z">
# 版本选择的理由（为什么不能 pip install 默认版：PR/tag/索引实测结论）：
#   <逐条写，含查证日期>
# ==========================================================================================
set -euo pipefail

VENV="${VENV:-</绝对路径/venv_<model>>}"
PYBIN="${PYBIN:-$(command -v python3)}"       # 或 conda：conda create -p "$VENV" python=3.12
UV="${UV:-$(command -v uv || true)}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN() { echo "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }

# ---------------- 体检（可独立运行：CHECK_ONLY=1） ----------------
# 体检 = 架构 registry 命中 + 关键包 import/版本 + 必要时的设备断言。
# 体检通过的输出同时是 requirements.txt 冻结与 model_card §5 的输入。
selfcheck() {
  "$VENV/bin/python" - <<'PY'
import importlib.metadata as md, importlib.util
# <按需改：本模型硬门槛>
import torch
from vllm.model_executor.models.registry import ModelRegistry as R
ARCH = "<ForConditionalGeneration 类名>"
hit = ARCH in R.get_supported_archs()
print("vllm       ", md.version("vllm"))
print("torch      ", torch.__version__, "cuda", torch.version.cuda)
print(f"registry    {ARCH} = {hit}")
for mod, dist in (("transformers", "transformers"), ("auto_round", "auto-round"),
                  ("compressed_tensors", "compressed-tensors"), ("lm_eval", "lm-eval")):
    if importlib.util.find_spec(mod) is None:
        print(f"{dist:18s} **缺失**"); continue
    print(f"{dist:18s} {md.version(dist)}")
if not hit:
    raise SystemExit(f"!! FAIL: 这个 vllm 不含 {ARCH}")
print("PASS")
PY
}

if [ "$CHECK_ONLY" = 1 ]; then
  [ -x "$VENV/bin/python" ] || { echo "!! $VENV 不存在；先不带 CHECK_ONLY 跑一次"; exit 1; }
  echo "### 体检 $VENV（不安装任何东西）"
  selfcheck; exit $?
fi

# ---------------- 前置检查 ----------------
[ -n "$UV" ] || { echo "!! PATH 里没有 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
df -h "$(dirname "$VENV")" | tail -1 | awk '{print "### 目标盘剩余: "$4}'

# 已存在且体检通过 → 只自检不重装（幂等）
if [ -x "$VENV/bin/python" ] && selfcheck >/dev/null 2>&1; then
  echo "### 环境已就绪 —— 直接自检，不重复安装"
  selfcheck; exit 0
fi

# 1) 建 venv
[ -x "$VENV/bin/python" ] || RUN "$PYBIN" -m venv "$VENV"
export VIRTUAL_ENV="$VENV"

# 2) 装依赖（uv 优先）。版本 pin 必须显式，来源特殊（nightly 索引/源码）要注明：
RUN "$UV" pip install -U pip setuptools wheel
RUN "$UV" pip install \
  "auto-round>=0.15.0" \
  "lm-eval" \
  "transformers>=<实测最低版本>"
# RUN "$UV" pip install --torch-backend=cu130 --pre \
#   --extra-index-url https://wheels.vllm.ai/nightly/cu130 "vllm==<已验证 dev 版本>"

# 3) 自检
echo "### 自检"
[ "$DRY_RUN" = 1 ] || selfcheck

cat <<EOF

### 完成。
  CHECK_ONLY=1 $0      # 随时体检
  体检通过后冻结依赖：  $VENV/bin/python -m uv pip freeze > requirements.txt
  （之后每次换/升依赖，重跑本脚本自检并重新冻结 requirements.txt）
EOF
