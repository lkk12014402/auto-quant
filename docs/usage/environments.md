# 环境构建：三种形态与安装内容（真源：`aqa/executors/_remote.py`、`aqa/executors/local_env.py`）

三种形态共享同一套 **staged inner run**：aqa 生成派生 `_run.yaml`（剥掉 env 块、
executor 强制 local），在目标环境里执行内层 `aqa run`。区别只在"环境怎么来"。

## 1. ssh docker（远程生产）

`executor.type: ssh`。流程：本仓库 tar → SFTP 上传 → 远端（可选跳板）→
`docker build`（或仅 pull base）→ `docker run` 跑容器内 bootstrap。

安装内容：

| 层 | 内容 | 说明 |
|---|---|---|
| base image | CUDA `nvcr.io/nvidia/pytorch:26.01-py3`；XPU `vllm/vllm-openai-xpu:nightly` | `executor.ssh.accelerator` 决定；`image.gpu/xpu` 可覆盖 |
| dockerfile（可选） | `docker/gpu.dockerfile` / `xpu.dockerfile`（OS 工具 + uv + Node + openclaw） | `image.dockerfile: local` 或路径；不建则直接用 base |
| 容器 bootstrap | `pip/uv install -e .`（aqa，必成功）+ 推导包集（失败仅警告） | uv 优先（`uv pip install --system`），无 uv 回退 pip |
| agent 后端（`agent.fix_loop/optimize` 开启时） | Node v22 → `openclaw@2026.6.8` →（有 opus tier）`@github/copilot` → `aqa.agent.provision` | provision 铺 ~/.openclaw、~/.copilot、skills |

**推导包集（`container_packages`）**：`auto-round` + `lm-eval` 必有；
`eval.backend=vllm` 加 `vllm`（XPU 除外）；`vllm_omni` 加
`vllm-omni, diffusers, datasets, transformers, sentencepiece, accelerate, pillow`；
任一 benchmark 走 evalscope 加 `evalscope`（+vllm）；`image.pip_install` 追加；
**`env.packages` 显式列表可整体覆盖推导**（vendor vllm 构建必须用它防覆盖）。

**torch 政策**：XPU 刻意不动（PyPI vllm 会冲掉 xpu 构建）；CUDA 用 NGC 自带。
非必要不动 torch/vllm；要动就 `env.packages` 显式 pin。

## 2. local docker（本机，day-0 常用）

`executor.type: local` + `env.type: docker`。**容器内逻辑与 ssh docker 完全一致**
（同一份 bootstrap 代码、同一份包推导）；区别在启动：本机 `docker run`、
repo 与 out_dir bind-mount、默认 host 网络、代理从宿主环境透传、
`CUDA_VISIBLE_DEVICES` 重映射为容器内 0..N-1。目前只支持 CUDA
（dockerfile 只解析 `gpu.dockerfile`）。

day-0 定制镜像用法：`image.gpu` 指向官方镜像、`dockerfile` 留空、
`env.packages` 里**不要 vllm**（用镜像自带）；示例 `examples/local_docker/`。

## 3. local venv（本机，复用/新建虚拟环境）

`executor.type: local` + `env.type: venv`。

| 步骤 | 行为 |
|---|---|
| 创建（缺失且 `create: true`） | **conda 优先**（`conda create -p <path> python=3.12`），无 conda 用 `python3 -m venv` |
| 安装 | **uv 优先**（`uv pip install --python <venv>/bin/python`），无 uv 回退 pip；`aqa`(editable) + 包集（同容器推导，`packages: auto` 时） |
| 复用 | venv 已存在则幂等跳过；存在但无 `aqa` 入口 → 报错 |
| agent provision | 与容器对齐：Node 解进 venv 前缀、npm 装 openclaw(+copilot)、`aqa.agent.provision`（best-effort） |
| 运行 | `<venv>/bin/aqa run _run.yaml`，PATH 前置 venv/bin；卡选择靠 `CUDA_VISIBLE_DEVICES`（不重映射） |

## 4. in-process

`executor.type: local` 无 env 块：当前进程直接跑，不构建任何环境。
要求当前环境已有所需（aqa + auto-round + lm-eval ± vllm/evalscope）。

## 密钥与环境变量转发

容器/venv 只收**白名单**变量（HF_TOKEN、MINIMAX_API_KEY、COPILOT_GITHUB_TOKEN、
GITHUB_TOKEN、HF_TOKENS 等）+ sink 引用的 `*_env`；ssh 密码（password_env）
明确不进容器。HF_DATASETS_CACHE/HF_HOME 类路径变量按 os.environ 继承
（venv 路径），容器内需经 image.pip_install/proxy 配置显式处理。
