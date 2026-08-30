# 派发（dispatch）用法：local current / venv / docker / ssh

> 内核：`aqa/executors/`（local.py / local_env.py / ssh.py / _remote.py）。
> 三条路径同一条流水线：包集合推导共用、agent backend 供给共用、secrets 同源、
> sink 上传在内层 run 里统一发生。切载体只改 `executor` 块。

## 四种模式

```yaml
executor:
  type: local
  env:
    type: current     # 默认：本进程内跑（要求当前环境已装好一切）
    # type: venv      # 在一个（可自动供给的）venv 里跑
    # path: ~/.venv-aqa-gpu
    # packages: auto  # auto = 按 run 配置推导；或显式列表
    # create: true    # venv 不存在时是否自动创建
    # type: docker    # 本地容器跑（image 配置在 executor.image）
    # mounts: [/data:/data]   # docker 额外挂载
  num_gpus: 4         # 显式卡数；否则数 CUDA_VISIBLE_DEVICES 个数
  required_env: [HF_TOKEN]   # 开工前的环境变量预检（缺则直接拒绝）
```

### local current
零隔离、最快。环境必须已经就绪（torch/vllm/auto-round/lm-eval…）。

### local venv
在 `env.path` 的 venv 里跑内层 `aqa run`；venv 缺失时自动供给
（`python3 -m venv` + `pip install -e <repo>` + 包集合）。**agent backend 供给**
与容器路径对齐：node 缺失时装进 venv 前缀（免 sudo）、npm `--prefix` 装
openclaw/copilot、`aqa.agent.provision` 同步 skills/密钥。同文件系统，
`output.dir` 原样生效。

### local docker
本地 `docker build`（`executor.image.dockerfile: local` → 仓内
`docker/gpu.dockerfile`；或给个路径；或空 = 直接用基础镜像）后 `docker run`。
实证要点（都已在代码里处理）：镜像自带定制 vllm 时 `packages` 里**排除 vllm**；
本地源码版依赖先打 wheel 再挂进容器（容器里没 git）；代理要显式转发；
多卡注意容器内 CVD 重映射；NCCL 用 `NCCL_NVLS_ENABLE=0`。

### ssh docker
生产验证路径。整条流水线在远端的容器里跑，结果拉回本机。
`executor.ssh: {host, gpus, auth, jump}`；`executor.ssh.accelerator: cuda|xpu`
切设备参数和基础镜像（XPU 走 `/dev/dri` 穿透 + `ZE_AFFINITY_MASK`）。

## secrets 与环境变量

- secrets 只在 yaml 里用 `*_env` 引用（如 `token_env: HF_TOKEN`），永不内联；
- 注入靠 `collect_secret_env` 白名单（HF_TOKEN/MINIMAX_API_KEY/COPILOT_GITHUB_TOKEN/…）+
  sink 里引用的 token_env；docker 走 `--env-file`，venv 走子进程环境；
- 长任务用 `setsid nohup` 脱离 session 启动（session 关闭会杀启动器）；
  日志在 `<output.dir>/logs/executor_*.log`，量化/评估都有 pty 进度条。

## 常用排障

- `docker run` 起不来：先 `executor.env.type: docker` + `dry-run`（`aqa run --dry-run`）看生成的 bootstrap；
- venv 供给失败：看 `<output.dir>/logs/executor_venv.log` 尾部；
- 容器里 vllm 版本不对：检查 `packages` 是不是把镜像定制 vllm 覆盖了。
