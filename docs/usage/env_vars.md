# 环境变量参考（aqa）

所有 aqa 及其子进程消费的环境变量。**纪律：密钥永不写进 yaml/git**——
env.sh 定义（已入 .gitignore），yaml 里用 `${VAR}` / `token_env` 引用；
上传前文本统一过 `aqa/report/scrub.py` 脱敏。

## 密钥类（env.sh 配置）

| 变量 | 用途 | 谁消费 | 备注 |
|---|---|---|---|
| `GPU_HOST` | ssh 执行器目标主机 | yaml `ssh.host: ${GPU_HOST}` → `executors/ssh.py` | 仅 ssh 形态需要 |
| `GPU_SSH_PASS` | ssh 密码 | yaml `password_env: GPU_SSH_PASS` → paramiko | **不进容器**；`SSHPASS` 只供人工 sshpass 打印命令 |
| `HF_TOKEN` | HF 模型下载 + sink token 兜底 | sinks / hub 库默认 / 容器白名单转发 | 至少要有读权限 |
| `LOCAL_RESULTS_DATASET` | 结果回传的 HF dataset repo | yaml `repo: ${LOCAL_RESULTS_DATASET:-...}` | 如 `lvkaokao/lb_local` |
| `LOCAL_RESULTS_HF_TOKEN` | dataset **写** token | yaml `token_env: LOCAL_RESULTS_HF_TOKEN` → `resolve_hf_token` | 需写权限；403 多半是没用它 |
| `MINIMAX_API_KEY` | fix-loop minimax tier | 容器白名单转发 → `aqa.agent.provision` | 配了 tiers 才需要 |
| `COPILOT_GITHUB_TOKEN` | fix-loop/optimize 的 opus tier（copilot 后端） | 容器白名单转发 → provision | 同上 |
| `GH_TOKEN` / `GITHUB_TOKEN` | `aqa publish-patch`（gh pr create） | gh CLI | 建议 env.sh 里配 `GH_TOKEN`；**白名单已含**，容器内可用 |
| `HF_TOKENS` | 多账号令牌（预留） | 容器白名单转发 | 当前 sink 不做逐账号 failover |

## 行为/路径类（按需，一般不配）

| 变量 | 用途 | 默认 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | local 形态选卡（**aqa 不重映射**，`num_gpus` 只是 TP 参数） | 全卡可见 |
| `AR_MODEL_FREE_SHARD_PARALLELISM` | auto-round model_free 的分片并发限流（大模型建议 1） | auto-round 内部默认 |
| `AQA_KNOWLEDGE_DIR` | knowledge 本地库路径 | `~/.cache/aqa/knowledge` |
| `XDG_CACHE_HOME` | 影响上面的默认推导 | `~/.cache` |
| `HF_HOME` / `HF_DATASETS_CACHE` | HF 模型/数据集缓存 | 注意：docker 写过的缓存目录会变 root 所有，calibration 下载会 Permission denied——指到自己可写的目录 |
| `LOCAL_HTTP_PROXY` / `LOCAL_HTTPS_PROXY` | 远端代理 | yaml `executor.proxy` 引用；本机不需要就别转发 |
| `VLLM_WORKER_MULTIPROC_METHOD` | vllm 后端自动设 `spawn`（`aqa/backends/vllm.py`），无需手配 | aqa 自理 |

## 各场景需要哪些（速查）

| 场景 | 必须 |
|---|---|
| 本机 in-process / local venv | `HF_TOKEN`（hub 模型时） |
| local docker / ssh docker | + `LOCAL_RESULTS_DATASET` + `LOCAL_RESULTS_HF_TOKEN`（要回传时） |
| fix-loop / optimize agent | + `MINIMAX_API_KEY` / `COPILOT_GITHUB_TOKEN`（按 tiers） |
| 补丁转 PR | + `GH_TOKEN`（gh auth 也行） |
| 知识库共享（references/lessons） | + `LOCAL_RESULTS_HF_TOKEN`（推送时） |

## 排错速查

- **403 Forbidden 上传**：用了只读 `HF_TOKEN` 而不是 `LOCAL_RESULTS_HF_TOKEN`
- **容器里缺密钥**：变量不在白名单（白名单见 `aqa/executors/_remote.py` 的
  `_SECRET_ALLOWLIST`）或 yaml 没经 `token_env` 引用
- **校准数据集下载 Permission denied**：缓存目录被 docker 写成 root 所有，
  换 `HF_DATASETS_CACHE` 指向可写目录
