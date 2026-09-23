# AGENTS.md — aqa (autoquant-agent) 使用指南（供外部 agent：copilot / claude / kimi 等）

本文件是使用本仓库做**模型量化 + 评估**的仓库级操作指令。目标读者是以
**owner 角色**驱动 aqa 的外部 agent。读完本文件再动手；凡本文件与代码冲突，
以代码为准（并修正本文件）。

**新任务入口：直接按下面「硬流程」五步建 plan 开工，不要从中间章节开始。**

## 硬流程：每个模型必走的五步 + 产物契约

**接到任何量化/评估任务，先读这段。** 后面各节是知识库，本段是执行顺序。
**开工第一件事：把下表五步转成你的 plan/todo**（每步一条，带产物与完成判据），
按序推进并即时更新状态——上一步产物缺失不得进入下一步。模板真源在
`templates/`（骨架，禁止写本机绝对路径；填好的实例在 `templates/examples/`、
`cookbooks/`、`docs/RUN_JOURNAL.md`）。所有 runtime 产物最终落进 run 的
`output.dir` 顶层，随 sink 回传（§6，文本自动脱敏）。

| 步 | 做什么 | 产物 | 完成判据 |
|---|---|---|---|
| ① 结构分析与量化方案 | 不下载权重读 config+index；**先查 skill / recipe / knowledge**（查阅清单是硬章节，逐条记采纳或偏离）；三件套探针定推理路径；出逐模块精度表 | `analysis.md` | 查阅清单填满；ignore/量化范围对 index.json 逐条核对；给出量化命令 + 评估计划 + 预期体积 |
| ② 环境 | 按①的推理路径结论：先找本机现有环境，没有则新建 venv（uv 优先，conda 优先于系统 python） | `setup_venv.sh` + `requirements.txt` | 自检通过（架构 registry 命中 = 硬门槛）；自检后 `uv pip freeze` 冻结；**之后每次换依赖必须重跑自检并重冻结** |
| ③ yaml 定稿 | 从 `templates/run.yaml` 出发按①②结论填全（baseline/quant/eval/gate/sink） | `run.yaml`（实际文件名可另起） | `aqa validate` 过 + 回传字段补全（§0 铁律）；后续单跑某环节也从此 yaml 派生 |
| ④ 运行与输出 | `aqa run`；输出结构守 layout 契约；发布权重（`provider.upload`）时写 model card | `model_card.md`（仅发布时必填） | 输出符合 `aqa/report/layout.yaml`（`aqa inspect` 核对）；card 守 `templates/model_card.md` 四条纪律并作为模型 repo README |
| ⑤ 回溯 | 对照产物写报告（不凭记忆）；经验全部路由落点 | `report.md` | sanity check 达标（体积/键 dtype/完备性）；经验路由表无空行；`aqa artifacts check <run_dir>` 全过 |

贯穿全程另有两份产物：`journal.md`（叙事日志：每步命令+依据+坑，**随手记
不是收尾补记**，粒度参照 `docs/RUN_JOURNAL.md`）与 `owner_actions.jsonl`
（机器可读动作，`aqa log-action` 追加）。journal 记"为什么"，log-action 记
"做了什么"，不重复。

收尾完成的定义 = ⑤的判据全过。⑤的经验路由包括**回流本仓**——cookbooks
实例、recipes pin、skills/AGENTS.md/打包层知识、patches，都要落到本仓
对应位置并标明路径（**git 提交交给 user，agent 不擅自提交**；只回传 HF
不回流本仓 = 知识只存了一半）。`aqa artifacts check
<run_dir>` 做确定性校验：产物文件存在、必备章节齐全、report 的 gate 结论与
`gate.json` 一致、发布权重时 `model_card.md` 在场。

## 文档地图

| 主题 | 位置 |
|---|---|
| **本文件（owner 操作总纲）** | `AGENTS.md` |
| yaml 字段参考 | `docs/usage/yaml_reference.md` |
| 环境构建（三形态 + 安装内容） | `docs/usage/environments.md` |
| 内部 agent 输入/输出（optimize + fix-loop） | `docs/usage/agents_io.md` |
| owner 与 aqa 的接口契约（含无 owner 的纯 CLI 模式） | `docs/usage/owner_interface.md` |
| 补丁转 PR（fork 同步 + worktree + gh） | `docs/usage/publish_patch.md` |
| 环境变量参考（密钥/行为/场景速查） | `docs/usage/env_vars.md` |
| 落盘结构契约（本地 + sink 远端） | `aqa/report/layout.yaml` + `docs/run_layout.md` |
| auto-round 知识刷新机制 | `docs/knowledge_refresh.md` |
| knowledge/session 沉淀机制 | `docs/usage/knowledge_sessions.md` |
| owner 收尾 skill（session 沉淀） | `skills/session-sediment/SKILL.md` |
| owner 回写 recipe 站 skill（量化完更新 INC-Quantization-Recipes） | `skills/recipe-page/SKILL.md` |
| day-0 新架构 playbook | `docs/day0_quantization_recipe.md` |
| auto-round 全参数参考（file:line 引用） | `docs/autoround_full_reference.md` |
| 结构分析 skill（逐模块精度判定） | autoround-quant-analysis/SKILL.md（部署机本地文件，不在本仓，向 owner 要路径） |
| 内部 skill（provision 给 aqa 的 agent） | `skills/{quantize,evaluate,fix-and-patch}/SKILL.md` |
| **产物模板（硬流程骨架）** | `templates/`（实例在 `templates/examples/` 与 `cookbooks/`） |

## 0. 角色定位与铁律

- **你是 owner，aqa 是执行器。** aqa 内部的 agent fix-loop / optimize agent
  只是执行器的一部分，它的命令推荐、修复、补丁你都要把关，不要盲目放行。
  **把关要看行为而不只是结果**：结果异常或拿不准时，必须翻这两个 agent 的
  session 文件——fix-loop 各轮修复的对话与命令在 `logs/agent_fixes/<phase>/`，
  optimize 的推荐过程在 `logs/agent_optimize/`；确认它改的是对的，而不是
  只看它最后说"修好了"。
- **先 yaml，后运行。** 任何量化/评估任务必须先写出运行 yaml 配置，再执行
  `aqa run <yaml>`。不允许徒手拼 auto-round / lm_eval 命令绕过 aqa。
  **写 yaml 时回传字段必须补全**（`aqa validate` 会做确定性提醒）：
  env.sh 提供了 `LOCAL_RESULTS_DATASET` + `LOCAL_RESULTS_HF_TOKEN` 就配
  `output.sinks: [local_fs, hf_dataset]`；`agent.fix_loop` 开着就配
  `agent.knowledge.repo`（knowledge 与 trajectories 同库回传）。
  不写 = 结果只留本机，违背"全程留痕回传"的铁律。
- **owner 侧全程留痕。** 你的分析过程、运行命令、日志路径、思考与结论、
  所有 fix/patch，都要落盘（推荐落进 run 的 `output.dir` 或 aqa_results 工作目录），
  并随结果回传（见 §6）。可参照本仓库 `docs/day0_quantization_recipe.md` 与
  `docs/RUN_JOURNAL.md` 的记录粒度。
- **确定性优先，agent 兜底。** 能用确定性探针（读 config、查注册表、看版本）
  回答的问题，不要让 agent 猜。
- **owner 保持灵活性，但偏离必须可复现。** 鼓励主动查证与实验：查 GitHub
  最新代码/相关 PR 与 issue、web search、安装更新版本的依赖、切换不满足
  需求的版本（transformers/vllm/auto-round/llm-compressor 等任何相关仓，
  不局限于 auto-round——查询方法见 `skills/fix-and-patch/SKILL.md` 的
  "Upstream lookup playbook"：git ls-remote 查最新 tag、GitHub search API
  按错误串搜 issue/PR、raw 定点读上游源码）。但每一次偏离默认都要落到
  可复现的形式：依赖变更固化进 yaml（`env.packages` 显式 pin），代码修复
  沉淀为 `patches/` 补丁，决策与依据记进 run 日志。**禁止不留痕的手改环境**——
  否则下一次 run 不可复现，上传的 manifest 也在说谎。
- **不与他人抢 GPU。** 启动前 `nvidia-smi` 确认空闲显存，忙就等或换卡。

## 1. 标准工作流（七步）

```
① 模型架构判定（day-0?）→ ② 选定运行环境 → ③ 写运行 yaml → ④ aqa run
→ ⑤ 把关 fix-loop（如开启）→ ⑥ gate/结果核对 → ⑦ 落盘与回传
→ ⑧（可选）补丁处置（publish-patch）+ session 沉淀（session-sediment）
   + recipe 站回写（recipe-page）
```

```bash
source env.sh                          # 密钥与目标 repo（见 §6，勿提交 git，已入 .gitignore）
aqa validate examples/<env>/<run>.yaml # 配置体检
aqa run examples/<env>/<run>.yaml --dry-run   # 打印计划不执行
aqa run examples/<env>/<run>.yaml      # 正式运行
```

除了 CLI，也可以在 owner 的脚本里以 API 形式驱动（与 `aqa run` 同一入口）：

```python
from aqa.config import load_run_config
from aqa.registry import EXECUTORS

cfg = load_run_config("examples/<env>/<run>.yaml")
executor = EXECUTORS.create(cfg.executor.type, **cfg.executor.options)
result = executor.run(cfg)   # result.state / result.output_dir
```

## 2. 运行环境：三种形态

yaml 中 `executor` 块决定形态。**三者共享同一套 staged inner run**：aqa 生成派生
`_run.yaml`（剥掉 env 块、executor 置 local），在目标环境里执行内层 `aqa run`
（`aqa/executors/_remote.py`、`aqa/executors/local_env.py`）。

### 2.1 ssh docker（远程，生产主力）
`executor.type: ssh`。aqa 把本仓库 tar 上传到远端，构建/拉取镜像后 `docker run`
跑内层。安装内容：
- base image：CUDA 默认 `nvcr.io/nvidia/pytorch:26.01-py3`；XPU 为
  `vllm/vllm-openai-xpu:nightly`（`executor.ssh.accelerator` 决定）
- 业务栈（容器 bootstrap）：`aqa`(editable) + `auto-round` + `lm-eval`，
  eval backend 为 vllm 时加 `vllm`，vllm_omni 时加对应栈；可用
  `executor.env.packages` 整体覆盖推导结果（vendor vllm 构建必须用这招防覆盖）
- agent 后端（`agent.fix_loop/optimize` 开启时）：Node v22 + `openclaw@2026.6.8`
  (+按需 `@github/copilot`) + `aqa.agent.provision` 铺配置与 skills
- **torch 保持不动**：XPU 是刻意的（PyPI vllm 会冲掉 xpu 构建）；CUDA 用 NGC 自带。
  非必要不动 torch/vllm；必须动时用 `env.packages` 显式清单，不用推导。

### 2.2 local docker（本机，day-0 常用）
`executor.type: local` + `env.type: docker`。容器内逻辑与 ssh docker **完全一致**
（同一份 bootstrap 代码）；区别只在启动方式（本机 `docker run`、repo/输出目录
bind-mount、host 网络、代理从宿主环境透传）。注意：local docker 目前只支持
CUDA（dockerfile 只解析 `gpu.dockerfile`）。
day-0 典型用法：model card 指定了官方定制镜像（如
`vllm/vllm-openai:qwen38-flash-next`）时，`image.gpu` 指向它、`dockerfile` 留空
直接用，`env.packages` 里**不要装 vllm**（用镜像自带定制版），示例见
`examples/local_docker/`。

### 2.3 local venv（本机，复用现有环境）
`executor.type: local` + `env.type: venv`。场景：你（用户）已有一个配置好的环境
（含已有 docker container 内），aqa 在其中建/复用一个虚拟环境。
- 新建环境时**优先 conda**（`conda create -p <path> python=3.12`），无 conda 用
  `python3 -m venv`；包安装**优先 uv**（`uv pip install`），无 uv 回退 pip
- `env.create: false` + `env.path` 指向既有环境时原样复用（幂等，不重装）
- `env.packages: auto` 时按与容器相同的推导装（auto-round/lm-eval/vllm 等）
- agent 后端 provision 与容器版对齐（Node 解进 venv 前缀，best-effort）
示例：`examples/local_venv/run_qwen38_flash_next.yaml`。

### 2.4 in-process（最简）
`executor.type: local` 且无 `env` 块：直接在当前进程跑，不做任何环境构建。
适合环境已就绪的纯评估（`examples/eval_only.yaml`）与 diffusion 示例。

## 3. 量化配置（`quant:` 段）

schema：`aqa/config.py` 的 `QuantConfig`（extra=forbid，写错字段会直接报错）。

| 字段 | 说明 |
|---|---|
| `scheme` | MXFP4(默认)/MXFP8/W4A16/NVFP4_E5M3/FP8_STATIC/GGUF 系等 |
| `method` | `model_free`(默认) \| `autoround`（=model-load 路径） |
| `algorithm` | 算法链：rtn/auto_round/awq/hadamard/svdquant（model_free 仅允许 rtn） |
| `iters` / `nsamples` | 0 = RTN；200 量级 = tuning |
| `export_format` | 默认 `llm_compressor`；也可 `auto_round` |
| `ignore_layers` / `layer_config` | 逗号分隔 ignore 清单 / 按层混合精度(JSON) |
| `quant_lm_head` / `opt_rtn` / `hadamard_type` / `enable_alg_ext` | 对应 auto-round 旗标 |
| `static_kv_fp8` / `static_attention_fp8` | **新增**：FP8 KV cache / FP8 attention（`--static_kv_dtype fp8` / `--static_attention_dtype fp8`）。需校准，仅 `method: autoround` |
| `device_map` | 默认多卡 auto；可显式覆盖 |
| `extra_args` | 原样透传 CLI 的逃生口 |

三种方式：`model_free`（免校准、快、不掉点场景首选）、RTN
（`method: autoround, iters: 0`）、tuning（`iters: 200`，慢但通常最优）。
**用户显式配置的字段是硬约束**，optimize agent 推荐也不得违反
（`aqa/agent/optimize.py` 的 constraint 校验）。

经验法则（详见 `skills/quantize/SKILL.md` 与 `docs/autoround_full_reference.md`）：
- 分析模型**不要加载权重**：读 `config.json` + `model.safetensors.index.json`。
- MoE：shared_expert、router/gate、attention 通常保高精度；ignore 清单要对着
  index.json 的实际层名逐条核对（错配 = 静默漏量或过量）。
- 每次量化后必做 sanity check：输出目录体积是否符合位宽预期（如 125B 模型
  W4A16≈181G、FP8≈240G、bf16 全量 360G），并抽查量化键 dtype。
  **体积 = 原始大小说明量化静默失效**（典型根因：auto-round 版本过旧不认
  fused MoE——融合专家拆解需 ≥0.15.0/#2072）。

### 3.1 结构分析（逐模块精度判定）

量化计划的本质是一张**逐模块精度表**（哪些保 bf16、哪些保 fp32、哪些进
mxfp4/mxfp8/w4a16）。判定方法（与 optimize agent 同源的免下载分析）：

1. 读 `config.json`：架构类型、dense vs MoE（`num_experts`/`moe_*`）、
   `layer_types`、MLA/MTP/vision 字段、`tie_word_embeddings`、
   **`quantization_config`（模型可能已是量化权重，如 FP8 源）**。
2. 读 `model.safetensors.index.json` + 抽查张量头的 **dtype/shape**：
   抓融合 3D 专家（`experts.gate_up_proj [E,2I,H]`）、小 N 层（out<128，
   MXFP8 推理核 `mm_mxfp8 requires N >= 128` 的雷）、已量化权重（F8_E4M3）。
3. 按决策表归类：**量化主力**=路由专家（MoE 50-90% 参数）/稠密 MLP；
   **保 bf16**=attention、shared_expert（保守档）、router/gate、embedding、
   linear_attn、hyper_connection、ple、mtp、vision、indexer；
   **保 fp32**=checkpoint 里 dtype 本来就是 fp32 的（部分 norm/scale，逐头确认，
   不要降级它们）；norm/conv/裸参数 auto-round 自动跳过。
4. **owner 必须复核这张表**——它是 optimize agent 的输入也是你的把关点；
   拿不准某模块语义时沿升级阶梯查证（auto-round 源码 → transformers
   modeling 源码 → GitHub 搜索 → web），详见
   `/home/kaokaolv/Pai-Megatron-Patch/autoround-quant-analysis/SKILL.md`。

### 3.2 多 recipe 对比（`aqa sweep`，用户不指定配置时）

用户没给死配置时，不要替用户拍脑袋选一个——用 sweep 跑多个候选档再对比：

```yaml
quant:
  recipe_sweep: true        # 内置候选集（由结构分析生成）
  # 或自定义：
  # recipe_sweep_candidates:
  #   - {name: conservative, scheme: MXFP4, method: model_free, ignore_layers: "..."}
  #   - {name: mixed, scheme: MXFP8, method: model_free, layer_config: {mlp.experts: {bits: 4, data_type: mx_fp}}}
  #   - {name: fp8kv, scheme: MXFP8, method: autoround, iters: 0, static_kv_fp8: true,
  #      eval_model_args: {kv_cache_dtype: fp8}}   # 仅本档评估时追加的 model_args
```

完整可运行示例：`examples/local_venv/run_qwen3_30b_a3b_sweep.yaml`（混合 +
FP8 KV + FP8 attention 三档，复刻手工三变体流程）。

```bash
aqa sweep <yaml> --dry-run   # 打印每档的量化命令与理由（owner 先审再放行）
aqa sweep <yaml>             # 逐档执行（各自独立 run 目录），最后出对比表
aqa sweep <yaml> --only conservative,tuning
```

内置候选档：**conservative**（专家独享，敏感族全 bf16，下限基线）、
**aggressive**（放开 shared_expert/稠密 MLP）、**mixed**（MoE：mxfp8 主体 +
专家 mxfp4）、**tuning**（保守层计划 + iters=200 校准）。
baseline 只测一次（第一档），`sweep_report.json` 含各档分数与命令。
**`aqa run` 遇到 recipe_sweep 会直接报错并指向 `aqa sweep`**。

### 3.3 量化知识的更新机制（references）

auto-round 迭代快，aqa 打包的量化知识会过期。知识分两层：

- **打包层**：`aqa/agent/recipes/*.md`、`docs/autoround_full_reference.md`
  （随 aqa 发布，可能旧）
- **增量层**：knowledge store 的 `references/` 子树（与 fix-knowledge 共用
  HF dataset，run 开始 `sync_down` 拉最新）——optimize agent 的 prompt 知识
  **优先读这一层**，读不到回退打包层

owner 的职责：**开始任务前用 `aqa knowledge show` 看 `auto_round_ref`
（文档验证时的 auto-round 版本），比你将用的 auto-round 旧就触发刷新**——
`aqa knowledge diff --auto-round-dir <repo>` 圈定变更范围，逐条复核后更新
文档并 `aqa knowledge push-references` 推送共享。完整流程与纪律见
`docs/knowledge_refresh.md`。fix-loop / optimize agent 读到的是同一套
references，所以你刷新一次，所有 agent 都受益。

**用不用增量层是可选的**：`agent.knowledge.use_references: false` 会把
optimize 锁回打包层（复现实验、排除知识漂移时用）；**刷新永远是 owner 手动
动作**，run 不会自动更新知识——不刷新就只是继续用现有层，不会有意外变化。




## 4. day-0：新架构判定与对策

拿到新模型先做**三件套支持探针**（确定性，勿猜）：

1. **transformers**：最新 release 是否认 `config.json` 的 `model_type`？
   （`python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(m)"`）
2. **vllm**：最新 release/main 是否注册该架构？（model card、vllm PR/issue 搜索）
3. **auto-round**：是否支持（含 fused MoE 等结构）？

判定细则：
- 模型权重目录里**自带 modeling 文件** → 算支持，`trust_remote_code=True` 即可
  （aqa 后端默认带此参数）。
- 任一不认 → 查 **model card 的推理 recipe**（可能指定 docker image / vllm 版本，
  如 `recipes.vllm.ai/<org>/<model>`）、transformers/vllm 的 PR 与 issue。
- 推理路径确定后选环境：官方定制镜像 → local docker（§2.2）；有明确版本要求
  → local venv 或 ssh docker + `env.packages` 显式 pin。
- 量化侧：model_free 也要 **double check 配置命中**（ignore_layers/layer_config
  与 index.json 层名对齐）；model_free 对不识别的结构可能静默全量跳过
  （用 §3 的体积 check 抓）。
- 需要的修复**沉淀为 patch** 放 `patches/`（该目录的 README 有布局对照与
  防呆说明），不要只在环境里手改不留痕。
- 完整 playbook：`docs/day0_quantization_recipe.md`（含 Qwen3.8-Flash-Next、
  GLM-5.3-Flash 实证）。

## 5. 评估（`eval:` 段）

### 5.1 lm_eval（默认 harness）
`eval.backend: vllm | hf`；benchmarks 列表（默认 piqa/mmlu/hellaswag）；
生成类任务（如 gsm8k，经 `generative_tasks` 或 recipe `type: generative`）自动带
`--apply_chat_template`。每任务参数单一真源在 `recipes/benchmarks/<task>.yaml`。
backend 额外参数走 `eval.lm_eval.model_args`（如 `reasoning_parser`、`max_num_seqs`）。
`eval.baseline: true` 会先测 bf16 原模型再测量化模型（gate 对比的前提）。

### 5.2 thinking 双模式（新）
thinking 模型开/关思考链分数差异巨大。`eval.thinking`：
- `auto`（默认）：检测 chat template 是否含 `enable_thinking` 开关，
  是则**开/关各测一遍**；否则单测（行为同旧版）
- `both` / `on` / `off`：强制双测 / 仅开 / 仅关
- `thinking_off_model_args`（默认 `{enable_thinking: false}`）定义"关"的传参，
  特殊模型可覆盖
产物分目录：`eval/`（on）与 `eval_thinking_off/`（off）；baseline 同理；
gate 按**同模式配对**判定（on 对 on、off 对 off），`gate.json` 含每模式明细。

### 5.3 evalscope（推理类 benchmark）
aime24/25/26、gpqa_diamond 走 `evalscope` harness（`recipes/benchmarks/` 已内置
recipe）：harness 自动 `vllm serve`（TP=num_gpus）→ `evalscope eval` → 收报告。
`eval.evalscope` 透传参数：`repeats`（avg@N）、`generation_config`、
`server_args`（vllm serve 参数）、`port` 等；thinking off 由 §5.2 统一注入
（请求级 `chat_template_kwargs`）。

### 5.4 agent 任务（harbor 风格）
仓库自带 agentic eval 栈（`aqa/agent_eval/`、`aqa/terminal_bench/`、
`aqa/swebench/`，轨迹为 Harbor ATIF 兼容格式）；详见各自模块 docstring 与
`aqa swebench --help`。

## 6. 结果落盘与回传（sink）

### 6.1 落盘结构契约

- **落盘结构是显式契约**：机器可读定义在 `aqa/report/layout.yaml`（single
  source of truth），任何 run 目录用 `aqa inspect <run_dir>` 校验完整性，
  不要凭记忆核对。人类可读版见 `docs/run_layout.md`。
- 本地：`output.dir`（`unique_dir: true` 自动加时间戳并维护 `latest` 软链）。

### 6.2 sink 什么时候触发（pipeline.py 精确时点）

| 时点 | 行为 |
|---|---|
| baseline 评估完 | 逐模式 dispatch 一次（`baseline*/` 目录） |
| run 正常完成 | `_write_terminal_state(DONE)` 后 dispatch **整个 run 目录** |
| run 失败 | `_writeback_on_failure`：终态写 FAILED，补一份带 failure 原因的 manifest，**照样 dispatch 部分产物**，状态标 `Quant Failed` / `Eval Failed`——失败的 run 在榜单上同样可见，不会静默消失 |

每个 sink 失败不中断 run（只告警）。`output.sinks` 默认只有 `local_fs`。

### 6.3 hf_dataset sink 上传什么、传到哪里

**远端内容 = 本地契约（`aqa/report/layout.yaml`）的 stage 子集 + 三个远端附加文件。**
stage 规则（`aqa/report/scrub.py`）：排除 `sink: false` 的条目（`quantized/`
权重、`.aqa_local_*/` staged 工件）与 `<artifact_name>/` 拷贝目录；保留的
**逐文本脱敏**——GitHub token 家族（ghp/gho/ghu/ghs/ghr/github_pat）、hf_、
sk-、AKIA、Bearer 等（所以 layout.yaml 里每个本地条目会原样出现在远端，
结构与 §6.1/`docs/run_layout.md` 一致，不再重复定义）。随后
`upload_folder(repo_type="dataset")`。远端布局：

```
<repo>（如 lvkaokao/lb_local）
├── results/<org>/<artifact>/run_<run_id>/      # = run 目录 stage 子集（结构同 layout.yaml）
├── results/<org>/<artifact>/results_<run_id>.json   # 附加①聚合记录：
│       # {status, model, scheme, method, run_id, results{bench: score}, num_files}
├── status/<org>/<request_filename>.json        # 附加②状态回写（榜单消费）
└── knowledge/                                   # 附加③共享修复知识（配了 knowledge repo 才有）
```

- `org` = 模型 id 的 org 段；`run_id` = 时间戳+短哈希（`_make_run_id`）
- **artifact 命名统一规范**（dataset 路径与 HF model repo 同名，真源
  `aqa/report/names.py`）：
  `<Model>-<Scheme>[-Mixed][-FP8KV][-FP8Attn][-CT][-RTN]-AutoRound`——
  `Mixed`=有 layer_config（前导 scheme 取最低比特）；`CT`=llm_compressor 格式；
  `RTN`=model_free 或 iters=0（紧邻 AutoRound，对齐已发布命名
  `...-W4A16-RTN-AutoRound`）；baseline/聚合分别用 `-BF16-Baseline` /
  `-AutoRound-sweep`。**存量上传的旧命名（`...-AutoRound-<scheme>-<method>`）
  不改**，只对新 run 生效。
- 实证样例（此前 ssh docker 线的真实上传）：
  `hf.co/datasets/lvkaokao/lb_local/tree/main/results/Qwen/Qwen3.5-35B-A3B-AutoRound-MXFP8-model_free/run_Qwen3.5-35B-A3B_MXFP8_20260821-052837_a49094`
- 配置项：`repo / path_in_repo(默认 results) / token_env / full_run /
  sanitize / status / disable_xet`（`aqa/report/sinks.py:107-132`）

启用方式（yaml）：

```yaml
output:
  sinks:
    - type: local_fs
    - type: hf_dataset
      repo: ${LOCAL_RESULTS_DATASET:-lvkaokao/lb_local}
      token_env: LOCAL_RESULTS_HF_TOKEN
```

### 6.4 owner agent 的产物怎么回传

**约定：放进 run 目录，sink 自然带走**——没有第二套机制，也不该有：

- 分析文档 / 决策记录 → run 目录顶层（如 `<Model>_quantization_analysis.md`）或 `logs/`
- owner 的 fix/patch → `aqa capture-patch <run_dir> --repo <git仓>` 把工作区
  diff 捕获到 `patches/owner/`（aqa 内部 fix-loop 的 patch capture 管的是它自己
  改的可编辑依赖，见 `aqa/agent/patch.py`；owner 手改用这条命令对齐）
- 值得上游的补丁 → `aqa publish-patch <patch文件> --repo-dir <fork本地仓>
  --title ...`（先 `--dry-run` 验证 apply）：自动同步 fork 基线、临时
  worktree 打补丁、push 分支、`gh pr create`。token 只走环境变量（GH_TOKEN）
- **run 期间的环境变更/额外命令/决策 → `aqa log-action <run_dir> --kind ...`
  追加进 `owner_actions.jsonl`**，run 结束时并入 `run_manifest.json` 的
  `owner_actions` 段（无人模式自然缺省）。接口契约全文见
  `docs/usage/owner_interface.md`
- 上传前所有文本过 `scrub.py` 脱敏（token/口令正则集），权重与 staged 工件
  （`.aqa_local_*/`）结构性排除——所以**放心放文本，别放权重**

### 6.5 量化模型权重

默认**不外传**。需要发布时配 `provider.upload: {repo, token_env}` →
`HfModelSink` 推到 HF **model** repo（自动创建仓库 + 生成模型卡）。
模型卡除精度表外带 **Reproduce & Environment** 章节：实际量化命令、评估命令、
关键包版本（auto-round/lm_eval/transformers/torch/vllm，取自本次 run 的
manifest）与推理提示；评估失败但量化成功时也会兜底推送。

### 6.6 密钥链（env.sh → yaml → sink）

`env.sh`（**勿提交 git**，已入 .gitignore）只定义变量；yaml 里 `${VAR}` /
`token_env` 引用；运行时 `os.environ` 消费：

| 变量 | 用途 |
|---|---|
| `GPU_HOST` / `GPU_SSH_PASS` | ssh 执行器地址与密码（密码不进容器） |
| `HF_TOKEN` | 模型下载 + sink token 兜底 |
| `LOCAL_RESULTS_DATASET` | hf_dataset sink 的目标 repo |
| `LOCAL_RESULTS_HF_TOKEN` | dataset 写 token（`token_env` 按名读取） |
| `MINIMAX_API_KEY` / `COPILOT_GITHUB_TOKEN` | agent 后端 tier（白名单转发进容器） |

## 7. examples 导航

```
examples/
├── ssh_docker/    # 远程生产（run_5000 系列、run_b60_xpu、run_ssh、muse spark b200）
├── local_venv/    # 本机 venv（run_qwen38_flash_next）
├── local_docker/  # 本机容器/day-0（qwen38 docker+tuning、glm53 全系、eval_*）
└── *.yaml         # in-process（eval_only、diffusion 系列、muse local、deepseek-ocr2）
```

写新 yaml 时**就近抄同类**：ssh 生产抄 `ssh_docker/run_5000_qwen3.8_27B.yaml`；
day-0 定制镜像抄 `local_docker/run_qwen38_flash_next_tuning.yaml`。

## 8. 验证与测试

```bash
aqa validate <yaml>            # 配置体检（pydantic schema）
aqa run <yaml> --dry-run       # 打印执行计划
aqa compare <baseline_dir> <eval_dir>   # 事后对比 accuracy.json
aqa artifacts check <run_dir>  # 硬流程产物校验（收尾必过）
pytest tests/ -q               # 全量单测（改动后必跑）
```

## 9. 常见坑（务必避开的血泪清单）

1. **量化体积 = 原模型** → 量化静默失效，查 auto-round 版本（fused MoE 需
   ≥0.15.0）与 ignore 清单命中。
2. **手工 patch 版本错位** → `patches/` 的 README 有两种文件布局对照；
   不要把 patch 手工移植到别的基线。
3. **vendor vllm 被 PyPI 覆盖** → 用 `env.packages` 显式清单。
4. **thinking 模型单测一边** → 用 `eval.thinking` 双模式，gate 按模式配对。
5. **权重误传 HF** → 权重只走 `provider.upload`；dataset sink 永远排除权重。
6. **env.sh 提交进 git** → 里面是明文密钥。
