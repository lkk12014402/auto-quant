# RFC 0001 — 可控、可复现、可 resume 的量化 Session 状态机（借鉴 Hyperloom）

状态：Draft · 作者：owner agent · 对标：[AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom)

> 本 RFC 回答四个长期痛点：**agent 量化过程不可控、不可复现；每阶段产物/内容的结构化
> 不固定；知识复用缺失；agent session 不能 resume。** 结论是：把现在靠 `AGENTS.md`
> 纪律要求 owner 手动保证的东西（按序推进、产物齐全才进下一步、留痕、把关、知识回流），
> **下沉成代码里的 Session 状态机 + 阶段门 + Recipe KB**。文档从"操作手册"退回"契约镜像"。

---

## 0. TL;DR

| 支柱 | 现状问题（代码实证） | 目标 | 对标 Hyperloom |
|---|---|---|---|
| S1 Session 一等对象 | `aqa_run.json` 只有 `{model, state}`；无阶段进度 | `session/` 下 `manifest.json`+`state.json`+`events.jsonl` | `manifest.json` / `state.json` / `storage/coordinator.db` |
| S2 真 resume | `--resume` 透传到 executor 后**被忽略**（见 §1-B） | 加载 state → 跳过 DONE 阶段 → 从首个非 DONE 重入 | `--resume-from $SESSION_DIR` |
| S3 阶段产物契约 | `layout.yaml` 只校验**终态**，不构成阶段门 | 每阶段 precondition→run→postcondition，产物即门 | phase 链 + `allowed_actions_for(phase)` |
| S4 Recipe KB | `recipe_store.pin/load` **全仓无调用方**（死代码） | arch/scheme→获胜配方的捕获 + 逐字段放宽热启动 | Recipe KB（relax one field at a time） |
| S5 把关内建 | 靠 owner 手翻 `logs/agent_fixes/` session 验证 | propose(agent) 与 decide(gate+Critic) 分离 | Coordinator/Critic/PolicyGate |
| S6 CLI/文档对齐 | AGENTS.md 引用 6 个未实现命令（见 §1-A） | 实现或从文档删除；AGENTS.md 瘦身 | "observable artifacts are the contract" |

落地顺序：**P0（诚实化）→ P1（Session 状态机 + 真 resume）→ P2（阶段门）→ P3（Recipe KB）→ P4（Critic/PolicyGate）**。

---

## 0.1 实现状态（本次提交 · 2026-10）

> **P0 + P1-core + P2 + P3 + P4 + S3 已落地**（分支 `auto-run`）。下表记录已落地项与其产物；设计正文（§2–§9）保持不变。

| 项 | 状态 | 落地文件 |
|---|---|---|
| `aqa inspect <run_dir>` 接 CLI（终态产物契约校验，INCOMPLETE 退 1） | ✅ | `aqa/cli.py` |
| `--resume` **诚实化** → 真 resume（local executor；ssh 显式报错，RFC §9.2） | ✅ | `aqa/cli.py`、`aqa/executors/local.py`、`aqa/executors/ssh.py` |
| Session 一等对象：`session/manifest.json`+`state.json`+`events.jsonl`（原子写、缺失即拒 resume） | ✅ | 新增 `aqa/session/`（`state.py`/`manifest.py`/`events.py`/`session.py`/`_io.py`） |
| `run_pipeline` 重构为 `build_phase_plan` 驱动的 phase 循环（enter→run→done，按产物跳过已完成阶段） | ✅ | `aqa/pipeline.py` |
| 真 resume：checkpoint 在场则跳过 quantize（最贵阶段），半成品重跑 | ✅ | `aqa/pipeline.py`（`_quant_checkpoint_present`、`should_skip`） |
| `aqa session {state\|report\|events}` operator 工具 | ✅ | `aqa/cli.py` |
| `read_eval_result`：从 `accuracy.json` 复原 `EvalResult` 供 resume 跳过 baseline/eval | ✅ | `aqa/report/__init__.py` |
| 测试：session 单测 + pipeline resume 集成 + CLI P0 | ✅ | `tests/test_session.py`、`tests/test_cli_p0.py` |
| AGENTS.md 瘦身：顶部「命令可用性」状态框 + 逐处标注未接入 CLI 的命令（`sweep`/`knowledge`/`log-action`/`capture-patch`/`publish-patch`），quick-ref 补 `inspect`/`session`/`--resume` | ✅ | `AGENTS.md` |
| **P2 阶段门**：`layout.yaml` 升级为**逐阶段契约**（每条目带 `phase` + 新 `session`/`session_quantize`/`session_done` 判据）；4 个机读阶段产物 `env_lock.json`(ENV)/`analysis.json`(ANALYZE)/`quant_plan.json`(QUANT)/`eval_report.json`(EVAL) 由 `run_pipeline` 落盘；`aqa inspect` 按 Phase 汇总 present/MISSING | ✅ | 新增 `aqa/report/phase_artifacts.py`；`aqa/report/layout.py`、`aqa/report/layout.yaml`、`aqa/pipeline.py`、`tests/test_phase_artifacts.py` |
| P2 写回文案对齐：QUANT 失败写回 `status=Quant Failed`（原误写 `Eval Failed`） | ✅ | `aqa/pipeline.py`（`_PHASE_FAIL_STATUS`/`_failure_status`） |
| **P3 Recipe KB**：`recipes/quant/*.json` 的 `RecipeStore.load/pin` 死代码接入 run loop——ANALYZE 阶段按**渐进字段放松**（model id→架构 family→scheme）查最近的已验证配方，命中即作为 **hot-start** 注入 agent-optimize（advisory，不改复现）；CLOSE 阶段 **strict pin-back**（仅 gate=GO 才写，落独立 `auto/` 命名空间，合并同模型多 scheme + union lessons/failures）。审计产物 `recipe_kb.json`（query+命中/未命中），`recipe_kb_hit` 事件 + `ANALYZE` decision | ✅ | 新增 `aqa/agent/recipe_kb.py`、`tests/test_recipe_kb.py`；`aqa/pipeline.py`（`_recipe_kb_hot_start`/`_recipe_kb_pin` + `hot_start` 线程）、`aqa/config.py`（`RecipeKBConfig`）、`aqa/recipe_store.py`（复用 `RepoDirStore`/`_safe_key`）、`aqa/report/layout.yaml` |
| **P4 Critic/PolicyGate + state 重播种**：fix-loop 的 keep/revert 本就由权威 **re-run** 决定（agent VERDICT 只驱动升级/防漂移），故 P4 为**默认 advisory 加法**——（1）**Critic**：把 agent 自述（FIXABLE/UNFIXABLE/ESCALATE）与真实 re-run 结果比对，写 6 态标定（confirmed/overclaimed/silent_fail/conceded/understated/escalated，`honest` 标志）为 `critic_calibration` 事件，纯观察不改行为；（2）**PolicyGate**：对 `combined_diff()` 做**具名规则**审查（`edit_tests`/`skip_tests`/`disable_cuda`/`touches_vcs_ci`/`mass_deletion`），只读不执行 diff，命中写 `policy_violation`（点名规则）；`enforce=true`（opt-in）才 `patch_capture.restore()` 回滚并写 `policy_revert`；（3）**state 重播种**：每轮 fix prompt 从 `state.json`（mission+baseline+current_best+decisions）而非 transcript 重新播种（`SessionState.prompt_seed()`）注入权威任务块。全部 hook 为 `None`/`""` 时 fix-loop 字节级等价旧行为 | ✅ | 新增 `aqa/agent/critic.py`、`aqa/agent/policy_gate.py`、`tests/test_critic.py`、`tests/test_policy_gate.py`、`tests/test_fix_loop_p4.py`；`aqa/agent/fix_loop.py`（`_emit`/`_policy_review`/`_critic_review` + 4 新参数）、`aqa/agent/prompt.py`（`state_seed`）、`aqa/session/state.py`（`prompt_seed`）、`aqa/config.py`（`PolicyGateConfig` + `AgentConfig.critic`/`policy_gate`）、`aqa/pipeline.py`（`_fix_hooks` + `session` 线程） |
| **S3 phase-action 守卫 + 有界 re-QUANT 回环**：补齐与 Hyperloom `PHASE_ALLOWED_ACTIONS`/`allowed_actions_for(phase)` 对标的**动作级 precondition 守卫**与**宏周期回环**——（1）`aqa/session/actions.py`（纯数据+谓词）：`PHASE_ALLOWED_ACTIONS`（每 phase 可提议动作白名单，`GATE=∅` 为 Coordinator 保留）/`allowed_actions_for`/`PHASE_ALLOWED_TRANSITIONS`（沿 plan 前进 + 唯一回边 `GATE→QUANT`，`EVAL→QUANT` 静默重量化**非法**）/`is_action_allowed`/`is_transition_allowed`/`PhaseActionDenied`；（2）**Coordinator 守卫**：`Session.guard_action(action, phase=当前)` 对非法动作写 `phase_action_denied`（+decision），`agent.action_guard.enforce=true` 才抛 `PhaseActionDenied`；`Session.enter()` 对每次相位切换标注 `phase_transition`（forward/loopback/irregular）；契约产物 `session/allowed_actions.json`（open_session 写，供外部 proposer 读）；（3）**fix prompt 注入**当前 phase 允许动作块（`build_fix_prompt(allowed_actions=…)`）；（4）**有界 `GATE→QUANT` re-QUANT 回环**（opt-in，默认关→单程行为不变）：gate 非 GO 时按 recipe-sweep 候选逐周期重量化→重评→重判，隔离到 `requant/cycle_<n>/`，keep-best 后 promote 到 canonical 路径；护栏移植 Hyperloom（`max_cycles`/`min_gain`/`no_gain_patience` 的 `GLOBAL_CONVERGENCE_NO_GAIN_CYCLES`），纯控制器 `_RequantController` 可单测 | ✅ | 新增 `aqa/session/actions.py`、`tests/test_actions.py`；`aqa/session/session.py`（`guard_action`+transition 标注+写契约）、`aqa/config.py`（`ActionGuardConfig`/`RequantConfig`）、`aqa/agent/prompt.py`+`aqa/agent/fix_loop.py`（`allowed_actions` 线程）、`aqa/pipeline.py`（`_RequantController`/`_run_requant_loop`/`_promote_cycle`/`_allowed_actions_block`）、`aqa/report/layout.yaml` |

**P4 启用说明**：Critic + PolicyGate 审查**默认开**（纯写事件，零行为变更）；PolicyGate **回滚**默认关，需显式 `agent.policy_gate.enforce: true` 才在命中具名规则时 `restore()` 回退该次 agent 编辑。关闭标定可设 `agent.critic: false`，关闭审查设 `agent.policy_gate.enabled: false`。

**S3 启用说明**：phase-action 守卫**默认开但 advisory**（`agent.action_guard.enabled=true`，只写 `phase_action_denied` 事件不拦截）；需 Coordinator 硬拦截时设 `agent.action_guard.enforce: true`（越界动作抛 `PhaseActionDenied`）。re-QUANT 回环**默认关**，设 `gate.requant.enabled: true`（+`max_cycles`/`min_gain`/`no_gain_patience`）才在 gate 非 GO 时触发；关闭即保持历史单程 quantize→eval→gate 行为逐字节不变。回环只走 `GATE→QUANT` 唯一合法回边，`EVAL` 内不含 `requantize`/`quantize` 动作，故外部/内部 proposer 都无法把"静默重量化"塞进 EVAL（绕过 gate / 乱账 / 断审计）。

**未接入 CLI 的命令**（留待后续 / owner 决定）：`sweep` 逻辑已在 `aqa/quantize/candidates.py`；`knowledge`/`log-action`/`capture-patch`/`publish-patch` 按 §7 分期补齐。
> 注：本分支存在**与 session 层无关的既有红测**（`test_sweep`/`test_owner_actions`/`test_local_env`/`test_redact`/`test_thinking_*`/`test_traces_*` 等，以及 `test_export_repair`/`test_hooks` 的 collect 失败）——本次改动**未新增**任何红测（已通过 stash 对照验证），这些属分支自身待修。

### 真实模型验证（2026-10-13 · `Qwen3.8-27B`）

在本机真实模型 `/home/kaokaolv/models/Qwen3.8-27B`（`qwen3_5` 多模态，bf16≈54GB）上，
用空闲 GPU 1 做了一次分层验证，**P0 + P1-core 全部符合预期**。完整轨迹（命令/依据/结果 +
真实 `state.json`/`events.jsonl` 内嵌）见 [`0001-validation-journal.md`](./0001-validation-journal.md)。

| 项 | 实测 |
|---|---|
| P0 `aqa inspect`（真实 dir） | INCOMPLETE + exit 1 + 产物清单 ✅ |
| P0 `--resume` 无 session | exit 1 + `no session manifest to resume …` ✅ |
| P1 计划构建 | quantize→`QUANT→EVAL→GATE→CLOSE`；pretrained→`EVAL→CLOSE` ✅ |
| P1 三件套落盘（原子写） | manifest/state/events 齐全 ✅ |
| P1 失败写回 | `QUANT=FAILED` + `phase_fail` 事件 ✅ |
| P1 `aqa session state\|report\|events` | 均可用 ✅ |
| P1 resume **重跑**未完成相位（FAILED） | 第二次 `phase_enter`，不跳过 ✅ |
| P1 resume **跳过**已完成相位（DONE+产物） | `phase_skip` + exit 0 ✅ |

> 环境/兼容性旁证（**与本改动无关**）：`~/.venv-aqa-gpu` 当时未装 `auto-round`/`lm_eval`，
> 故真实量化/评估未实际执行——反而顺带覆盖了**失败路径**与**无 benchmark 完成路径**。
> 一处 cosmetic（QUANT 失败时写回文案误写 `status=Eval Failed`）已在 P2 修复：新增
> `_PHASE_FAIL_STATUS`/`_failure_status(failed_phase, reason)`，按真实失败相位输出
> `status=Quant Failed`，与 `state.json` 的 `QUANT=FAILED` 对齐（已真实重跑复验）。

### 真实全链路量化验证（2026-10-13 · `Qwen3.8-27B` W4A16）

P4 收尾后在同一真实模型上跑了一次**完整的** quantize→eval→gate→close 周期（此时 venv 已装齐
`auto-round 0.15.0`/`lm_eval 0.4.12`/`vllm 0.27.0`，故这次是**真实执行**，非失败旁证）。config
`W4A16 / model_free`（免校准 RTN）、单基准 `piqa` 带 bf16 baseline、agent off、GPU 1（不抢卡）。
run_id `Qwen3.8-27B_W4A16_20261013-223001_e8f3e1`。

| 项 | 实测 |
|---|---|
| 量化（auto-round model_free RTN） | 400/607 层量化，`visual`/`linear_attn.in_proj`/`lm_head` 全精度保留（符合 `ignore_layers`），导出 llm_compressor 19GB，tuning 437s ✅ |
| baseline piqa（bf16，vllm） | `acc=0.8090`（acc_norm 0.8156）✅ |
| W4A16 piqa（compressed-tensors，vllm MacheteLinearKernel 可加载） | `acc=0.8058`（acc_norm 0.8188）✅ |
| gate | `GO`（gap −0.33pt，阈值 2pt）✅ |
| P1 状态机 | `state.json` phases BASELINE/QUANT/EVAL/GATE/CLOSE 全 `DONE`；`events.jsonl` = `session_start`×1 + `phase_enter`×5 + `phase_done`×5 ✅ |
| P2 阶段产物 | `analysis.json`(qwen3_5 骨架 64 层 multimodal)、`env_lock.json`、`quant_plan.json`(含冻结命令+`command_source=builder`)、`eval_report.json`(quant/source/gap 合成) 全落盘 ✅ |
| `aqa inspect` | 各相位产物 present；唯一 `logs/` MISSING → INCOMPLETE（见下）|

> 两点**诚实说明**（均非 P4 回归）：
> 1. `aqa inspect` 报 INCOMPLETE 仅因 `logs/` 缺失——`logs/executor_*.log` 只由 **venv/docker dispatch** 路径（`local_env.py`）写；本次用裸 in-process `executor.type: local`（无 `env:` 块）+ agent off，既无 executor 子进程、也无 agent_optimize/agent_fixes 产物，故 `logs/` 合理缺席。这是 `logs/ (always)` 契约对裸本地模式偏严的既有缺口，与 P0–P4 无关（本改动从未触及 executor 日志）。
> 2. **未见** `critic_calibration`/`policy_violation` 事件——因 `agent.fix_loop: false` 本次零失败、fix-loop 从未触发；P4 这两条钩子只在 fix-loop 失败时发事件（已由 `tests/test_critic.py`/`test_policy_gate.py`/`test_fix_loop_p4.py` 共 26 项覆盖）。同理 agent off → 无 P3 `recipe_kb.json` 热启动决策。
> 环境修复留痕：`~/.cache/huggingface/datasets` 为 root 所有、不可写 lock → piqa 评估曾报 `PermissionError`；改用 `HF_HOME=~/aqa_hf_cache`（executor 透传白名单内）+ 拷入已缓存 piqa 后复跑通过。run config/log 存 session workspace。

---

## 1. 诊断：文档承诺 vs 代码实现的漂移

全部结论可在当前代码复现。

### A. 过程不可控 —— AGENTS.md 建立在不存在的命令上

`aqa/cli.py` 实际注册的命令：`run / validate / eval / compare / list / swebench / traces`
（源码另有 `artifacts check`）。而 `AGENTS.md` 的硬流程、owner 接口、知识刷新三大节
依赖以下命令，**它们在 `aqa/cli.py` 里没有实现**：

| AGENTS.md 引用 | 实际状态 |
|---|---|
| `aqa inspect <run_dir>` | 只有库函数 `aqa/report/layout.py:inspect_run_dir`，**未接 CLI** |
| `aqa sweep <yaml>` | 未实现（§3.2 整节、`recipe_sweep` 分支都指向它） |
| `aqa knowledge show/diff/push-references` | 未实现（§3.3 知识刷新全靠它） |
| `aqa log-action <run_dir>` | 未实现（硬流程贯穿产物 `owner_actions.jsonl` 靠它） |
| `aqa capture-patch` / `aqa publish-patch` | 未实现（§6.4 owner 回传靠它们） |

外部 agent 照 AGENTS.md 执行，会在第一处卡住或开始"幻觉命令"。这是"不可控"的根因。

### B. session resume 是空承诺

`aqa/cli.py:run` 接受 `--resume` 并 `executor.run(cfg, dry_run=..., resume=resume)`，但：

- `aqa/executors/local.py:LocalExecutor.run(self, run_cfg, *, dry_run, resume)` —— **形参 `resume` 完全未被使用**，直接 `run_pipeline(run_cfg, out_dir)` 从头跑。
- `aqa/executors/ssh.py:SshExecutor.run` —— 同样忽略 `resume`。
- `aqa_run.json` 仅 `{"model":..., "state": running|done|failed}`，**没有阶段进度**。

后果：eval 在数小时量化之后崩溃 → 整条流水线从 baseline 重新开始、重新量化。

### C. 没有阶段状态机 / 落地的 session 状态

`aqa/pipeline.py:run_pipeline` 是线性 Python 控制流：

```
baseline eval → _resolve_model(=quantize) → evaluate → gate → build_manifest → sinks
```

过程中的 baseline、current_best、每轮 fix 的决策只活在 transcript 与分散日志里，
没有一个 `state.json` 把"我在哪个阶段 / 已决策什么 / 当前最优"固化下来。

### D. 知识复用只有 fix 层，recipe 层是死代码

- `aqa/agent/knowledge.py:KnowledgeStore` 做得好：按**错误签名**跨模型复用 patch/lesson，
  `find_patches` 有 4 级 rank，HF dataset 回传。
- 但**没有 Recipe KB**：没有"arch/scheme → 获胜配方 + 分数 + 教训"的捕获与热启动查找。
- `aqa/recipe_store.py:RepoDirStore.pin/load` 在 `registry.py` 注册了，但 **grep 全仓无调用方**。
  `optimize.py` 的 docstring 说"pin 其提议的 recipe"，实际没有任何代码调 `.pin()`。
- 结果：`quant.recipe: auto` 每次从零跑 optimize；"什么配方成功过"散落在 `cookbooks/`（散文）
  和外部 recipe 站，不可查询、不自动复用。

### E. 把关责任外包给人

`AGENTS.md §0` 要求 owner "结果异常时翻 `logs/agent_fixes/<phase>/` 和 `logs/agent_optimize/`
确认 agent 改的是对的"。即 harness 把本应由 harness 承担的把关，交给了人的纪律。
aqa 目前信任 agent 自报的 `VERDICT`（`aqa/agent/verdict.py`）。

### F. AGENTS.md 过载

27.6KB，三套重叠且编号打架的流程（硬流程**5 步** / §1 标准工作流**七步** / §1 代码块 **①–⑧**），
参考资料（quant 字段、eval 模式、sink 布局、env vars、day-0）与执行步骤混排。

---

## 2. Hyperloom 借鉴映射（已核对其源码）

| Hyperloom 机制 | 源码锚点 | 解决 aqa 的 |
|---|---|---|
| Session 一等对象：`manifest.json`(schema 版本化、原子写、缺失即拒 resume) | `inference_optimizer/session/manifest.py`（`build/write/load_manifest`，`SCHEMA_VERSION`） | S1、S2 |
| `state.json` 状态机（load-or-init、from_dict/to_dict） | `orchestrator/state/shared_state.py:SharedState.load_or_init/state_path` | S1 |
| `--resume-from $SESSION_DIR`：保留 baseline/current best/搜索状态/事件历史，清陈旧 stop_reason | `agent_backends/session_resume.py` + `inference_optimizer/SKILL.md` Resume 段 | S2 |
| 显式 phase 链 + 每阶段 allowlist | `PRELUDE→ENABLEMENT→FRAMEWORK→KERNEL→SWEEP→CLOSE`，`machine_state.PHASE_ALLOWED_ACTIONS` / `allowed_actions_for(phase)` | S3 |
| Coordinator 跑 benchmark、agent 预测收益**只校准不决定 keep**、失败即回滚 | optimization-loop 文档 "Benchmarks are run by the coordinator, not the model" | S5 |
| Critic（每次 keep-or-revert 写学习记录）、Specialist（只返回 reviewed diff，不决策） | 角色表 Orchestration/Critic/Specialist | S5 |
| PolicyGate：隔离 worktree + 统一 diff 门 + 策略检查 + Critic 签字 + 复测；拒绝点名规则 | optimization-loop "Risky source edits go through..." | S5 |
| Recipe KB：开跑前读、收尾写；逐字段放宽查找做热启动；入库严格、失败是证据非否决 | README "Recipe KB" 段 | S4 |
| Operator 工具 `session <breakdown\|report\|events\|state>` | README operator tools | S6 |

---

## 3. 目标设计

### 3.1 Session 作为可复现、可 resume 的一等对象（S1/S2）

在 run 的 `output.dir` 下新增 `session/` 子树（与现有产物并存，不破坏 layout 契约）：

```
<run_dir>/
  session/
    manifest.json      # 复现清单：schema_version / run_id / 解析后 run.yaml / env lock / 栈版本
    state.json         # 状态机：phase + phases{} + baseline + current_best + decisions[] + gate
    events.jsonl       # 追加式事件流（取代"发生了什么"散落在日志里）
  ...（quantized/ eval/ baseline/ gate.json run_manifest.json 等现有产物不变）
```

- 新增 `aqa/session/state.py`，仿 `SharedState`：

```python
class Phase(StrEnum):
    ANALYZE = "ANALYZE"; ENV = "ENV"; PLAN = "PLAN"
    QUANT = "QUANT"; EVAL = "EVAL"; GATE = "GATE"
    PUBLISH = "PUBLISH"; CLOSE = "CLOSE"

@dataclass
class SessionState:
    schema_version: int
    run_id: str
    phase: Phase
    phases: dict[Phase, Status]       # PENDING|RUNNING|DONE|FAILED|SKIPPED
    baseline: dict | None
    current_best: dict | None
    decisions: list[dict]             # {phase, what, why, ref}
    gate: dict | None

    @classmethod
    def load_or_init(cls, session_dir: Path) -> "SessionState": ...
    def commit(self, session_dir: Path) -> None: ...   # 原子写
```

- `manifest.json` 迁移/升级现有 `aqa/report/manifest.py:RunManifest`（已有 `schema_version`、
  `_package_versions`、`_accelerator`），加入解析后的 run.yaml 与 env lock，开跑即写一次
  （不再只在成功收尾写）。

**真 resume**（`aqa/executors/local.py`）：

```python
def run(self, run_cfg, *, dry_run=False, resume=False):
    session_dir = out_dir / "session"
    if resume:
        if not (session_dir / "manifest.json").exists():
            raise AqaError("no session to resume under %s" % out_dir)  # 宁可拒绝也不假装
        state = SessionState.load_or_init(session_dir)
    else:
        state = SessionState.new(run_cfg, run_id)
    run_pipeline(run_cfg, out_dir, state=state)   # pipeline 按 state 跳过 DONE 阶段
```

### 3.2 run_pipeline 重构为 phase 循环（S1/S3）

把 `run_pipeline` 的线性流重构成显式阶段循环。**每个阶段三段式**：
`precondition（所需产物在场且校验）→ run → postcondition（产物落盘 + commit state + append event）`。

```python
def run_pipeline(run_cfg, out_dir, state):
    plan = build_phase_plan(run_cfg)        # 按 provider/eval.baseline/provider.upload 裁剪阶段
    for phase in plan:
        if state.phases[phase] == DONE:     # resume：跳过已完成
            continue
        require_artifacts(phase, out_dir)   # 阶段门：缺前置产物即拒绝进入
        state.enter(phase); state.commit(session_dir)
        run_phase(phase, run_cfg, out_dir, state)
        verify_artifacts(phase, out_dir)    # postcondition：产物齐全才算 DONE
        state.done(phase); state.commit(session_dir)
```

阶段切分直接来自现有 `run_pipeline` 已经在做的事，不是凭空设计——见 §4 的产物对照表。

### 3.3 阶段产物契约：从"终态校验"升级为"阶段门"（S3）

- 扩展 `aqa/report/layout.yaml`：给每条目加 `phase` / `produced_by` / `required_for` 字段，
  让它同时是**产物清单**和**阶段依赖图**。
- 每阶段产出**机器可读 + 散文双份**：现在只有 `.md` 的加 `.json` 兄弟（见 §4）。
- `inspect_run_dir` 增加"某阶段是否可进入"查询；把 `aqa inspect` 真正接上 CLI。

### 3.4 Recipe KB（S4）

新增 `aqa/agent/recipe_kb.py`，复用 `KnowledgeStore` 的 HF dataset 回传管道
（dataset 里 `references/` 旁边放 `recipes/`）。

- 一行 = `key{arch, model_type, scheme, precision, hw} → recipe + result + lessons[] + failures[]`。
- **热启动查找**：逐字段放宽（model→arch→model_type→scheme→precision→hw），命中即作为
  optimize agent 的起点 prompt（过去失败作为证据注入，不硬否决）。
- **接活死代码**：`CLOSE` 阶段调 `recipe_store.pin()` 严格入库（仅过 gate 的），
  `ANALYZE` 阶段调 `recipe_store.load()` / `recipe_kb.lookup()` 读。

### 3.5 Critic / PolicyGate 内建（S5）

- 分离 **propose（agent）** 与 **decide（确定性 gate + Critic）**：agent 的 `VERDICT` 与预测收益
  只作为校准记录进 `events.jsonl`，**keep/revert 由 coordinator 复测 + gate 裁决**（aqa gate 已是确定性，升为唯一裁决者）。
- optimize/fix-loop 的 prompt **从 `state.json` 重新播种**（mission + 已决策 + current_best），
  而非从 transcript——Hyperloom 的抗漂移核心。
- 把 `aqa/agent/patch.py` 的 patch capture 正式包成 **PolicyGate**：隔离 worktree → 统一 diff →
  策略检查 → 复测 → 通过才成为新基线；拒绝时在 `events.jsonl` 点名规则。

### 3.6 CLI 与 AGENTS.md 对齐（S6）

- **要么实现、要么删除**那批承诺命令。优先实现控制面核心：
  `aqa session {state|report|events|resume}`、`aqa inspect`、`aqa sweep`、`aqa log-action`；
  `knowledge / capture-patch / publish-patch` 次之。
- **AGENTS.md 瘦身**：顶部只留**唯一**一份 phase 契约（= 代码里的 `Phase` 枚举，1:1 对应产物），
  参考资料全部下沉 `docs/`。单一真源 = 代码枚举，文档镜像它。

---

## 4. 每个 Phase 的产物契约（依据 auto-quant 现有代码核对）

**这是本 RFC 的核心表。** 左侧 owner 输入产物来自 `AGENTS.md` 硬流程①–⑤；右侧运行时产物来自
`pipeline.py` / `layout.yaml` / `manifest.py` 现在**已经在写**的文件。"新增"列是本 RFC 要补的
机器可读中间产物。

| Phase | 对应硬流程 | 前置产物（precondition） | 现有运行时产物（代码已写） | 新增机器可读产物 | 完成判据（postcondition） |
|---|---|---|---|---|---|
| **ANALYZE** | ① 结构分析与量化方案 | model id | `logs/agent_optimize/`（`optimize.py` 的 8 段 recipe 文档）、`analysis.md`(owner) | `analysis.json`（逐模块精度表）、`quant_plan.json` | 查阅清单填满；ignore 对 index.json 逐条核对；Recipe KB 已查 |
| **ENV** | ② 环境 | `analysis.json` | `setup_venv.sh`+`requirements.txt`(owner)、`logs/executor_*.log` | `env_lock.json`（栈版本冻结，复用 `manifest._package_versions`） | 架构 registry 命中；`uv pip freeze` 冻结 |
| **PLAN** | ③ yaml 定稿 | `analysis.json`,`env_lock.json` | `run.yaml`(owner)、`aqa_run.json`(running) | `session/manifest.json`、`session/state.json` | `aqa validate` 过；回传字段补全 |
| **QUANT** | ④a 运行 | 冻结的 `run.yaml` | `quantized/`、`quantized/quantize.log`、`.quant_command.json`（`record_quant_command`）、`logs/agent_fixes/<phase>/`、`patches/<phase>/`、`patch_snapshot/` | `state.decisions[]` 追加量化决策 | 量化命令已冻结；体积 sanity（位宽符合预期） |
| **EVAL** | ④b + ⑥ | `quantized/`（+ baseline） | `eval/accuracy.json`(`write_accuracy_json`)、`baseline/`、`eval_thinking_off/`、`eval/<task>/*.eval.log` | `eval_report.json`（source/quant/gap 合成） | 有 benchmark 结果且非全 0 |
| **GATE** | ⑥ | `eval/accuracy.json`,`baseline/` | `gate.json`（`evaluate_gate` 的 verdict+reasons） | `state.gate` | gate verdict 与 report 结论一致 |
| **PUBLISH** | ④ 发布（可选） | `gate=GO` | HF model repo（`HfModelSink`）、`model_card.md`(owner) | 上传 url 记进 `state` | card 守四条纪律；Reproduce 段齐全 |
| **CLOSE** | ⑤ 回溯 + ⑦⑧ | 以上全部 | `run_manifest.json`(`write_manifest`)、`report.md`(owner)、sink dispatch、session 沉淀 | Recipe KB 写回一行、`events.jsonl` 封口 | `aqa artifacts check` 全过；经验路由无空行 |

贯穿全程：`journal.md`（叙事）与 `owner_actions.jsonl`（`aqa log-action` 追加）——后者 §1-A 指出当前未实现，P0 补。

> **P2 落地细化（2026-10）**：上表"新增机器可读产物"列已由 `aqa/report/phase_artifacts.py`
> 落地，`run_pipeline` 在对应相位写盘、`layout.yaml` 按 `phase` 归位、`aqa inspect` 逐阶段校验。
> 两处对原表的务实收敛：
> 1. **`analysis.json`** 落地为**确定性骨架**（architectures/model_type/层数(含多模态 `text_config` 嵌套)/多模态标志 + 计划量化范围 scheme/method/ignore），只读 `config.json`、不下权重；**完整的逐模块精度表**需校准，归 `agent-optimize`（`agent.optimize=true` 时写入 `logs/agent_optimize/`），`analysis.json` 以 `note` 字段明示该依赖。
> 2. **`quant_plan.json`** 归 **QUANT** 相位（而非 ANALYZE）：它合并 recipe 字段与**实际冻结命令**（读 `.quant_command.json`），命令只有进入 QUANT 后才存在；`when: session_quantize`，仅量化 run 强制。
> 落盘时机：`env_lock.json`/`analysis.json` 在 `open_session` 后立即写（确定性、无 GPU、任何相位崩溃都在场）；`quant_plan.json` 在 QUANT 收尾后写；`eval_report.json` 在 CLOSE 写（此时 `result`/`baseline`/`gate` 在作用域内）。所有 writer 均 best-effort（失败只记日志、绝不中断 run）。

> **P3 落地细化（2026-10）**：Recipe KB 把 `recipe_store.pin/load` 死代码接进 run loop（`aqa/agent/recipe_kb.py`）。
> - **hot-start（读）**：QUANT 进入时（`run_cfg.agent.optimize && agent.recipe_kb.enabled`）按**渐进字段放松**查最近的已验证配方——`model id`（本地路径取末段 slug，`/home/u/models/Qwen3.8-27B` 与 HF id `Qwen/Qwen3.8-27B` 同 slug）→ 架构 family（`model_type`／散文 `architecture` 首 token）→ 请求的 `scheme` → 任意 scheme；curated 配方在同级胜过 auto-pinned。命中渲染成 **advisory** 提示块前插到 agent-optimize 的 `user_context`（"adapt paths / re-validate / 失败是证据不是否决"），**不**改变可复现性（冻结命令仍由 `.quant_command.json` 记录）。审计落 `recipe_kb.json`（query + 命中/`null`），并记 `recipe_kb_hit` 事件 + `ANALYZE` decision。
> - **pin-back（写）**：CLOSE 阶段仅当 `quantize && gate=GO && agent.recipe_kb.pin` 时写回，落 `recipes/quant/auto/`（独立命名空间，**绝不覆盖**手写 curated），同模型多次运行**合并**为一行（加/覆盖该 scheme，union lessons/failures）。默认 `pin=false`——常规 run 不写仓库。
> - **启用**：run yaml 里 `agent.optimize: true`（KB 只在走 agent-optimize 的量化 run 生效）即默认开 `recipe_kb.enabled`；要让成功 run 回写知识库，显式 `agent.recipe_kb.pin: true`。`agent.recipe_kb.root` 可覆盖 curated 目录（默认仓内 `recipes/quant`）。KB 全链路 best-effort（任何异常降级为 `hot_start=""`，绝不中断 run）。

> **P4 落地细化（2026-10）**：§3.5 的三件事（propose/decide 分离、state 重播种、PolicyGate）落地为 **fix-loop 的默认 advisory 加法**（`aqa/agent/critic.py` + `aqa/agent/policy_gate.py` + `aqa/pipeline.py:_fix_hooks`）。关键务实收敛：**keep/revert 本就是确定性的**——现有 fix-loop 的"保留/回退"由 `run_phase()` 的权威**复测**决定，agent `VERDICT` 从来只驱动升级 tier 与防漂移，**从不**决定保留。因此 P4 不改这条决策链，而是在其周围加三个观察/护栏层：
> 1. **Critic = propose vs decide 的事后标定**（不是裁决者）。每次复测后把 agent 自述（`AgentAnalysis.verdict`）与真实 `result.ok`/`error_class` 比对，产出 6 态：`confirmed`（声称可修且过）/`overclaimed`（声称 FIXABLE 但复测仍挂，**不诚实**）/`silent_fail`（无 verdict 且挂）/`conceded`（UNFIXABLE 且挂，诚实让步）/`understated`（UNFIXABLE 却过）/`escalated`。`honest=false` 的三态（overclaimed/silent_fail/understated）即"自述与现实不符"的校准信号，写 `critic_calibration` 事件进 `events.jsonl`——纯学习记录，零行为变更。
> 2. **state 重播种**：每轮 fix prompt 由 `SessionState.prompt_seed()` 从 `state.json`（run_id/phase + baseline「不得回退」+ current_best + 最近 N 条 decisions，newest-first）渲染一个**权威 mission 块**，前插到 `build_fix_prompt`（`state_seed` 参数）——agent 的任务来自 state 而非可能漂移的 transcript。
> 3. **PolicyGate = 具名规则 diff 审查**。对 `PatchCapture.combined_diff()`（**只读**，绝不执行 diff）按 5 条默认 deny 规则正则匹配：`edit_tests`（改 `**/tests/**`、`test_*.py`）/`skip_tests`（`@pytest.mark.skip|xfail`、`pytest.skip(`）/`disable_cuda`（stub `torch.cuda.is_available`、强置 `device='cpu'`）/`touches_vcs_ci`（`.github/`、CI yaml）/`mass_deletion`（净删 > `max_net_deletions`，默认 200）。命中写 `policy_violation`（**点名规则 + detail**）；仅当 `agent.policy_gate.enforce=true` 才 `patch_capture.restore()` 回滚该次编辑并写 `policy_revert`——默认 advisory（只记录不回滚）。
> **安全缺省**：四个 hook（`critic`/`policy_gate`/`on_event`/`state_seed`）在 `None`/`""` 时 fix-loop 与 P3 前字节级等价；`_fix_hooks` 由 `run_cfg.agent.critic`/`policy_gate` 构造，event sink 走 `session.event()`（commit state + append events.jsonl），全部 best-effort。

**两点说明，回答"产物是不是在 run_pipeline 这里确认"：**

1. **是的，phase 切分与每阶段产物契约在 `run_pipeline` 重构时固定下来**，并同时落到两个真源：
   升级版 `layout.yaml`（静态契约，`aqa inspect` 校验）与 `state.json` 的 `phases{}`（运行时进度）。
2. **产物不是凭空约定，是对现有代码产物的归位 + 补洞**：上表"现有运行时产物"列的每个文件，
   `pipeline.py` / `manifest.py` / `report/*` 现在都已经在写；本 RFC 只是（a）把它们按阶段归类，
   （b）给只有散文的阶段补一个 `.json` 兄弟，（c）让"前置产物缺失即拒绝进入下一阶段"由代码强制
   （现在是 `AGENTS.md` 的 owner 纪律）。

---

## 5. 关键数据结构

```jsonc
// session/state.json
{
  "schema_version": 1,
  "run_id": "Qwen3-30B_MXFP8_20261013-...",
  "phase": "EVAL",
  "phases": { "ANALYZE":"DONE","ENV":"DONE","PLAN":"DONE","QUANT":"DONE",
              "EVAL":"RUNNING","GATE":"PENDING","PUBLISH":"PENDING","CLOSE":"PENDING" },
  "baseline": { "mean_score": 0.72, "per_task": {"mmlu":0.78,"gsm8k":0.66} },
  "current_best": { "scheme":"MXFP8-Mixed", "mean_score":0.70, "size_gb":31 },
  "decisions": [
    {"phase":"ANALYZE","what":"experts→mxfp4, attn/shared_expert→bf16",
     "why":"index.json 逐层核对；shared_expert 保 bf16 掉点<0.3%","ref":"analysis.json"}
  ],
  "gate": null
}
```

```jsonc
// Recipe KB 一行（dataset recipes/ 子树）
{ "key": {"arch":"Qwen3MoE","model_type":"qwen3_moe","scheme":"MXFP8","precision":"mixed","hw":"H200x8"},
  "recipe": {"method":"autoround","iters":0,"ignore_layers":"...","layer_config":{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}},
  "result": {"mean_score":0.70,"size_gb":31,"gate":"GO"},
  "lessons": ["shared_expert 保 bf16 掉点<0.3%"],
  "failures": ["iters=200 在 H200x8 OOM"] }
```

---

## 6. P0 + P1 对标 Hyperloom 的哪些功能（直接回答）

### P0 — 诚实化（低风险，先消除"照文档执行就崩"）

| P0 动作 | 对标 Hyperloom |
|---|---|
| 实现 `aqa inspect`（接 `inspect_run_dir`）、`aqa session state` | operator 工具 `session <breakdown\|report\|events\|state>`；"observable session artifacts are the contract, private helper names are not" |
| `--resume` 在真 resume 落地前**显式报错**而非假装成功 | resume 前置校验："The CLI refuses to start if `manifest.json` or `state.json` is missing in that session dir" |
| AGENTS.md 删除/标注未实现命令（§1-A 清单） | "observable artifacts / subprocess JSON bridges are the contract" |

### P1 — Session 状态机 + 真 resume（对标三件核心）

| P1 交付 | 对标 Hyperloom 源码/契约 |
|---|---|
| `session/manifest.json`（schema 版本化、原子写、缺失即拒 resume） | `inference_optimizer/session/manifest.py`：`build_manifest`/`write_manifest`(原子 `os.replace`)/`load_manifest`(缺失 raise)、`SCHEMA_VERSION` |
| `session/state.py:SessionState.load_or_init` + `state.json` | `orchestrator/state/shared_state.py:SharedState.load_or_init/state_path/from_dict/to_dict` |
| `aqa run --resume` 跳过 DONE 阶段、从首个非 DONE 重入，保留 baseline/current_best | `agent_backends/session_resume.py` + SKILL.md："Resume preserves baseline, current best, params-search state, event history...; clears stale stop_reason and crash_count" |
| `run_pipeline` 重构为显式 phase 循环（§3.2） | phase 链 `PRELUDE→…→CLOSE` + runtime contract "Create or resume a session directory" |
| `session/events.jsonl` 追加事件流 | session events / `storage/coordinator.db` 的可观测事件 |

> 简言之：**P0 对标 Hyperloom 的"operator 工具 + resume 前置校验"**（让表面即契约、宁拒绝不假装）；
> **P1 对标 Hyperloom 的 `manifest.json` + `state.json` + `--resume-from` + phase 链**（Session 成为可
> load/commit/resume 的一等对象）。S3 的阶段门、S4 的 Recipe KB、S5 的 Critic/PolicyGate 分别在
> P2/P3/P4 对标 `allowed_actions_for(phase)` / Recipe KB / Coordinator-Critic-PolicyGate。

---

## 7. 分期落地

| 期 | 交付 | 触碰模块 | 风险 | 价值 |
|---|---|---|---|---|
| **P0** ✅ | `aqa inspect`/`session state` 接 CLI；`--resume` 显式报错；AGENTS.md 删未实现命令 | `cli.py`、`AGENTS.md` | 极低 | 立刻消除"照文档崩" |
| **P1** ✅ | `session/state.py` + manifest/state/events；`run_pipeline` phase 循环；真 resume | `pipeline.py`、`executors/*.py`、`report/manifest.py` | 中（重构 pipeline） | 解决 B/C + resume |
| **P2** ✅ | layout.yaml 升级阶段门（逐条目 `phase` + `session`/`session_quantize`/`session_done` 判据）；`analysis.json`/`quant_plan.json`/`env_lock.json`/`eval_report.json` 由 `run_pipeline` 落盘；`aqa inspect` 按 Phase 汇总校验 | `report/phase_artifacts.py`(新)、`report/layout.*`、`pipeline.py` | 中 | 产物结构固定 |
| **P3** ✅ | `recipe_kb.py` + 热启动查找（渐进放松）+ 接活 `recipe_store.pin/load`；ANALYZE 查 KB 注入 agent-optimize，CLOSE strict pin-back（gate GO → `auto/`）；审计 `recipe_kb.json` | `agent/recipe_kb.py`(新)、`pipeline.py`、`config.py`、`recipe_store.py`、`report/layout.yaml` | 中 | 知识复用 |
| **P4** ✅ | Critic/PolicyGate 内建（默认 advisory：Critic 写 6 态标定事件、PolicyGate 具名规则审查 diff，`enforce` opt-in 才回滚）；fix prompt 从 `state.json` 重播种 | `agent/critic.py`(新)、`agent/policy_gate.py`(新)、`agent/fix_loop.py`、`agent/prompt.py`、`session/state.py`、`config.py`、`pipeline.py` | 高 | 追平 Hyperloom 把关 |
| **S3** ✅ | phase→`allowed_actions` 动作级守卫（`actions.py`：白名单+合法转移+`PhaseActionDenied`；`Session.guard_action`/`enter` 标注；契约产物 `allowed_actions.json`；fix prompt 注入）+ 有界 `GATE→QUANT` re-QUANT 回环（opt-in，`_RequantController`/`_run_requant_loop`/`_promote_cycle`，护栏对标 Hyperloom 宏周期） | `session/actions.py`(新)、`session/session.py`、`config.py`、`agent/prompt.py`、`agent/fix_loop.py`、`pipeline.py`、`report/layout.yaml` | 中（回环触碰 pipeline，默认关） | 可控/可审计的 proposer 动作 + 有界重量化 |

---

## 8. 风险与取舍

- **向后兼容**：`session/` 与现有 `aqa_run.json`/`run_manifest.json` 并存；P1 不删旧文件，
  `aqa_run.json` 继续作为 `layout.py:_conditions` 的 `done` 判据，直到 P2 迁移 `inspect` 读 `state.json`。
- **resume 的幂等边界**：量化/评估是重副作用阶段；resume 以"产物齐全 = DONE"为准，
  半成品（如 `quantized/` 存在但 `quantize.log` 未见结束标记）视为未完成，重跑该阶段。
- **Recipe KB 入库污染**：沿用 Hyperloom "入库严格"——只有过 gate 的 run 才写回；
  失败只作为 `failures[]` 证据注入 prompt，不作硬否决。
- **不推倒现有线性流**：P1 的 phase 循环是对 `run_pipeline` 的包装式重构，每阶段函数体
  基本复用现有 `_run_quantize`/`_run_evaluate`/`evaluate_gate`/`build_manifest`。

---

## 9. 开放问题

1. `state.json` 与 `run_manifest.json` 的边界：manifest = 复现清单（不可变意图+环境），
   state = 可变进度+决策。是否需要第三份 `storage/*.db`（Hyperloom 有 `coordinator.db`）？
   建议 P1 先用 `events.jsonl`，有查询需求再引 sqlite。
2. ssh executor 的 resume：容器是临时的，resume 语义是"远端重启后从 staged session 续"还是
   "本地 driver 续、远端重跑未完成阶段"？建议 P1 先只在 local executor 落地 resume，ssh 延后。
3. `aqa sweep` 与 Session 状态机的关系：每档一个独立 session，还是一个父 session 带子阶段？
   建议每档独立 session + 一个 `sweep_report.json` 汇总（贴近现状）。
