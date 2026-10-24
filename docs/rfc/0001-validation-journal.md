# RFC 0001 — P0/P1 真实模型验证日志

> 任务：在真实本地模型上验证 RFC 0001 已落地的 **P0（`aqa inspect` + 诚实 `--resume`）**
> 与 **P1-core（Session 状态机 + phase 循环 + 真 resume + `aqa session` 运营命令）**
> 是否符合预期。粒度对标 `docs/RUN_JOURNAL.md`（命令 + 依据 + 结果）。
> 时间：2026-10-13（本机 H200）。分支：`auto-run`。

## 环境基线

- 模型：`/home/kaokaolv/models/Qwen3.8-27B` —— `Qwen3_5ForConditionalGeneration`
  （`model_type: qwen3_5`，**多模态**：text 64 层/5120 hidden + vision），bf16 ≈ 54GB、18 shards，root 只读。
- GPU：8×H200（143GB）。验证只用**空闲卡 GPU 1**（`CUDA_VISIBLE_DEVICES=1`），
  未触碰他人占用的 0/5；验证结束 GPU 1 回落 0 MiB、无残留进程。
- venv：`~/.venv-aqa-gpu`。**关键坑**：`aqa` 可执行文件指向开发仓
  `/home/kaokaolv/auto_agent/auto-quant`，**不是审阅仓**。要跑本次改动必须：
  `cd /home/kaokaolv/aqa/auto-quant && ~/.venv-aqa-gpu/bin/python -m aqa.cli ...`
  （CWD 优先导入审阅仓的 `aqa`）。
- 工作区：`/home/kaokaolv/aqa/p0-validation/`（临时，验证后已清理；下文产物为留痕）。

## 验证策略：分层，隔离"P0 机制"与"模型/harness 兼容性"

- **Tier 1（无 GPU）**：真实模型上验证 config 解析、计划构建、resume 诚实性。
- **Tier 2（空闲 GPU）**：真实有界 run 产出真实 `session/`，演示 inspect / session 命令
  / resume 的"跳过已完成相位"与"重跑未完成相位"。
- **硬约束**：MM 模型在本分支量化/评估失败属**环境/兼容性问题，不等于 P0 回归**，分开记。

---

## Tier 1 — 真实模型上的 config / 计划 / resume 诚实性（无 GPU）

run.yaml（要点）：`provider.type=quantize`，`quant{scheme:MXFP4, method:model_free}`，
`eval{benchmarks:[piqa], backend:vllm, baseline:false, limit:8}`，
`agent{fix_loop:false, optimize:false}`（确定性、无 agent、无网络）。

| 步骤 | 命令 | 结果 |
|---|---|---|
| 1a validate | `aqa validate run.yaml` | `OK: model=…Qwen3.8-27B backend=vllm benchmarks=['piqa']` —— 真实 MM 配置可解析 |
| 1b dry-run | `aqa run --dry-run run.yaml` | `[dry-run] would run pipeline …` → `state=done` |
| 1c resume 诚实性 | `aqa run --resume run.yaml`（无 session） | **exit 1** + `error: no session manifest to resume at …/out/session/manifest.json` |

**结论**：P0 的 `--resume` 诚实化成立——无 session 时**显式报错退 1**，不再静默从头跑。

---

## Tier 2a — 真实 run：QUANT 失败路径 + inspect + session 命令 + resume 重跑

命令：`CUDA_VISIBLE_DEVICES=1 aqa run run.yaml`

实况日志（节选）：

```
[INFO] aqa.providers: == quantize (MXFP4 / model_free) ==
[INFO] aqa.quantize: [CMD] auto-round --model_name …/Qwen3.8-27B --scheme MXFP4 --model_free …
[WARNING] aqa.pipeline: run failed (auto-round is not installed; `pip install auto-round`);
          writing back partial results + status=Eval Failed
error: auto-round is not installed; `pip install auto-round`   # exit 1
```

**环境发现①**：`~/.venv-aqa-gpu` **未装 auto-round** → 真实量化无法在本机跑（命令已正确构造）。
这恰好触发 P1 的**失败写回**路径。产出的真实 Session trace：

`session/state.json`（QUANT 真实失败）：

```json
{
  "schema_version": 1,
  "run_id": "Qwen3.8-27B_MXFP4_20261013-164922_10c834",
  "phase": "QUANT",
  "phases": { "QUANT": "FAILED", "EVAL": "PENDING", "GATE": "PENDING", "CLOSE": "PENDING" },
  "baseline": null, "current_best": null, "decisions": [], "gate": null,
  "created_at": "2026-10-13T23:49:22Z", "updated_at": "2026-10-13T23:49:22Z"
}
```

`session/events.jsonl`（含随后一次 resume → 第二条 `phase_enter QUANT`，**不跳过** FAILED 相位）：

```jsonl
{"ts":"2026-10-13T23:49:22Z","type":"session_start","run_id":"Qwen3.8-27B_MXFP4_20261013-164922_10c834","plan":["QUANT","EVAL","GATE","CLOSE"]}
{"ts":"2026-10-13T23:49:22Z","type":"phase_enter","phase":"QUANT"}
{"ts":"2026-10-13T23:49:22Z","type":"phase_fail","phase":"QUANT","error":"auto-round is not installed; `pip install auto-round`"}
{"ts":"2026-10-13T23:50:53Z","type":"phase_enter","phase":"QUANT"}
{"ts":"2026-10-13T23:50:53Z","type":"phase_fail","phase":"QUANT","error":"auto-round is not installed; `pip install auto-round`"}
```

验证到的点：

- **计划构建正确**：quantize 配置 → `QUANT→EVAL→GATE→CLOSE`（正确省略 BASELINE：`baseline:false`；省略 PUBLISH：无 upload）。
- **三件套落盘**：`manifest.json`（冻结完整 run_config = resume 前置）+ `state.json` + `events.jsonl`，原子写。
- **失败写回**：`_fail_current_phase` → `QUANT=FAILED` + `phase_fail` 事件。
- `aqa inspect out` → `status: INCOMPLETE` + **exit 1** + 产物清单（present/MISSING）。
- `aqa session state|report|events out` → 三个运营命令均正常（分别输出 state.json 美化、相位表、事件流）。
- **resume 重跑未完成相位**：`aqa run --resume run.yaml` → QUANT=FAILED（无 checkpoint 产物）
  → `should_skip=False` → **重入**（第二条 `phase_enter QUANT`，复用同一 `run_id`），再次失败。
  证明"半成品重跑"语义。

---

## Tier 2b — 真实 run：EVAL 完成 → resume **跳过**已完成相位（核心价值）

为取得一个真实 **DONE** 相位（auto-round 缺失导致 QUANT 无法成功），改用 `provider.type=pretrained`
（无量化，只评估）：`CUDA_VISIBLE_DEVICES=1 aqa run run2.yaml`。

实况：

```
[INFO] aqa.pipeline: == evaluate ==
[WARNING] aqa.eval.runner: harness lm_eval unavailable; skipping piqa
state=done output=…/out2
```

**环境发现②**：`~/.venv-aqa-gpu` **未装 lm_eval** → benchmark 被跳过，EVAL 相位"完成"但 `mean_score=0.0`。
但相位状态机照常推进并落盘 `eval/accuracy.json`：

`session/state.json`：

```json
{
  "schema_version": 1,
  "run_id": "Qwen3.8-27B_eval_20261013-165123_62ac2d",
  "phase": "CLOSE",
  "phases": { "EVAL": "DONE", "CLOSE": "DONE" },
  "baseline": null, "current_best": { "mean_score": 0.0 }, "decisions": [], "gate": null,
  "created_at": "2026-10-13T23:51:23Z", "updated_at": "2026-10-13T23:51:23Z"
}
```

随后 `aqa run --resume run2.yaml`：

```
[INFO] aqa.pipeline: == resume: eval present, skipping ==
state=done output=…/out2   # exit 0
```

`session/events.jsonl`（最后三行为 resume：**EVAL 被跳过**）：

```jsonl
{"ts":"2026-10-13T23:51:23Z","type":"session_start","run_id":"Qwen3.8-27B_eval_20261013-165123_62ac2d","plan":["EVAL","CLOSE"]}
{"ts":"2026-10-13T23:51:23Z","type":"phase_enter","phase":"EVAL"}
{"ts":"2026-10-13T23:51:23Z","type":"phase_done","phase":"EVAL"}
{"ts":"2026-10-13T23:51:23Z","type":"phase_enter","phase":"CLOSE"}
{"ts":"2026-10-13T23:51:23Z","type":"phase_done","phase":"CLOSE"}
{"ts":"2026-10-13T23:52:07Z","type":"phase_skip","phase":"EVAL","reason":"eval/accuracy.json present"}
{"ts":"2026-10-13T23:52:07Z","type":"phase_enter","phase":"CLOSE"}
{"ts":"2026-10-13T23:52:07Z","type":"phase_done","phase":"CLOSE"}
```

**结论（核心价值达成）**：EVAL=DONE 且 `eval/accuracy.json` 在场 → resume **跳过**该相位
（`should_skip(EVAL, artifact_present)=True`，`phase_skip` 事件带 reason），退 0。
pretrained 配置的计划为 `EVAL→CLOSE`（正确省略 QUANT/GATE）。

---

## 验收矩阵

| 项 | 期望 | 实测 | 结论 |
|---|---|---|---|
| P0 `aqa inspect`（真实 dir） | INCOMPLETE + exit 1 + 产物清单 | 符合 | ✅ |
| P0 `--resume` 无 session | exit 1 + 明确报错 | 符合 | ✅ |
| P1 计划构建（quantize / pretrained） | `QUANT→EVAL→GATE→CLOSE` / `EVAL→CLOSE` | 符合 | ✅ |
| P1 三件套落盘（manifest/state/events，原子写） | 齐全 | 符合 | ✅ |
| P1 失败写回（QUANT FAILED） | state FAILED + phase_fail | 符合 | ✅ |
| P1 `aqa session state\|report\|events` | 均可用 | 符合 | ✅ |
| P1 resume **重跑**未完成相位（FAILED） | 第二次 phase_enter，不跳过 | 符合 | ✅ |
| P1 resume **跳过**已完成相位（DONE + 产物） | phase_skip + exit 0 | 符合 | ✅ |

## 环境/兼容性发现（与本次改动无关，分开记）

1. `~/.venv-aqa-gpu` 未装 **auto-round** → 真实 MXFP4/model_free 量化无法在本机执行（命令已正确构造）。
2. `~/.venv-aqa-gpu` 未装 **lm_eval** → benchmark 被跳过，EVAL 以 `mean_score=0.0` 计为完成。
3. **cosmetic**：QUANT 失败时写回告警文案写成 `status=Eval Failed`（既有写回标签），
   但 `state.json` 正确记录 `QUANT=FAILED`——建议后续把写回标签与真实失败相位对齐。

> 本次改动**未新增红测**（早先已用 `git stash` 对照验证）。两项依赖缺口（auto-round / lm_eval）
> 不影响 P0/P1 机制验证——反而顺带覆盖了**失败路径**与**无 benchmark 完成路径**。

## 关于 traces

本仓 `aqa traces` 指**agent ReAct 会话 → ATIF（SFT/RL 视图）**（见 `docs/usage/trajectories.md`）。
本次为**人工运营验证**、不产 ATIF 会话，故 `aqa traces export` 不适用；
相位状态机的真实轨迹以上文 `state.json`/`events.jsonl` 原样内嵌留痕。
