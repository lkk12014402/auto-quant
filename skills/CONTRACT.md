# Skill interaction contract (external-agent control loop)

Every aqa skill follows the same five-stage loop. It is the shared discipline
an external agent (Claude Code / Codex / Copilot CLI driving aqa) must hold —
the analog of Quark's Intake→Route→Plan→Confirm→Execute and Hyperloom's
"agent proposes, deterministic harness decides". Read this once per session;
each skill body assumes it.

The golden rule, inherited from `AGENTS.md`: **先 yaml 后运行, 全程留痕, 不抢卡,
route through `aqa`, never git-commit (that's the user's), 拿不准先查证.**

## The five stages

1. **Intake** — identify the goal (quantize / evaluate / quantize+evaluate /
   reproduce a recipe / fix a failure / publish). Collect only what's missing:
   `model` (HF id or local path), target `scheme`, benchmarks, hardware/GPUs.
   Detect **plan-only vs plan+execute**.
2. **Route** — pick the skill from `skills/SKILLS.md` (intent → skill). Say the
   choice in one sentence. Prefer composing existing skills; do not re-implement
   another skill's body inline.
3. **Plan** — before any heavy action:
   - **Consult the Recipe KB first** (this is `AGENTS.md` 硬流程 ① "先查
     skill / recipe / knowledge", now a command):
     `aqa recipe lookup <model> --scheme <S>` → adapt the validated command,
     `ignore_layers`, lessons, and past failures to THIS run. **Adapt, never
     copy; re-validate.** Cold start (no hit) is fine — proceed from structure.
   - Read the model structure from `analysis.json` (ANALYZE phase) — module
     families, MoE vs dense, multimodal, planned quant scope. Never fabricate
     module names; derive them from the real safetensors header / index.
   - Draft the machine-readable plan: `quant_plan.json` (scheme/method/algorithm/
     ignore) and the `run.yaml` (from `templates/run.yaml`). Show defaults,
     assumptions, expected size, and risks.
   - **Record the plan narrative**: `aqa journal add -r <run_dir> -k plan -p PLAN
     -t "<one-line>" -b "<why/assumptions/risks>"`. This is the reasoning behind
     the numbers in `quant_plan.json` — captured into the run, not left in chat.
4. **Confirm** — STOP for explicit user approval before any of:
   installing packages or building a venv · running quantization · running
   evaluation · uploading weights (`provider.upload`) · applying an editable
   source patch · overwriting an existing run dir. Present the frozen `run.yaml`
   and command. (This is Quark's mandatory Confirm gate.)
5. **Execute or Summarize** — execute only after confirm, via `aqa run` /
   `aqa sweep` (route through aqa; never call `auto-round` / `lm_eval` by hand as
   the primary path). Otherwise hand back a runbook. **Deterministic code
   measures and decides keep/revert; the agent never decides a keep from a
   predicted number** (Hyperloom principle). Verify output against
   `aqa inspect` + `aqa artifacts check`. **When a pit forces a change of plan,
   journal it as a revision** — `aqa journal add -r <run_dir> -k revise
   --supersedes <prior id> -t "..." -b "<what broke + the fix>"` — so the
   plan-then-revise trail survives (append-only; the original plan entry stays).

**Recovery rule** — on any failure, route to `skills/fix-and-patch` and return:
(a) the blocking reason, (b) the missing artifact / unmet precondition,
(c) the smallest next action that unblocks. Do not silently proceed on a broken
model or a failed eval. **Journal the pit + fix as a `revise` entry** (above) so
the recovery is part of the run's provenance.

## Phase → artifact IO (the typed contract between steps)

Artifacts are first-class contracts (one producer each). A downstream step reads
its inputs from a named file, never by guessing. Full map: `aqa/report/layout.yaml`
(verify any run with `aqa inspect <run_dir>`).

| Phase | Reads | Writes (primary) |
|---|---|---|
| PLAN | `run.yaml` | `aqa_run.json` (run state) |
| ANALYZE | `config.json` + safetensors header | `analysis.json` (identity + module families + planned scope), `recipe_kb.json` (hot-start audit — only when internal optimize on) |
| ENV | manifest probes | `env_lock.json` (frozen soft/hardware stack) |
| QUANT | `analysis.json`, Recipe KB hit | `quant_plan.json` + `.quant_command.json`, `quantized/` (+ `quantize.log`) |
| BASELINE | source (bf16) model | `baseline/accuracy.json` |
| EVAL | `quantized/` | `eval/accuracy.json`, `eval_report.json` (bf16 vs quant gap) |
| GATE | `eval_report.json` | `gate.json` (GO/NO-GO/INCONCLUSIVE + reasons) |
| CLOSE | all of the above | `recipe.yaml` (portable manifest + `quantization_details`), `run_manifest.json`, `logs/pipeline.log` |
| VERIFY | manifest + this run's artifacts | `reproduction_report.json` (input/structure/numbers; only `aqa reproduce`) |
| PUBLISH | `model_card.md` + checkpoint | HF model repo README (only `provider.upload`) |

Cross-cutting (written throughout by the driving agent, not one phase): `journal.md`
+ session `events.jsonl` via `aqa journal add` — the plan and its revisions.

Owner-facing 硬流程 (narrative artifacts, see `AGENTS.md`): `analysis.md` →
`setup_venv.sh`/`requirements.txt` → `run.yaml` → `model_card.md` (publish only)
→ `report.md`, with `journal.md` (why — via `aqa journal add`) +
`owner_actions.jsonl` (what) throughout.

## Knowledge an agent consults, and how it is reached

Knowledge is **pulled at the step that needs it**, not loaded up front:

| Knowledge | How the agent reaches it |
|---|---|
| Pinned per-model recipes (`recipes/quant/*.json`) | `aqa recipe lookup/list/show` (GPU-free) |
| Scheme defaults (`recipes/schemes/*.yaml`) | referenced inside `skills/quantize` |
| Benchmark defs (`recipes/benchmarks/*.yaml`) | `aqa list benchmarks`; used by eval |
| Worked examples (`cookbooks/*.sh`) | path reference from a skill body |
| Deep references (`docs/autoround_full_reference.md`, `docs/day0_quantization_recipe.md`) | path reference (progressive disclosure) |

## What the agent decides vs. what code decides

- **Agent (LLM):** scheme/algorithm choice, `ignore_layers` reasoning, MoE/dense
  handling, which benchmarks, interpreting gaps, root-causing failures.
- **Deterministic code (aqa):** running the quant/eval jobs, measuring scores,
  the gate verdict (keep/revert), structure ground-truth in `quantization_details`,
  artifact completeness (`aqa inspect` / `aqa artifacts check`).

A number the agent *predicts* is a hypothesis for the plan; only a number aqa
*measures* decides a keep.
