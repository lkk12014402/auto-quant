---
name: reproduce
description: >
  Reproduce a published INC recipe variant and verify it. Use when the user says
  "reproduce this recipe", "rerun this quantization/eval from recipe.yaml", "does
  this model card's recipe still reproduce", or when a published checkpoint needs
  an input/structure/accuracy check. Works through the CLI or the Python API.
---

# Reproduce skill

Replay **one variant** of an INC `model-recipe` manifest (`schema_version 1.0`)
and report whether it reproduces on this box. The run reuses aqa's normal phase
machine (`QUANT → EVAL → CLOSE → VERIFY`) and writes
`reproduction_report.json`.

## When to use
- The user hands you a `recipe.yaml` / INC manifest and asks "can we reproduce it".
- A published quantized model needs its recipe checked before you trust/copy it.
- You want to verify an existing checkpoint without re-quantizing (`eval-only`).

## Intake
- `manifest` path, optional `variant` id (auto if the manifest has one).
- Target environment: the manifest's `environments.quantization/evaluation`
  package pins are **expectations**, not an installer. Run in an environment that
  matches them; mismatches are recorded (and can fail the verdict).
- Mode: `full` (replay quantization + evaluation) or `eval-only` (evaluate an
  existing checkpoint / `hf_url`).
- Ask before heavy GPU work. Use `--dry-run` first to review the resolved plan.

## Procedure
1. **Inspect the manifest** — `--dry-run` prints model source, substituted
   quantization script, evaluation backend/benchmarks/model args, expected scores.
2. **Review before execute** — the manifest is a shell recipe. A valid manifest is
   not permission to run arbitrary scripts; confirm the commands with the user and
   run in an isolated/matched environment.
3. **Run**:
   ```bash
   aqa reproduce recipe.yaml --variant mxfp4-v1 --mode full \
       --out ./reproduce/mxfp4-v1
   # no checkpoint for quantization:
   aqa reproduce recipe.yaml --variant mxfp4-v1 --mode eval-only \
       --checkpoint org/Model-MXFP4-AutoRound --out ./reproduce/eval-only
   ```
4. **Read the report** (`<out>/reproduction_report.json`):
   - `input_equivalence` — model source / quant command / package versions;
   - `quantization_reproduction` — per-module expected vs got scheme;
   - `evaluation_reproduction` — per-task expected vs got, abs+rel tolerance;
   - `verdict` — `reproduced` / `not_reproduced` / `inconclusive`.
5. **Do not auto-fix the recipe.** A mismatch is reported as-is; changing a recipe
   is a normal Coordinator/owner decision, not part of the reproduce path.

## Python API (external agents)
```python
from aqa.reproduce import plan_reproduction, reproduce_manifest

plan = plan_reproduction("recipe.yaml", variant="mxfp4-v1")   # no GPU
report = reproduce_manifest(
    "recipe.yaml", variant="mxfp4-v1", mode="full",
    out_dir="./reproduce/mxfp4-v1", abs_tol=0.01, rel_tol=0.02,
    on_input_mismatch="warn", force=False)
assert report["verdict"] in {"reproduced", "not_reproduced", "inconclusive"}
```
`reproduce_manifest` returns the report dict even when the verdict is
`not_reproduced`; it raises `AqaError` only for unusable input (bad manifest,
missing variant/command, non-empty out dir without `force`).

## Safety / semantics
- **Read-only**: no Recipe-KB pin, no weight upload, no agent self-heal, no gate.
- **Draft model card**: reproduce writes `model_card.md` when the run has
  evaluation accuracies. Without `provider.upload.signoff` it is a draft with
  template placeholders (`{base_repo_id}` / `{member name} <{member email}>`) and
  is never uploaded. A publishable card needs a user-confirmed sign-off.
- `eval-only` skips the quantization check (`status: skipped`).
- Exit codes: `0` reproduced, `1` not_reproduced, `2` inconclusive.
- Default output dir is `./reproduce/<manifest>-<variant>`; it is gitignored.
- Environment/package mismatch is `warn` by default; `--on-input-mismatch fail`
  makes it `not_reproduced`.
