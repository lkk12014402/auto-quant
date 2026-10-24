# Skill routing catalog (intent → skill)

The index an external agent uses to route a user request to the right skill.
Analog of Quark's `skills_catalog.md`. Each skill is auto-discovered by the
harness via its `description` frontmatter (trigger surface); this table is the
human-readable map. Always run the loop in `skills/CONTRACT.md`.

**Start here:** read `AGENTS.md` (root entrypoint; the 硬流程 5-step + doc map),
then route with the table below.

| User intent (trigger phrases) | Skill | Primary output |
|---|---|---|
| "quantize model X", pick a scheme (MXFP4/MXFP8/W4A16/NVFP4/GGUF), MoE vs dense, `ignore_layers`, mixed-precision `layer_config` | [`quantize`](quantize/SKILL.md) | `quant_plan.json` + pinned recipe + `auto-round` command |
| "evaluate / benchmark model X", pick benchmarks, harness (native/lm_eval/evalscope), backend (hf/vllm/vllm_xpu), merge scores | [`evaluate`](evaluate/SKILL.md) | `eval/accuracy.json` |
| a quantize/evaluate **step failed**; root-cause, fix (env or editable source), retry with tier escalation, capture a patch | [`fix-and-patch`](fix-and-patch/SKILL.md) | patch + retry |
| "publish results / 更新/回刷 recipe 文档", sync a run into the INC-Quantization-Recipes site | [`recipe-page`](recipe-page/SKILL.md) | recipe-site pages |
| "reproduce this recipe / verify a published checkpoint", replay a `recipe.yaml` variant and check input/structure/accuracy | [`reproduce`](reproduce/SKILL.md) | `reproduction_report.json` |
| END-of-task wrap-up: sediment the owner session + verify aqa's internal sessions into the trajectories store | [`session-sediment`](session-sediment/SKILL.md) | sedimented session |

## Typical compositions

- **Quantize + evaluate end-to-end:** `quantize` → (run) → `evaluate`, gated by
  `eval_report.json`. One `aqa run` with `quant` + `eval` + `gate` does this;
  the skills inform the plan. Finish with `session-sediment`.
- **A step broke:** any skill → `fix-and-patch` (Recovery rule) → resume.
- **Reproduce a published recipe:** `aqa reproduce <recipe.yaml> --variant <id>`
  (`full` or `eval-only`) → read `reproduction_report.json`; use `--dry-run`
  first. Do not hand-edit the recipe on a mismatch — report it.

## Before you run anything

1. `aqa recipe lookup <model> --scheme <S>` — hot-start from a validated recipe
   (adapt, don't copy; re-validate). See `skills/CONTRACT.md` §Plan.
2. Freeze the plan into a `run.yaml` (`templates/run.yaml`), `aqa validate` it.
3. Confirm with the user before heavy/irreversible actions, then `aqa run`.

Knowledge locations (consulted on demand, not loaded up front): `recipes/`
(schemes, pinned recipes, benchmarks), `cookbooks/` (worked examples),
`templates/` (scaffolds), `docs/` (deep references). The pinned recipe corpus is
queried via `aqa recipe`; everything else is reached by the path a skill names.
