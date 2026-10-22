---
name: quantize
description: >
  Choose and produce an optimal auto-round quantization command + recipe for a
  given model. Use when a run needs a scheme/method decision, MoE vs dense
  handling, mixed-precision layer_config, or ignore_layers selection.
---

# Quantize skill

Turn a model + target scheme into a **correct, optimal `auto-round` command** and
a human-readable **quantization recipe**, then pin it as an artifact so later runs
reuse a validated recipe instead of re-deriving it.

## When to use
- The user asks to quantize a model (MXFP4 / MXFP8 / W4A16 / NVFP4 / GGUF ...).
- A scheme is chosen but the model is a MoE / has unusual layer naming / needs
  mixed precision on experts or attention.

## Inputs
- `model` (HF repo id or local path), `quant.scheme`, `quant.method`.
- `quant.algorithm` (optional) — the auto-round algorithm chain (see below).
- Recipe corpus: `recipes/schemes/*.yaml` (scheme definitions) and
  `recipes/quant/*` (per-model pinned recipes).

## Algorithm selection (`quant.algorithm` → `--algorithm`)
A comma-separated chain composing **preprocessors** (stackable, run before
quantization) with **exactly one block-quantizer**:
- Block-quantizers (pick one; default `auto_round`, or `rtn` when `iters:0`):
  `rtn`, `auto_round` (SignRound tuning; add `iters:200`, `enable_alg_ext:true`
  for the auto-round v2 extension).
- Preprocessors (stackable): `awq` (activation-aware smoothing), `hadamard`
  (rotation; `hadamard_type: hadamard|random_hadamard|quarot_hadamard`),
  `svdquant` (low-rank decomposition).
- Examples: `awq,auto_round`, `hadamard,auto_round`, `awq,hadamard`, `svdquant,rtn`.
- `opt_rtn: true|false` toggles optimized-RTN rounding.
- Constraints (validated at load): only one block-quantizer; `enable_alg_ext`
  needs an auto_round token; `method:model_free` is calibration-free (RTN only).

## Procedure
1. **Analyse structure without downloading weights**: read `config.json` +
   safetensors index; classify dense vs MoE, list tensor names, expert layout,
   and check how checkpoint names map to runtime module names.
2. **Consult the recipe corpus** (`recipes/schemes/<scheme>.yaml`, prior pinned
   recipes) for known-good settings.
3. **Emit the command** via `aqa/quantize` (or an agent-optimized command). Use
   `--model_name <model>` explicitly. Add `--ignore_layers` / `layer_config`
   for mixed precision when the structure requires it. Resolve ignore tokens
   against AutoRound's actual `model.named_modules()` names; checkpoint keys
   may use a converted prefix (for example, `backbone.*` while modules are
   named `model.*`).
4. **Write `quantization_recipe.md`** (conclusion / structure / module list /
   built-in handling / config decisions / recommended command / accuracy
   expectation / variants) and **pin** the machine-readable recipe via a
   `RecipeStore` (default `repo_dir` -> `recipes/quant/<org>__<model>.json`).

## Accuracy / safety rules
- Never fabricate tensor names — derive them from the actual index. For
  `ignore_layers`, separately verify the runtime module-name namespace and any
  checkpoint/model prefix conversion; index hits or log strings alone do not
  prove a module was excluded. After export, compare safetensors dtypes with
  the planned precision allocation.
- Treat `model.safetensors.index.json` as the checkpoint map: verify every mapped
  key is present in its selected shard, detect unreferenced physical shards and
  duplicate keys, and check index metadata totals against indexed tensors.
  Unreferenced leftovers must not inflate the size check; remove them only after
  confirming they are outside the index map. Duplicate tensor copies are safe
  only when the index selects an occurrence and every duplicate has identical
  dtype, shape, and payload.
- `group_size` for MX schemes is fixed at 32; export `llm_compressor` for vLLM.
- If `static_kv_fp8` creates new K/V scale parameters and AutoRound's fused-MoE
  blockwise/meta loader reports `Failed to materialize ... k_scale/v_scale ...
  still on meta`, use its documented `AR_DISABLE_META_LOAD=1` whole-model CPU
  loading path, then verify every exported KV scale is finite and nonzero.
  This trades host RAM for materialization; enable it in response to that
  failure, not by default. Verified examples: the Gemma-4-26B-A4B and
  Nemotron-3-Nano-30B-A3B cookbooks.
- The pinned recipe is the single source of truth; do not hard-code recipe
  knowledge in code paths.
