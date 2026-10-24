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
> Follow the five-stage loop in [`../CONTRACT.md`](../CONTRACT.md)
> (Intake→Route→Plan→Confirm→Execute). Steps below are the Plan→Execute detail.

1. **Analyse structure without downloading weights**: read `config.json` +
   safetensors index; classify dense vs MoE, list module names, expert layout.
2. **Consult the Recipe KB first** — run
   `aqa recipe lookup <model> --scheme <scheme>` (GPU-free; the external-agent
   analog of agent-optimize hot-start). It surfaces the closest validated
   command, `ignore_layers` rationale, lessons, and past failures. **Adapt to
   this run's paths and re-validate; never copy blindly.** Cold start (no hit) is
   fine — proceed from structure. Also read `recipes/schemes/<scheme>.yaml` for
   scheme defaults.
3. **Emit the command** via `aqa/quantize` (or an agent-optimized command). Use
   `--model_name <model>` explicitly. Add `--ignore_layers` / `layer_config`
   for mixed precision when the structure requires it.
4. **Write `quantization_recipe.md`** (conclusion / structure / module list /
   built-in handling / config decisions / recommended command / accuracy
   expectation / variants) and **pin** the machine-readable recipe via a
   `RecipeStore` (default `repo_dir` -> `recipes/quant/<org>__<model>.json`).
   Browse the existing corpus with `aqa recipe list` / `aqa recipe show`.

## Accuracy / safety rules
- Never fabricate module names — derive them from the actual index.
- `group_size` for MX schemes is fixed at 32; export `llm_compressor` for vLLM.
- **Native-MX-packed MoE experts (e.g. gpt-oss) must not use `method: model_free`**:
  when expert weights are already stored as MX blocks (`*_blocks` + `*_scales`),
  `model_free` treats the packed scales as ordinary Linear weights and exports a
  self-inconsistent checkpoint (vLLM then fails with `KeyError experts.*.w2_bias`).
  Use the model-load path (`method: autoround`, `iters: 0` for RTN) so the
  `fused_moe`/`gpt_oss` handler dequantizes then requantizes.  `model_free` is for
  fp8 sources or unquantized oversized models only.  Full evidence:
  `aqa/agent/recipes/auto_round_recipe.md` §1.4.
- If `static_kv_fp8` creates new K/V scale parameters and AutoRound's fused-MoE
  blockwise/meta loader reports `Failed to materialize ... k_scale/v_scale ...
  still on meta`, use its documented `AR_DISABLE_META_LOAD=1` whole-model CPU
  loading path, then verify every exported KV scale is finite and nonzero.
  This trades host RAM for materialization; enable it in response to that
  failure, not by default. Verified examples: the Gemma-4-26B-A4B and
  Nemotron-3-Nano-30B-A3B cookbooks.
- The pinned recipe is the single source of truth; do not hard-code recipe
  knowledge in code paths.  The canonical auto-round domain text lives in
  `aqa/agent/recipes/auto_round_recipe.md`; this SKILL keeps the operating
  procedure and safety red lines, not a second copy of the domain rules.
