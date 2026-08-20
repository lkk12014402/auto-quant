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
- Recipe corpus: `recipes/schemes/*.yaml` (scheme definitions) and
  `recipes/quant/*` (per-model pinned recipes).

## Procedure
1. **Analyse structure without downloading weights**: read `config.json` +
   safetensors index; classify dense vs MoE, list module names, expert layout.
2. **Consult the recipe corpus** (`recipes/schemes/<scheme>.yaml`, prior pinned
   recipes) for known-good settings.
3. **Emit the command** via `aqa/quantize` (or an agent-optimized command). Use
   `--model_name <model>` explicitly. Add `--ignore_layers` / `layer_config`
   for mixed precision when the structure requires it.
4. **Write `quantization_recipe.md`** (conclusion / structure / module list /
   built-in handling / config decisions / recommended command / accuracy
   expectation / variants) and **pin** the machine-readable recipe via a
   `RecipeStore` (default `repo_dir` -> `recipes/quant/<org>__<model>.json`).

## Accuracy / safety rules
- Never fabricate module names — derive them from the actual index.
- `group_size` for MX schemes is fixed at 32; export `llm_compressor` for vLLM.
- The pinned recipe is the single source of truth; do not hard-code recipe
  knowledge in code paths.
