---
name: quantize
description: >
  Choose and produce an optimal auto-round quantization command + recipe for a
  given model. Use when a run needs a scheme/method decision, MoE vs dense
  handling, mixed-precision layer_config, ignore_layers selection, or a
  diffusion/multimodal model (DiT text-to-image, VLM) quantization.
---

# Quantize skill

Turn a model + target scheme into a **correct, optimal `auto-round` command** and
a human-readable **quantization recipe**, then pin it as an artifact so later runs
reuse a validated recipe instead of re-deriving it.

## Ground truth (read before emitting any command)
- **Reference doc**: `/home/kaokaolv/auto_agent/auto-quant/docs/autoround_full_reference.md`
  — full scheme / data_type / algorithm / CLI-arg / export-format tables, all
  verified against source with `file:line` citations. Consult it first.
- **Diffusion / VLM models**: `/home/kaokaolv/auto_agent/auto-quant/docs/autoround_diffusion_reference.md`
  — DiT/VLM quantization architecture, per-model verdicts (Z-Image-Turbo,
  FLUX.2-*, DeepSeek-OCR-2, ...), and the vllm-omni deployment matrix.
- **Day-0 models**（transformers/vllm/auto-round 有任一不认的新架构）: `/home/kaokaolv/auto_agent/auto-quant/docs/day0_quantization_recipe.md`
  — 通用 playbook：三件套注册表探针 / 推理路径信息源与决策矩阵 / model_free vs
  调优选型（含 model_free 静默回退陷阱）/ ignore 清单双重检查纪律 /
  docker-first 环境策略 / 导出保真三坑（config 漂移、字面量分片键、孤儿文件）。
  附录 A 是 Qwen3.8-Flash-Next 的完整实证（含 RTN 先行建下限的实证依据）。
- **Source repo**: `/home/kaokaolv/auto_agent/auto-round` (baseline commit
  `bd9a13ce`). When the doc doesn't cover a flag or you are unsure, grep the
  source (`auto_round/cli/`, `auto_round/schemes.py`) — never invent flags.
- The local repo is the **refactored** auto-round: scheme + `alg_configs` driven.
  Several things from older docs/README do NOT apply (see "Pitfalls" below).

## When to use
- The user asks to quantize a model (MXFP4 / MXFP8 / W4A16 / NVFP4 / GGUF ...).
- A scheme is chosen but the model is a MoE / has unusual layer naming / needs
  mixed precision on experts or attention.
- The model is a diffusion pipeline (has `model_index.json`) or a VLM.

## Inputs
- `model` (HF repo id or local path), `quant.scheme`, `quant.method`.
- `quant.algorithm` (optional) — the auto-round algorithm chain (see below).
- Recipe corpus:
  - `recipes/schemes/*.yaml` (scheme definitions) and `recipes/quant/*`
    (per-model pinned recipes, machine-readable).
  - `/home/kaokaolv/auto_agent/recipes/*.md` — human-written per-model
    quantization analyses with validated commands (GLM-5.1, LongCat-Flash,
    DeepSeek-V4-Flash-0731, Gemma-4-31B, Qwen3.5-397B-A17B, Qwen3.8-27B,
    Qwen3.8-2.4T-A95B, Qwen3.8 vLLM troubleshooting, MXFP4 general survey).
    **Caveat: these were validated against auto-round 0.15.0 (commit
    `f17d9cd4`); the ground-truth repo here is the newer refactored build.
    Re-verify version-specific behavior (e.g. model_free internals) before
    reusing.**

## Recipe-writing methodology (follow the pinned per-model docs)
1. **Never load weights to analyse.** Read `config.json` +
   `model.safetensors.index.json` (+ safetensors headers for shapes/dtypes).
   For diffusion pipelines read `model_index.json` + `transformer/config.json` +
   the transformer safetensors index instead.
2. **Classify the model**: dense vs MoE (routed/shared experts, fused 3-D
   expert tensors?), text vs multimodal vs diffusion, MTP/speculative modules,
   and whether the source checkpoint is BF16 or already pre-quantized (FP8/MXFP —
   then model_free dequant→requant applies).
3. **Enumerate quantizable 2-D modules** with shapes: check group_size
   divisibility (MX=32, NVFP4=16, W4A16=128) and `out_features ≥ 128` if any
   layer would fall into MXFP8 (`mm_mxfp8` crashes below that — keep N=64
   projections like `in_proj_a/b` in BF16).
4. **Simulate pattern matching before running**: apply ignore_layers +
   auto-appended patterns (lm_head/head, `.gate.`, embed, conv, nontext
   keywords) + layer_config over the FULL tensor index and produce a
   per-module precision assignment table. No module's fate should be a guess.
5. **Choose the method by scale**: huge models (≳100B) → `--model_free` RTN;
   smaller dense models → calibrated `--iters 200` (better accuracy).
   Large DiT (FLUX.1/FLUX.2-dev) → `iters 0` first (calibration costs
   nsamples × num_inference_steps full denoising passes).
6. **Verify the export**: the `ignore` list in the exported
   `quantization_config.json` must cover every Linear that is actually stored
   unquantized — a mismatch makes vLLM build quantized layers for plaintext
   weights (`KeyError: '<layer>.weight'`). For vllm-omni DiT checkpoints,
   additionally verify `block_name_to_quantize` is present in the transformer's
   `quantization_config`.
7. **Evaluate**: lm-eval vs BF16 baseline; MXFP4 RTN should keep ≥99%
   accuracy on well-tested recipes. For DiT: DrawBench + CLIP score vs BF16
   baseline (`examples/run_diffusion_flux1.yaml`).

## Algorithm selection (`quant.algorithm` → `--algorithm`)
A comma-separated chain composing **preprocessors** (stackable, run before
quantization) with **exactly one block-quantizer**:
- Block-quantizers (pick one; default `auto_round`, or `rtn` when `iters:0`):
  `rtn`, `auto_round` (SignRound tuning; add `iters:200`; add
  `enable_alg_ext:true` → CLI `--enable_alg_ext` for the SignRoundV2 extension —
  **recipes do NOT enable it automatically, it must be explicit**).
- Preprocessors (stackable): `awq` (activation-aware smoothing; requires
  `nblocks:1`, skips MoE experts by default), `hadamard` (per-linear rotation;
  CLI `--rotation_type hadamard|random_hadamard`, `--rotation_backend
  auto|inplace|transform`; `transform` backend only supports MXFP4/NVFP4,
  `inplace` backend can only export `fake`), `svdquant` (low-rank decomposition;
  forces `format:svdquant_nunchaku`, diffusion-oriented, FLUX.1 only).
- Architecture-level rotation (QuaRot) is **Python-API only**:
  `rotation_config="quarot"`.
- Examples: `awq,auto_round`, `hadamard,auto_round`, `svdquant,rtn`.
- opt_rtn (imatrix-weighted scale/zp search) is ON by default when `iters:0`;
  `--disable_opt_rtn` forces pure RTN, `--enable_opt_rtn` forces it on.
  `auto-round-rtn` = iters 0 + disable_opt_rtn; `auto-round-opt-rtn` = iters 0 +
  opt_rtn on.
- Constraints (validated at load): only one block-quantizer; `enable_alg_ext`
  needs an auto_round token; `method:model_free` is calibration-free (RTN only,
  supported schemes: W2A16(G32/G64)/W4A16(_MIXED)/W8A16/MXFP4/MXFP8/NVFP4_E5M3/BF16).

## Diffusion / multimodal models (DiT → vllm-omni)
- Only `pipe.transformer` (DiT) / `pipe.unet` (SD) is quantized; text encoders
  and VAE stay untouched. Multi-transformer models (Wan2.2) are quantized one
  transformer at a time.
- Calibration dataset defaults to `coco2014` (LLM pile-10k is force-replaced);
  `batch_size` is forced to 1.
- Export for vllm-omni: **`--format auto_round` mandatory**; `fake` cannot be
  reloaded. vllm-omni scheme support: **W4A16 ✅ (PR #1777), MXFP8 ✅ (PR #3782),
  MXFP4 ✅ (PR #5544, merged 2026-08-25)** — all three load on vllm-omni main
  (release pending; installs older than 2026-08-25 reject MXFP4).
- The exported transformer `config.json` must carry `quantization_config` with
  `quant_method="auto-round"` AND `block_name_to_quantize`; if auto-round did not
  write it, post-process the JSON.
- SVDQuant→Nunchaku is FLUX.1-only; its adapter auto-detection misses FLUX.2
  (`Flux2Transformer2DModel` does not contain substring "fluxtransformer").
- Per-model verdicts and smoke-test ordering (klein-4B → 9B → dev): see the
  diffusion reference doc, section 4.

## Procedure
1. **Analyse structure without downloading weights**: read `config.json` +
   safetensors index (diffusion: `model_index.json` + `transformer/` subfolder);
   classify dense vs MoE vs DiT, list module names, expert layout.
2. **Consult the recipe corpus** (`recipes/schemes/<scheme>.yaml`, prior pinned
   recipes) for known-good settings.
3. **Emit the command** via `aqa/quantize` (or an agent-optimized command). Use
   `--model_name <model>` explicitly. Add `--ignore_layers` / `layer_config`
   for mixed precision when the structure requires it.
4. **Write `quantization_recipe.md`** (conclusion / structure / module list /
   built-in handling / config decisions / recommended command / accuracy
   expectation / variants) and **pin** the machine-readable recipe via a
   `RecipeStore` (default `repo_dir` -> `recipes/quant/<org>__<model>.json`).

## What auto-round already handles (do NOT hand-configure)
- MoE router/gate: modules ending in `.gate` are auto-ignored for MoE models
  (special_model_handler fallback); several models have richer predefined
  ignore lists (GLM, Step3.5, Kimi-K2.5, BAGEL, ...).
- lm_head: not quantized unless `--quant_lm_head` (and forced off for tied
  embeddings); layers with shapes not divisible by the group size are auto-kept
  in fp16.
- VLM/multimodal: only the text tower is quantized unless
  `--quant_nontext_module`.
- Diffusion: pipeline detection, transformer/unet targeting, and coco2014
  calibration are automatic — you only choose scheme/iters/steps/guidance.

## Pitfalls (verified against source / real runs — do not relearn these)
- `enable_quarot_opt` does NOT exist. Use `rotation_config="quarot"` (Python).
- CLI `--rotation_type quarot_hadamard` is rejected by config validation; use
  `hadamard` / `random_hadamard` (or `inplace_quarot_hadamard` via API).
- `--fp_layers` is only a CLI alias of `--ignore_layers`; the Python API has
  no `fp_layers` parameter.
- MXFP4 CAN export to `auto_round` format too (weights identical to
  llm_compressor layout), but use `llm_compressor` for vLLM deployment.
- `auto_gptq` / `auto_awq` are export formats only (int schemes only) — the
  algorithm is still SignRound tuning.
- MX schemes: `group_size` fixed at 32; NVFP4: 16.
- **`mlp.gate` substring-matches `mlp.gate_proj`** — on dense models this
  silently excludes all gate_proj from quantization. For router-only matching
  use the anchored form `mlp.gate$`; on dense models simply omit it.
- **Export/weights consistency** (vLLM `KeyError: '<layer>.weight'`): if a
  Linear is stored plaintext but not listed in `ignore`, vLLM builds it as a
  quantized layer and fails. When ignoring a tower (e.g. `visual`), make sure
  the exported ignore list covers ALL its Linears (check both
  `config.json` and `quantization_config.json`).
- **MXFP8 `mm_mxfp8 requires N >= 128`**: any Linear with out_features < 128
  (e.g. Gated-DeltaNet `in_proj_a/b`, N=64) must be BF16-ignored if it would
  land in an MXFP8 group. MXFP8 is also MX-block FP8, NOT per-channel FP8 —
  it cannot byte-replicate Quark `fp8_e4m3 per_channel` recipes.
- **model_free scheme whitelist** is fixed (W2A16(G32/G64)/W4A16(_MIXED)/
  W8A16/MXFP4/MXFP8/NVFP4_E5M3/BF16) — `MXFP4_RCEIL` is NOT accepted in
  model_free. model_free forces `iters=0`.
- In model_free, predefined skips (`.gate.`, `embed`, `conv`,
  `shared_expert_gate.`) take priority over layer_config — a router can never
  be quantized there (that's a safety feature).
- Huge shards: cap `AR_MODEL_FREE_SHARD_PARALLELISM=1` first (each worker may
  dequantize MXFP; default up to 10 workers can OOM host RAM).
- llm_compressor export groups layers by effective weight bits and rebuilds
  full presets: `{mlp.experts:{bits:4,data_type:mx_fp}}` under default MXFP8
  yields experts = full MXFP4 **W4A4**, not W4A8.
- In the current repo, recipes (`auto-round-best` etc.) do NOT auto-enable
  `enable_alg_ext` — pass it explicitly when chasing accuracy (W2A16/MXFP4).

## Accuracy / safety rules
- Never fabricate module names — derive them from the actual index.
- Never fabricate CLI flags — every flag in an emitted command must exist in
  the reference doc or the source repo.
- The pinned recipe is the single source of truth; do not hard-code recipe
  knowledge in code paths.
