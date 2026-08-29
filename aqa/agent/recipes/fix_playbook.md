<!-- Fix-agent skill / troubleshooting playbook for the quantize + evaluate phases.
This file is packaged with aqa and embedded (best-effort) into EVERY fix prompt as
reference knowledge. It is safe to edit/extend: keep entries mechanism-level and
verifiable. Per-run box/model-specific notes belong in `agent.context` instead. -->

# aqa fix playbook (quantize + evaluate)

Reference knowledge, NOT instructions. Confirm each item against THIS traceback before
acting; the deterministic classifier and your own EVIDENCE_CMD always win over this list.

## Ignore / fp-protect layer name matching
- `--ignore_layers` / `--fp_layers` do **substring** matching on weight names.
- Never pass a bare leaf like `gate` — it also matches `mlp.experts.gate_up_proj` and
  `shared_expert.gate_proj`, silently skipping the quantization target. Use the qualified
  form `mlp.gate` to hit only the router.
- A protect-name that matches NO real weight is a no-op; at load time a mismatch between
  the ignore-list and actual weight names raises. Verify against section 3 of the recipe
  (the safetensors-index module names), not from memory.

## MXFP8 / MXFP4 fused-kernel shape constraints (flashinfer / mm_mxfp8)
- MX matmul kernels require the fused output dim N to meet a minimum (commonly N >= 128).
- vLLM fuses sibling projections: e.g. Mamba/linear-attn `in_proj_a` + `in_proj_b`
  (out=32 each) fuse to `in_proj_ba` (N=64) which is < 128 -> the engine aborts on load.
- Fix mechanism: keep such small fused projections in FP16, e.g.
  `--ignore_layers "linear_attn.in_proj_a,linear_attn.in_proj_b"` (add to, don't replace,
  the model_free auto-ignore set). Confirm the offending N in the error, then assert the
  layer is in the checkpoint's quantization_config ignore-list as your smoke test.

## model_free auto-ignore
- auto_round `--model_free` auto-adds conv1d / embed / rotary / norm / visual to ignore,
  but NOT small fused attention projections — those must be added explicitly (see above).

## Router / MoE
- Routers (`mlp.gate`, `moe.gate`, `shared_expert_gate`) are FP4/FP8-sensitive -> keep FP16.
- Routed experts (`mlp.experts.*`, fused `gate_up_proj`/`down_proj`) are the quantization
  target and the bulk of params -> NEVER fp-protect them; drop bits only via `--layer_config`.

## Device / backend (XPU vs CUDA)
- On Intel XPU: select devices with `ONEAPI_DEVICE_SELECTOR=level_zero:...` / `ZE_AFFINITY_MASK`,
  NOT `CUDA_VISIBLE_DEVICES`. Prefer `enforce_eager=True` for compressed-tensors MX loads.
- Custom model code calling `.cuda()` / `.float()` on XPU: replace with `.to("xpu")` /
  `.to(other.dtype)`. Such code under
  `~/.cache/huggingface/modules/transformers_modules/` IS editable.

## OOM / memory
- Large-model load OOM: prefer `--device_map auto`; for model_free memory pressure,
  `AR_MODEL_FREE_SHARD_PARALLELISM=1` shards the pack. Lower `batch_size` before touching
  precision.

## Evaluate phase (lm-eval-harness)
- `acc == 0` across tasks usually means a broken load (dtype/format mismatch, wrong
  chat template, or a silently-skipped generation), not a real "0% model" — treat it as a
  FAILURE and root-cause the load, don't accept the number.
- Prefer a tiny `--limit` smoke over a full task suite while iterating; the harness re-runs
  the full evaluation authoritatively after you return.

## Smoke-test discipline (see PROTOCOL)
- Verify the SPECIFIC mechanism you changed at ~30s: import the edited module, read the
  quantization_config JSON, read a safetensors HEADER (not a full load), or call just the
  failing kernel once on a dummy tensor of the offending shape. Do not re-quantize or boot
  the whole engine as smoke.

## Diffusion (vllm-omni text-to-image)
- vllm-omni loads an auto_round DiT checkpoint only when the transformer's `config.json`
  carries `quantization_config` with `quant_method="auto-round"` AND `block_name_to_quantize`
  (e.g. "transformer_blocks,single_transformer_blocks"); if auto-round's diffusion export
  did not write it, post-process the JSON to add it (verify by re-reading the JSON).
- MXFP4 DiT checkpoints load on vllm-omni upstream main (PR #5544 merged 2026-08-25);
  older local checkouts predating the merge still reject them -> use W4A16/MXFP8 there
  instead of patching the loader.
- auto-round forces `batch_size=1` for diffusion calibration; do not try to raise it.
- Generation failing at engine boot: confirm vllm-omni is actually installed and that the
  model's `model_index.json` names a pipeline class vllm-omni supports.
- **RTN auto-routes to model_free**: `--iters 0 --disable_opt_rtn` triggers auto-round's
  model-free path ("Auto-routing to model-free quantization"). For diffusion models that
  path can SILENTLY QUANTIZE 0 LAYERS (transformer config lacks `model_type` for layer
  detection; shard prefetch 404s at repo root) while still stamping a quantization_config
  — the exported checkpoint is then unquantized weights + a quantizer config (vllm-omni
  will fail to load it). ALWAYS pass `--disable_model_free` for diffusion RTN, and verify
  `Quantized layers (N)` with N>0 in the log before trusting the checkpoint.
- **`FileNotFoundError: '<model-id>/model_index.json'`** on diffusion quantize: the
  regular diffusion loader only accepts a LOCAL directory (`utils/model.py` opens
  `<path>/model_index.json` directly, no hub resolution). Download first
  (`hf download <id> --local-dir ...`) and pass the local path as `model`.
- **vllm-omni `no module or parameter named 'text_encoder.embed_tokens'`**: auto-round
  re-saves non-quantized components itself and can mangle them — Z-Image's Qwen3 text
  encoder loses the inner `model.` prefix on every weight (source `model.layers.N.*`
  → exported `layers.N.*`). aqa's post-export repair (`repair_diffusion_export`)
  auto-restores mismatched components from the source model and writes
  `aqa_export_repair.json`; if you see this error outside aqa, copy the pristine
  `text_encoder/` (and any mismatched component) from the source model into the export.
- **vllm-omni `KeyError: '<block>.N.adaLN_modulation.0.qweight'`（或类似小层键名）**:
  vllm-omni 刻意把某些小型 DiT 层保持全精度（模型文件里 `quant_config=None`，如
  Z-Image 的 `adaLN_modulation` 与 `t_embedder.mlp`，注释 "precision-sensitive"）。
  auto-round 会把它们连同所在 block 一起量化 → checkpoint 是打包的 qweight，引擎端
  却要明文 weight（或反向）。量化时排除：`--ignore_layers adaLN_modulation`（去
  vllm-omni 对应模型文件里搜 `quant_config=None`，忽略清单与它们对齐）。
- Full source-verified reference (quantization architecture, per-model verdicts,
  vllm-omni scheme matrix): `docs/autoround_diffusion_reference.md` in this repo.
