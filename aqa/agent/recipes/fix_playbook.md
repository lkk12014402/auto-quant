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
- **MXFP8 MoE has a separate shape bug at TP>1** (vLLM
  `.../quantization/utils/flashinfer_utils.py` `_shuffle_mxfp8_moe_weights`): the FlashInfer
  TRTLLM scale swizzle pads the SF **column count up to a multiple of 4**, then the code
  `.reshape(w2_scale.shape)`s back to the UNPADDED shape ->
  `RuntimeError: shape '[512, 2560, 10]' is invalid for input of size 15728640`.
  Deterministic predicate: `cols = moe_intermediate_size / TP / 32`; crashes iff
  `cols % 4 != 0` (Qwen3.8-Flash-Next: 20/TP -> TP=1 ok, TP=2/4 crash). Verify on CPU with
  `flashinfer.shuffle_matrix_sf_a(torch.zeros(2560, 10, dtype=uint8).cuda(), 128)` ->
  numel 30720 (=2560*12), while `(2560,20)` -> 51200 (no pad).
  Fixes, in order of preference: quantize experts to MXFP4 instead (the W4A4 MoE path swizzles
  to `padded_N` explicitly and is not affected); else keep W8A8 and override the MoE backend
  (`LLM(kernel_config={"moe_backend":"marlin"})`) — probe which alternates are actually
  installed (`deep_gemm`, `humming_kernels` often are NOT; `triton` is rejected on CUDA for
  MXFP8 MoE), don't hardcode a backend name.
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
- **Some giant tensors impose a minimum TP, independent of weights.** Qwen3.8-Flash-Next's PLE
  n-gram table (51.2B params = 95.37 GiB bf16) needs a SECOND full copy at TP=1 during
  first-forward/compile -> OOM even at `gpu_memory_utilization=0.95` (weights alone DO fit:
  219.94 GiB of 268.6 GiB). At TP>=2 it is vocab-sharded (`PLEVocabParallelEmbedding`), so the
  peak halves. Diagnostic: the OOM alloc size equals the tensor's byte size exactly
  (95.37 GiB = 51_200.25M x 2B), and it does NOT shrink when `max_num_batched_tokens` drops
  (measured identical at 16384 and 2048) — so don't "fix" it by lowering the batch.

## Evaluate phase (lm-eval-harness)
- `acc == 0` across tasks usually means a broken load (dtype/format mismatch, wrong
  chat template, or a silently-skipped generation), not a real "0% model" — treat it as a
  FAILURE and root-cause the load, don't accept the number.
- Prefer a tiny `--limit` smoke over a full task suite while iterating; the harness re-runs
  the full evaluation authoritatively after you return.
- **`--model_args` forms are version-dependent and one is a trap.** Current lm_eval
  (`_cli/utils.py` `MergeDictAction`) tries `json.loads` on the FIRST token and, if the string
  contains `{`, raises `Invalid JSON: ...` instead of falling back to `k=v` parsing. So a
  comma-joined `k=v` string that embeds any dict value (e.g.
  `...,kernel_config={"moe_backend":"marlin"}`) fails, while the same string without braces works.
  Pass the whole thing as ONE JSON object to be safe — it also fixes types for free
  (ints/bools/nested dicts). Validate cheaply without booting vLLM:
  `from lm_eval._cli.utils import try_parse_json; try_parse_json(s)`.

## Smoke-test discipline (see PROTOCOL)
- Verify the SPECIFIC mechanism you changed at ~30s: import the edited module, read the
  quantization_config JSON, read a safetensors HEADER (not a full load), or call just the
  failing kernel once on a dummy tensor of the offending shape. Do not re-quantize or boot
  the whole engine as smoke.
