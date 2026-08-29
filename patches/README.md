# auto-round patches (day-0 Qwen3.8-Flash-Next)

`autoround_qwen4exp_fixes.patch` — two fixes against auto-round main @ `bd9a13ce`
(0.15.0-dev), required for the tuning path on `qwen4_exp` (Qwen3.8-Flash-Next).
Apply inside the auto-round repo:

```bash
cd <auto-round repo>
git apply /path/to/autoround_qwen4exp_fixes.patch
pip wheel . --no-deps -w /tmp/wheels   # container 无 git，源码直装会挂在 setup.py 的 git describe
```

## 内容

1. **`compressors/layer_config/resolver.py` — scheme 属性剥离范围限定。**
   `apply_plan_to_model` used to `delattr` any module attribute colliding with
   a `QuantizationScheme` field name (`bits`, `group_size`, ...) on **every**
   module. Qwen4-Exp's hyper-connection norm (`Qwen4ExpTextRMSNorm`) has a real
   structural `group_size` attribute (grouped RMSNorm) that its forward needs —
   deleting it crashes the first calibration forward. The patch restricts the
   cleanup to modules that can actually carry a quant plan (supported layer
   types / explicit plan entries).

2. **`utils/device.py` — 被忽略 nn.Embedding 叶子散布到预算最空闲的卡。**
   The multi-GPU tuning dispatch used to dump every non-quantized leftover
   module onto device 0. Qwen4-Exp's PLE n-gram embedding materializes as a
   single ~102GB `nn.Embedding` and OOMed device 0. The patch relocates leftover
   `nn.Embedding` leaves to the device with the largest remaining budget
   (AlignDevicesHook covers IO for module-call embeddings). Non-embedding leaves
   keep the old behavior because their weights may be consumed functionally by
   the parent module (e.g. `causal_conv1d_fn(self.conv1d.weight)`), which module
   hooks cannot intercept.

## 验证

- auto-round unit tests: no new failures (pre-existing ones verified against
  pristine main via `git stash`).
- End-to-end: Qwen3.8-Flash-Next W4A16 tuning (iters=200) completes 48/48
  blocks on 4xH200; tiny 3-layer qwen4_exp repro passes on 2 GPUs (EXIT=0).

## 上游

Candidates for an intel/auto-round upstream PR together with the tiny-model
reproductions (see `docs/day0_quantization_recipe.md` appendix A, pit #2/#4).
