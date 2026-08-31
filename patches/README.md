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

## 另外两份（GLM-5.3-Flash day-0）

- `torch_186659_compiled_backward_context.patch` — PyTorch 上游修复
  [pytorch#186659](https://github.com/pytorch/pytorch/pull/186659) 的一行移植
  （`_cc.run` → `_cc.copy().run`），修 compiled backward 的 contextvars 重入
  （[pytorch#185372](https://github.com/pytorch/pytorch/issues/185372)）。
  打在 torch 2.13.0 的 `torch/_functorch/_aot_autograd/runtime_wrappers.py`；
  对容器镜像用单文件 bind-mount 盖入即可，不用重建镜像。
- `transformers_glm5_next_mask_bool.patch` — transformers 5.16.1 的
  `glm5_next` 建模里四处 `~mask` 反转补 `.bool()`（与文件内 176 行既有写法对齐；
  eager 下无操作），修 dynamo fake-tensor 下 bf16 mask 反转炸
  （`TorchRuntimeError: invert on bf16`）。打 wheel 后安装
  （pip 会覆盖 bind-mount，所以不能走挂载）。

两者的实证与机制详见 `docs/day0_quantization_recipe.md` 附录坑 #3 与
GLM-5.3-Flash 的 run journal（`aqa_results/glm53_flash_day0/RUN_JOURNAL.md`）。

## 第三份（GLM-5.3-Flash day-0，拟上游 PR）

`autoround_shard_writer_converter_reverse.patch` — auto-round main @ `bd9a13ce`，
修 tuning 即时保存（ShardWriter）对 `WeightConverter` 融合权重只做**改名**不做
**切分**：glm5_next 加载时 `conversion_mapping` 把 q/k/v_conv1d 拼成单个
`conv1d`（Concatenate），导出时融合张量被整体写到第一个源名 `q_conv1d.weight`
下，k/v 丢失；vllm（glm5next kda.py 期望分拆 ColumnParallelLinear）缺权重
**静默随机初始化** → 加载正常但精度全随机。补丁按 transformers
`revert_weight_conversion` 的单张量版实现：名字命中反向 WeightConverter 时应用
其逆向操作（Concatenate→Chunk）逐源名写回。附单测（融合 conv1d 往返切分）。
实证：修复前 GLM-5.3-Flash tuning 评估 piqa/hellaswag/mmlu = 0.49/0.25/0.24
（随机），修复后 = 0.832/0.689/0.862（与 RTN 持平）。
分支：`fix/shard-writer-converter-reverse`（commit 7026d988，待推 fork 开 PR）。
