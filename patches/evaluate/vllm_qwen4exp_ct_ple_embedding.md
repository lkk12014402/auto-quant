# vLLM qwen4_exp: compressed-tensors 产物在 PLE embedding 处被拒（回归自 #54371）

**文件**：`vllm_qwen4exp_ct_ple_embedding.patch`（对 `vllm/models/qwen4_exp/nvidia/ngram_embedding.py`，`git apply -p1`）
**影响版本**：vLLM main ≥ `3116c5d06b`（#54371 "[Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism"，2026-09-09）
起，**含 2026-09-22 的 nightly `9815732c1` 与当日 main**。
**不受影响**：#54371 之前的树（本机 vendor 树 `0.28.1rc1.dev312+g41848caa6` 基线是 9/2 → 没有这个问题，
所以 8/29 的 MXFP4-Mixed 参考产物与 9/21 的 M1 实跑都是在那棵树上过的）。

## 症状

评估/服务一加载就崩，且崩在**建模块阶段**（还没读权重，约 90 秒就出）：

```
File ".../vllm/models/qwen4_exp/nvidia/model.py", ... in __init__
    self.ple_embedding = Qwen4ExpNGramEmbedding(
File ".../vllm/models/qwen4_exp/nvidia/ngram_embedding.py", ... in from_quant_config
    raise NotImplementedError(
NotImplementedError: Qwen4Exp PLE embedding does not support quantization config CompressedTensorsConfig
```

任何 auto-round `--format llm_compressor`（= compressed-tensors）的 Qwen3.8-Flash-Next 产物
（MXFP4-Mixed / MXFP8 / W4A16-CT…）都中；`--quantization` 不写也一样，因为 config 是从产物的
`quantization_config` 推出的。

## 根因

`from_quant_config()` 是 #54371 从 `ple_layer.py` 搬进 `ngram_embedding.py` 的白名单化重写。搬之前的判据是
"**不是** serialized-FP8 的 `Fp8Config` 就返回 None（= 不量 PLE）"：

```python
# vendor tree @ 9/2, ple_layer.py:_get_ple_embedding_quant_method
def _get_ple_embedding_quant_method(quant_config, prefix):
    if not isinstance(quant_config, Fp8Config):
        return None            # ← CT 走这里，n-gram 表保持 BF16，正常加载
    if not quant_config.is_checkpoint_fp8_serialized:
        return None
    ...
```

搬之后变成"只认 `None` / ModelOpt / `Fp8Config`，其余一律 raise"，compressed-tensors 从没被加进白名单
⇒ 默认放行的语义被反转成默认拒绝。PLE 的 n-gram 表（本模型 51.2 B / 102 GB，embedding 查表）在任何
CT 产物里都不可能是量化目标（auto-round 把 128 个 `...ple_embedding.ngram_embedding.shard_<i>` 全列进
`ignore`，落盘 dtype 是 BF16），所以这纯粹是**类型白名单漏了一类 config**，不是能力冲突。

搜索记录（2026-09-22，owner）：`"PLE embedding does not support"` 与 `qwen4_exp compressed-tensors`
在 vllm-project/vllm 的 issue/PR 里 **0 命中** → 未上报的回归。值得单独提 PR（改动就是把 CT 加回放行分支）。

## 修法

在 `from_quant_config()` 里、`if not isinstance(quant_config, Fp8Config)` 之前，把 CT 重新放行成"不量 PLE"：

```python
if isinstance(quant_config, CompressedTensorsConfig):
    ...
    return Qwen4ExpPLEUnquantizedEmbeddingMethod()
```

要点（为什么不是照抄下面 `Fp8Config` 分支的 ignore 匹配）：产物里的 `ignore` / `targets` 名字是
**HF checkpoint 风格**（`model.language_model.layers.9.mlp.gate`），而这里收到的 `prefix` 是 **vLLM 模块路径**
（`...language_model.model.layers...` 一族），文本匹配会把合法产物也判成"没被 ignore"→ 继续 raise。
所以本 patch 与 #54371 之前一致地按"CT 从不量化 embedding"放行，并留一条 `debug_once` 把真实 prefix
打出来（要核实名字形态时把 `VLLM_LOGGING_LEVEL=DEBUG` 打开即可看到）。
真出现"CT 量化了 embedding"的产物（目前没有这种导出），权重装载会在 dtype/shape 上硬失败，不会被静默放过。

## 应用

```bash
# 只作用于本 run 专用的那个 venv（不动 /models/lkk/muse_support/vllm 那棵共享 vendor 树）
SP=/models/lkk/venv_qwen38fn_fp8kv/lib/python3.12/site-packages
cd "$SP" && git apply -p1 --check /models/auto-quant/patches/evaluate/vllm_qwen4exp_ct_ple_embedding.patch \
            && git apply -p1 /models/auto-quant/patches/evaluate/vllm_qwen4exp_ct_ple_embedding.patch
```

`git apply` 基线核对过：patch 是对 commit `9815732c16ca076c60b65fccd787167d6a9c5328` 的原始 blob 生成的，
在 pristine 副本上 `--check` 通过、应用后与 venv 里实际生效的文件逐字节相同。

## 验证（本机实测）

- 打前：`qwen38_flash_next/mxfp4_mixed_fp8kv` 第一次运行在 14:05:33 崩于上述 `NotImplementedError`（TP=2、FP8 KV）。
- 打后：`Qwen4Exp PLE embedding` 不再 raise，模型进入权重装载/引擎就绪阶段。
