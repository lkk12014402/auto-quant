# Patch: compressed-tensors 支持嵌套（VLM）config 的 attention 维度读取

**目标包/版本**：`compressed-tensors 0.17.0`（实测基线，`muse_new` conda env）
**文件**：`compressed_tensors/utils/helpers.py`
**应用位置**：site-packages 根目录，`patch -p1`

## 症状

带 `static_kv_fp8` / `static_attention_fp8` 的量化导出（即 aqa 的 v2/v3 变体）在
装载阶段崩溃，与量化方案本身无关：

```
File ".../compressed_tensors/quantization/lifecycle/initialize.py", line 276, in initialize_attn_qparams
File ".../compressed_tensors/utils/helpers.py", line 438, in get_num_attn_heads
ValueError: Cannot determine num_attention_heads from config. Config must define
either `num_attention_heads` or both `hidden_size` and `head_dim`. Qwen3_5Config {...}
```

复现体（本机现成）：`Qwen/Qwen3.5-9B` + `scheme MXFP4` + `method autoround, iters: 0`
+ `static_kv_fp8: true`，导出目录
`results/qwen3_5_9b/_diag_nolowmem/quantized/Qwen3.5-9B-mxfp-w4g32/`。

## 根因

`get_num_attn_heads` / `get_num_kv_heads` / `get_head_dim` 只从**传入 config 对象自身**
读这些字段：

```python
if hasattr(config, "num_attention_heads"): ...
```

而 Qwen3.5 / Qwen3-VL 这类模型是 **composite config**：
`Qwen3_5Config.sub_configs = {text_config, vision_config}`，`num_attention_heads=16`、
`num_key_value_heads=4`、`head_dim=256`、`hidden_size` **只存在于 `text_config`**，
且 transformers 的 composite config **不会**把它们代理到 root。
实测 `hasattr(root, "num_attention_heads") == False` ⇒ 三个 getter 全部落到 raise。

只有走静态 KV/attention scale 时才会碰到这条路：
`model_free`（v1）不写 `kv_cache_scheme`，所以永远不触发 —— 这正是
"v1 正常、v2/v3 崩"的全部原因，与 MXFP4/MXFP8 无关。

## 修复

新增 `_attn_config_scope(config)`：当 root 上没有这三个属性、而 `text_config` 上有时，
下沉一层再读。三个 getter 统一使用它。

**对扁平 config 零行为变化**（root 有属性 ⇒ 原样返回 root），因此对现有所有
LLM 量化流程无影响；错误信息仍打印原始 `config`，便于定位。

## 验证（本机实测）

```
嵌套复合 config: 16 4 256            # 修复前 raise
同一对象作 root:  16 4 256            # text_config 直接传入，行为不变
合成扁平:         8 2 64             # 无回归
回退除法:         64                 # hidden_size//num_attention_heads 分支保留
缺失仍正确报错                         # 真缺字段不会被静默吞掉
```

补丁本身：正向 `--dry-run` 通过；应用后与现场文件**逐字节一致**；`-R` 反向 `--dry-run`
通过（可干净回退）。

## 应用方式

```bash
SP=/models/miniforge3/envs/muse_new/lib/python3.12/site-packages
cd "$SP" && patch -p1 < /models/auto-quant/patches/quantize/compressed_tensors_nested_config_attn_dims.patch
# 回退：patch -R -p1 < 同一文件
```

## 复现建议 / 注意

- 这是 **site-packages 内的已安装包**（非 editable），换环境或重装 compressed-tensors
  会丢；重装后需重新打。**先核对版本是 0.17.0**，不要盲目移植到别的版本
  （见 `patches/README.md` 的"两种文件布局"事故记录）。
- 上游方向：该修复是通用性的（任何 composite config 模型都需要），值得提 PR 给
  `compressed-tensors`；本机只是 owner 侧的临时沉淀。
- 本补丁解决的是"导出之后装载"。model-load **导出本身**还有两个独立前置条件，
  见 `OWNER_JOURNAL.md §15b`：① auto-round 需 `--disable_low_cpu_mem_usage`
  （否则 meta 骨架导致键名前缀翻倍）；② 导出嵌套一层目录，需 aqa 侧
  `resolve_export_dir()` 解析。
