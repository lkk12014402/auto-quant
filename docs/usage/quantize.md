# 量化（quantize）用法

> 内核：`aqa/quantize/`（auto-round 命令构建）+ `aqa/providers/`（quantize provider）。
> 判断层：`skills/quantize/SKILL.md`（agent 的领域知识）。

## 最小配置

```yaml
model: Qwen/Qwen3-0.6B          # HF repo id 或本地路径
provider:
  type: quantize
  quant:
    scheme: MXFP4               # MXFP4 | MXFP8 | W4A16 | NVFP4_E5M3 | ...
    method: model_free          # model_free（免校准 RTN 快路径）| autoround（真加载调优）
```

## quant: 字段速查

| 字段 | 对应 auto-round | 说明 |
|---|---|---|
| `scheme` | `--scheme` | MXFP4 / MXFP8 / W4A16 / NVFP4_E5M3 / BF16 / W2A16(G32/G64)… |
| `method` | `--model_free`（省略即不传） | `model_free` 免校准；`autoround` 真加载+校准调优 |
| `iters` | `--iters` | 调优迭代数（0 = 纯 RTN）；`method: autoround` 时才有意义 |
| `nsamples` | `--nsamples` | 校准样本数（默认 128） |
| `algorithm` | `--algorithm` | 算法链：`rtn`/`auto_round` 二选一为量化器，`awq`/`hadamard`/`svdquant` 为可叠加预处理器 |
| `enable_alg_ext` | `--enable_alg_ext` | SignRound 扩展（"auto-round v2"），recipe 不会自动开 |
| `ignore_layers` | `--ignore_layers` | 高精度保留清单（子串匹配） |
| `layer_config` | `--layer_config` | 混合精度 per-layer 覆盖 |
| `quant_lm_head` | `--quant_lm_head` | 默认不量化 lm_head |
| `device_map` | `--device_map` | 多卡自动分片（>1 GPU 时自动 `auto`） |
| `extra_args` | 原样追加 | 如 `["--disable_torch_compile"]` |
| `export_format` | `--format` | `auto_round`（vllm-omni 部署必须）/ `llm_compressor` / `fake` |

## 算法链组合规则（config 加载即校验）

- 预处理器（`awq`/`hadamard`/`svdquant`）可叠加，量化器（`rtn`/`auto_round`）只能一个；
- `rtn,auto_round` 非法（两个量化器）；`awq,auto_round`、`hadamard,auto_round`、`svdquant,rtn` 合法；
- `method: model_free` 是免校准的——只能配 `rtn`，配 `auto_round` 会在 config 加载时报错。

## 两条路径怎么选

| | model_free | autoround（调优） |
|---|---|---|
| 加载模型 | 不加载（流式权重替换） | 真加载 + 校准前向 + block 级调优 |
| 耗时 | 分钟~小时级 | 大模型可达 10 小时+ |
| 精度 | 大模型上常已近乎无损 | 有提升空间时更好，也可能无收益 |
| 适用 | 超大模型 / day-0 先建下限 | 中小模型 / 追精度上限 |

**纪律：大模型先跑 model_free RTN 建下限，再决定要不要为调优付费。**

## 导出与验证

- 量化产物在 `output.dir/quantized/`；tuning 路径可能多套一层 `<model>-w4g128/`（aqa 的 provider 会自动解析嵌套）；
- 导出后自动做结构恢复（`restore_config_structure`：源 config + quantization_config 嫁接，防 transformers 重序列化漂移）；
- 相位钩子 `quant_sanity` 默认开着：声称 auto-round 导出但没有任何 `qweight` 打包张量会直接判失败；
- 推理冒烟：`python scripts/smoke_vllm.py <量化产物路径>`（TP 加载 + 3 条确定性生成，几分钟内暴露加载类问题）。

## 相关知识库

- agent 量化知识：`aqa/agent/recipes/auto_round_recipe.md`（LLM）+ `auto_round_diffusion.md`（diffusion）；
- 完整 scheme/算法/CLI 参数表（源码级）：`docs/autoround_full_reference.md`；
- day-0 新架构：`docs/usage/day0.md`。
