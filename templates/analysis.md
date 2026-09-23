# <Model> 结构分析 + 量化方案（第①步产物）

> 模板使用：替换所有 `<...>`；章节不许删（不需要的写「本次无，原因」）。
> 铁律：**不下载权重做分析**——只读 `config.json` + `model.safetensors.index.json`
> + 抽查张量头 dtype/shape。所有数字必须实测（注明命令/日期），不许凭记忆。
> 实例参照：`cookbooks/Qwen3.8-27B-quantization-recipe.md`、
> `cookbooks/GLM-5.3-Flash-quantization-recipe.md`。

- 模型：`<HF id 或本地路径>`（源 dtype：`<bf16/fp8/...>`，`<N>` 分片，`<实测体积>`）
- 日期 / 机器：`<YYYY-MM-DD>` / `<主机+GPU 型号×数量>`
- 用户需求：`<目标 scheme / 算法 / 部署约束 / 精度预算——用户说什么就是什么，显式配置是硬约束>`

## 0. 查阅清单（必填，带日期/版本）

开工前必须查完以下项并记录结论（采纳了哪条 / 为什么偏离）。这一章是 owner
复核入口：方案有没有依据，看这里，不用翻 session。

| 查阅对象 | 实际查到的（版本/日期） | 采纳 / 偏离及理由 |
|---|---|---|
| `skills/quantize/SKILL.md` | | |
| `skills/evaluate/SKILL.md` | | |
| 结构分析 skill（autoround-quant-analysis/SKILL.md，部署机路径见 AGENTS.md 文档地图） | | |
| `recipes/schemes/<scheme>.yaml` | | |
| `recipes/quant/` 同模型/同架构历史 pin 配方 | `<有=路径；无=本次收尾要补 pin>` | |
| knowledge 增量层：`aqa knowledge show` 的 `auto_round_ref` | `<文档版本 vs 将用的 auto-round 版本；旧则先走 docs/knowledge_refresh.md 刷新>` | |
| 同类实证：`cookbooks/` 相近架构 recipe / `docs/day0_quantization_recipe.md` | | |
| 上游查证（model card / vllm PR·issue / transformers release） | | |

## 1. 架构判定（以 config.json 为准，不看模型名）

| 项 | 值 | 说明 |
|---|---|---|
| `architectures` / `model_type` | | |
| dense vs MoE | `<experts 数 / top-k / shared / moe_intermediate_size；dense 写「无 MoE」>` | |
| 层结构 | `<总层数；混合注意力（linear/full 分布）；MTP；vision 塔>` | |
| hidden / inter / head 维度 | | |
| 词表 / 上下文 / `tie_word_embeddings` | | |
| 权重 dtype 分布 | `<全 bf16 / 原生 FP8（二次压缩风险！）/ 混有 fp32 件>` | |
| **`quantization_config`** | `<源模型可能已是量化权重——是则换源或论证二次量化>` | |

## 2. 三件套支持探针（确定性，勿猜）

| 件 | 探针命令 | 结果 |
|---|---|---|
| transformers | `python -c "from transformers import AutoConfig; AutoConfig.from_pretrained(...)"`（最新 release 版本号） | ✅/❌（❌ → 查 model card 推理 recipe / PR） |
| vllm | registry 是否含该架构（本地版 + 最新 release + main） | ✅/❌（❌ → 官方定制镜像？含支持的最低 tag？） |
| auto-round | 是否支持该结构（fused MoE 拆解需 ≥0.15.0/#2072） | ✅/❌ |

**推理路径结论**：`<环境形态：local venv / local docker（官方镜像名）/ ssh docker>`，
以及为什么（哪些约束逼出来的：CUDA kernel 级改动不可 patch、PyPI 落后等）。

## 3. 模块 / 张量清单与参数占比

从 index.json 实测统计（附统计命令/脚本）。产出**逐模块精度表**——这是量化
计划的本质，也是 owner 的核心把关点：

| 模块（实际层名模式） | 出现次数 | 参数量 | 占比 | 目标精度 | 归类理由 |
|---|---|---|---|---|---|
| `<mlp.experts.*.down_proj 等>` | | | | `<MXFP4/MXFP8/...>` | 量化主力 |
| | | | | bf16 | 敏感族（attention/router/embed/...） |
| | | | | fp32 | checkpoint 本来就是 fp32，不降级 |

注意：
- 层名必须对着 index.json 逐条核对，子串陷阱写出来（如 `mlp.gate` 误伤
  `mlp.gate_proj`；`b_proj` 误伤 `q_b_proj`）。
- 标出已知雷：小 N 层（MXFP8 kernel `N>=128`）、3D 融合专家张量、
  vllm 端融合名/非融合名歧义。

## 4. 量化方案

| 决策 | 取值 | 依据（指回 §0 查阅清单或实测） |
|---|---|---|
| scheme / method / algorithm | | |
| iters / nsamples | | |
| export_format | | |
| ignore_layers（最终清单，逐条核对过） | | |
| layer_config（混合精度时） | | |
| static_kv_fp8 / static_attention_fp8 | | |
| 预期产物体积（按位宽估算） | | 收尾 sanity check 的基准 |

**生成的量化命令**（aqa 确定性 builder 产出或推导过程记录）：

```bash
<auto-round ... 或 aqa run 的等价说明>
```

## 5. 评估计划

- baseline：`<是否测 bf16（gate 对比前提）/ 用官方已发表参照值（注明出处与不可比风险）>`
- benchmarks 与口径：`<逐任务 metric 口径；生成类任务的 chat template 处理>`
- thinking：`<auto/both/on/off——模型 chat template 有无 enable_thinking 开关>`
- backend 与关键 model_args：`<TP/max_num_seqs/reasoning_parser/...>`
- gate 阈值：`<max_accuracy_drop / min_accuracy——注意阈值是口径绑定的>`

## 6. 风险与未决项

- `<逐项列出：可能掉点的模块、没把握的兼容性、需要冒烟验证的假设>`
