# GLM-5.3-Flash 结构分析 + AutoRound MXFP4-Mixed / MXFP8 量化方案

- 本地目录：`/models/lkk/cri_models/GLM-5.3-Flash` —— **这是原生 block-FP8 版**（`du` 306 GiB = **328.3 GB**（header 实算）/ 62 分片，`quantization_config.quant_method=fp8`，`weight_block_size=[128,128]`）
- 架构：`Glm5NextForConditionalGeneration` / `model_type: glm5_next`（VL 多模态 + MoE + KDA 线性注意力 + 稀疏 MLA + hyper-connection + MTP）
- 参考产物：https://huggingface.co/INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound （下文 **HF 参考产物**，已逐项反解并复现）
- 先例：`/models/auto-quant/recipes/quant/zai-org__GLM-5.3-Flash.json`（W4A16 线，2026-08-31 验证）
- 工具链与 Qwen3.8-27B / Qwen3.8-Flash-Next 同一套：auto-round 0.16.0 @ env `muse_verify`，分析器 `audit_precision_plan.py`
- **验证状态**：结构/精度分配审计 = 已完成；HF 参考产物反解 = 已完成（37 152 层逐项吻合）；**G1/G1+ 已实跑**（BF16 源已到位）；官方 FP8 覆盖范围 = **safetensors header 逐张量实测**；**vLLM 支持 = 本机实测不支持**，上游结论为源码/tag 级核实，未端到端起服务
- ⚠ **本轮实跑推翻了 §3/§4 的三处预测**：plus 口径实际是空转的、锚定 `o_proj` 的正则写法失效（已在 `gen_glm_ignore.py` 修正）、层 0 的 dense MLP 被静默漏掉。依据见 **§4.1（auto-round 内置 predefined ignore）** 与 **§3.1（与官方 FP8 的集合 diff）**；**§4.2 是本机对 auto-round 的补丁**，打完之后 `*_plus` 的量化集合与官方 block-FP8 **双向 diff 为 0**

---

## 0. 结论速览

| 方案 | 组合 | 精度分配 | 产物体积 | 结论 |
|---|---|---|---|---|
| **G0 MXFP4-Mixed** | `MODE=mxfp4_mixed`（默认） | routed experts=MXFP4(W4A4)，dense MLP+shared_experts=MXFP8，注意力全 BF16 | **≈183.33 GB** | ✅ **基准方案 / 要出货就用它**；4-bit 37 152 层 + 8-bit **138** 层，与 HF 参考产物逐项吻合（§4.2 补丁修好了原先少掉的 3 层 dense MLP） |
| **G0+ 含稀疏MLA** | `MODE=mxfp4_mixed_plus` | G0 + 官方 block-FP8 也量了的 48 个稀疏MLA投影 → MXFP8 | ≈182.16 GB（省 1.17 GB） | ❌ **不要用于推理**：范围确实与官方对齐了（§4.2 补丁），但同样的 48 个 MXFP8 注意力投影会让 vLLM 加载期反量化算式崩（§6.5.1，已在 G1+ 上实测）；G0+ 走的是同一批张量，结论直接继承 |
| G1 纯 MXFP8 | `MODE=mxfp8` | 量化区全 MXFP8 | **≈339.16 GB**（37 290 层） | ❌ 比官方 FP8(328.3 GB) 大 **3.3 %**；注意旧版"339 vs 306 亏 33 GB"是 GB/GiB 混比 |
| G1+ | `MODE=mxfp8_plus` | G1 + 稀疏MLA | **338.07 GB**（37 338 层，重跑产物 `…-MXFP8-Plus-AutoRound-aligned`） | ⚠ 范围与官方**逐层一致（双向 diff 0，§3.2）**，但**实测无法加载**：vLLM 的注意力反量化把 MXFP8 的 scale 布局算错 → `RuntimeError: size of tensor a (512) must match b (16384)`（**§6.5.1**）。且比官方大 9.75 GB（+2.97 %，全是 scale 粒度）。**只能当范围对齐的实验品** |

> 8-bit 层数：旧版预测 aligned 138 → plus 186；**打补丁前实跑两者都是 135**（`layers.0.mlp` 吃掉层 0 的 3 个 dense MLP，裸 `self_attn` 让 plus 完全空转）；**§4.2 的本地补丁之后恢复 138，plus 也真正到 186**。
> plus 的 ignore 不写在脚本里，由 `gen_glm_ignore.py` 按 config 现算：**KDA 层与稀疏MLA 层的输出投影同名 `self_attn.o_proj`**，只能按层号（`linear_attn_config.kda_layers`）锚定 —— 且锚定模式**必须用裸点**，`layers\.(0|1)` 这种手写转义会被 `to_standard_regex` 二次转义成永不相配（实测，已修，见 §4）。

三件事必须先知道：

1. **本机原有 vLLM 不认这个模型**（`muse_verify` 的 vendor 树实测 `NOT registered`），但 **本轮已按 §7 建好可用环境**：`/models/lkk/venv_glm53`（vllm `0.29.1rc1.dev471+g82daf9f57`，registry 实测 `Glm5NextForConditionalGeneration=True`，torch 2.13.0+cu130，deep_gemm `2.8.0+e1f418c`）。上游支持由 PR #53906 提供，**已于 2026-09-03 合并**，最低 tag **v0.29.1rc0**、稳定 tag **v0.30.0**；但 **PyPI 最新只有 0.29.0（不含）**、`:v0.30.0` 镜像也没发 → 只能走 **nightly**（`wheels.vllm.ai/nightly/cu130` 或 `vllm/vllm-openai:nightly`）。注意 **v0.30.0 比 main 少 10 条 glm5next 修复**，建议直接跟 nightly；官方专用 tag `glm53-flash*` 停在 09-09，已不是首选。纯 pip 路线还要**手动补 `deep_gemm` 与 `flashinfer-cubin`**（vllm wheel 不带）。详见 §6，环境构建 §7。
2. **量化输入源不对**。pinned recipe 白纸黑字写着"不要从原生 FP8 版二次压缩；FP8 版留作量化范围的对齐参照"，而本地这份就是 FP8 版。默认脚本要求 BF16 源（`zai-org/GLM-5.3-Flash-BF16`，642.6 GB header 实算 / `du` 605 GiB；`/models` 现余 **1.2 TB**，2026-09-21 实测），要用 FP8 源必须显式 `SOURCE=fp8`，脚本会二次确认。
3. 这个模型**不踩 Qwen3.8-Flash-Next 的两个坑**：(a) 被量化的集合里没有 N<128 的小投影（GLM 真正被 shape 卡住的只有 KDA 的 **`self_attn.b_proj`，N=64**；`f_a/g_a` N=128、`f_b/g_b` K=128 都合规。注意生成器里写的是 **锚定的 `self_attn.b_proj`**，裸 `b_proj` 会子串误伤 `q_b_proj`/`kv_b_proj`，实测踩过）；**稀疏 MLA 的 4 个投影 shape 全部合规**（`q_a` [1536,4096]、`q_b` [16384,1536]、`kv_a` [512,4096]、`o_proj` [4096,8192]，N/K 皆为 32 的倍数且 N≥128 → 它们量不上与 shape 无关，见 §3.1）；(b) experts 是 **MXFP4** → 走 `CutlassExpertsMxfp4`，不经过那个 MXFP8 MoE scale-swizzle 形状 bug；即使退到 G1（experts MXFP8），`cols = moe_intermediate/(TP·32) = 2048/TP/32 = 64/TP` 在 TP=1/2/4/8 下分别是 64/32/16/8，**全是 4 的倍数，也不会触发**。

---

## 1. 模型结构

### 1.1 config 特征

| 项 | 值 |
|---|---|
| `architectures` / `model_type` | `Glm5NextForConditionalGeneration` / `glm5_next`（`glm5_next_text`） |
| 层数 | 45（+ 第 45 层作为 MTP 层，张量名里就是 `model.language_model.layers.45.*`） |
| 注意力混合 | **34 层 `linear_attention`（KDA）+ 11 层 `deepseek_sparse_attention`（稀疏 MLA，层号 3,7,…,43）**；`indexer_types` 全 `full` |
| MoE | `n_routed_experts=288`、`num_experts_per_tok=8`、`n_shared_experts=1`、`moe_intermediate_size=2048`、`moe_router_dtype=float32`、`first_k_dense_replace=3`（前 3 层 dense MLP，`intermediate_size=12288`） |
| MLA | `kv_lora_rank=512`、`q_lora_rank=1536`、`v_head_dim=256`、`qk_nope_head_dim=256`、`qk_rope_head_dim=0`、`mla_use_nope=true` |
| DSA indexer | `index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`、`index_kpool=4`、`index_kpool_compress=true`、`index_share_for_mtp_iteration=true` |
| Hyper-connection | `mhc=true`、`hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-6`（每层 `hc_attn_*` / `hc_ffn_*`） |
| MTP | `eh_proj`、`shared_head.*`、`enorm`/`hnorm`（第 45 层） |
| 多模态 | `visual` 塔（24 blocks + merger + patch_embed + downsample） |
| 其它 | `hidden_size=4096`、`vocab_size=154880`、`max_position_embeddings=1 048 576`（1M 上下文） |
| 本地源 dtype | `F8_E4M3` 37 338 个权重 + `F32` 37 629 个 `weight_scale_inv` + `BF16` 1 141 个 → **block-wise FP8**（128×128，per-block float32 scale） |

### 1.2 张量命名与实例数（实测）

| 模式 | 实例 | 说明 |
|---|---|---|
| `model.language_model.layers.N.mlp.experts.K.{gate,up,down}_proj` | 12 384 ×3 | **per-expert 独立 2-D 张量**（不是 Qwen 那种 fused 3-D），43 层 × 288 |
| `…layers.N.mlp.shared_experts.{gate,up,down}_proj` | 43 ×3 | 共享专家 |
| `…layers.{0,1,2}.mlp.{gate,up,down}_proj` | 3 ×3 | dense MLP |
| `…layers.N.mlp.gate` | 43 | **router**（+ `e_score_correction_bias`） |
| KDA 层（34 个）`…self_attn.{q,k,v}_proj`、`{q,k,v}_conv1d`、`f_a_proj/f_b_proj`、`g_a_proj/g_b_proj`、`b_proj`、`A_log`、`dt_bias` | 各 34 | 线性注意力件，**命名也在 `self_attn.` 下** |
| 稀疏 MLA 层（12 个含 MTP）`…self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,o_proj}` | 各 12 | 低秩 MLA |
| `…self_attn.indexer.{wk,weights_proj}` | — | DSA 打分 |
| `…layers.N.hc_attn_{base,fn,scale}` / `hc_ffn_{base,fn,scale}` | 各 45 | hyper-connection |
| `layers.45.{eh_proj, shared_head.*}` | 1 | MTP |
| `model.visual.blocks.N.*`、`merger.*`、`patch_embed.proj`、`downsample` | 336+ | 视觉塔 |
| `lm_head.weight` / `model.language_model.embed_tokens.weight` | 1 / 1 | `[154880, 4096]` 各 634 M |

### 1.3 参数量与占比（总 **321.342 B**；bf16 等价 642.7 GB；本地 FP8 落盘 328.3 GB，`du` 报 306 GiB）

| 模块 | 参数 | 占比 | bf16 体积 |
|---|---|---|---|
| **Routed experts（43 层 × 288）** | **311.673 B** | **96.99 %** | 623.3 GB |
| Attention（KDA 34 + 稀疏 MLA 11 + MTP） | 6.092 B | 1.90 % | 12.2 GB |
| Shared experts | 1.082 B | 0.34 % | 2.2 GB |
| `lm_head` | 0.634 B | 0.20 % | 1.3 GB |
| `embed_tokens` | 0.634 B | 0.20 % | 1.3 GB |
| Vision tower | 0.564 B | 0.18 % | 1.1 GB |
| Dense MLP（层 0-2） | 0.504 B | 0.16 % | 1.0 GB |
| Indexer（DSA 打分） | 0.090 B | 0.03 % | 0.2 GB |
| Hyper-connection（`hc_*`） | 0.035 B | 0.01 % | 0.1 GB |
| MTP（`eh_proj`/`shared_head`） | 0.034 B | 0.01 % | 0.1 GB |
| Norms / 其它 | ~0 B | 0.00 % | — |

- **每 token 激活 ≈ 17 B**（pinned recipe 记 18B，同口径）：dense 部分 ~7.5 B + `8/288 × 311.7 B` 的专家激活。
- 量化本质就是压那 96.99% 的 experts：`experts/6` + 其余 ≈ **183 GB** 就是现实下限。
- 与 Flash-Next 的关键差别：**这里没有一个"关不掉的 102 GB 巨型 embedding 表"**，所以压缩比能到 3.5×（vs Flash-Next 被 PLE 封顶在 2×）。

---

## 2. HF 参考产物反解

`quantization_config.json` 实测：

| 项 | 值 |
|---|---|
| `format` / `quant_method` | `mixed-precision` / `compressed-tensors` |
| `group_0` | **W4A4**，`group_size=32`，`scale_dtype=torch.uint8`(E8M0)，动态激活；targets = **37 152** 个 `layers.*.mlp.experts.*.{gate,up,down}_proj`（= 43×288×3） |
| `group_1` | **W8A8**，g32，targets = `["Linear"]`（其余全部） |
| `ignore` | 676 项：`self_attn.*`（o_proj 46、KDA 的 q/k/v/conv1d/b/f/g 各 34）、`mlp.gate` 43、`visual` 全部、`hc_*`、`norm`、`embed_tokens`、`lm_head`、MTP 件 |
| `kv_cache_scheme` | `null` |
| README 报告的分数 | BF16 avg **0.8399**（gsm8k .9735 / mmlu .8666 / piqa .8292 / hellaswag .6903）→ MXFP4 avg **0.8366** = **99.61%** |

README 里的生成命令（G0 就是它的复现）：

```bash
auto-round --model_name zai-org/GLM-5.3-Flash-BF16 --model_free --scheme MXFP8 \
  --ignore_layers "lm_head,embed_tokens,visual,hc_,indexer,self_attn,mlp.gate.,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d" \
  --layer_config "{mlp.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor --output_dir ./GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound
```

**复现校验（`audit_precision_plan.py` 的预测，实跑有偏差）**：同一 ignore + layer_config：

```
bits=4    311.654 B (97.0%)  层数 37152      ← 与 group_0.targets 数量完全一致
bits=8      1.535 B ( 0.5%)  层数 138        ← dense MLP 9 + shared_experts 129
bits=16     8.070 B ( 2.5%)  层数 572
产物 ≈183.46 GB；vLLM MX kernel 约束校验 OK
```

> ⚠ 这 138 层是**审计脚本的预测**，它不认识 auto-round 的 predefined ignore（§4.1）。**补丁前实跑只有 135**：层 0 的 `mlp.{gate,up,down}_proj` 被裸 `layers.0.mlp` 静默保成 BF16，产物 ≈183.6 GB（+0.15 GB）。**§4.2 的本地补丁已修好，回到 138**（≈183.33 GB）。4-bit 的 37 152 层自始至终不受影响，experts 主体一直与参考产物逐层吻合。
> 根因是审计脚本没有把 `get_predefined_ignore_layers_from_config(config)` 并进 ignore 集合 —— 修了它，这类偏差才能在前置审计里看见。

---

## 3. 与官方 FP8 范围的对齐（G0+ / G1+ 可选口径）

官方 FP8（本地这份）量化了 **37 338** 个张量 = routed experts 37 152 + **186** 个其它。下表是 **safetensors header 逐张量实测**（`du` 306 GiB = 328.3 GB；`quantization_config` 为 `{quant_method: fp8, fmt: e4m3, activation_scheme: dynamic, weight_block_size: [128,128], modules_to_not_convert: 1509 条}`）：

| 官方 FP8 额外覆盖 | 层数 | G0 的处理 |
|---|---|---|
| `mlp.shared_experts.{gate,up,down}` | 129 | ✅ 同样量化（落在 group_1 的 MXFP8） |
| dense MLP（层 0-2） | 9 | ⚠️ **实跑只量到层 1、2 共 6 个**；层 0 的 3 个被 auto-round predefined `layers.0.mlp` 静默吃掉（§4.1） |
| 稀疏 MLA 的 `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`o_proj` | 48 | ❌ G0 保 BF16（被 `self_attn` 覆盖）；**plus 口径也保 BF16**，因为 auto-round 硬塞裸 `self_attn`（§4.1） |

官方在 `self_attn.` 下的切分（12 个稀疏 MLA 层 = 3,7,…,43 **+ MTP 层 45**；34 个 KDA 层）：

| 模块 | 官方 dtype | 个数 |
|---|---|---|
| `self_attn.{q_a,q_b,kv_a_proj_with_mqa,o}_proj` | **F8_E4M3** | 12 × 4 = 48 |
| `self_attn.kv_b_proj` | BF16（官方也没量） | 12 |
| `self_attn.indexer.*`、`{q_a,kv_a}_layernorm` | BF16 | 96 |
| KDA 层 `self_attn.{q,k,v,o}_proj`、`*_conv1d`、`{b,f_a,f_b,g_a,g_b}_proj`、`o_norm` | BF16 | 34 × 各 |

⇒ **更正本文旧版 §6.5 的一句话**："MLA 的 `o_proj` 是明文" 不成立 —— 12 个稀疏 MLA 层的 `o_proj` **是 FP8**，明文的只有 KDA 那 34 个（`modules_to_not_convert` 里 `o_proj` 恰好 34 条、`kv_b_proj` 12 条、`q_a/q_b/kv_a_proj_with_mqa` 0 条，与 header 完全自洽）。

差异不只是那 48 个投影，而是 **51 层 / 1.359 B 参数**（48 + 层 0 dense MLP 3 个；转 MXFP8 只省 **1.32 GB** = 1.359 B × 0.969 B/参数，其中那 48 个占 1.17 GB）。而源码层面这 48 个投影**在 vLLM 里根本不会有量化权重通路**：`common/model.py:330` 构造 `Glm5NextMLAAttention` 时显式传 `quant_config=None`，模型侧永远是 BF16；vLLM 只在加载期把官方的 block-FP8（`float8_e4m3fn` + `weight_scale_inv`）**反量化回 BF16**（`_FP8_ATTN_PROJS`，`common/model.py:1238-1246`）。我们的 CT/MXFP8 也会撞进这条反量化通路，但因为 scale 是 `[N, K/32]` 而不是 `[N/bs, K/bs]`，`_dequant_fp8_block` 把 dim=0 也 `repeat_interleave` 了 32 倍 → **加载期直接 shape 崩**（实测日志与逐张量对照见 **§6.5.1**）。
**这条与 INC 无关**：我们导 `llm_compressor`，走的是 vLLM 的 compressed-tensors 通路，先例里 `KeyError qweight` 那个 INC 症状不适用，不该拿它当依据（我先前那么写过，已更正）。
所以 G0+/G1+ 已做成可选口径 `MODE=mxfp4_mixed_plus|mxfp8_plus`，**以实跑为准**：实跑结论是 plus 与 aligned 逐层相同（§4.1），连"能不能加载"这一步都没走到。

⇒ **G0 保持参考产物的范围不是"保守"，是唯一可行**；官方 FP8 多覆盖的这 1.2 GB 拿不到，也不用想要。

另外注意：`--layer_config` 只给 `bits/data_type` 时，llm_compressor 会按有效 weight bits 重建整组 preset → 得到的是 **W4A4**（激活也 4 bit），与参考产物一致，**别补 `act_bits:8`**。

### 3.1 与官方 FP8 的量化集合严格 diff（实跑 + header 复核）

复现脚本 `_diff_official_vs_ours.py`（只读 header 与 auto-round 的 `_PatternMatcher`，不加载权重）。口径 **G1 / MODE=mxfp8**（纯 8-bit，与官方同位宽才好比范围）：

| 集合 | 层数 | 参数量 |
|---|---|---|
| 官方 block-FP8 量化集 | 37 338 | 314.397 B |
| 我们实跑（aligned 与 plus **完全相同**） | 37 287 | 313.038 B |
| 官方量化、我们保 BF16 | **51** | **1.359 B** |
| 我们量化、官方保 BF16 | **0** | 0 |

明细：`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`o_proj` 各 12（=48）+ `layers.0.mlp.{gate,up,down}_proj`（=3）。**我们的量化集合是官方的真子集**，没有任何一层是官方明文而我们量了的。

体积（header 实算，GB = 1e9；`du` 报的是 GiB，别混）：

| 产物 | 体积 |
|---|---|
| BF16 源 | 642.6 GB（`du` 605 GiB） |
| 官方 block-FP8 | **328.3 GB**（`du` 306 GiB） |
| 我们 MXFP8，范围与官方逐层对齐（37 338 层） | **338.1 GB** |
| 我们 MXFP8，实跑（37 287 层） | **339.4 GB** |

⇒ 结论要换个说法：**"MXFP8 不如官方 FP8" 与那 51 层无关**。就算把 51 层全补回来，MXFP8 仍大 9.8 GB，因为 scale 粒度差两个数量级 —— MX 是 1 B E8M0 / 32 权重（+3.1 %），官方是 1 个 F32 / 128×128 权重（+0.024 %）。旧版写的"339 vs 306 = 亏 33 GB"是拿十进制 GB 比 GiB，真实差距 ≈ **+3 %**。

**产物级复核**（`GLM-5.3-Flash-MXFP8-Plus-AutoRound`，已跑完，BF16 源）：

| 核对项 | 实测 |
|---|---|
| F8_E4M3 权重张量 / U8 `weight_scale` | **37 287 / 37 287** —— 与 §3.1 预测的量化层数一字不差 |
| `self_attn.*` 下被量化的层 | **0 个**（plus 口径确认空转） |
| `ignore` 里 `layers.0.mlp.gate_proj` / `layers.1.mlp.gate_proj` | **True / False** —— `first_k_dense_replace` 回落成 1 的直接证据 |
| `ignore` 里 `layers.3.self_attn.q_a_proj` / `o_proj` | **True / True** —— 稀疏 MLA 投影全被裸 `self_attn` 吃掉 |
| `config.json` → group_0 | `targets=["Linear"]`, `num_bits=8`, `format=mxfp8-quantized`, `scale_dtype=torch.uint8`, `group_size=32` |
| 文件字节合计 | **339.4 GB**（= header 累加，与预测一致）；注意该 NFS 卷 `du` 报 **323 GiB**，多出的 ~7 GiB 是块分配开销 |

> ⚠ 比体积一律用**文件字节 / header 累加**，别用 `du`：官方 FP8 的 306 是 GiB 且与 header 吻合（328.3 GB），而我们的产物在这个卷上 `du` 会虚高 2 %。

⇒ **这 48 个投影此前量不上，跟矩阵形状无关**。实测四个投影 `q_a [1536,4096]` / `q_b [16384,1536]` / `kv_a [512,4096]` / `o_proj [4096,16384]`，N、K 全是 32 的倍数且 N ≥ 128，vLLM MXFP8 kernel 的两条约束（`K%32==0`、`N>=128`）全过；GLM 里唯一被**矩阵形状**卡住的注意力件是 KDA 的 `b_proj`（N=64）。此前的两层卡点：① auto-round 的 predefined 裸 `self_attn`（§4.1，量化阶段，已由 §4.2 补丁解除）；② vLLM 建模期 `quant_config=None` + 加载期反量化（§6.5）。
> ⚠ 补丁后实跑推翻了 ②的**失败方式**：不是"CT 的 `weight_scale` 进不了兼容通路"，而是**进得去、然后被 block-FP8 的反量化算式按错误布局解释**（`[N, K/32]` 当成 `[N/bs, K/bs]`）→ 加载期 shape 崩。所以严格说："矩阵形状无关，但 **scale 布局有关**"，详见 **§6.5.1**。

### 3.2 逐模块对比：我们的 MXFP8(重跑) vs 官方 block-FP8

对象：`GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned`（§4.2 补丁之后重跑，120 分片）vs `GLM-5.3-Flash`（官方 block-FP8，62 分片）。

**方法**：直接扫两侧**全部 `*.safetensors` 分片的 JSON header**（不加载权重），按模块类别聚合张量数 / 参数量 / 落盘字节。两侧张量总数都是 **76 108**。量化判定 = 该层的 `*.weight` 是否存成 `F8_E4M3`。复现：

```bash
/models/miniforge3/envs/muse_verify/bin/python compare_quant_scopes.py \
    GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned GLM-5.3-Flash
/models/miniforge3/envs/muse_verify/bin/python verify_glm_scope.py GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned
```

dtype 直方图（两侧权重侧完全同形）：

```
官方  {'F32': 37629, 'BF16': 1141, 'F8_E4M3': 37338}      # F32 主要是 weight_scale_inv（128×128 一块）
我们  {'F32': 291, 'BF16': 1141, 'F8_E4M3': 37338, 'U8': 37338}   # U8 = weight_scale（E8M0，1×32 一组）
```

**表 1 · 逐模块量化状态**（`★` = 官方量化、而补丁前的 ignore 口径会漏掉的；`└` = `self_attn.` 下的细分）

| 模块 / 类别 | 权重张量 | 参数量 B | 官方量化 | 官方权重 dtype | 我们量化 | 我们权重 dtype |
|---|---|---|---|---|---|---|
| routed experts (MoE ×288) | 37 152 | 311.654 | **37152/37152** | F8_E4M3 | **37152/37152** | F8_E4M3 |
| `└` KDA q/k/v_proj | 102 | 3.423 | 0/102 | BF16 | 0/102 | BF16 |
| `└` KDA o_proj | 34 | 1.141 | 0/34 | BF16 | 0/34 | BF16 |
| shared_experts | 129 | 1.082 | **129/129** | F8_E4M3 | **129/129** | F8_E4M3 |
| `★└` 稀疏MLA o_proj | 12 | 0.805 | **12/12** | F8_E4M3 | **12/12** | F8_E4M3 |
| embed_tokens | 1 | 0.634 | 0/1 | BF16 | 0/1 | BF16 |
| lm_head | 1 | 0.634 | 0/1 | BF16 | 0/1 | BF16 |
| 视觉塔 visual | 347 | 0.564 | 0/347 | BF16 | 0/347 | BF16 |
| dense MLP (层 0-2) | 9 | 0.453 | **9/9** | F8_E4M3 | **9/9** | F8_E4M3 |
| `★└` 稀疏MLA q_a/q_b/kv_a_proj | 36 | 0.403 | **36/36** | F8_E4M3 | **36/36** | F8_E4M3 |
| `└` 稀疏MLA kv_b_proj | 12 | 0.201 | 0/12 | BF16 | 0/12 | BF16 |
| `└` KDA b/f/g 小投影 | 170 | 0.116 | 0/170 | BF16 | 0/170 | BF16 |
| `└` DSA indexer | 84 | 0.090 | 0/84 | BF16 | 0/84 | BF16 |
| MoE router `mlp.gate` | 43 | 0.051 | 0/43 | BF16 | 0/43 | BF16 |
| hyper-connection `hc_*` | 270 | 0.035 | 0/270 | F32×180, BF16×90 | 0/270 | F32×180, BF16×90 |
| MTP (`eh_proj`/`shared_head`/`enorm`/`hnorm`) | 4 | 0.034 | 0/4 | BF16 | 0/4 | BF16 |
| `└` KDA `*_conv1d` | 102 | 0.003 | 0/102 | BF16 | 0/102 | BF16 |
| RMSNorm / 其它 1-D | 93 | ~0 | 0/93 | BF16 | 0/93 | BF16 |
| `└` MLA layernorm | 24 | ~0 | 0/24 | BF16 | 0/24 | BF16 |
| `└` KDA o_norm | 34 | ~0 | 0/34 | BF16 | 0/34 | BF16 |
| `└` KDA `A_log` / `dt_bias` | 34 / 34 | ~0 | 0 | F32 | 0 | F32 |
| `└` router `e_score_correction_bias` | 43 | ~0 | 0/43 | F32 | 0/43 | F32 |
| **合计** | **38 770** | **321.323** | **37 338** | | **37 338** | |

**表 2 · 体积归集**（GB = 1e9，header 实算，含 scale 与明文；按类别）

| 模块 / 类别 | 官方 block-FP8 | 我们 MXFP8 | Δ |
|---|---|---|---|
| routed experts (MoE ×288) | 311.730 | **321.393** | **+9.663** |
| `└` KDA q/k/v_proj | 6.845 | 6.845 | 0 |
| `└` KDA o_proj | 2.282 | 2.282 | 0 |
| shared_experts | 1.082 | 1.116 | +0.034 |
| `★└` 稀疏MLA o_proj | 0.806 | 0.830 | +0.025 |
| embed_tokens | 1.269 | 1.269 | 0 |
| lm_head | 1.269 | 1.269 | 0 |
| 视觉塔 visual | 1.127 | 1.127 | 0 |
| dense MLP (层 0-2) | 0.453 | 0.467 | +0.014 |
| `★└` 稀疏MLA q_a/q_b/kv_a_proj | 0.403 | 0.415 | +0.012 |
| `└` 稀疏MLA kv_b_proj | 0.403 | 0.403 | 0 |
| `└` KDA b/f/g 小投影 | 0.232 | 0.232 | 0 |
| `└` DSA indexer | 0.179 | 0.179 | 0 |
| MoE router `mlp.gate` | 0.101 | 0.101 | 0 |
| hyper-connection `hc_*` | 0.071 | 0.071 | 0 |
| MTP 件 | 0.067 | 0.067 | 0 |
| `└` KDA `*_conv1d` | 0.007 | 0.007 | 0 |
| **合计** | **328.33 GB** | **338.07 GB** | **+9.75 GB（+2.97 %）** |
| 其中 **scale 开销** | 0.077（37 338 个 F32，128×128 一块） | **9.825**（37 338 个 U8，1×32 一组） | **+9.748** |

**表 3 · 量化集合差异（逐层名比对）**

```
官方量化而我们没量化: 0
我们量化而官方没量化: 0
=> ✅ 两者量化集合逐层完全一致（37 338 层）
```

从这张表能读出三件此前只能推测的事：

1. **范围问题已经彻底关闭**：37 338 层、双向 diff 为 0，`hc_*`/indexer/router/visual/embed/lm_head/MTP 这些明文集合也与官方逐条一致。产物 `quantization_config.ignore` 从补丁前的 679 条降到 **628 条（−51）**，正好等于 §3.1 那 51 层差集 —— 三个独立口径互相咬合。
2. **+9.75 GB 的差距 100 % 来自 scale 粒度，与量化了哪些层无关**。两侧**权重字节完全相同**（都是 314.397 GB 的 `F8_E4M3`）；差的全在 scale：官方 1 个 F32 / 128×128 = 0.077 GB，我们 1 个 U8 / 1×32 = 9.825 GB，而 9.825 = 314.397 / 32 ✓。换句话说 **MXFP8 相对 block-FP8 的"格式税"是 3.1 %**，这 3 % 花不掉，除非换成 block-scale 语义（那就不是 MXFP8 了，vLLM 的 `mxfp8-quantized` 走的是 1×32 E8M0）。
3. **注意力里我们和官方"没量"的部分是同一批**：KDA 的 102 个 q/k/v_proj + 34 个 o_proj（合计 4.56 GB 明文）、`kv_b_proj` 12 个（0.40 GB）、indexer 84 个 —— 两侧字节数逐行相等。所以任何"再压注意力"的想法都要先回答 §6.5 那个部署通路问题，而不是再调 ignore 清单。



---

## 4. ignore 清单逐项依据

ignore 清单**不写在脚本里**，由 `gen_glm_ignore.py <model_dir> --profile aligned|plus` 按产物 config 现算：

```
# aligned（= HF 参考产物口径）
lm_head,embed_tokens,visual,hc_,indexer,mlp.gate.,eh_proj,enorm,hnorm,shared_head,norm,A_log,
dt_bias,self_attn.b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d,self_attn

# plus（G0+/G1+）：去掉 self_attn，改为逐名保留 KDA 专属 + kv_b_proj，
# 并按层号锚定 KDA 的 o_proj（KDA 与稀疏MLA 的输出投影同名，子串分不开）
...,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.kv_b_proj,
(language_model.)?layers.(0|1|2|4|...|44).self_attn.o_proj
```
层号列表取自 `config.text_config.linear_attn_config.kda_layers`。模式里**不能含逗号**（`--ignore_layers` 以逗号分隔）。
★ **锚定正则必须写裸点，不能自己加 `\.`**：`to_standard_regex` 对含元字符的模式走"只转义裸点"分支，
已转义的 `\.` 会被叠成 `\\.`（正则语义=字面反斜杠+任意字符），整条模式**静默失配**。实测对比：

```
输入 r"(language_model\.)?layers\.(0|1|2|4)\.self_attn\.o_proj"
  -> .*(language_model\\.)?layers\\.(0|1|2|4)\\.self_attn\\.o_proj.*   # 层0/4 全 False（失效）
输入 r"(language_model.)?layers.(0|1|2|4).self_attn.o_proj"
  -> .*(language_model\.)?layers\.(0|1|2|4)\.self_attn\.o_proj.*       # 层0/4 True、层3/45 False（正确）
```

| 模式 | 为什么保 BF16 |
|---|---|
| `mlp.gate.` | **router**（288 选 8，`moe_router_dtype=float32`）。**必须带尾点**：裸 `mlp.gate` 会子串命中 dense 层的 `mlp.gate_proj`（层 0-2 共 3 个），把 dense MLP 静默漏掉。尾点写法在 model-free 里被特化成 `(?:\.|$)` 锚定（`model_free_utils.py:500-511`），确实有效（实测 `mlp.gate_proj` False、`mlp.gate` True） |
| `self_attn`（仅 aligned） | GLM 把 **KDA(34) 和稀疏 MLA(12) 的投影全放在 `self_attn.` 下**，一把覆盖。plus 口径下拆成逐名 + 层号锚定（见上）。注意 auto-round 原先也会塞一条裸 `self_attn`（§4.1），**已被 §4.2 的本地补丁换成"KDA 精确名 + 34 条层号锚定 o_proj"**；在没打补丁的 auto-round 上，plus 拆得再细也白拆 |
| `self_attn.b_proj`（**不是裸 `b_proj`**） | 裸串会子串误伤 `q_b_proj` / `kv_b_proj`——实测在 plus 口径下让 `q_b_proj` 静默少量化 12 层，正是本文 §4 一直警告的那类坑。另：`b_proj` 是全模型唯一 N<128（N=64）的注意力投影，本来就量不了 |
| `indexer` | DSA 打分件（`wk`、`weights_proj`），决定选哪些 KV block，量化敏感 |
| `hc_` | hyper-connection（`hc_attn_base/fn/scale`、`hc_ffn_*`）；含 `[hc_mult×hidden]` 这类小形状 |
| `eh_proj` / `enorm` / `hnorm` / `shared_head` | MTP 结构件；评测不开 spec decode，先保明文 |
| `visual` | 视觉塔；文本评测不走，且 mm 路径无对应量化 kernel |
| `norm` / `A_log` / `dt_bias` / `conv1d` | 1-D 参数与 conv（auto-round 预定义 skip 也会跳过 `conv`/`embed`，写出来是为了与参考产物逐项对齐、便于核对） |
| `lm_head` / `embed_tokens` | 输出层与输入表（各 634 M） |

### 4.1 ★ auto-round 给 `glm5_next` 硬编码的 predefined ignore（会盖掉你的一切设计）

量化的**第一步日志**就有它，别忽略这行：

```
INFO model_free.py L794: Using predefined ignore_layers from config: indexer, layers.0.mlp, self_attn, weights_proj
```

注册处 `auto_round/special_model_handler.py:1086-1096`：

```python
register_ignore_layers(
    matchers=[ModelTypeMatcher(r"glm5_next", mode="full")],
    ignore_layers=[get_glm_flash_ignore_layers, "weights_proj", "indexer", "self_attn"],
)
```

三条必须知道的性质（全部源码 + 实测核实）：

1. **并集、无开关**（补丁前的原始行为）。model-free 走 `compressors/model_free.py:791-795` 的 `self.ignore_patterns.extend(predefined)`；tuning 通路 `compressors/base.py:1005-1011` 同样 `self.ignore_layers += ...`。`--ignore_layers` 只能**加**不能**减**，CLI 上没有任何 flag 能关掉它 —— 所以要么改注册表（§4.2），要么走下面那条 layer_config 精确点名的绕行通道。
2. **裸 `self_attn` 是子串匹配**（无元字符 → `.*self_attn.*`），于是 **plus 口径整体空转**：实测 `MODE=mxfp8` 下 aligned 与 plus 的量化集**逐层相同**（都是 37 287 层），48 个稀疏 MLA 投影一个都没量上。
3. **`layers.0.mlp` 是个 bug 级回落**。`get_glm_flash_ignore_layers`（`special_model_handler.py:1047-1052`）读 `model.config.first_k_dense_replace`，而 model-free 构造的伪 config 是 `SimpleNamespace(**config.json)`（**顶层**，`model_free.py:257`）；GLM 的这个字段在 `text_config` 里（=3），顶层没有 → 回落 `num_dense_layer = 1` → **只保层 0 的 dense MLP，层 1、2 照常量化**。

**不改库时的绕行通道**：`should_ignore` 里"**layer_config 的 key 与完整层名精确相等**时优先于 ignore"（`model_free_utils.py:513-524`）。注意 `mlp.experts` 这类**子串 key 绕不过**，必须是 `model.language_model.layers.3.self_attn.q_a_proj` 这种全名。实测把 51 个全名塞进 `--layer_config` ⇒ 量化集变成 37 338 层，与官方 FP8 **双向 diff 为 0**。

对本文各方案的实际影响（**补丁前**的实测）：

| 口径 | 预测（旧版审计） | 实跑 | 差额来源 |
|---|---|---|---|
| G0 8-bit 层数 | 138 | **135** | `layers.0.mlp` 吃掉 3 |
| G1/G1+ 量化层数 | 37 290 / 37 338 | **37 287 / 37 287** | 同上 + 裸 `self_attn` 吃掉 48 |

⇒ **`audit_precision_plan.py` 目前不合并 predefined**（它只吃 `--ignore_layers`），所以它的层数预测与实跑必然差 3~51 层。要修就在审计里 `from auto_round.compressors.model_free import get_predefined_ignore_layers_from_config` 并同样做并集，这类静默偏差就能在前置审计暴露。

### 4.2 ★ 本机补丁：把 `glm5_next` 的 predefined 改成与官方 checkpoint 精确对齐（2026-09-21）

改的是本地 editable 安装的那份源码 `/models/lkk/auto-round`（`muse_verify` 里 `pip -e` 装的正是这个目录，**改完立即生效，不用重装**）：

- 新增 `_config_field(config, key)`：兼容三种形状 —— 真 `PreTrainedConfig`、`SimpleNamespace` 包一层、以及 `text_config` 是**普通 dict** 的多模态 config（这正是上面第 3 条踩的坑）。
- 新增 `get_glm5_next_attn_ignore_layers(model)`：从 `text_config.linear_attn_config.kda_layers`（回退 `layer_types`）生成 **34 条 `layers.{i}.self_attn.o_proj`** 精确名 + 9 个 KDA 专属投影 + `kv_b_proj` + `o_norm`。**故意不加反斜杠**：这些名字没有正则元字符，`to_standard_regex` 会转义点号并包成子串匹配，而 `.self_attn.o_proj` 这个后缀保证 `layers.4` 不会误配 `layers.44`。**拿不到层布局时回退成裸 `self_attn`** —— 宁可少量，绝不静默多量。
- `glm5_next` 的注册块里移除裸 `self_attn` 与 `get_glm_flash_ignore_layers`（官方量化了全部 3 层 dense MLP）；**`glm_moe_dsa` / `Glm4MoeLite` 那条规则原样不动**。

改后 predefined 实际下发（`compress_layer_names` 压缩显示）：

```
indexer, layers.[0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44].self_attn.o_proj,
self_attn.b_proj, self_attn.f_a_proj, self_attn.f_b_proj, self_attn.g_a_proj, self_attn.g_b_proj,
self_attn.k_proj, self_attn.kv_b_proj, self_attn.o_norm, self_attn.q_proj, self_attn.v_proj, weights_proj
```

四个 MODE 的结果（`_diff_official_vs_ours.py` 同口径，header 实算体积）：

| MODE | 量化层数（4bit/8bit） | 与官方 FP8 集合 diff | 体积 |
|---|---|---|---|
| `mxfp4_mixed` (G0) | 37 290（37 152 / **138**） | 少 48 / 多 0 | ≈183.33 GB |
| **`mxfp4_mixed_plus` (G0+)** | 37 338（37 152 / **186**） | **少 0 / 多 0 ✅** | ≈182.16 GB |
| `mxfp8` (G1) | 37 290（0 / 37 290） | 少 48 / 多 0 | ≈339.16 GB |
| **`mxfp8_plus` (G1+)** | 37 338（0 / 37 338） | **少 0 / 多 0 ✅** | **338.07 GB**（已实跑，逐 header 复核见 §3.2） |

> 上表除 `mxfp8_plus` 是**产物实测**外，其余三个是 `_diff_official_vs_ours.py` 的**范围级预测**（同一套 `_PatternMatcher` + header 字节公式，实测偏差 < 0.1 GB）。

G0 的 8-bit 也从 135 回到 **138**，与 HF 参考产物逐项吻合（4-bit 37 152 层自始至终没受影响）。

测试与检查：`test/unit/common/models/test_special_model_handler.py` 里改写 `test_glm5_next_matcher` 并加 3 个用例（精确对齐 / 仅 `layer_types` / 无布局回退）。本机各 env 都没有 pytest，用 `sys.modules['pytest']` 打桩直接跑该类，**13 passed / 0 failed**；`black --check` 与 `isort --check` 干净（line-length 120）；**`pre-commit` 未安装，完整 hook 链没跑** —— 提交前建议补一次 `pre-commit run --files auto_round/special_model_handler.py test/unit/common/models/test_special_model_handler.py`。

两条必须记住的边界：

- ⚠ **只对齐了"量化范围"，部署结论反而更硬了**：补丁后重跑的 `mxfp8_plus` 已**实测加载失败**，原因是 vLLM 的注意力反量化把 MXFP8 的 `[N, K/32]` scale 当成 block-FP8 的 `[N/bs, K/bs]` 网格（**§6.5.1**，含逐张量对照与修法评估：修好也不省显存）。所以**要出货用 `MODE=mxfp4_mixed`（G0）**；`*_plus` 到此为止，只是范围对齐的实验品。
- ⚠ 这是**本地改动**，`git pull` / 重新 clone auto-round 会被覆盖。补丁点集中在三处：`_GLM5_NEXT_ATTN_IGNORE`、`get_glm5_next_attn_ignore_layers`、以及 `glm5_next` 那个 `register_ignore_layers` 块。上游化时把"官方 checkpoint 的 self_attn 切分"作为依据附进 PR 说明（`modules_to_not_convert` 34×`o_proj` / 12×`kv_b_proj` / `q_a`-`q_b`-`kv_a` 0 条，与 header 互证）。


**三条清单互抄警告**：

| | Qwen3.8-27B（dense） | Qwen3.8-Flash-Next（qwen4_exp） | GLM-5.3-Flash（glm5_next） |
|---|---|---|---|
| `mlp.gate` | **禁写**（误伤 `gate_proj`） | 必写（router） | 必写，且**必须带尾点 `mlp.gate.`**（有 dense 层） |
| 小投影名 | `linear_attn.in_proj_a/b` | 同左 | `self_attn.{b,f_a,f_b,g_a,g_b}_proj` |
| experts 存储 | 无 | **fused 3-D** `[512,1280,2560]` | **per-expert 2-D**，43×288×3 |
| 压缩上限 | ~2.8× | ~2.0×（PLE 表封顶） | **~3.5×**（experts 占 97%） |
| MXFP8-MoE swizzle bug | 不涉及（无 MoE） | **涉及**（cols=20/TP，TP2/4 触发） | **不涉及**（experts 是 MXFP4；即便 MXFP8，cols=64/TP 恒为 4 的倍数） |

---

## 5. 量化命令

### 5.1 G0 · MXFP4-Mixed（推荐）

```bash
export PATH=/models/miniforge3/envs/muse_verify/bin:$PATH
export HF_HOME=/models/huggingface
export AR_MODEL_FREE_SHARD_PARALLELISM=1

auto-round \
  --model_name /models/GLM-5.3-Flash-BF16 \
  --model_free \
  --scheme MXFP8 \
  --layer_config '{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}' \
  --ignore_layers lm_head,embed_tokens,visual,hc_,indexer,self_attn,mlp.gate.,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d \
  --format llm_compressor \
  --device_map auto \
  --output_dir /models/lkk/cri_models/GLM-5.3-Flash-MXFP4-Mixed-AutoRound
```

### 5.2 源准备（BF16，本地缺）

```bash
df -h /models                                   # 现有 2.0 TB 空闲，装得下 642 GB
hf download zai-org/GLM-5.3-Flash-BF16 --local-dir /models/GLM-5.3-Flash-BF16
```
用本地 FP8 源属于**二次量化**（auto-round 的 model_free 有 `weight_scale_inv` 反量化路径，技术上能跑），
精度不可对标参考产物；脚本里 `SOURCE=fp8` 会明确警告并等 5 秒。

### 5.3 一键脚本（含源检查 + 前置审计 + 导出核对）

```bash
cd /models/lkk/cri_models
DRY_RUN=1 ./quant_GLM-5.3-Flash-mxfp.sh                        # 只看审计表+命令
./quant_GLM-5.3-Flash-mxfp.sh                                  # G0  aligned（需 BF16 源）
MODE=mxfp4_mixed_plus ./quant_GLM-5.3-Flash-mxfp.sh            # G0+ 含稀疏MLA，预期加载失败待验
MODE=mxfp8 ./quant_GLM-5.3-Flash-mxfp.sh                       # G1（脚本警告"比官方 FP8 还大"）
MODE=mxfp8_plus ./quant_GLM-5.3-Flash-mxfp.sh                  # G1+
SOURCE=fp8 DRY_RUN=1 ./quant_GLM-5.3-Flash-mxfp.sh             # 用本地 FP8 源做审计预演

# 生成器可单独查（想手改 ignore 时先看它算出什么）
python3 gen_glm_ignore.py ./GLM-5.3-Flash --profile plus

# plus 产物只验加载，别急著跑全量：
ENVBIN=/models/lkk/venv_glm53/bin SMOKE_ONLY=1 TP=4 ./eval_GLM-5.3-Flash-quant.sh GLM-5.3-Flash-MXFP4-Mixed-Plus-AutoRound
```

---

## 6. vLLM 支持判定

### 6.1 本机：**不支持**（实测）

```
$ python -c "from vllm.model_executor.models.registry import ModelRegistry as R; \
             print('Glm5NextForConditionalGeneration' in R.get_supported_archs())"
False
```
- `/models/lkk/muse_support/vllm`（`0.28.1rc1.dev312+g41848caa6`）：`vllm/models/` 下只有 `qwen4_exp / deepseek_v4 / deepseek_v32 / kimi_k3 / minimax_m3 / hy_v4 / dots3_note / inkling / common`，**没有 glm5next**；全树 grep `glm5_next|Glm5Next` 零命中。
- 另外三个 venv（`.venv-muse-main`、`.venv-qwen35-9b`、`.venv-zimage`）的 registry 命中数也都是 0。
- `eval_GLM-5.3-Flash-quant.sh` 里有 registry 预检，环境不对会**干净地拒绝**而不是跑到一半崩。

### 6.2 上游（2026-09-21 逐项核实）

| 检查 | 结果 |
|---|---|
| registry 含 `Glm5Next` | **v0.29.0 → 0 命中**；**v0.30.0 → 4 命中**；main → 4 命中 |
| 模块路径 | `vllm.models.glm5next`（`registry.py:123` `Glm5NextForCausalLM`、`:429-432` `Glm5NextForConditionalGeneration`、`:693` `Glm5NextMTPModel`）——注意是 **glm5next**，不是 `glm5_next` |
| PyPI | 最新 **0.29.0**（GitHub releases 也只到 v0.29.0，2026-09-09）→ **pip 装不到含支持的版本** |
| v0.30.0 状态 | tag 已切（`9ed533eb`），但**无 GitHub release、无 PyPI 包、无 `vllm/vllm-openai:v0.30.0` 镜像** |
| 镜像 | `vllm/vllm-openai:nightly` 与 `cu129-nightly` 今天（09-21）刚构建 → 含 main 的支持；`vllm/vllm-openai:glm53-flash-x86_64-cu130` last_updated **2026-09-09T13:31Z**，是官方为这个模型出的 **cu130** 镜像（正对本机 B300） |
| 支持 PR | `vllm-project/vllm#53906` **已合并**：`merged_at=2026-09-03T16:40:36Z`，merge commit `98ed0856`，94 files / +17926 −265 |
| **为什么不能只 patch Python** | 该 PR 改了 `csrc/libtorch_stable/cache_kernels.cu`(+9/−4)：`gather_and_maybe_dequant_cache` 的 `head_dim` 白名单从 `{320,576}` 放宽到 `{320,512,576}`（**512 就是 GLM 的 MLA 隐头宽**）→ 已装好的 0.29.0 wheel 里这个 C++ 检查仍是 320/576，覆盖 .py 文件没用 |
| 最低可用 tag | **v0.29.1rc0**（探测 `vllm/models/glm5next/__init__.py`：v0.29.1rc0/v0.30.0/v0.30.0rc1/rc2 = 200；v0.29.0 及更早、v0.28.x = 404）；稳定 tag = **v0.30.0** |
| **v0.30.0 不是最新的** | `commits?path=vllm/models/glm5next`：v0.30.0 只到 9/15（8 条），main 有 18 条 —— **9/16~9/20 的 10 条不在 v0.30.0**，含 #57477（kpool tail 按 padded stride 寻址的 bugfix，9/20）、#57701（indexer decode workspace，省 3072 MiB，9/19）、#57327（cooperative top-k，9/18）→ **要跑就装 nightly(=main)，别锁 v0.30.0** |
| 旧记录（8-31）"必须用定制镜像" | **已过时**：支持已进 main 与 v0.29.1rc0+，专用镜像 `glm53-flash*` 全停在 09-09，已被 nightly 取代（见 §6.3） |

### 6.3 选择矩阵

| 路径 | 命令 | 现在能用？ | 备注 |
|---|---|---|---|
| pip release | `uv pip install vllm==0.29.0` | ❌ 0.29.0 不含 glm5next | 等 v0.30.0 上 PyPI；但那时**仍缺 9/16-9/20 的 10 条修复**，不如 nightly |
| **pip nightly wheel** | `uv pip install --torch-backend=cu130 --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130 "vllm>=0.29.1rc1.dev0"` | ✅ **本机已实测装成**（`/models/lkk/venv_glm53`：vllm 0.29.1rc1.dev471+g82daf9f57、registry Glm5Next=True、torch 2.13.0+cu130）。要手动补两个依赖 | 已验证 `https://wheels.vllm.ai/nightly/cu130/vllm/` 存在 `vllm-0.29.1rc1.dev470+g15859bb3a-…manylinux_2_28_x86_64.whl`（09-21 生成）；**必须另装 `deep_gemm`（vLLM fork ref `e1f418c2`）与 `flashinfer-cubin==0.6.18.post1`**，否则 DSA indexer 直接 RuntimeError |
| **官方镜像（推荐）** | `docker run --gpus all --network host -v /models:/models -e HF_HOME=/models/huggingface -it vllm/vllm-openai:nightly bash` | ✅ 一次给齐 | nightly = main 全量修复 + torch 2.13/cu130 + flashinfer 0.6.18.post1 + tilelang 0.1.12 + **DeepGEMM** + cubin。容器里只补 `auto-round`/`lm-eval`，**不要装 vllm** |
| 专用镜像（**已不推荐**） | `vllm/vllm-openai:glm53-flash-x86_64-cu130` | ⚠️ 能用但旧 | 全部 `glm53-flash*` tag 停在 **2026-09-09**，缺 9/10 FlashKDA prefill、9/14 NoPE prefill、9/19 workspace、9/20 kpool tail 修复 |
| 源码构建 | 从 main / v0.30.0 编译 | ⚠️ 可行但最贵 | 要编 CUDA 扩展 + 单独编 DeepGEMM；除非要锁版本 |

另外：`latest` 镜像与 `v0.29.0` 同 digest（`082ca6f0…`，09-09）→ **`latest` 不含支持**，别想当然。GLM 的 config 由 vLLM in-tree 提供（`transformers_utils/config.py:109` `glm5_next="Glm5NextConfig"`），**推理不需要 `--trust-remote-code`**；该架构也还没进 `docs/models/generative_models.md`（docs PR #55785 未合），所以别指望官方文档，以上述行号为准。

### 6.5 vLLM 侧"强制明文"的模块清单（这解释了 §4 为什么必须这么写）

这些不是保守选择，是 vLLM 代码里写死的；产物若把它们打包了，加载期直接挂：

| 模块 | 代码位置 | 行为 |
|---|---|---|
| 稀疏 MLA 投影 `q_a/q_b/kv_a_proj_with_mqa/o_proj` | `common/model.py:330` 构造 `Glm5NextMLAAttention` 时显式 `quant_config=None`，注释 "MLA projections are BF16 in checkpoint" | vLLM **只支持一种"注意力已量化"的落盘格式**：block-FP8。`common/model.py:1238-1246` 的 `_FP8_ATTN_PROJS` 在**加载期把它们反量化成 BF16**（模型侧永远是 BF16，量化只省文件不省显存）。**实测：CT 的 MXFP8 也会进这条通路然后崩** —— 见下面 §6.5.1 |
| KDA 全部投影 | `common/kda.py:189-194` 构造期临时把 `vllm_config.quant_config` 置 None（注释："KDA projections remain BF16 because fp8 checkpoints omit their scales"） | 同上，必须明文 |
| indexer | `common/attention.py:258-265`、`common/model.py:1169-1196`：`fp8 wk` 加载时反量化回 BF16 以与 `weights_proj` 融合；indexer 的 K cache 只接受 `fp8`/`mxfp4`（`v1/attention/backends/mla/indexer.py:60-69`，`mxfp4` 还要 Blackwell DC → sm103 ✓） | 权重必须明文 |
| hyper-connection `hc_*` | `layers/mhc.py:667,702`：是 `nn.Parameter` + scratch 硬编码 bf16，**完全不经 quant method** | 必须明文 |
| `visual` 塔 | `common/model.py:1096-1105` 显式 `quant_config=None`（注释：继承全局 fp8 会把塔量化成 NaN image features） | 必须明文 |
| MTP `eh_proj` / `enorm` / `hnorm` / `shared_head.norm` | `common/mtp.py:49` 是普通 `nn.Linear`（不接 quant_config）；官方 ckpt 的 `modules_to_not_convert` 正是这么列的 | 必须明文 |

⇒ **所以 MXFP4/MXFP8 只能落在 `mlp.experts.*`（+ 可选 dense MLP / shared_experts 的 gate/up/down）**，也就是 §5 的 G0 形状。

> 顺带更正旧记录的两处说法：这些"必须明文"**不是 vLLM 的限制失误**，而是按官方 checkpoint 的形状刻意写死的——官方 `zai-org/GLM-5.3-Flash` 的 `quantization_config` 是 `{quant_method: fp8, fmt: e4m3, activation_scheme: dynamic, weight_block_size: [128,128], modules_to_not_convert: 1509 条}`。
> **但旧版写"`o_proj` 与全部 `visual` 是明文"是错的**（header 实测，见 §3）：12 个**稀疏 MLA 层**（含 MTP 层 45）的 `o_proj` **是 F8_E4M3**，与 `q_a/q_b/kv_a_proj_with_mqa` 一起走那条 block-FP8 反量化通路；明文的是 **34 个 KDA 层的 `o_proj`** 和全部 `kv_b_proj` / `visual`。`modules_to_not_convert` 里 `o_proj` 恰好 34 条、`kv_b_proj` 12 条、`q_a`/`q_b`/`kv_a_proj_with_mqa` 0 条 —— 三个来源互证一致。

#### 6.5.1 实测：MXFP8 的注意力投影不是"被拒绝"，是"把 vLLM 的反量化算式撑爆"

`GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned`（§4.2 补丁后重跑，量化集与官方逐层一致）在 TP=4 加载即失败，日志 `test_mxfp8_aligned.log`：

```
vllm/models/glm5next/common/model.py:831   if _try_load_fp8_attn_proj(
vllm/models/glm5next/common/model.py:1295     weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
vllm/models/glm5next/common/model.py:1231     out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
RuntimeError: The size of tensor a (512) must match the size of tensor b (16384) at non-singleton dimension 0
Loading safetensors checkpoint shards:   5% Completed | 6/120
```

**本文此前"`weight_scale` 不叫 `weight_scale_inv`，所以兼容层不会触发"的推断是错的**，源码里明写了要同时接受两种名字（`model.py:1265-1267`）：

```python
is_scale = "weight_scale_inv" in name or name.endswith(".weight_scale")   # 我们的 U8 scale 照样进得来
...
block_size = weight_fp8.shape[1] // scale_inv.shape[1]                     # ← 把 scale 的形状当成 [N/bs, K/bs]
```

真正的原因是 **scale 的布局语义不同**，实测两侧同一层（layer 3）的 header：

| 投影 | weight | 官方 scale_inv（F32，2-D block 网格） | 我们 scale（U8，1×32 沿 K） | `block_size` 推导 | `repeat_interleave(bs, dim=0)` 后 |
|---|---|---|---|---|---|
| `kv_a_proj_with_mqa` | [512, 4096] | [4, 32] | **[512, 128]** | 官方 128 ✓ / 我们 **32** | 官方 [512,4096] ✓ / 我们 **[16384, …]** ❌ |
| `q_a_proj` | [1536, 4096] | [12, 32] | [1536, 128] | 32 | [49152, …] ❌ |
| `q_b_proj` | [16384, 1536] | [128, 12] | [16384, 48] | 32 | [524288, …] ❌ |
| `o_proj` | [4096, 16384] | [32, 128] | [4096, 512] | 32 | [131072, …] ❌ |

`_dequant_fp8_block` 对 **dim=0 和 dim=1 都做 `repeat_interleave(block_size)`**，这只有当 scale 的第 0 维是 `ceil(N/bs)`（block-FP8 语义）时才成立；MXFP8 的第 0 维是**完整的 N**（group 只沿 K 切），于是 N 被无谓放大 32 倍 → 第一层 `kv_a_proj_with_mqa` 就报 `512 vs 16384`（= 512 × 32），与日志逐字吻合。

三点结论：

1. **是的，就是量化 MLA 那 48 个投影导致的。** 未量化的 `GLM-5.3-Flash-MXFP8-Plus-AutoRound`（37 287 层，注意力全 BF16）同一套脚本、同一 TP=4 正常加载。
2. **即使把这段算式修好也不划算**：`_FP8_ATTN_PROJS` 的设计就是"**模型侧永远是 BF16，加载期反量化**"（`model.py:1238-1241` 的注释原文），所以这 48 层量化**只减文件体积（≈1.17 GB），一点显存都不省**。为一个零收益的项目去改 vLLM 的 kernel 入口，不值得。
3. **推理侧的口径就此定死**：注意力（KDA + 稀疏 MLA）全部保 BF16。要出货用 `MODE=mxfp4_mixed`（G0，注意力全明文）；想与官方"层数一致"地跑 MXFP8，用 `MODE=mxfp8`（补丁后 37 290 层 = 官方 − 48），它与已验证可加载的那份只差层 0 的 3 个 dense MLP，而那 3 个不在 `_FP8_ATTN_PROJS` 名单里。

### 6.4 服务侧硬约束（源码级，行号来自 tag v0.30.0 / main）

| 约束 | 代码位置 | 后果 / 处置 |
|---|---|---|
| **`block_size` 必须是 `index_kpool × 32` 的倍数**（本模型 kpool=4 → **128**） | `vllm/platforms/cuda.py:428`、`glm5next/common/attention.py:116,135-143`（原文 "must be a multiple of index_kpool * 32"） | 脚本默认 `block_size=128`，并从产物 `index_kpool` 现算校验 |
| **`deep_gemm` 是硬依赖，但不在 vllm wheel 的 requires_dist 里** | `models/glm5next/nvidia/sparse_indexer.py:662` `RuntimeError("Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.")` | 官方镜像用 `tools/install_deepgemm.sh` 从 vLLM fork 现编（ref `e1f418c2`）；纯 pip 必须自己装 → `setup_vllm_env_glm53.sh` 已包含，eval 脚本做 import 预检 |
| **KDA 状态页必须塞进 MLA 页** | `v1/core/kv_cache_utils.py:1249-1254` → `ValueError: the mamba state page (N bytes) does not fit the MLA page (M bytes); increase tensor parallelism or use a wider KV cache dtype` | 这才是 TP / KV dtype 与"能不能起服务"的真实耦合；`max_num_seqs` 与 mamba 块数按 `max_num_seqs×(1+num_spec)` 预留（`v1/kv_cache_interface.py:1034-1045`）→ 默认保守 256 |
| **`max_model_len` 必须显式封顶** | `models/glm5next/nvidia/sparse_indexer.py:160-163`：indexer decode logits 预留 ≈ `max_num_seqs×max_model_len×4B`；1M 上下文轻松上 GiB。**收这个问题的修复 #57701（省 3072 MiB）在 main，不在 v0.30.0** | 脚本默认 8192；要长上下文记得同时看 KV pool 余量 |
| **PP 实质禁用** | `common/model.py:735-737`（"PP is gated off for GLM-5.3-Flash"）+ `kv_cache_utils.py:1258-1264`；解禁在 open PR #57171/#57121 | 脚本对 `PIPELINE_PARALLEL!=1` 直接拒绝 |
| **TP 必须整除 64 / 64** | `common/model.py:693`（attn heads）、`common/kda.py:199`（KDA heads）、`common/attention.py:436`（MLA heads）；288 experts 侧**无**整除断言（走通用 FusedMoE/EPLB，#55119 已为本模型开启） | TP=4 全满足；TP=3 会被脚本拦下 |
| **dtype 必须 bf16** | `layers/mhc.py:667,702` hyper-connection scratch 硬编码 `torch.bfloat16`；`FLASHMLA_SPARSE.supported_dtypes=[bfloat16]` | 已在 model_args 固定 |
| **KV dtype：sm103 允许 fp8，但绝不能用 `*_ds_mla` 打包布局** | `csrc/libtorch_stable/cache_kernels.cu:937,948` 断言 `pe_dim must be 64 for fp8_ds_mla`，而 GLM 是 **NoPE**（`qk_rope_head_dim=0`）→ open issue #57578/#55773 未修；`FLASHINFER_MLA_SPARSE`（`capability.major==10`）显式拒绝 `fp8_ds_mla`，且专为 NoPE 开了口子：要求 `qk_nope_head_dim=256 && kv_lora_rank=512`（`flashinfer_mla_sparse.py:127-140`）**正是 GLM 的形状** | 脚本只拦 `*ds_mla*`，`auto/fp8/fp8_e4m3` 放行（修正我先前"一律禁 KV 量化"的说法） |
| **不开 expert parallel** | CT 的 W4A4 MXFP4 MoE 只在 CUTLASS/Marlin 间选（`compressed_tensors_moe_w4a4_mxfp4.py:54-70`），CUTLASS 要 `ep_size==1`（`experts/cutlass_moe.py:1041-1044`） | 脚本拒绝 EP=1 |
| **不开 spec decode（现阶段）** | MTP 支持量化权重（`common/mtp.py:45,68-69` 透传 quant_config；`config/speculative.py:64` 有 `glm5_next_mtp`），**但** `mtp.py:49` 的 `eh_proj` 是普通 `nn.Linear` 不接 quant_config → 必须明文；且量化 ckpt + `shared_head.head` 有 open 的加载期误报 **#57532**（修复 #57552/#55442 未合） | 评测阶段不开；要开先确认 ignore 覆盖 `eh_proj/enorm/hnorm/shared_head.*` |
| `reasoning_parser=glm45` | 与 Qwen 系的 `qwen3` 不同 | 脚本默认 glm45，可 `REASONING_PARSER=` 覆盖 |

> 与 8/31 day-0 记录的差别：当时把 `max_num_seqs<=256` 归因为"KDA 状态缓存 512 块 < 默认 1024"，本次读源码得到的机制是**状态页/MLA 页尺寸耦合 + `max_num_seqs×(1+num_spec)` 块预留**，保守值 256 不变，但归因已更正。

---

## 7. 环境构建（系统 python venv + uv）

脚本：`./setup_vllm_env_glm53.sh`。**本机已按此路径实跑成功**，成品环境 `/models/lkk/venv_glm53`
（实测体检输出，全项 `== verified`）：

| 组件 | 已验证版本 |
|---|---|
| vllm | `0.29.1rc1.dev471+g82daf9f57`（`wheels.vllm.ai/nightly/cu130`） |
| torch | `2.13.0+cu130`（`torch.version.cuda == 13.0`） |
| deep_gemm | `2.8.0+e1f418c`（官方 `tools/install_deepgemm.sh` 源码编） |
| flashinfer / flashinfer-cubin | `0.6.18.post1` |
| tilelang / compressed-tensors / lm-eval | `0.1.12` / `0.17.0` / `0.4.13` |
| transformers | `5.17.0`（装 lm-eval 从 5.16.1 顶上来的副作用；vllm 要求 >=5.10.4 无碍） |
| 系统前置 | `libdw-dev 0.190-1.1ubuntu0.1`（缺了 DeepGEMM 编译必挂）、nvcc/cmake/ninja |
| registry | `Glm5NextForConditionalGeneration / ForCausalLM / MTPModel` = **True** |
| device | `torch.cuda.get_device_capability(0) == (10, 3)` = sm103 |

脚本用法（版本已锁到上面这套，且**幂等**）：

```bash
./setup_vllm_env_glm53.sh                # 建/补齐；已就绪则只做体检直接退出（不会重复装 25 GB）
CHECK_ONLY=1 ./setup_vllm_env_glm53.sh   # 只体检（14 s），漂移会标 DRIFT
PIN=0 ./setup_vllm_env_glm53.sh          # 不锁版本，跟随 moving nightly
CHANNEL=release ./setup_vllm_env_glm53.sh  # 等 v0.30.0 上 PyPI 后
DRY_RUN=1 ./setup_vllm_env_glm53.sh      # 只打印命令
```
锁定的 `vllm==0.29.1rc1.dev471+g82daf9f57` 若在 nightly 索引里已被滚动淘汰，脚本会自动退回
`vllm>=0.29.1rc1.dev0` 并打印说明（体检随即标 DRIFT，提醒你去核 registry 与 deep_gemm）。

手工等价步骤：

```bash
# 0) 前置：/usr/bin/python3 = 3.12.3（不用 conda）；uv 0.12.13；/models 剩 2.0 TB
/usr/bin/python3 -c "import venv,ensurepip;print('venv OK')"; uv --version

# 1) 系统 python 建 venv
/usr/bin/python3 -m venv /models/lkk/venv_glm53
export VIRTUAL_ENV=/models/lkk/venv_glm53

# 2) vLLM —— 必须 cu130（B300=sm103），且只能走 nightly（release 0.29.0 不含支持）
uv pip install --torch-backend=cu130 --pre \
   --extra-index-url https://wheels.vllm.ai/nightly/cu130 "vllm>=0.29.1rc1.dev0"

# 3) vllm wheel 不带、但 glm5next 必需的两个依赖
uv pip install --extra-index-url https://flashinfer.ai/whl/ "flashinfer-cubin==0.6.18.post1"
# DeepGEMM 用 vLLM 官方脚本装：它自带权威 DEEPGEMM_GIT_REF（与
# cmake/external_projects/deepgemm.cmake 同步），做 clone --recursive + checkout + bdist_wheel + uv install
# ★ 构建依赖预检（实测踩过缺 libdw-dev）：
ls /usr/include/elfutils/libdwfl.h || echo "缺 libdw-dev：apt-get install -y libdw-dev（系统级改动；本机已装）"
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh -o /tmp/idg.sh
# ★ PATH 前置 venv/bin：官方脚本内部是 `python3 setup.py bdist_wheel`，
#   不前置会用**系统** python/系统 torch 的头文件去编（本机恰好都是 2.13.0+cu130 才侥幸能过）
VIRTUAL_ENV=$VIRTUAL_ENV PATH=$VIRTUAL_ENV/bin:$PATH bash /tmp/idg.sh --cuda-version 13.0

# 4) 量化/评测侧（版本交给 resolver，别手钉）
uv pip install "auto-round>=0.15.0" "lm-eval" "transformers>=5.16.1" accelerate

# 5) 自检：registry + deep_gemm + cuda（不加载权重）
$VIRTUAL_ENV/bin/python - <<'PY'
import importlib.util, torch, vllm
from vllm.model_executor.models.registry import ModelRegistry as R
reg = R.get_supported_archs()
print("vllm", vllm.__version__, "| torch", torch.__version__, torch.version.cuda)
print({n: n in reg for n in ["Glm5NextForConditionalGeneration","Glm5NextForCausalLM","Glm5NextMTPModel"]})
print("deep_gemm", bool(importlib.util.find_spec("deep_gemm")))
if torch.cuda.is_available(): print("cc", torch.cuda.get_device_capability(0))
PY
```

要点：
- **`--torch-backend=cu130` 不能省**：默认解析可能给到 cu12x，sm103 上缺 kernel；装完核对 `torch.version.cuda`。
- **`deep_gemm` 漏装的症状在加载期**：`RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.`（`sparse_attn_indexer_kpool.py:918-921`），不是 import 期，很容易误判成模型/权重问题。官方镜像是用 `tools/install_deepgemm.sh` 从 vLLM fork 现编的（ref `e1f418c2`）。
- **`flashinfer-cubin` 被 v0.30.0 的 `requirements/cuda.txt` 从 wheel 依赖里排除**，必须手动装（SM100+ 的 MXFP4/MXFP8 快路径要用）。
- **registry 自检是硬门槛**：打印 `False` 就说明装到的版本不含支持，别往下走。
- 建议**跟 nightly 而不是锁 v0.30.0**：v0.30.0 缺 9/16-9/20 的 10 条 glm5next 提交（含 3 GiB indexer workspace 修复与 kpool tail 寻址 bugfix）。v0.30.0 上 PyPI 后脚本 `CHANNEL=release` 可用，但那时仍应比对 main。
- 有 docker 权限时**首选 `vllm/vllm-openai:nightly`**（DeepGEMM/cubin/tilelang 全给齐）；注意 **`latest` = v0.29.0 同 digest，不含支持**。

---

## 8. 风险与排查清单

| 症状 | 根因 | 处置 |
|---|---|---|
| `Model architecture Glm5Next... not supported` / registry 预检 False | 本机 vLLM 无该架构；pip release 也还没有 | 用 §6.3 的 nightly 或 `glm53-flash-x86_64-cu130` 镜像 |
| 引擎初始化炸在 mamba/KDA 缓存 | `max_num_seqs` > 状态缓存块数 512 | `MAX_NUM_SEQS<=256`（脚本默认，已有守卫） |
| 想直接拿本地 FP8 目录量化 | 那是**二次量化**源 | 默认拒绝，需 `SOURCE=fp8` 显式确认；结论不能对标参考产物 |
| `fatal error: elfutils/libdwfl.h: No such file or directory`（DeepGEMM 编译中途挂） | `third-party/deep_jit/include/deep_jit/utils/exception.hpp:3` 依赖 libdw；Ubuntu 24.04 默认无 `libdw-dev` | `apt-get install -y libdw-dev`（系统级改动，需确认；**本机 2026-09-21 已装 0.190-1.1ubuntu0.1**）后重跑；脚本已加构建依赖预检，会先告诉你缺什么 |
| DeepGEMM wheel 编出来了但装不进/import 异常 | 官方脚本内部是 `python3 setup.py bdist_wheel`，PATH 里是**系统** python 时就按系统 torch 头文件编（本机恰好同为 2.13.0+cu130 才没炸） | 调用时 `PATH=$VIRTUAL_ENV/bin:$PATH`（脚本已加） |
| `failed to find branch, tag, or commit 'e18418c2…'` | 在调用方硬钉 DeepGEMM sha 且抄错一位（正确的是 `e1f418c2…`，GitHub API 校验 200；错的那位返回 422） | **别自己钉 sha**：调 vLLM 官方 `tools/install_deepgemm.sh`，ref 由上游与 `cmake/external_projects/deepgemm.cmake` 同步维护 |
| `RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.` | vllm wheel 的 requires_dist **不含 deep_gemm**，而 DSA indexer 硬依赖它 | 用官方脚本编装（`setup_vllm_env_glm53.sh` 已含）；或直接用 `vllm/vllm-openai:nightly`（自带） |
| `assert block_size % (index_kpool*32) == 0` / "must be a multiple of index_kpool * 32" | `cuda.py:428`、`attention.py:116,135-143`：本模型 kpool=4 → block_size 必须 128 的倍数 | `BLOCK_SIZE=128`（脚本默认并校验） |
| `ValueError: the mamba state page (N bytes) does not fit the MLA page (M bytes)` | KDA 状态页与 MLA 页尺寸耦合（`kv_cache_utils.py:1255`） | 按提示**加大 TP** 或用更宽的 KV dtype；别硬调 max_num_seqs |
| 1M 上下文一开就 OOM，且开销与模型大小无关 | indexer decode logits 按 `max_num_seqs×max_model_len×4B` 预留（`models/glm5next/nvidia/sparse_indexer.py:160-163`），修这个问题的 #57701 **不在 v0.30.0** | `MAX_MODEL_LEN` 显式封顶（脚本默认 8192）；或跟 nightly |
| `pe_dim must be 64 for fp8_ds_mla` | GLM 是 NoPE（`qk_rope_head_dim=0`），`*_ds_mla` 打包布局断言 pe_dim==64（`cache_kernels.cu:937,948`，open #57578/#55773） | **别传 `fp8_ds_mla`/`nvfp4_ds_mla`**；sm103 用 `auto`/`fp8` 走 `FLASHINFER_MLA_SPARSE`（脚本只拦这两种布局） |
| 把 `mlp.gate` 写成裸串 | 误伤层 0-2 的 `mlp.gate_proj`（dense MLP） | 用 `mlp.gate.` 尾点锚定；跑前置审计看命中表 |
| 用裸 `b_proj` 保护 KDA 小投影 | 子串同时命中 `q_b_proj`/`kv_b_proj`；plus 口径下实测让 `q_b_proj` 少量化 12 层，且不报错 | 写锚定的 `self_attn.b_proj`；ignore 一律由 `gen_glm_ignore.py` 生成并看审计命中表 |
| **ignore 写对了却没生效**（本次三连踩） | auto-round 对 `glm5_next` 有**内置 predefined ignore**：裸 `self_attn`、`layers.0.mlp`、`indexer`、`weights_proj`，与你的 `--ignore_layers` **取并集且无开关**（§4.1） | 先看日志那行 `Using predefined ignore_layers from config: …`；要硬上只能把**完整层名**写进 `--layer_config`（`should_ignore` 里精确 key 优先），子串 key 无效 |
| 层号锚定的正则莫名失效（KDA `o_proj` 反而被量） | `to_standard_regex` 对含元字符的模式"只转义裸点"，你手写的 `\.` 被叠成 `\\.` → **永不相配**（§4） | 模式里**一律写裸点**：`(language_model.)?layers.(0|1|…).self_attn.o_proj`；`gen_glm_ignore.py` 已按此修正 |
| 审计层数与实跑对不上（138 vs 135、37 338 vs 37 287） | `audit_precision_plan.py` 只吃 `--ignore_layers`，不合并 predefined（§4.1） | 审计里 import `get_predefined_ignore_layers_from_config(config)` 并并集；或改看产物 `config.json` 的 group targets 计数收尾 |
| 层 0 的 dense MLP 静默没被量化 | `get_glm_flash_ignore_layers` 从**顶层** config 读 `first_k_dense_replace`，GLM 该字段在 `text_config` → 回落 1（§4.1） | **§4.2 的补丁已从 glm5_next 规则里移除这条**（官方量化全部 3 层 dense MLP）；不想改库就把 3 个全名写进 `--layer_config` |
| **`…-aligned` 产物加载即崩**（补丁后 `MODE=*_plus` 的真症状）：`RuntimeError: The size of tensor a (512) must match the size of tensor b (16384)`，栈在 `glm5next/common/model.py:1231 _dequant_fp8_block`，约 5 % 分片处 | vLLM 把稀疏 MLA 的 `q_a/q_b/kv_a/o_proj` 一律**加载期反量化成 BF16**（`_FP8_ATTN_PROJS`），且**同时接受 `weight_scale` 这个名字**；但它假设 scale 是 block-FP8 的 `[N/bs, K/bs]` 网格，MXFP8 的是 `[N, K/32]` → dim=0 被多放大 32 倍（§6.5.1） | **别用 `MODE=*_plus` 推理**。用 `MODE=mxfp4_mixed`（G0，交付）或 `MODE=mxfp8`（补丁后 37 290 层，与可加载那份只差层 0 的 3 个 dense MLP，不碰 `_FP8_ATTN_PROJS`）。真想要那 48 层，需给 `_dequant_fp8_block` 加"scale 第 0 维已是 N"的分支 —— 但**显存零收益**（模型侧本来就是 BF16），只为文件小 1.17 GB，不建议 |
| 以为"`weight_scale` 名字不对，所以 vLLM 不会走反量化" | 错。`model.py:1265-1267` 明写 `is_scale = "weight_scale_inv" in name or name.endswith(".weight_scale")`，两种都收（本文 §6.5 旧版就是这么推断的，已更正） | 判断通路要以源码为准，不能只看 scale 的**变量名**；真正的差异在 scale 的**形状语义** |
| 重跑 `MODE=*_plus` 把旧产物盖掉 | `quant_GLM-5.3-Flash-mxfp.sh` 的 `OUT` 只按 LABEL 拼名字，`GLM-5.3-Flash-MXFP8-Plus-AutoRound`（37 287 层、补丁前）已存在，auto-round 直接往同目录写 | 重跑前先挪走旧目录，或改 `LABEL`/`OUT`；对比"补丁前 vs 后"时两边都要留着 |
| 抄了 Qwen 系的 ignore | 两边小投影名完全不同（`linear_attn.in_proj_a/b` vs `self_attn.{b,f,g}_proj`） | 三份清单各自独立，见 §4 对比表 |
| 选了纯 MXFP8 | 同范围下 338.1 GB vs 官方 328.3 GB（**+3 %**，差在 1/32 E8M0 scale；旧版"339 vs 306 亏 33 GB"是 GB/GiB 混用） | 没有理由；只做 G0 |
| 以为稀疏 MLA 那 4 个投影"shape 量不了" | **不是 shape**：`[1536,4096]`/`[16384,1536]`/`[512,4096]`/`[4096,8192]` 全满足 `K%32==0` 与 `N>=128`；真正卡点是 predefined（量化期）+ vLLM `quant_config=None`（部署期）（§3.1） | GLM 里唯一被 shape 卡住的注意力件是 KDA `b_proj`（N=64） |
| `KeyError ...qweight` / 加载期报稀疏 MLA 投影 | 那些投影超出"可部署交集" | 保持 `self_attn` 全保（G0 就是这样），别追那 1.2 GB |
| 想量化 PLE/视觉/路由 | router/hc/indexer 在 vLLM 侧多被硬编码 `quant_config=None` | 保 BF16（G0 已如此） |

---

## 附录 A · 复现本文数字

```bash
cd /models/lkk/cri_models
IGN="lm_head,embed_tokens,visual,hc_,indexer,self_attn,mlp.gate.,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d"

python3 audit_precision_plan.py --model ./GLM-5.3-Flash --plan-only-table                      # §1.3
python3 audit_precision_plan.py --model ./GLM-5.3-Flash --scheme MXFP8 --ignore_layers "$IGN" \
  --layer_config '{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}'                              # §2 / G0
python3 audit_precision_plan.py --model ./GLM-5.3-Flash --scheme MXFP8 --ignore_layers "$IGN"  # §0 / G1

# ★ 本节这些数字（138/37 338/338 GB）都是"审计预测"，实跑差 3~51 层，原因见 §4.1。
#   对账用下面两个脚本，它们直接调 auto-round 自己的匹配器 + 读 safetensors header：
/models/miniforge3/envs/muse_verify/bin/python _probe_ignore.py ./GLM-5.3-Flash-BF16 plus \
    '{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}'      # §4.1：predefined 额外吃掉了哪些层
/models/miniforge3/envs/muse_verify/bin/python _diff_official_vs_ours.py   # §3.1：集合 diff + 体积精算
# 产物级核对（量化跑完用它：范围是否与官方 block-FP8 一致，48 个稀疏MLA投影到不到位）
/models/miniforge3/envs/muse_verify/bin/python verify_glm_scope.py GLM-5.3-Flash-MXFP8-Plus-AutoRound
# §3.2 的两张表就来自这里（直接扫全部 *.safetensors 头，按模块聚合张量数/参数量/字节）
/models/miniforge3/envs/muse_verify/bin/python compare_quant_scopes.py \
    GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned GLM-5.3-Flash
# 官方 FP8 覆盖范围（§3 表）——只看 header，不加载权重：
python3 - <<'PY'
import json,os,struct,collections
D='GLM-5.3-Flash'; idx=json.load(open(f'{D}/model.safetensors.index.json')); m={}
for s in sorted(set(idx['weight_map'].values())):
    with open(os.path.join(D,s),'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        for k,v in json.loads(f.read(n)).items():
            if k!='__metadata__': m[k]=v['dtype']
print(collections.Counter(m.values()))                     # F8_E4M3 应 = 37338
print({k for k,d in m.items() if d=='F8_E4M3' and '.self_attn.' in k and k.endswith('.weight')
       and int(k.split('.layers.')[1].split('.')[0])==3})  # 层3(稀疏MLA) 的 4 个投影
PY

# 上游支持核实（本文 §6.2 的来源）
for tag in v0.29.0 v0.30.0 main; do
  echo -n "$tag -> "; curl -s "https://raw.githubusercontent.com/vllm-project/vllm/$tag/vllm/model_executor/models/registry.py" | grep -c Glm5Next
done
curl -s https://pypi.org/pypi/vllm/json | python3 -c "import json,sys;print(json.load(sys.stdin)['info']['version'])"
```

## 附录 B · 三个模型的对比（同一套方法，三套方子）

| | Qwen3.8-27B | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|---|
| 类型 | dense VL + GDN | MoE VL + GDN + PLE + DSA | MoE VL + KDA + 稀疏 MLA + HC + DSA |
| 总参数 / 激活 | 27.8 B / — | 180.0 B / ~6.7 B | 321.3 B / ~17 B |
| 量化主体 | MLP 61.6% + GDN 20% | routed experts 67.1% | routed experts **97.0%** |
| 压缩上限 | 2.8× | 2.0×（PLE 102 GB 封顶） | **3.5×** |
| 本机 vLLM | ✅ 已端到端跑通 | ✅ 需 `moe_backend=marlin`（TP>1） | ❌ 需 nightly / 定制镜像 |
| 参考产物复现 | — | 73 728 层 4-bit 吻合 | **37 152 层 4-bit 吻合** |
