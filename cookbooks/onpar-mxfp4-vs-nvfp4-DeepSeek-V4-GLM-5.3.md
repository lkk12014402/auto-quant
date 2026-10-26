# MXFP4-on-par-NVFP4：DeepSeek-V4-Flash-0731 与 GLM-5.3-Flash 三方对照

日期：2026-10-25/26 · 主机：sc09super22-b200（8×B200）· owner: kaokaolv

本文对比**两个模型家族 × 三个产物**：

- **NVIDIA**：`nvidia/DeepSeek-V4-Flash-0731-NVFP4`、`nvidia/GLM-5.3-Flash-NVFP4`
- **我们**：`DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-RTN-AutoRound`、`GLM-5.3-Flash-BF16-MXFP4-Mixed-CT-RTN-AutoRound`
- **intel-ai**：`intel-ai/DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound`、`intel-ai/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`

「on-par」在本任务里的定义：**量化模块集合与 NVIDIA 一致**，scheme 做映射
（NVFP4 → MXFP4；若存在 FP8 权重则 FP8 → MXFP8）。

---

## 0. 证据来源（全部来自实际产物，不是推测）

| 来源 | 方法 |
|---|---|
| NVIDIA 两个 checkpoint | `config.json` / `hf_quant_config.json` / `.quant_summary.txt` + safetensors header 的 HTTP Range 读取（逐张量 dtype/shape）；NVFP4 模块集合由 `weight_scale_2` 张量反查 |
| intel-ai 两个 checkpoint | `config.json` 的 `quantization_config`（config_groups/ignore）+ `model.safetensors.index.json` 的 tensor 数/分片数 + model card |
| 我们的两个产物 | 量化 run 的 `quantize.log` 汇总（Quantized/Ignored 层数）、`audit_export.py`（只读 header 的 dtype/scale 形状审计）、`accuracy.json` |
| 精度 | lm_eval 0.4.13 + vLLM（TP=2，`kv_cache_dtype=fp8`，全量样本：piqa 1 838 / hellaswag 10 042 / mmlu 14 042 / gsm8k 1 319） |

---

## 1. 总览

| | NVIDIA NVFP4 | 我们的 MXFP4 | intel-ai MXFP4-Mixed |
|---|---|---|---|
| **DeepSeek-V4-Flash-0731** | 源：官方原生 FP4+FP8；NVFP4 experts 33 024 + 原生 FP8 390 + 原生 MXFP4 MTP 2 304 + BF16 其余；**166.88 GB / 138 365 tensors / modelopt** | MXFP4 33 024 + MXFP8 390 + MXFP4 MTP 2 304 + BF16 其余；**167.08 GB / 72 317 / compressed-tensors** | 模块集合与 NVIDIA 完全相同（35 718 层）；**167.08 GB / 72 317 / compressed-tensors** |
| **GLM-5.3-Flash** | 源：BF16；NVFP4 **36 297**（36 288 routed experts + 9 dense MLP）+ BF16 其余；FP8 仅 KV cache；**204.42 GB / 147 661 / modelopt** | MXFP4 **36 297**（集合与 NVIDIA 双向 diff 0）+ BF16 其余；fp8 KV；**194.89 GB / 75 067 / compressed-tensors** | MXFP4 **37 152**（含 layer45 MTP experts）+ MXFP8 **138**（dense MLP 9 + shared experts 129）+ BF16 其余；**~183.3 GB / 76 060 / compressed-tensors** |

---

## 2. DeepSeek-V4-Flash-0731：三方逐模块对照

| 模块（43 主层 + 3 MTP） | NVIDIA NVFP4 | 我们 | intel-ai |
|---|---|---|---|
| `layers.*.ffn.experts.*.w{1,2,3}` | 33 024 × NVFP4（group 16，E4M3 scale + F32 `weight_scale_2`/`input_scale`） | 33 024 × **MXFP4**（group 32，E8M0；**bit-exact passthrough**） | 33 024 × MXFP4 |
| `mtp.*.ffn.experts.*` | 2 304 × **原生 MXFP4**（未动） | 2 304 × **MXFP4**（passthrough） | 2 304 × MXFP4 |
| `attn.{wq_a,wq_b,wkv,wo_a,wo_b}` / `ffn.shared_experts.*` / `attn.indexer.wq_b` / `mtp.0.main_proj` | 390 × **原生 FP8**（128×128 block，ue8m0） | 390 × **MXFP8**（per-32 E8M0，scale 只是展开，值不变） | 390 × MXFP8 |
| `attn.compressor.*` / `indexer.compressor.*` / `indexer.weights_proj` | BF16 | BF16 | BF16 |
| `ffn.gate`（router）/ `hc_*` / `embed` / `head` / `mtp.2.markov_head` / `confidence_head` | BF16/F32 | BF16（ignore 里显式写出） | BF16（196 项 ignore） |
| 量化层数 / ignore | 35 328 + 390 | **35 718 / 196** | 35 718 / 196 |
| 产物体积 / tensors | 166.88 GB / 138 365 | 167.08 GB / 72 317 | 167.08 GB / 72 317 |

**关键结论（DeepSeek）**

- 模块集合与 NVIDIA **完全一致**；与 intel-ai 也几乎同构（层数、ignore、体积、tensor 数都相同）。
- **权重数值等价**：NVIDIA 的 NVFP4 是对源 MXFP4 nibble 的无损 bit-cast；我们的 MXFP4 是
  model_free 的 bit-exact passthrough；FP8→MXFP8 只把 128×128 block scale 展开成 per-32。
  因此两边的 4-bit 专家值与 390 条线性层的值都等于官方 checkpoint。
- 差异只存在于**运行时与格式**：NVFP4（group 16 + 静态 amax 激活）vs MXFP4（group 32 + 动态激活）；
  以及 modelopt 内核路径 vs compressed-tensors + `wo_a` bmm patch。
- 体积 +0.2 GB 全部是 per-32 scale 的开销。

**精度（4 任务全量；DeepSeek）**

| | gsm8k | mmlu | piqa | hellaswag | mean |
|---|---:|---:|---:|---:|---:|
| 官方 checkpoint（我们同机跑） | 0.9356 | 0.8631 | 0.8264 | 0.6802 | 0.8263 |
| **我们 MXFP4** | **0.9439** | **0.8643** | **0.8210** | **0.6784** | **0.8269** |
| 官方 checkpoint（intel card 报告） | 0.9583 | 0.8649 | 0.8341 | 0.6818 | 0.8348 |
| intel-ai MXFP4 | 0.9568 | 0.8631 | 0.8232 | 0.6788 | 0.8305 |

我们在**同机同口径**下比官方基线 **+0.06 pp**；intel 的公开数字相对其基线为 −0.43 pp
（两台机器/协议细节不同，绝对值不可跨比，只能看相对基线的损失）。

---

## 3. GLM-5.3-Flash：三方逐模块对照

| 模块 | NVIDIA NVFP4 | 我们 | intel-ai |
|---|---|---|---|
| `layers.0-2.mlp.{gate,up,down}_proj` | 9 × NVFP4 | 9 × **MXFP4** | 9 × **MXFP8** |
| `layers.3-44.mlp.experts.*.{gate,up,down}_proj` | 36 288 × NVFP4 | 36 288 × **MXFP4** | 36 288 × MXFP4 |
| `layers.45.mlp.experts.*`（MTP） | **BF16**（`layers.45*` 整体 exclude） | **BF16** | 864 × MXFP4 |
| `mlp.shared_experts.*` | **BF16** | **BF16** | 129 × **MXFP8** |
| `self_attn.*`（KDA 34 + 稀疏 MLA 11 + indexer） | BF16 | BF16 | BF16 |
| `mlp.gate`（router） | BF16 | BF16 | BF16 |
| `visual.*` | BF16 | BF16 | BF16 |
| `embed_tokens` / `lm_head` / norms / hc / conv1d | BF16 | BF16 | BF16 |
| KV cache | **FP8**（`kv_cache_quant_algo: FP8`） | fp8（推理时 `--kv-cache-dtype fp8`） | 未量化 |
| 量化层数 | 36 297 | **36 297（双向 diff 0）** | 37 290 |
| 产物体积 / tensors | 204.42 GB / 147 661 | 194.89 GB / 75 067 | ~183.3 GB / 76 060 |

**关键结论（GLM）**

- NVIDIA 这个模型**没有 FP8 权重**：`quant_algo: NVFP4` + `kv_cache_quant_algo: FP8`，
  `.quant_summary` 中启用的权重量化器只有 dense MLP 与 routed experts。所以「FP8 → MXFP8」
  在这一模型上没有作用对象，只有 KV cache 用 fp8。
- 我们的模块集合与 NVIDIA **0-diff**（36 297 vs 36 297，missing 0 / extra 0）。
- NVIDIA 比我们大 **9.5 GB** 且 tensors 多一倍：NVFP4 用 group-16 scale（305B/16 ≈ 19 GB）
  加每模块两个 F32 标量；MXFP4 用 group-32（305B/32 ≈ 9.5 GB）。
- 与 DeepSeek 不同，GLM 的源是 BF16，两边用不同量化器（modelopt vs auto-round RTN），
  所以**权重数值不等价**。
- intel-ai 的 GLM 产物**不在 on-par 定义内**：多量化了 993 个张量
  （layer45 MTP experts 864 + shared experts 129 = 76 060 − 75 067），并把 dense MLP
  做成 MXFP8 而非 NVFP4 对应的 MXFP4。

**精度（4 任务全量；GLM）**

| | gsm8k | mmlu | piqa | hellaswag | mean |
|---|---:|---:|---:|---:|---:|
| intel card 的 BF16 参考（同协议） | 0.9735 | 0.8666 | 0.8292 | 0.6903 | 0.8399 |
| **我们 MXFP4** | **0.9750** | **0.8588** | **0.8275** | **0.6825** | **0.8359** |
| intel-ai MXFP4 | 0.9735 | 0.8581 | 0.8313 | 0.6836 | 0.8366 |

本机 BF16 基线未跑：643 GB 权重在 TP=2（2×183 GB）OOM，且本机只剩 2 张空闲卡（TP=4 不可用）。
我们相对同协议 BF16 参考为 **99.53%**；与 intel MXFP4 参考逐项差 ≤0.38 pp、mean −0.06 pp。

---

## 4. 我们 vs NVIDIA：差异来源一览

| 维度 | DeepSeek | GLM |
|---|---|---|
| 量化模块集合 | 一致 | 一致（0-diff） |
| 权重数值 | **等价**（都是源 nibble/scale 的保真重编码） | 不等价（各自从 BF16 量化） |
| 4-bit scale | NVFP4 g16 vs MXFP4 g32 | 同左 |
| 激活 | 静态 amax（NVFP4）vs 动态（MXFP4 W4A4） | 同左 |
| MTP | 原样保留（FP8/MXFP4）vs 重编码为 MXFP8/MXFP4 | NVIDIA 整层 BF16；我们同样 BF16 |
| KV cache | 原生 FP8 布局 vs fp8 runtime | FP8 runtime，两边一致 |
| 体积差 | +0.2 GB（per-32 scale） | −9.5 GB（g32 更省 scale） |
| tensors 差 | 138 365 vs 72 317（NVFP4 每层 3 个标量张量） | 147 661 vs 75 067（同因） |
| 服务路径 | modelopt vs compressed-tensors + `wo_a` bmm patch | modelopt_fp4 vs compressed-tensors（需 vLLM 命名空间 targets） |

---

## 5. intel-ai 两个模型互比

| 维度 | intel DeepSeek | intel GLM |
|---|---|---|
| 源格式 | 原生 FP4+FP8 混合 | BF16 |
| 4-bit（MXFP4） | 33 024 experts + 2 304 MTP experts | 37 152 experts（**含 layer45 MTP**） |
| 8-bit（MXFP8） | **390 条源 FP8 线性层**（attention / shared experts / indexer.wq_b / MTP linears） | **138 条**：dense MLP 9 + shared experts 129 |
| attention | **MXFP8**（源就是 FP8） | **BF16**（源 BF16） |
| MTP | experts 4-bit（源就是 MXFP4） | 整层 experts 也 4-bit（源 BF16） |
| 量化层数 | 35 718 | 37 290 |
| 体积 / tensors | 167.08 GB / 72 317 | ~183.3 GB / 76 060 |
| 相对基线 | −0.43 pp（0.8305 vs 0.8348） | −0.33 pp（0.8366 vs 0.8399） |

一句话：**intel 的 DeepSeek 版本是「对齐源 checkpoint 自己的量化边界」**（源有 FP8 权重家族，
所以有 MXFP8）；**intel 的 GLM 版本是「自选一套 8-bit 集合」**（源没有 FP8 权重，选 dense MLP +
shared experts 做 MXFP8），所以两者的 8-bit 模块完全不同，且 GLM 版超出了 NVIDIA 的范围。

---

## 6. 配置/服务层面的一个关键差异（GLM）

- intel-ai 的 GLM `quantization_config` 使用 **HF 命名空间**（`model.language_model.*`）+
  `targets: ["Linear"]`。vLLM 的 GLM 实现把前缀重映射成 `language_model.model.*` / `visual.*` /
  `language_model.lm_head.`，且模型没有 `packed_modules_mapping`，因此这套配置在
  我们使用的 vLLM（nightly `46d2b23ac`）上会失配（BF16 模块被当作 MXFP4 → `KeyError`）。
- 我们的 GLM 产物把 targets 显式写成 vLLM 命名空间：
  `language_model.model.layers.{0,1,2}.mlp.{gate_up_proj,down_proj}` + `"RoutedExperts"`，
  并只 ignore `language_model.model.layers.45.mlp.experts`，可直接加载。
- DeepSeek 产物不存在该问题（其 vLLM 模型的前缀与 auto-round 导出的名字可匹配），
  仅需 `wo_a` 的 CT MXFP8 bmm patch。

---

## 7. 产物与复现路径（本机）

| 模型 | 权重 | 审计/报告 |
|---|---|---|
| DeepSeek-V4-Flash-0731 MXFP4 | `/home/kaokaolv/dsv4_0731/runs/quant_main/quantized`（167.08 GB） | `.../quant_main/report.md`、`.../quant_main/eval/accuracy.json`、全量在 `runs/eval_quant_full` |
| GLM-5.3-Flash MXFP4 | `/home/kaokaolv/glm53/runs/quant_main/quantized`（194.89 GB） | `.../quant_main/report.md`、全量在 `runs/eval_quant_full/eval/accuracy.json` |
| 审计脚本 | `dsv4_0731/scripts/audit_export.py`、`glm53/scripts/audit_glm_export.py` | 只读 safetensors header，输出 PASS/FAIL |
| aqa 回流 | `cookbooks/DeepSeek-V4-Flash-0731-MXFP4-MXFP8-onpar-NVFP4.md`、`cookbooks/GLM-5.3-Flash-MXFP4-onpar-NVFP4.md`、`recipes/quant/{deepseek-ai__DeepSeek-V4-Flash-0731,zai-org__GLM-5.3-Flash}.json`、`patches/evaluate/vllm_dsv4_ct_mxfp8_bmm_wo_a.*` | 在可写 clone `/home/kaokaolv/aqa/latest/auto-quant`（待 user commit） |

---

## 8. 结论

1. **DeepSeek-V4-Flash-0731**：量化范围与 NVIDIA 完全一致，且 4-bit 专家值与 390 条 FP8 线性层的值
   **与官方 checkpoint 数值等价**；同机 4 任务 mean 0.8269 vs 官方 0.8263（+0.06 pp）。
   与 intel-ai 产物几乎同构。
2. **GLM-5.3-Flash**：量化范围与 NVIDIA **0-diff**（36 297），但权重由不同量化器从 BF16 得到；
   4 任务 mean 0.8359（同协议 BF16 参考的 99.53%，与 intel MXFP4 参考差 −0.06 pp）。
   intel-ai 的 GLM 产物多量化了 993 个张量、dense MLP 用 MXFP8，**不满足 on-par 定义**。
3. 两个 intel-ai 模型的差别根源是**源 checkpoint 格式不同**：DeepSeek 有 FP8 权重家族（因此有 MXFP8），
   GLM 没有（FP8 只在 KV cache），其 8-bit 集合是自选的。
