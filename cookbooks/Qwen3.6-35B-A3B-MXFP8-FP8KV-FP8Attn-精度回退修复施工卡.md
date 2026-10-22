# Qwen3.6-35B-A3B：MXFP8 + FP8 KV + FP8 Attention 精度回退定因与修复施工卡（v2）

**状态：owner 于 2026-10-14 13:16 批准继续施工；S0–S5 已完成。** `aqa inspect RUN` 与 `aqa artifacts check RUN` 均为 `status: OK`；三项 loglikelihood task 恢复到历史 FP8KV-only REF 的 0.4 pp 内，但 GSM8K strict 残差未定因，未宣称四任务精度问题全部修复。
S4 路线 A（同一 RUN checkpoint + `TRITON_ATTN`）三项 loglikelihood 任务均恢复到历史 FP8KV-only REF 的 0.4 pp 内：
PIQA 0.821001、MMLU 0.838271、HellaSwag 0.634435。GSM8K 正确 stop 后 strict-match 为
0.899166/1,319、flexible-extract 0.795299/1,319；strict 仍比 REF 低 5.838 pp，残差未定因，
不宣称四任务精度问题全部修复。首轮 thinking/loglikelihood 冲突与错误 stop 产物保留、不计分。
本卡取代原卡“量化范围是主要嫌疑”的旧结论；原 run 与产物不覆盖、不重做量化或 baseline。

**Owner 决策（2026-10-14）：**

| 决策项 | 选择 | 对计划的影响 |
|---|---|---|
| 是否按本卡执行 | 批准（仍只用 GPU0–3，不跑 BF16 baseline） | — |
| D3（上游补丁 vllm-project/vllm#53754 验证 FlashInfer） | **若 TRITON_ATTN 有效就直接用 TRITON_ATTN** | D3 降为条件项：只有 D1 未恢复时才考虑，默认不做 |
| H2 成立时是否自动重量化（路线 B） | **先停下回报，由 owner 决定** | 路线 B 不自动执行 |
| S4 是否在同一 0.31 路径完整重测 REF | **否** | S4 仅测修复后的候选；跨路径对比以 D2 子集结果校正 |

- 原 run：`/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-FP8Attn_20261013-1600-GPU0-3-no-baseline/`（下称 `RUN`）
- 同模型 FP8KV-only 参考：`/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-AutoRound_20261013/`（下称 `REF`）
- 原施工卡：`cookbooks/Qwen3.6-35B-A3B-MXFP8-FP8KV-FP8Attn-施工卡.md`

## 1. 现状与更新后的诊断（2026-10-14 只读）

| 任务 | REF BF16（历史 0.29 路径） | REF FP8KV-only candidate | 原 RUN 默认 FlashInfer | S4 route A：TRITON_ATTN | S4 − REF candidate |
|---|---:|---:|---:|---:|---:|
| GSM8K strict | 0.9568 | 0.9575 | 0.1638*（旧 stop 错误） | 0.8992 | −5.84 pp |
| PIQA | 0.8237 | 0.8248 | 0.8020 | 0.8210 | −0.38 pp |
| MMLU | 0.8366 | 0.8378 | 0.4138 | 0.8383 | +0.04 pp |
| HellaSwag | 0.6349 | 0.6333 | 0.5795 | 0.6344 | +0.11 pp |

REF 是历史 candidate，不是本次 baseline；S4−REF 是跨评估路径参考、不是正式 gate。S4 样本数为 GSM8K/PIQA/MMLU/HellaSwag = 1,319/1,838/14,042/10,042。S4 GSM8K flexible-extract 为 0.7953，REF 为 0.7930（+0.23 pp）；原错误 stop 的 0.1638 不作为有效分数。

### 1.1 首要假设 H1：vLLM 0.31 的 FlashInfer 后端在 SM120 上误用 `q_scale`（运行时问题）

证据链如下：

1. **checkpoint 差异**：RUN 有 10 个 `q_scale`（0.029–0.039）；REF 没有 `q_scale`。两者的 20 个
   `k_scale/v_scale` 数值几乎一致（差异 <3%），因此 KV scale 不是差异来源。
2. **加载**：vLLM 0.31 的 compressed-tensors KV method 将 `q_scale` 初始化为 1，再从 checkpoint
   加载为约 0.03。S1 probe 已逐层确认 10 个 full-attention 层的 q/k/v scales 均加载且与 checkpoint 相符。
3. **消费**：SM120 上 FlashInfer 的 `supports_quant_query_input` 只对 SM100 开启，`get_q_data_type`
   也保持 BF16 query。但 native prefill/decode wrapper 仍传入 `q_scale=layer._q_scale_float`；
   FlashInfer 0.7 会无条件执行 `sm_scale *= q_scale`。结果是 10 层 full-attention 的 logits
   约被缩小 26–34 倍，注意力接近均匀分布。REF 的 `q_scale` 默认为 1，因此不受影响。
4. **上游**：vllm-project/vllm#53481（SM120 FLASHINFER 输出退化，TRITON_ATTN/FLASH_ATTN 正常）；
   修复 PR vllm-project/vllm#53754（open，仅在 query 为 FP8 时传入 `q_scale`）。
5. **仓内 Gemma-4-26B 对照**：FP8KV-only 与 FP8Attn 两份 checkpoint 的权重量化范围相同（ignore 都是 317），
   FP8Attn 版只多了 `attention_input_activations` 和 `q_scale`。在相同 vLLM 0.29、FLASHINFER、PP3 条件下，
   MMLU 分别为 0.7441 与 0.2295；同一 FP8Attn checkpoint 改用 TRITON_ATTN 后恢复到 0.7441。
   Qwen 只有 10/40 层是 full attention，其余为 GDN，所以退化幅度小于 Gemma 的全面崩溃，这一差异与机制吻合。
6. **TRITON_ATTN**：CUDA 上 `supports_quant_query_input=True`，会先用 `q_scale` 把 query 静态量化为 FP8，
   只在 query 确实为 FP8 时才传 `q_descale`。因此 SM120 上**只有 TRITON_ATTN 会真正运行 FP8 Attention**。
   Nemotron FP8KV+FP8Attn 使用 TRITON_ATTN，并已通过 runtime probe，相对 BF16 的变化 <0.5 pp。

**定因结论：vLLM 运行时 backend/`q_scale` 路径是原始 MMLU 等 loglikelihood 严重回退的主要原因。**
同一 RUN checkpoint 切换至 TRITON_ATTN 后 MMLU-50 从 0.4182 恢复至 0.8484；S4 三项完整
loglikelihood task 也都在 REF 的 0.4 pp 内。量化范围更宽仍可能造成小幅变化，但证据不支持其为严重回退主因。

### 1.2 次要假设（修复 H1 后需量化残差）

| 编号 | 假设 | 当前证据 | 结论 |
|---|---|---|---|
| H2 | 量化范围更宽：RUN 比 REF 多量化 250 个 Linear（linear_attn 90、shared_expert 120、self_attn 40） | 同一 RUN checkpoint 改用 TRITON_ATTN 后三项 loglikelihood 恢复到 REF 0.4 pp 内；量化范围差异不能解释原始大幅回退 | 不触发路线 B；无重量化 |
| H3 | dense MXFP8 GEMM kernel 差异 | D1 与 S4 都选用 MXFP8 GEMM kernel，S4 分数恢复；D4 未触发 | 不支持其为此次主要原因；不执行 D4 |
| H4 | vLLM 版本、lm-eval adapter 与 TP/DP 拓扑变化 | D2 在 0.31/TP4/EP 下接近历史 REF；D1/S4 使用 RUN checkpoint + TRITON_ATTN 后恢复 | 不支持其解释原始 loglikelihood 崩溃；仍是 GSM8K REF 对照的混杂变量 |
| H5 | GSM8K stop 包含 `Question:`，会截断 thinking | AQA vLLM harness 已修复自定义 `gen_kwargs` 透传；有效 S4 log 仅含 `<|im_end|>`，旧错误 stop 结果不计 | 初始近零分由 stop 配置解释；strict 残差 −5.838 pp 仍待批准的 generation 路径 A/B |

## 2. 执行边界

- GPU：**只用物理 GPU0–3**（同属 NUMA 0，PCIe，无 NVLink），不查询也不使用 GPU4–7。每个 GPU 阶段开始前确认
  0–3 无 compute 进程且显存 <1 GiB；被占用时每 5 分钟复查一次，不抢卡。
- 不跑 BF16 baseline；不上传、不发布权重；不提交 Git。网络异常时使用 `http_proxy/https_proxy=http://proxy-ir.intel.com:912`。
- 环境：conda env `qwen36-mxfp8`（vLLM 0.31.0、lm-eval 0.4.13、FlashInfer 0.7.0.post1），uv 管包。
  D3 需要修改 vLLM 时，只在克隆环境 `qwen36-mxfp8-fifix` 中操作，不改主环境。
- 固定评估口径（每个 D 实验只改变一个变量）：原生 `vllm` adapter，TP4/DP1/EP，`kv_cache_dtype=fp8`，
  `max_model_len=131072`，batch 32，`max_num_seqs=32`，`max_num_batched_tokens=8192`，lm-eval seed 42、
  vLLM engine seed 1234（原始评估日志确认；S1 单独的 vLLM smoke 也固定 engine seed 1234）。
  OOM 时 batch 依次降为 16、8、4、2、1，并在 journal 中记录。lm-eval 原生 adapter 不支持 DP>1 与 EP 同时开启，
  所以保持 TP4/DP1/EP。
- 产物：诊断结果写到 `RUN/diag_precision_20261014/`，最终评估写到 `RUN/eval_fix/`（或路线 B 的新 run 目录）；
  原 `eval/`、`accuracy.json` 和 `report.md` 中的历史段落不覆盖。

## 3. 施工步骤与验收标准

### 3.0 S0 只读准备（2026-10-14，未占用 GPU）

产物目录：`RUN/diag_precision_20261014/`

| 产物 | 内容 | 结果 |
|---|---|---|
| `mmlu_subset_offline.py` + `d0_offline_mmlu50.json` | 基于现有 samples 离线计算 MMLU 每个 subject 前 50 题（共 2,850 题）的准确率与逐题配对统计 | RUN **0.4182**、REF FP8KV-only **0.8460**、REF BF16 **0.8481**。RUN 与 REF 的预测一致率 41.9%，McNemar χ²≈1009；REF 与 REF-BF16 的一致率 98.0%，χ²≈0.57（噪声底：两者各自独有的正确题分别为 25/19）。**子集差距 42.8 pp，判别力满足 ≥30 pp 门槛** |
| `vllm_probe/sitecustomize.py` | 从 Nemotron 复用，hook `CompressedTensorsKVCacheMethod.process_weights_after_loading`，逐层记录 impl、`supports_quant_query_input`、`query_quant`、q/k/v scale | 已运行并校验 RUN 的 10 层 full-attention；细节见 S1 完成记录 |
| `validate_probe.py` | 适配 Qwen 的 10 层 full-attention 校验器；checkpoint 缺少 `q_scale` 时按 1.0 期望；输出 `fp8_query_effective` 与 `bf16_query_with_non_unit_q_scale` 两个判定字段 | FlashInfer 与 TRITON_ATTN 两组校验均通过，scale 与 checkpoint 匹配 |
| `s1_probe_smoke.py` | 以原评估相同的 engine 参数（TP4/EP、fp8 KV、131072、seq32、batched 8192、bf16、无 prefix cache、seed 1234）加载模型，执行 3 条贪心 smoke 并记录 top-5 logprob | 两后端均成功完成模型加载和三条生成。此前缺少 `__main__` 保护与 Conda bin 未进入 PATH 的问题均已修正；重试时将 Conda env bin 前置到 PATH |
| `s1/flashinfer_pathfixed_20261014.*` + `s1/triton_pathfixed_20261014.*` | RUN checkpoint 默认 FlashInfer 与显式 TRITON_ATTN 的 runtime probe、校验结果、生成 smoke 与运行日志 | 各 40 行覆盖 10 层×4 rank，scale 均匹配 checkpoint，FP8 KV 生效。FlashInfer 为 `FlashInferImpl`、`supports_quant_query_input=false`、`query_quant=null`、`fp8_query_effective=false`、`bf16_query_with_non_unit_q_scale=true`；TRITON_ATTN 为 `TritonAttentionImpl`、`supports_quant_query_input=true`、`query_quant=QuantFP8`、`fp8_query_effective=true`、`bf16_query_with_non_unit_q_scale=false`。两后端均完成 3 条 smoke，结果写入对应 JSON；运行后 GPU0–3 均空闲。旧 `flashinfer_resume_20261014.*` 是 PATH 故障时的部分探针记录，保留作历史 |
| `configs/eval_*.yaml` + `s0_validate_dryrun.log` | D1/D2/D4 的 MMLU-50 与 PIQA 全量配置、S4 四任务全量配置及共享基配置；逐个执行 `aqa validate` 与 `aqa run --dry-run` | 8 个 YAML 全部通过。D1/D2/D4 按任务拆成 MMLU 与 PIQA 两份，因为 AQA 的 `eval.limit` 是全局字段；各自继承同一 TP4/EP、FP8 KV、128K、batch32、lm-eval seed42 配置。vLLM engine seed 为 lm_eval adapter 默认的1234 |

S0 已完成。#53754 patch 仍按 owner 决策仅在 TRITON_ATTN 未能启动或 D1 不恢复时下载；默认路线不做 D3。

| 步骤 | 工作与产物 | 验收标准 |
|---|---|---|
| **S0 只读准备（无 GPU；①② 已完成）** | ✅① 从 RUN 与 REF 现有 MMLU samples 离线计算子集分数（每个 subject 前 50 题，共 2,850 题；与 lm-eval `--limit 50` 一致）；✅② 改写 Nemotron 的 probe，用于 Qwen：逐层导出 `_q/_k/_v_scale_float`、attention impl、`supports_quant_query_input`、`query_quant` 状态；✅③ 生成 D1/D2/D4 的 MMLU-50 与 PIQA 全量 eval-only YAML，以及 S4 四任务全量 YAML；☐④ 仅在 TRITON_ATTN 未启动或 D1 不恢复时下载 vllm-project/vllm#53754 diff，保存为 `patches/vllm_0.31.0_flashinfer_bf16_query_q_scale.patch` | ✅子集差距 42.8 pp（≥30 pp）；✅8 个 YAML 均通过 `aqa validate` 与 `aqa run --dry-run`（日志 `diag_precision_20261014/s0_validate_dryrun.log`）；✅D1/D2/D4 每次只与原始 RUN 改变一个诊断变量；D3 patch 暂不需要 |
| **S1 Runtime probe（GPU0–3，已完成）** | ✅ RUN checkpoint 分别使用默认 FLASHINFER 和 `attention_backend: TRITON_ATTN`，执行 scale/query-quant probe 与 3 条确定性生成 smoke。运行文件前缀为 `s1/flashinfer_pathfixed_20261014` 与 `s1/triton_pathfixed_20261014`；调用时将 Conda env bin 前置到 PATH | ✅ 两组均 40 行（10 层×4 rank），q/k/v scales 与 checkpoint 一致、FP8 KV 生效，生成 smoke 成功。FlashInfer 保持 BF16 query 却加载非单位 q_scale；TRITON_ATTN 启用 `QuantFP8` query，排除该路径上的 H1 机制。结果支持 H1 的运行时机制判断；精度影响由 S2 分数 A/B 定量验收 |
| **S2 定因 A/B（GPU0–3，已完成）** | ✅D1：RUN+TRITON_ATTN，MMLU-50 **0.8484/2,850**、PIQA **0.8183/1,838**；✅D2：REF+默认 FlashInfer，MMLU-50 **0.8474/2,850**、PIQA **0.8205/1,838**。D1 相对旧 RUN 子集 +43.02 pp MMLU、+1.63 pp PIQA；D1 与 D2 差 −0.105 pp MMLU、−0.218 pp PIQA；D2 相对历史 REF +0.14 pp MMLU、−0.43 pp PIQA。MMLU 配对：RUN/D1 预测一致率41.93%、McNemar χ²=1016.68；REF/D1一致率97.61%、χ²=0.735；D1/D2一致率97.51%、χ²=0.073。结果支持 H1 为主因且不支持 H4；D3/D4 均未触发 | ✅四个评估完整；样本数与 `accuracy.json` 确认；配对结果在 `d1_d2_mmlu50_paired.json`，逐题预测与 score 均留存。按 §4 唯一选择 H1→路线 A |
| **S3 修复路线选择（已完成）** | ✅选择路线 A：保留 RUN checkpoint，后续完整评估显式设置 `attention_backend: TRITON_ATTN`。依据：D1 恢复到 REF 水平，D2 在同一 0.31 默认 FlashInfer 路径也复现 REF 分数；D1/D2 的差异低于阈值，排除 H4，H2 无需触发重量化。D3/D4 不执行 | ✅路线与依据已写入 journal 和 `owner_actions.jsonl`；不改权重、不重量化、不提交，进入 S4 candidate-only 验证 |
| **S4 完整四任务评估（GPU0–3，已完成；GSM8K strict 阈值未达）** | ✅修复 AQA vLLM harness 自定义 `gen_kwargs` 透传，`pytest tests/test_eval.py` 22 项通过。✅GSM8K 的实际 task log 确认 `until` 仅为 `<|im_end|>`；strict-match **0.899166 ±0.008294**、flexible-extract **0.795299 ±0.011114**（n=1,319）。✅非生成 task 用独立配置、移除 thinking 参数，结果为 PIQA **0.821001/1,838**、MMLU **0.838271/14,042**、HellaSwag **0.634435/10,042**；实际 backend `TRITON_ATTN`、FP8 KV、TP4/EP、131072、batch32。首轮合并任务的 thinking/loglikelihood 冲突日志保留、不计分。S4 输出分别位于 `eval_fix/eval/gsm8k/` 与 `eval_fix/loglikelihood_corrected/` | ✅四项结果与样本数已核验；相对 REF candidate 的差值为 GSM8K −5.838 pp、PIQA −0.381 pp、MMLU +0.043 pp、HellaSwag +0.110 pp。三项 loglikelihood 在 ±0.4 pp 内；GSM8K strict 未满足每项下降 ≤2 pp 的原参考阈值，残差因 REF/S4 vLLM 版本、adapter、并行拓扑与 thinking 后处理不同而未归因。S1 probe 确认 TRITON_ATTN 上 FP8 Q+KV scale 真正生效。candidate-only；不跑 baseline、不重测 REF、不重量化、不宣称完全修复 |
| **S5 回溯与沉淀（已完成）** | 已更新 RUN `report.md`、`journal.md`、本卡及原 v1 卡；已追加 `aqa log-action` 完成记录；经验回流已覆盖 `AGENTS.md`、`skills/evaluate/SKILL.md`、AQA harness/test/docs。未产生上游或模型 patch | ✅`aqa inspect RUN` 与 `aqa artifacts check RUN` 均为 `status: OK`；report 经验路由表无空行、结论均有对应日志/JSON；不提交 Git（交由 owner） |

## 4. S2 判定规则

阈值：MMLU-50 子集的配对差 >1.5 pp，或 PIQA 差 >1 pp，视为“显著”。

| 结果 | 结论 | 下一步 |
|---|---|---|
| D2 ≈ REF 历史子集，且 D1 ≈ D2 ≫ RUN | **H1 成立**：主因是 vLLM FlashInfer/`q_scale` 运行时问题；量化权重不是主因 | 路线 A → S4 |
| D1 明显好于 RUN，但仍显著低于 D2 | H1 只是部分原因 → 执行 D4 | D4 若补齐差距，判为 H3（0.31 dense kernel），后续评估禁用该 kernel；否则判为 H2 → **暂停并回报 owner**，由 owner 决定是否走路线 B |
| D1 未恢复，或 TRITON_ATTN 无法启动 | TRITON_ATTN 路线不可用 | 此时才执行 D3；若 D3 恢复，用打过补丁的克隆环境走 FLASHINFER，并归档 patch |
| D2 显著低于 REF 历史子集 | H4：vLLM 0.31 或 adapter 路径存在偏移 | 路线 C：暂停并回报，另行决定评估路径 |
| D1 与 RUN 基本相同 | H1 不成立 | 暂停并回报，重新排查（逐层 logits 对比 BF16 dequant 参考） |

## 5. 风险与预案

- TRITON_ATTN 在 Qwen GDN 混合架构上的吞吐可能低于 FlashInfer，显存也可能更高；OOM 时按 batch 序列降级，不改 TP。
- D3 的 patch 基于上游 main，可能无法干净应用到 0.31.0；先运行 `patch --dry-run`，不适用时将 D3 降级为跳过，不手工移植。
- 路线 B 需要重新量化（参照上次，约 40–60 分钟，使用 GPU0–2）；只有 S2 判为 H2 且 owner 批准后才执行。
- S1 脚本必须保留 `if __name__ == "__main__"` 保护（vLLM 多卡会 spawn 子进程）；首次尝试正是因为缺少该保护而在 engine 初始化前失败。
- 本卡所有 Qwen 结论只基于 candidate；没有新的 BF16 baseline，不能给出正式 GO/NO-GO 判定。
