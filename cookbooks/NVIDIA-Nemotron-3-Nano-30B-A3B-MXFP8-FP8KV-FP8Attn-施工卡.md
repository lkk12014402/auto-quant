# NVIDIA-Nemotron-3-Nano-30B-A3B：MXFP8 + 静态 FP8 KV + 静态 FP8 Attention 施工卡

**状态：执行完成（用户已确认；①–⑤与 AQA 硬流验收通过，2026-10-14）。** 本卡是 FP8 Attention 变体，不覆盖
KV-only 记录 `cookbooks/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-施工卡.md`。

## 0. 目标与硬约束（用户已确认）

| 项 | 决定 |
|---|---|
| 源模型 | `/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`（HF `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`，63.16 GB，13 shards，6,243 keys） |
| 量化 | AutoRound `MXFP8`，`method=autoround`、RTN（`iters=0`），`--static_kv_dtype fp8 --static_attention_dtype fp8`（granularity 保持默认 `tensor`，vLLM 只支持 per-tensor q/k/v scale），导出 `llm_compressor` |
| 量化范围 | 沿用上轮 **B 档**（对齐 NVIDIA 官方 FP8）：路由专家、shared expert、非紧邻 attention 的 Mamba `in_proj/out_proj` → MXFP8；attention q/k/v/o、router、embeddings、lm_head、conv1d、第 4/11/18/25/32/41 层 Mamba 投影保 BF16；F32 标量不降级 |
| 评估 | 仓库 lm_eval 四任务 gsm8k / piqa / mmlu / hellaswag 全量；`max_model_len=131072`；候选 `kv_cache_dtype=fp8`；GSM8K thinking-on、`max_gen_toks=2048`、`until=<|im_end|>`；seed 42 |
| Attention 后端 | 候选评估用 **TRITON_ATTN**（SM120 上唯一会消费静态 q_scale 的后端）；smoke 失败则回退 FLASHINFER，并在报告中标注"FP8 Attention 未生效，仅 MXFP8+FP8KV runtime" |
| Baseline | **复用** 10-13 的 BF16 baseline（同环境、同 eval yaml、batch32、128K、TP4×DP1+EP、BF16 KV；mean 0.7660014348），不重跑；报告注明来源与 sha256 |
| batch | `batch_size=32` 起步，仅**真实 OOM** 时 16→8→4→2→1 |
| GPU | 只用物理 GPU 0–3（NUMA0；0↔1、2↔3 为 PIX）。量化默认**单卡 GPU0**（上轮峰值 VRAM 18.2 GB），真实 OOM 才扩到 0,1；评估 TP2×DP2+EP、PP=1。每阶段前 `nvidia-smi` 确认空闲，忙则等待 |
| 环境 | 复用 conda 环境 `nemotron3-mxfp8-20261013`（uv 管包），不新建、不改包 |
| 网络 | 失败时 `export http_proxy=http://proxy-ir.intel.com:912 https_proxy=http://proxy-ir.intel.com:912` |
| 回传 | 运行时仅 `local_fs`（未启用 HF dataset sink）；用户后续要求发布，已单独上传到 INCModel4 model repo；不 git commit |
| Run 目录 | `/home/xuehaosu/aqa_results/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-FP8Attn-AutoRound_20261014`（artifact 名预期 `…-MXFP8-FP8KV-FP8Attn-CT-RTN-AutoRound`） |

## 1. 调研结论（2026-10-14）

| 项 | 结论 | 依据 |
|---|---|---|
| 架构 | `nemotron_h`：52 层 = 23 Mamba2 + 23 MoE（128 专家 top-6 + 1 shared，relu2 非门控）+ 6 Attention（层 5/12/19/26/33/42，GQA kv_heads=2，head_dim 128） | `config.json`、上轮 header 审计 |
| 上游版本 | vLLM 0.31.0、AutoRound 0.16.0、lm-eval 0.4.13 均为最新 release；transformers 最新 5.19.0，但 vLLM 0.31 要求 `<5.18` → 保持 5.17.0 | `git ls-remote` |
| 现有环境 | `nemotron3-mxfp8-20261013`：torch 2.13.0、vLLM 0.31.0、AutoRound 0.16.0、transformers 5.17.0、lm-eval 0.4.13、compressed-tensors 0.17.0、uv 在 env/bin；无 mamba_ssm（走 reference fallback） | 本机探针 |
| AutoRound 静态 attention | `attention_quant_ctx` 给类名含 "attention" 且有 `k_proj` 的模块（`NemotronHAttention` 命中）挂 hooked attention，标定 `q_scale`；依赖原 `_attn_implementation` 在 `ALL_ATTENTION_FUNCTIONS` 中——NemotronH 原生 `_supports_sdpa=True`，默认 sdpa 可用（`eager` 不在表中，若被设为 eager 会 KeyError） | `auto_round/experimental/attention.py`、`transformers/.../modeling_nemotron_h.py:958` |
| vLLM 加载 q_scale | `get_cache_scale_mapper` 默认规则 `.q_scale → .attn.q_scale`，导出键 `backbone.layers.N.mixer.q_scale` → `model.layers.N.mixer.attn.q_scale`；CT loader 会跳过独立 Attention config group（只告警），scale 仍按 KV scheme 加载 | `vllm/.../quantization/base_config.py:210-242`、`compressed_tensors.py:230` |
| vLLM 消费 q_scale | `Attention` 仅在 `impl.supports_quant_query_input` 且 kv fp8 时做 query FP8 量化：**FLASHINFER 仅 SM100 家族为 True（5090=SM120 为 False）**；**TRITON_ATTN 在 CUDA 上为 True** → 本次用 `--attention-backend TRITON_ATTN` | `flashinfer.py:1918`、`triton_attn.py:525`、`attention.py:468` |
| 预期告警 | AutoRound 不导出 `prob_scale`，vLLM 会打印 "uncalibrated q_scale … and/or prob_scale 1.0"——只要 q_scale≠1.0 且无 "Checkpoint does not provide a q scaling factor"，即不影响结论，报告中说明 | `kv_cache.py:145-205` |
| 上轮已知坑 | ① `AR_DISABLE_META_LOAD=1`（meta loader 不物化新建 scale）；② Mamba reference fallback 需 `--seqlen 2048 --batch_size 1`；③ ignore 须用运行时 `model.layers.*` 名；④ `--disable_trust_remote_code` / vLLM 不开 remote code；⑤ SM120 自动 Marlin MoE 报 `Invalid thread config` → 显式 `--moe-backend humming`；⑥ lm_eval vLLM adapter 不支持 DP>1+EP → 外置 vLLM + `local_completions`；⑦ AQA 调 `auto-round` 需 env bin 前置 PATH | 上轮施工卡 / journal |
| 参考产物 | HF `INCModel4/…-MXFP8-FP8KV-CT-RTN-AutoRound`：5,968 F8_E4M3 + 5,968 U8 + 206 BF16 + 81 F32；payload 33,616,949,680 B | 用户给的 README |
| 偏离 | 外部 `autoround-quant-analysis/SKILL.md` 无读权限；`aqa knowledge`/references 未接入；`eval.thinking` 未接线 → 均记入 `analysis.md` | AGENTS.md |

## 2. 逐模块精度表（预期，以导出 header 为准）

| 模块 | 精度 | 预期 tensor 数 |
|---|---|---:|
| 路由专家 up/down、shared expert、17 层 Mamba in/out_proj | MXFP8（F8_E4M3 + U8 group-32 scale） | 5,968 + 5,968 |
| Attention q/k/v/o、第 4/11/18/25/32/41 层 Mamba 投影、router、embeddings、lm_head、conv1d、norm 等 | BF16 | 206 |
| `A_log`、`D`、`e_score_correction_bias`（原 F32） | F32 | 69 |
| 静态 k_scale / v_scale（6 层） | F32 | 12 |
| **静态 q_scale（6 层，新增）** | F32 | **6** |

ignore 清单（与上轮 attempt05 一致）：
```text
lm_head,backbone.embeddings,mixer.q_proj,mixer.k_proj,mixer.v_proj,mixer.o_proj,mixer.gate,mixer.conv1d,
model.layers.{4,11,18,25,32,41}.mixer.{in_proj,out_proj}
```
**体积预期**：≈33.6 GB（+6 个标量 scale），可接受 32–36 GB；接近 63 GB = 静默失效，立即停止。

## 3. 五步施工与验收标准

### ① 结构分析与量化方案 → `analysis.md`（已完成）

| 子任务 | 内容 |
|---|---|
| 1.1 | 按 `templates/analysis.md` 填写查阅清单：AGENTS.md、`skills/{quantize,evaluate}`、`recipes/schemes/mxfp8.yaml`、四个 benchmark recipe、`recipes/quant/nvidia__…json`、上轮 KV-only 施工卡/recipe、Qwen3.6 与 Gemma FP8Attn 施工卡、NVIDIA FP8 `hf_quant_config.json`、本卡第 1 节源码证据；逐条记采纳/偏离 |
| 1.2 | 复用并重跑上轮 `index_header_audit.py`（只读 header）确认源 6,243 keys 未变 |
| 1.3 | 三件套探针：AutoConfig（`trust_remote_code=False` 原生）、vLLM registry、AutoRound CLI 含 `--static_attention_dtype`；静态确认 sdpa 为默认 attn impl |
| 1.4 | 定稿精度表、ignore（含 checkpoint `backbone.*` ↔ 运行时 `model.*` 映射说明）、量化命令、评估计划、体积预期 |

**验收**：查阅清单无空行；源 key 数 6,243 与 index 对齐；ignore 逐条命中运行时模块名且不误伤（`layers.4` vs `layers.4x`）；给出完整命令 + 评估计划 + 33.6 GB 预期；三项偏离已记录。

### ② 环境 → `setup_venv.sh` + `requirements.txt` + `logs/environment_selfcheck.json`（已完成）

| 子任务 | 内容 |
|---|---|
| 2.1 | 复制上轮 `setup_venv.sh`，`CHECK_ONLY=1` 运行（不安装/不改包） |
| 2.2 | 自检：torch CUDA 13.0 / SM120；AutoConfig 识别 `nemotron_h`；vLLM `ModelRegistry` 命中 `NemotronHForCausalLM`（**硬门槛**）；`auto-round --help` 含两个 static 参数；`vllm serve --help` 含 `--attention-backend`、`--moe-backend`；四 task recipe 与 `local_completions` 可用；`uv pip check` |
| 2.3 | `uv pip freeze > requirements.txt` 并与上轮 diff |

**验收**：硬门槛全过；freeze 与上轮一致（如有差异须解释）；未改动任何 conda 环境。

### ③ YAML 定稿 → `run.yaml`（canonical）+ `run_smoke.yaml` + `eval_candidate.yaml` + `baseline/`（已完成，复用）

| 子任务 | 内容 |
|---|---|
| 3.1 | 由上轮 `run.yaml` 派生：新增 `static_attention_fp8: true`；`device_map: "0"`；`extra_args: --seqlen 2048 --batch_size 1 --nsamples 128 --disable_trust_remote_code`；`output.dir` 指向新 run 目录、`sinks: [local_fs]`、`agent.fix_loop/optimize: false`、`gate.max_accuracy_drop: 0.02` |
| 3.2 | eval 段同上轮（`local_completions`、batch 32、`max_length 131072`；lm-eval adapter 实际上限为 131071 token，服务端硬上限为 131072，不超过 128K；GSM8K gen_kwargs），关闭本次 pipeline baseline，复用现有 baseline |
| 3.3 | `run_smoke.yaml`：同量化段 + 有界 direct-vLLM smoke（piqa/gsm8k `limit` 少量；TP2×DP1+EP；`kv_cache_dtype: fp8`、`moe_backend: humming`、`attention_backend: TRITON_ATTN`、`trust_remote_code: false`、`max_model_len 131072`） |
| 3.4 | baseline 复用：把 10-13 `baseline/accuracy.json` 复制进新 run `baseline/`，记录源路径与 sha256（`aqa log-action`） |

**验收（通过）**：三份 YAML `aqa validate` 全过；`aqa run run_smoke.yaml --dry-run` 正常；AQA builder 精确命令包含 `--scheme MXFP8 --algorithm rtn --iters 0 --static_kv_dtype fp8 --static_attention_dtype fp8 --format llm_compressor --device_map 0` 及 seqlen/batch/nsamples；ignore 与①一致；baseline SHA256 与源一致。

### ④ 运行与输出（进行中）

**④a 量化 + pipeline smoke（已完成）**：AQA 同一进程依次做单卡量化和 TP2 smoke；启动时 `CUDA_VISIBLE_DEVICES=0,1`，量化配置强制 `--device_map 0`（只在物理 GPU0 上量化），后续 smoke 使用 GPU0–1。AutoRound 导出与 AQA PIQA `limit=1` smoke 均成功；pipeline gate 因未运行 baseline 为预期的 `INCONCLUSIVE`，不代表全量评估结论。

**④a 验收**：
- 体积 32–36 GB；`quantization_config` 为 mxfp8 + `kv_cache_scheme`（float8, static, tensor）；含 attention config group（记录 vLLM 跳过告警属预期）；
- header 审计：**5,968 F8_E4M3 + 5,968 U8**；BF16 206；F32 = 69 + 12 k/v + **6 q_scale**；18 个 scale 均有限、非零、非 1.0；
- 受保护的 24 个 attention 权重与 12 个相邻 Mamba 投影仍为 BF16；导出键与源 index 同构、无缺键/融合残留、无 `q_max` 等标定临时张量；index 与 shard 一致。
- 初次导出额外留下 shards 1–6：已确认无 index 引用且其所有 12,224 个 tensor 名均由 index 指向的 shards 7–13 覆盖，之后定点移除。index metadata 已重算为 7 shards / 33,616,949,704 bytes / 32,530,809,426 parameters；`quantized_header_audit.json` 当前 PASS，模型 payload 约 33.6 GB。

**④b 候选 runtime smoke（GPU0–3 → TP2×DP2+EP，已通过）**：AQA TP2 smoke 与 full-topology 独立 vLLM runner 均通过：
- 服务日志：`Using TRITON_ATTN attention backend`、`FlashInferCutlassMxfp8LinearKernel`、Humming MoE、`kv_cache_dtype=fp8`；**无** "Checkpoint does not provide a q scaling factor"、**无** "q_scale … not loaded"、无 KV scale 1.0 告警；
- vLLM worker 探针在 `CompressedTensorsKVCacheMethod.process_weights_after_loading` 后记录 6 个 attention 层 `_q_scale_float/_k_scale_float/_v_scale_float`、KV dtype、attention impl、`supports_quant_query_input` 和 `query_quant`；validator 逐层与导出 checkpoint scales 比对；
- chat 生成 `finish_reason=stop`、输出合理。

**④b 验收**：scale/header 检查、运行时探针与正常生成全部通过 → 判定"FP8 Attention 在 runtime 生效"。若 TRITON_ATTN 启动/生成失败 → 先定位；只有确认此硬件/版本路径不能工作后才回退 FLASHINFER，并在报告标注"FP8 Attention 未生效"。

**本次 runtime 证据**：24 条 worker probe（6 attention 层、每层 4 rank）；layers 5/12/19/26/33/42 的 q/k/v scales 与 checkpoint 一致，KV dtype=FP8、impl=Triton、`supports_quant_query_input=true`、`query_quant=QuantFP8`；chat smoke `finish_reason=stop`。详见 run 目录 `logs/attention_runtime_probe_validation_20261014T030411Z.json` 与 `logs/vllm_candidate_triton_smoke_20261014T030411Z.json`。

**④c 候选全量评估（GPU0–3）**：外置 `vllm serve <quantized> --tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --moe-backend humming --attention-backend TRITON_ATTN --kv-cache-dtype fp8 --max-model-len 131072 --max-num-batched-tokens 2048 --max-num-seqs 32 --no-enable-prefix-caching`，`aqa eval eval_candidate.yaml --out eval`（沿用上轮 `run_external_eval.sh`，参数化 attention backend）。

**④c 验收**：`eval/accuracy.json` 四任务齐全，样本数 GSM8K 1,319 / PIQA 1,838 / MMLU 14,042 / HellaSwag 10,042；batch 32（如降档须有 OOM 日志）；服务日志再次确认 TRITON_ATTN + FP8 KV；GSM8K 样本审计（空回答、≥2048 token 保守指示）；结束后 GPU0–3 回落空闲、端口释放；`aqa inspect` 通过。

**④c 完成记录（2026-10-14 11:32 +08）**：
- 全量候选 `eval/accuracy.json` 四项齐全，样本数 GSM8K 1,319 / PIQA 1,838 / MMLU 14,042 / HellaSwag 10,042；batch 32、`num_concurrent=32`，runner 退出码 0，全量评估日志无 OOM。
- 分数：GSM8K strict `0.9529946929`、flexible `0.9575435936`；PIQA `0.8101196953`；MMLU `0.7097279590`；HellaSwag `0.5972913762`；四项 mean `0.7675334309`。
- GSM8K 样本审计：2,638 条 filter 记录对应 1,319 个唯一样本，空回答 0；7 个唯一回答达到 2,048 token 上限，最长 2,048。
- 完成后 vLLM / AQA eval 进程退出、8000 端口释放；GPU0–3 回到 2 MiB / 0%。完整评估输出已替换 smoke 的 local_fs 副本，旧 smoke 保存在 `logs/pipeline_smoke_eval/`。

**⑤ 完成结果**：对 BF16 baseline 与旧 MXFP8+FP8KV 参考均完成 `aqa compare`；基于 BF16 baseline 的 `max_accuracy_drop=0.02` gate 为 GO（MMLU 最大下降 `0.002421`）。`report.md`、完整 `run_manifest.json` 与 FP8Attn recipe 变体已回写；`aqa inspect` 和 `aqa artifacts check` 均为 OK，报告 / gate 一致且必备章节、经验路由表齐全。recipe JSON 解析和 `git diff --check` 通过；未提交 Git。

仓库单测结果单独记录：目标量化环境没有 pytest，因此使用一次性 uv 测试环境（未改动量化环境）。全量收集因 `test_export_repair.py` 无法导入 `repair_diffusion_export`、`test_hooks.py` 无法导入 `HOOKS` 而中止；排除这两个收集失败模块后为 **569 passed / 59 failed**（52.29 s）。这些问题位于当前已有的工作区代码 / 测试组合，不涉及本次新增 recipe JSON 或文档；未做无关代码修复。

全程：`journal.md` 随手记（命令+依据+坑），`aqa log-action` 记环境/额外命令；后续发布量化权重时生成英文 `model_card.md` 并作为 Hub `README.md`。

### ⑤ 回溯 → `report.md` + `gate.json` + `run_manifest.json`

| 子任务 | 内容 |
|---|---|
| 5.1 | `aqa compare baseline eval`；三列对比表：BF16（复用）/ MXFP8+FP8KV（10-13，FLASHINFER）/ 本次 MXFP8+FP8KV+FP8Attn（TRITON_ATTN），含 Δpp 与比值；gate 0.02 出 GO/NO-GO；注明部署协议对比（KV dtype、拓扑、attention 后端不同） |
| 5.2 | sanity 证据：体积、dtype 计数、键完备性、18 个 scale 值与 runtime 加载证据 |
| 5.3 | 经验路由表（无空行）：本卡回填 → `cookbooks/`；量化 pin → `recipes/quant/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json` 增 FP8Attn 变体；"SM120 FlashInfer 不做 query FP8、需 TRITON_ATTN 才消费 q_scale" → `skills/evaluate/SKILL.md`；patch（如有）→ `patches/` |
| 5.4 | `aqa inspect <run_dir>` + `aqa artifacts check <run_dir>` |

**验收**：report 的 gate 结论与 `gate.json` 一致；`aqa inspect` / `aqa artifacts check` 均 OK；经验路由表各行已填并落到本仓对应路径；GPU 空闲；**不 git commit**（交给用户）。

## 4. 风险与回退

| # | 风险 | 探测点 | 回退 |
|---|---|---|---|
| R1 | AutoRound hooked attention 与 NemotronH/sdpa 或 meta-load 冲突，q_scale 未物化 | ④a 日志、header | 保持 `AR_DISABLE_META_LOAD=1`；查源码定位，必要时沉淀 patch 并先征询 |
| R2 | Mamba reference fallback 显存峰值 | ④a | 维持 seqlen 2048 / batch 1；真实 OOM 扩到 GPU0,1 |
| R3 | TRITON_ATTN 与 hybrid Mamba / FP8 KV 在 SM120 不兼容或精度异常 | ④b smoke | 回退 FLASHINFER 并标注 FP8Attn 未生效 |
| R4 | q_scale 名称映射失败（静默用 k_scale 代替） | ④b 告警 + 离线探针 | 停下修 loader 映射或征询，不得宣称生效 |
| R5 | FP8 query 量化导致掉点超 gate | ④c / ⑤ | 如实报 NO-GO；可建议补跑 FLASHINFER 消融（需用户同意） |
| R6 | TRITON_ATTN 吞吐较低致评估变慢 | ④c | 保持 batch 32，耐心等待；不因超时降 batch |
| R7 | GPU 被他人占用 | 每阶段 `nvidia-smi` | 等待，不抢卡、不用 4–7 |

## 5. 时间与资源

环境自检 10 min；量化 30–45 min（上轮 RTN 478 s + 整模 CPU 加载，峰值 RAM ≈113 GB，现可用 177 GB）；smoke 20–30 min；
候选全量 3–5 h（TRITON_ATTN 可能更慢）；回溯 30 min。合计约 **5–7 h**。磁盘余量 348 GB，产物约 34 GB。

## 6. 后续 Hugging Face 发布（用户追加指示，2026-10-14）

| 项 | 结果 |
|---|---|
| Model repo | [intel-ai/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-FP8Attn-CT-AutoRound](https://huggingface.co/intel-ai/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-FP8Attn-CT-AutoRound) |
| README | 英文卡包含模型/精度信息、静态 q/k/v scale、RTX 5090 serving 参数和完整 AutoRound 量化命令；评估保留率及变化以未舍入的 quantized/BF16 比值计算，显示两位小数；无 Auto-Quant/AQA 内部信息或凭据 |
| 上传来源 | 本卡 run 内的量化导出目录；18 个文件，共 33,647,250,027 bytes；不含量化日志或其它 run 产物 |
| 远端核验 | commit `1829906ebc255fc4a8c8bee30e2cc421abb83def`；远端 19 个文件（含 Hub 管理的 `.gitattributes`）；18 个本地文件大小一致，8 个 LFS 对象 SHA256 与本地匹配，其余 10 个下载后内容 SHA256 匹配 |
| 凭据与运行配置 | 用 `/home/xuehaosu/hf_token`（权限 0600）在本机进程内认证，未输出/写入密钥；原 run YAML 未配置 `provider.upload`，故以本次单独授权的发布脚本执行，不重跑量化评估；HF dataset sink 仍未启用 |
