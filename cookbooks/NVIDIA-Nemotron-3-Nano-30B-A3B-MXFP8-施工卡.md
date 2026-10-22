# NVIDIA-Nemotron-3-Nano-30B-A3B：纯 MXFP8（BF16 KV）施工卡

**状态：已完成（①–③、④a–④c、⑤均通过；canonical `aqa_run.json` 的内置 eval 失败状态如实保留）。** 本卡是纯 MXFP8 变体，不覆盖
`NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-施工卡.md` 与 `…-MXFP8-FP8KV-FP8Attn-施工卡.md`。

## 0. 目标与硬约束（用户已确认）

| 项 | 决定 |
|---|---|
| 源模型 | `/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`（HF `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`，59 GiB / 63.16 GB，13 shards，6,243 keys，31.58B 参数） |
| 量化 | AutoRound **MXFP8**，`method=autoround`、`algorithm=rtn`、`iters=0`、导出 `llm_compressor`；**不下发** `--static_kv_dtype` / `--static_attention_dtype` |
| 量化范围 | **B 档**（对齐 NVIDIA 官方 FP8，与前两个变体同一 ignore 清单）：路由专家、shared expert、非紧邻 attention 的 Mamba `in_proj/out_proj` → MXFP8；attention q/k/v/o、router、embeddings、lm_head、conv1d、第 4/11/18/25/32/41 层 Mamba 投影保 BF16；F32 张量不降级 |
| 评估 | 仓库 lm_eval 四任务 **gsm8k / piqa / mmlu / hellaswag** 全量（不加 limit）；vLLM `max_model_len=131072`（128K 上限，四任务不是 128K 压测）；GSM8K thinking-on（模板默认）、`max_gen_toks=2048`、`until=<\|im_end\|>`、5-shot multiturn；seed 42 |
| KV cache | 候选 **`kv_cache_dtype=auto`（BF16 KV）**，与 baseline 同 KV → 本次是纯权重/激活量化对比 |
| Baseline | **复用** 10-13 BF16 baseline（同环境、同 eval 配置、batch32、128K、TP4×DP1+EP、BF16 KV；mean 0.7660014348），不重跑；report 注明来源路径与 sha256 |
| batch | `batch_size=32` 起步；仅在**真实 GPU OOM** 时按 16→8→4→2→1 递减（超时/kernel 错误不算 OOM） |
| GPU | 只用物理 GPU **0–3**（NUMA0；0↔1、2↔3 为 PIX，跨对 NODE；5090 PCIe 无 NVLink）。量化默认**单卡 GPU0**（FP8Attn 轮带校准峰值 18.2 GB），真实 OOM 才扩到 0,1（同 PIX 对）；评估 **TP2×DP2 + EP、PP=1**（TP 对 = {0,1}/{2,3}，均为 PIX）。每个 GPU 阶段前 `nvidia-smi` 确认空闲（显存 ≤512 MiB 且利用率 ≤5%），忙则 30 s 轮询等待，不抢卡、不碰 4–7 |
| 环境 | 复用 conda 环境 `nemotron3-mxfp8-20261013`（uv 管包），不新建、默认不改包 |
| 网络 | 直连失败时 `export http_proxy=http://proxy-ir.intel.com:912 https_proxy=http://proxy-ir.intel.com:912`（本地服务设 `no_proxy=127.0.0.1,localhost`） |
| 回传 | 仅 `local_fs`（无 `env.sh`，不配 hf_dataset）；**不上传 HF、不 git commit** |
| Run 目录 | `/home/xuehaosu/aqa_results/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-AutoRound_20261014`（artifact 名预期 `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-MXFP8-CT-RTN-AutoRound`，以 `aqa/report/names.py` 实际生成为准） |

## 1. 调研结论（2026-10-14）

| 项 | 结论 | 依据 |
|---|---|---|
| 架构 | `nemotron_h`：52 层 = 23 Mamba2 + 23 MoE（128 路由专家 top-6 + 1 shared，relu2 非门控仅 up/down）+ 6 Attention（层 5/12/19/26/33/42，GQA kv_heads=2，head_dim 128）；`tie_word_embeddings=false`；无 MTP/vision；源权重无 `quantization_config` | `config.json`、10-13 header 审计 |
| 上游版本 | 最新 release：vLLM **0.31.0**（仅 0.31.1rc0/0.32.0rc0 预发布）、AutoRound **0.16.0**、lm-eval **0.4.13**；transformers 最新 5.19.0，但 vLLM 0.31 要求 `<5.18` → 保持 5.17.0 | PyPI / `git ls-remote`（经代理） |
| 复用环境 | `nemotron3-mxfp8-20261013`：torch 2.13.0、vLLM 0.31.0、AutoRound 0.16.0、transformers 5.17.0、lm-eval 0.4.13、ninja 1.13.2、uv 0.13.0（env/bin 内） | 本机 `uv pip list` |
| 量化路由 | AutoRound 0.16 在 `iters=0` + `llm_compressor` + MX-FP 且 `disable_opt_rtn=True` 时会**自动转 model_free**；本次按用户选择走模型加载路径，YAML 不设 `opt_rtn`（即不加 `--disable_opt_rtn`），故不触发自动 model_free；①中核实 MXFP8 下 opt_rtn 是否实际调用校准数据并记录 | `auto_round/autoround.py:161-200`、`utils/model.py:3167` |
| ignore 命名空间 | AutoRound 按 `model.named_modules()` 匹配：Mamba 相邻层用 `model.layers.*`，checkpoint 键为 `backbone.layers.*`；10-13 attempt04 用错前缀导致 12 个投影被量化 | FP8KV 施工卡 / recipe JSON pitfalls |
| 导出缺陷 | AutoRound ShardWriter 曾残留 6 个未被 index 引用的旧 shard，且有字节一致的重复张量 → 需按 index 规范化（参照 `normalize_attempt05_export.py`） | 10-13 `attempt05_export_normalization.json` |
| SM120 推理 | 原生 `NemotronHForCausalLM`、`trust_remote_code=false`；MoE 自动 Marlin 报 `Invalid thread config`，**显式 `--moe-backend humming`** 通过；Linear 用 `FlashInferCutlassMxfp8LinearKernel`；attention FLASHINFER（无 FP8 KV/q_scale，不受 AGENTS.md 坑 #8 影响） | 10-13 vLLM 日志 |
| AQA 现状 | `aqa run` 不能在阶段间重启外置 vLLM 服务；`eval.thinking` 未接入；`aqa knowledge` / references 未接入 → 量化走 `aqa run`，全量评估走外置 vLLM + `aqa eval` 分阶段，均如实记录 | AGENTS.md §2.3/§3.3/§5.2、前两卡 |
| 参照分数 | BF16 0.766001；MXFP8+FP8KV 0.766297；MXFP8+FP8KV+FP8Attn 0.767533（均 GO） | `recipes/quant/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json` |
| 资源 | GPU 0–7 当前全部空闲；磁盘 `/` 余 290 GB（输出约 34 GB） | `nvidia-smi`、`df -h` |

## 2. 逐模块精度表（B 档，沿用 10-13 header 审计实数）

| 模块 | 精度 | Keys / 参数量 |
|---|---|---:|
| 路由专家 `mixer.experts.N.{up,down}_proj` | **MXFP8** | 5888 / 29,374.8M |
| Shared expert `mixer.shared_experts.{up,down}_proj` | **MXFP8** | 46 / 459.0M |
| Mamba `in_proj/out_proj`（17 层，非紧邻 attention） | **MXFP8** | 34 / ≈658.0M |
| Mamba `in_proj/out_proj`（层 4/11/18/25/32/41） | BF16 | 12 / ≈232.2M |
| Attention `q/k/v/o_proj`（层 5/12/19/26/33/42） | BF16（KV cache BF16） | 24 / 140.5M |
| `lm_head`、`backbone.embeddings` | BF16 | 2 / 704.6M |
| Router `mixer.gate.weight` | BF16 | 23 / 7.9M |
| `gate.e_score_correction_bias`、`A_log`、`D` | **F32（不得降级）** | 69 |
| `conv1d`、`dt_bias`、各 norm | BF16 / 自动跳过 | 余下 |

ignore 清单（与前两变体逐字一致）：

```text
lm_head,backbone.embeddings,mixer.q_proj,mixer.k_proj,mixer.v_proj,mixer.o_proj,mixer.gate,mixer.conv1d,
model.layers.4.mixer.in_proj,model.layers.4.mixer.out_proj,model.layers.11.mixer.in_proj,model.layers.11.mixer.out_proj,
model.layers.18.mixer.in_proj,model.layers.18.mixer.out_proj,model.layers.25.mixer.in_proj,model.layers.25.mixer.out_proj,
model.layers.32.mixer.in_proj,model.layers.32.mixer.out_proj,model.layers.41.mixer.in_proj,model.layers.41.mixer.out_proj
```

**体积预期**：30.49 GB FP8 + 0.95 GB E8M0 scale + 2.2 GB BF16/F32 ≈ **33.6 GB（≈31.3 GiB）**，接受区间 32–36 GB；
与 FP8KV 变体相比仅少 12 个 K/V scale（无 F32 k/v_scale）。**接近 63 GB = 量化静默失效，立即停止。**

## 3. 五步施工与验收标准

**进度**：①结构分析、②环境、③YAML、④a 量化与导出审计、④b 生成 smoke、④c 四任务全量评估、⑤回溯与 hard-flow 检查均完成。

### ① 结构分析与量化方案 → `analysis.md`

| 子任务 | 内容 |
|---|---|
| 1.1 | 按 `templates/analysis.md` 填写；查阅清单逐条记采纳/偏离：`skills/quantize`、`skills/evaluate`、`recipes/quant/nvidia__…json`、`docs/autoround_full_reference.md`、两张 Nemotron 施工卡、Nemotron recipe md、NVIDIA 官方 FP8 `hf_quant_config.json`、模型 README；references 层未接入记偏离；结构分析 skill 为部署机外部文件，未提供则记偏离 |
| 1.2 | 复跑 10-13 的 `index_header_audit.py`（只读 header，不加载权重）→ `logs/index_header_audit.json`，确认源模型未变 |
| 1.3 | 三件套探针：`AutoConfig`（`trust_remote_code=False/True` 两路）、vLLM `ModelRegistry` 命中 `NemotronHForCausalLM`、AutoRound 0.16 NemotronH 非门控专家拆解；核实 MXFP8+RTN 不加 `--disable_opt_rtn` 时走 regular 路径、opt_rtn 是否用校准 |
| 1.4 | 定稿精度表、ignore 清单（`model.named_modules()` 命名空间）、量化命令、评估计划、体积预期 |

**验收**：查阅清单无空行；header 6,243 keys 与 index 完全对齐；每条 ignore 在运行时模块名中命中且不误伤（`layers.4.` 不匹配 `layers.4x`）；
明确 modeling 路径（原生，`trust_remote_code=false`）；给出完整量化命令 + 评估计划 + 体积 33.6 GB。

### ② 环境 → `setup_venv.sh` + `requirements.txt` + `logs/environment_selfcheck.json`

| 子任务 | 内容 |
|---|---|
| 2.1 | `setup_venv.sh` 记录复用逻辑（`mamba`/conda 环境 `nemotron3-mxfp8-20261013` 存在即复用，不存在才按 10-13 pin 用 uv 重建） |
| 2.2 | 自检：torch CUDA 13.0 可见 SM120；`AutoConfig` 识别 `nemotron_h`；vLLM registry 命中（**硬门槛**）；AutoRound CLI 可用；四个 lm_eval task 与 recipe 存在；`local_completions` backend 可导入；`aqa` 指向本仓（editable） |
| 2.3 | `uv pip freeze > requirements.txt`、`uv pip check`；任何换包 → 先问用户、重跑自检、重冻结、`aqa log-action` |

**验收**：硬门槛全部通过；版本与 §1 一致；`uv pip check` clean；freeze 与 10-13 的 requirements 差异为空（或差异已解释）；其他环境未被改动。

### ③ YAML 定稿 → `run.yaml`（canonical）+ `eval_candidate.yaml`

| 子任务 | 内容 |
|---|---|
| 3.1 | 从 10-13 `run.yaml` / `templates/run.yaml` 派生：`scheme: MXFP8`、`method: autoround`、`algorithm: rtn`、`iters: 0`、`export_format: llm_compressor`、**删除 `static_kv_fp8`**、`device_map: "0"`、`extra_args: [--seqlen 2048, --batch_size 1, --nsamples 128, --disable_trust_remote_code]`、ignore 如上 |
| 3.2 | eval：`backend: local_completions`、四 task、`batch_size: "32"`、`baseline: false`（复用）、gsm8k `gen_kwargs {max_gen_toks: 2048, until: "<\|im_end\|>"}`、`model_args {tokenizer: <源模型>, max_length: 131073, tokenized_requests: true, num_concurrent: 32, …}`、seed 42（与 10-13 `eval_baseline.yaml` 逐项对齐；lm_eval 把此值规范化为有效 131072） |
| 3.3 | `gate.max_accuracy_drop: 0.02`；`output.sinks: [local_fs]`；`agent.fix_loop/optimize: false`；`executor: {type: local, num_gpus: 1}`（省略 `env`，用 `$ENV_PREFIX/bin/aqa` 直接调用） |
| 3.4 | 复制 10-13 `baseline/` 到新 run 目录（`baseline/accuracy.json` + 样本日志），记录源路径与 sha256 |

**验收**：`aqa validate` 通过；`aqa run --dry-run` 输出路径正确；builder 命令含 `--scheme MXFP8 --algorithm rtn --iters 0 --format llm_compressor --device_map 0`，**不含** `--static_kv_dtype` / `--static_attention_dtype` / `--disable_opt_rtn` / `--model_free`；
eval 参数与 baseline 逐项一致（diff 仅 model 路径与 KV 无差异）；baseline sha256 与源一致。

### ④ 运行与输出

**④a 量化（GPU0）**：`nvidia-smi` 确认空闲 → `CUDA_VISIBLE_DEVICES=0 $ENV_PREFIX/bin/aqa run run.yaml`。本次 AutoRound 量化成功；AQA 内置 GSM8K eval 因未启动外置服务而 `ConnectionRefusedError`，`aqa_run.json` 如实保留 `failed`，初始失败 manifest 另存 `logs/run_manifest_initial_pipeline_failure.json`，不伪报端到端成功。不预设 `AR_DISABLE_META_LOAD=1`（该变量为静态 KV scale 物化所需）；若出现 meta tensor 报错再加并记 log-action。真实 OOM 才扩到 `device_map 0,1`。
完成后按 index 规范化导出（清除未引用旧 shard、校验重复张量字节一致、校正 `total_size`），脚本与 JSON 落 `logs/`。

**验收**：
- 实际命令（`.quant_command.json`）与 ③ 一致；体积 32–36 GB；
- header 审计 `logs/quantized_header_audit.json`：**5,968 个 `F8_E4M3` 权重 + 5,968 个 `U8` scale（group 32）**；attention 24 / 相邻 Mamba 12 / lm_head / embeddings / router 为 BF16；69 个源 F32 张量仍为 F32；**无 k_scale/v_scale/q_scale**；
- 导出键与源 6,243 键同构（逐专家命名，无融合残留、无缺键）；所有 shard 均被 index 引用，每键存在于指定 shard，`total_size` 与实际一致；
- `config.json` 的 `quantization_config` 为 compressed-tensors mxfp8，**无 `kv_cache_scheme`**。

**④b 生成冒烟（GPU 0–3）**：外置 `vllm serve <quantized> --tensor-parallel-size 2 --data-parallel-size 2 --pipeline-parallel-size 1 --enable-expert-parallel --moe-backend humming --kv-cache-dtype auto --max-model-len 131072 --max-num-seqs 32 --max-num-batched-tokens 2048 --no-enable-prefix-caching --dtype bfloat16`；先发一条中英文 chat 请求 + 一条 GSM8K 题。本次 smoke PASS：双语回答含正确 `2`，GSM8K Janet ducks 样例输出 `####18`；server 日志确认 `HUMMING`、`FlashInferCutlassMxfp8LinearKernel`、`trust_remote_code=False`、BF16 KV (`auto`) 与 max length 131072。

**验收**：服务就绪；日志确认 `HUMMING` MxFp8 MoE、`FlashInferCutlassMxfp8LinearKernel`、`kv_cache_dtype=auto`、`trust_remote_code=False`；回答连贯且 GSM8K 题答案正确；无 NaN/乱码。若 Humming 失败，记录后逐个尝试其他显式 MoE backend，不擅自改量化范围。

**④c 候选全量评估（GPU 0–3，同 ④b 服务配置）**：执行 `bash run_external_eval.sh candidate`，服务 ready 后 bilingual/GSM8K smoke 通过，再以 `aqa eval eval_candidate.yaml --out eval` 完成四任务。batch size 保持 32，没有 OOM 或降档。

**验收（已通过）**：`eval/accuracy.json` 四任务齐全，样本数 **gsm8k 1319 / piqa 1838 / mmlu 14042 / hellaswag 10042**；batch 32；GSM8K 去重后 baseline/candidate 各 1,319 道唯一题且无空回答（`logs/gsm8k_sample_audit.json`）；服务退出后 GPU 0–3 均回落至 2 MiB / 0%，端口 8000 已释放。

| task | BF16 baseline | 纯 MXFP8 | Δ（百分点） |
|---|---:|---:|---:|
| GSM8K strict-match | 0.948446 | 0.946171 | -0.2274 |
| PIQA | 0.805223 | 0.811752 | +0.6529 |
| MMLU | 0.712149 | 0.710013 | -0.2136 |
| HellaSwag | 0.598188 | 0.598287 | +0.0100 |
| **四任务均值** | **0.766001** | **0.766556** | **+0.0554** |

`gate.json` 为 GO，阈值 `max_accuracy_drop=0.02`，最大单项下降 0.002274（0.2274 pp）。外置评估完成不改变 canonical `aqa_run.json=failed`；具体失败原因、staged 结果与两种状态的区分见 `report.md`、`run_manifest.json` 和失败快照。

**全程**：`journal.md` 随手记（命令 + 依据 + 坑）；环境/额外命令/决策用 `aqa log-action` 写 `owner_actions.jsonl`。不发布权重，故不写 model card。

### ⑤ 回溯 → `report.md` + `gate.json` + `run_manifest.json`

| 子任务 | 内容 |
|---|---|
| 5.1 | **完成**：`aqa compare baseline eval` 结果落在 `logs/compare_candidate_vs_baseline.txt`；报告列出逐任务 delta、candidate/BF16 百分比、BF16/FP8KV/FP8KV+FP8Attn 参照及协议差异；`gate.json` 为 GO |
| 5.2 | **完成**：体积、dtype、键完备性、index/shard、无静态 scale、Humming/MXFP8 runtime 与 GSM8K 样本审计证据均写入 `report.md` |
| 5.3 | **完成**：施工卡与 Nemotron cookbook 已回填；quant recipe 新增 `autoround_rtn_bf16kv`；发现的 dry-run 状态副作用记入 `AGENTS.md` §9 第 9 条；无依赖修复、HF knowledge/trajectory 或外部 recipe 发布 |
| 5.4 | **完成**：`aqa inspect`、`aqa artifacts check` 均 OK；canonical/candidate YAML、JSON、脚本语法检查通过。尝试运行 `tests/test_artifacts.py` 与 `tests/test_layout.py`，但复用环境没有 pytest；未为本次文档/recipe 修改安装依赖。最终 `aqa run --dry-run` 曾将状态写成 running，已依据原始失败记录恢复为 failed；没有重新执行 pipeline |

**验收（已通过）**：`report.md` gate 结论与 `gate.json` 一致；`aqa inspect`、`aqa artifacts check` 均 OK；recipe JSON 可解析；经验路由表所有状态列均填写；`aqa_run.json` 保持 failed、staged gate 为 GO；GPU 0–3 空闲；**未 git commit、未上传 HF**（提交交给你）。

## 4. 风险与回退

| # | 风险 | 探测点 | 回退 |
|---|---|---|---|
| R1 | AutoRound 自动转 model_free，ignore 前缀失配 | ④a 日志 `Auto-routing to model-free`、header 审计 | 确认未传 `--disable_opt_rtn`；必要时显式 `--disable_model_free`（记偏离） |
| R2 | Mamba2 reference fallback 大块分配 OOM | ④a 日志 | 已带 `--seqlen 2048 --batch_size 1`；仍 OOM 扩到 GPU 0,1 |
| R3 | ignore 未命中运行时模块 → 12 个相邻投影被量化 | header 审计 | 停止，修正清单后在干净目录重跑 |
| R4 | 未引用旧 shard / 重复张量虚增体积 | index vs 磁盘 | 按 index 规范化，保留规范化记录 |
| R5 | SM120 MXFP8 MoE kernel 失败 | ④b 日志 | 显式 Humming；失败则试其他 backend 并记录 |
| R6 | 候选与复用 baseline 协议不一致 | ③ eval 参数 diff | 不一致即改候选配置对齐；无法对齐则问用户是否重跑 baseline |
| R7 | GPU 被他人占用 | 每阶段前 `nvidia-smi` | 等待，不抢卡、不用 4–7 |

## 5. 时间与资源预估

① 20 min；② 10 min（复用）；③ 15 min；④a 量化 20–40 min + 规范化/审计 10 min；④b 冒烟 15 min；④c 全量评估 3–5 h；⑤ 30 min。
总计约 **5–7 h**。GPU：量化 1 卡（GPU0），评估 4 卡（GPU 0–3）；磁盘约 34 GB。
