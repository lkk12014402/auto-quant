# NVIDIA-Nemotron-3-Nano-30B-A3B MXFP8 + FP8 KV 施工卡

**量化/评估状态：已完成（2026-10-14；gate GO）；后续公开模型发布：已完成。** 用户确认量化范围 **B 档（对齐 NVIDIA 官方 FP8）**、
**新建 conda 环境**、GSM8K **thinking-on / max_gen_toks=2048**；原量化阶段为**仅本地落盘**（不上传 HF、不提交 Git），后续发布请求另行授权。
①–⑤ 的实测产物均已完成；`aqa inspect` 与 `aqa artifacts check` 均通过。BF16 baseline mean 为
`0.7660014348`，MXFP8 + static FP8 KV mean 为 `0.7662972486`；最差单项下降 0.2290 pp，低于
`max_accuracy_drop=0.02`。注意：最初完整 `aqa run` 保持 `failed`（见 `aqa_run.json`），量化和全量评估通过
外置 vLLM + `aqa eval` 分阶段完成；没有伪报完整 AQA pipeline 成功。详见 run 目录 `report.md`。
后续按用户授权准备了全英文、无 AQA/auto-quant 内部信息的模型 README。首次私有上传受 INCModel4 存储额度限制；用户随后明确授权公开，模型目录最初以旧名 `INCModel4/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-CT-RTN-AutoRound` 上传，commit `68348e84fa9cb86dde6e20eb1c364e0aacff0317`。2026-10-19 先按统一命名规则改为 `INCModel4/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-CT-AutoRound`，随后迁移至 `intel-ai/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-CT-AutoRound`，远端 README 已同步更新。初次上传时远端 18 个模型文件与本地大小和哈希一致，README 字节级一致；仅额外保留仓库 `.gitattributes`。

## 0. 目标与硬约束

- 源模型：`/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`（59 GiB / 63.16 GB，13 shards，6243 keys，31.58B 参数）。
- 量化：AutoRound **MXFP8**，`method=autoround`、RTN（`iters=0`）、`export_format=llm_compressor`，
  **必须实际下发 `--static_kv_dtype fp8`**；不启用 static FP8 attention。
- 评估：仓库 `lm_eval` 四个 task **gsm8k / piqa / mmlu / hellaswag**（全量，不加 limit）；
  vLLM `max_model_len=131072`（128K 上限，不代表四个任务是 128K 压测）；
  **候选 MXFP8 用 `kv_cache_dtype=fp8`**，BF16 baseline 用 `auto`（BF16 KV），对比标注为"部署协议对比"。
- batch：`eval.batch_size=32` 起步，仅在**真实 GPU OOM** 时按 16→8→4→2→1 递减（超时不算 OOM）。
- GPU：只用物理 GPU **0–3**（同属 NUMA0；0↔1、2↔3 为 PIX，跨对为 NODE；5090 PCIe 无 NVLink）。
  每个 GPU 阶段开始前 `nvidia-smi` 确认空闲，忙则等待；**禁止触碰 GPU 4–7**（当前 4/5/6 有他人负载）。
  量化默认 **GPU 0,1**（PIX 同交换机），OOM 才扩到 0,1,2；评估服务用 0–3，**TP2 × DP2 + EP，PP=1**。
- 环境：mamba 新建 conda 环境 + uv 装包，vLLM 用**最新 0.31.0**（模型卡要求 `vllm>=0.12.0`，最新版满足）。
- 网络：直连失败时 `export http_proxy=http://proxy-ir.intel.com:912 https_proxy=http://proxy-ir.intel.com:912`，并记入 journal/action log。
- 运行目录（拟）：`/home/xuehaosu/aqa_results/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-AutoRound_20261013`
  （artifact 名按 `aqa/report/names.py` 生成，预期形如 `...-MXFP8-FP8KV-CT-RTN-AutoRound`）。

## 1. 调研结论（2026-10-13）

| 项 | 结论 | 依据 |
|---|---|---|
| 架构 | `nemotron_h` / `NemotronHForCausalLM`：52 层 = 23 Mamba2 + 23 MoE + 6 Attention（GQA，kv_heads=2）；MoE 每层 128 路由专家（top-6）+ 1 shared，**非门控 relu2 MLP（仅 up/down_proj）**；无 MTP、无 vision；`tie_word_embeddings=false`；`mamba_ssm_cache_dtype=float32` | `config.json`、index、header 审计 |
| Attention 层号 | 5, 12, 19, 26, 33, 42（`hybrid_override_pattern` 中 `*`） | config |
| 上游最新版本 | vLLM **0.31.0**、auto-round **0.16.0**、transformers 5.19.0、lm-eval 0.4.13 | `git ls-remote` / PyPI |
| transformers | ≥5.3.0 原生支持；原生实现把专家融合为 3D `Parameter [E,I,H]`（加载时转换）；模型目录另带 4.55 时代的 remote code（逐专家 Linear） | 模型 README、`transformers/models/nemotron_h` |
| auto-round 0.16 | `moe_experts_interface` 已显式支持 NemotronH 非门控专家（`has_gate=False`）拆解；ShardWriter 带反向权重转换（导出键回到逐专家命名） | `auto_round/modeling/fused_moe/moe_experts_interface.py:281`、`shard_writer.py` |
| vLLM 0.31 | 注册 `NemotronHForCausalLM`；compressed-tensors `W8A8Mxfp8MoEMethod` 支持非门控 w13（`is_act_and_mul=False`）；后端自动选择（DeepGEMM/Triton/Marlin/Emulation…），**SM120 上 relu2 实际落到哪个后端需实测** | `compressed_tensors_moe_w8a8_mxfp8.py`、`fused_moe/oracle/mxfp8.py` |
| NVIDIA 官方 FP8 参考 | modelopt FP8 + FP8 KV；排除：`lm_head`、6 层 attention 的 q/k/v/o、紧邻 attention 前一层（4/11/18/25/32/41）的 Mamba `in_proj/out_proj`、全部 `conv1d` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/hf_quant_config.json` |
| 推理建议 | `vllm serve ... --trust-remote-code --max-model-len 262144`，reasoning parser `nano_v3`；默认 `enable_thinking=True` | 模型 README |
| 聊天模板 | `<|im_start|>/<|im_end|>`（id 11，为 eos 之一）+ `<think></think>`，与 Qwen 系一致 | tokenizer / chat_template |
| 现有环境 | `autoround-test`（vLLM 0.29）、`qwen36-mxfp8`（vLLM 0.31，无 mamba_ssm）——按用户决定**不复用**，新建 | `conda env list` |
| 仓库知识 | 无 Nemotron 先例；最近同类为 `cookbooks/Qwen3.6-35B-A3B-MXFP8-FP8KV-施工卡.md`（外部 vLLM server + `local_completions`、GSM8K `until` 覆盖、ninja PATH 坑） | 本仓 |
| 回传 | 仓库无 `env.sh` → 仅 `local_fs` sink（用户确认），在 analysis/report 记为偏离 | — |
| knowledge 增量层 | `aqa knowledge` 未接入（AGENTS.md §3.3）→ 仅查打包层，记偏离 | AGENTS.md |

## 2. 逐模块精度表（B 档，Header 审计实数）

| 模块 | 精度 | Keys / 参数量 |
|---|---|---:|
| 路由专家 `mixer.experts.N.{up,down}_proj` | **MXFP8** | 5888 / 29,374.8M |
| Shared expert `mixer.shared_experts.{up,down}_proj` | **MXFP8** | 46 / 459.0M |
| Mamba `in_proj/out_proj`（17 层，非紧邻 attention） | **MXFP8** | 34 / ≈658.0M |
| Mamba `in_proj/out_proj`（层 4/11/18/25/32/41） | BF16 | 12 / ≈232.2M |
| Attention `q/k/v/o_proj`（层 5/12/19/26/33/42） | BF16（KV cache → FP8 静态 scale） | 24 / 140.5M |
| `lm_head`、`backbone.embeddings` | BF16 | 2 / 704.6M |
| Router `mixer.gate.weight`（Parameter，非 Linear） | BF16 | 23 / 7.9M |
| `gate.e_score_correction_bias`、`A_log`、`D` | **F32（保持，不得降级）** | 69 |
| `conv1d`、`dt_bias`、各 norm | BF16 / auto-round 自动跳过 | 余下 |

计划 ignore 清单：source index audit 逐条核对 checkpoint tensor key，并显式处理
checkpoint/runtime module 前缀映射；AutoRound 的实际命中须按 `model.named_modules()`
命名空间核实。当前 `aqa run --dry-run` 不输出层命中清单，最终必须用导出 dtype 验收。

```text
lm_head,backbone.embeddings,mixer.q_proj,mixer.k_proj,mixer.v_proj,mixer.o_proj,mixer.gate,mixer.conv1d,
model.layers.4.mixer.in_proj,model.layers.4.mixer.out_proj,model.layers.11.mixer.in_proj,model.layers.11.mixer.out_proj,
model.layers.18.mixer.in_proj,model.layers.18.mixer.out_proj,model.layers.25.mixer.in_proj,model.layers.25.mixer.out_proj,
model.layers.32.mixer.in_proj,model.layers.32.mixer.out_proj,model.layers.41.mixer.in_proj,model.layers.41.mixer.out_proj
```

AutoRound 0.16.0 的 ignore 匹配对象是 `model.named_modules()`，所以这六层使用 `model.layers.*` 模块名；checkpoint index 导出键则为 `backbone.layers.*`（model/checkpoint 前缀映射）。attempt04 实测旧版 `backbone.layers.*` 字符串未命中运行时模块，12 个本应保留 BF16 的投影实际导出为 F8_E4M3。修正后不能只验日志或 source index 字符串，必须从新导出 safetensors header 确认这 12 个权重仍为 BF16。

**体积预期**：量化 30.49B 参数 → 30.49 GB FP8 + 0.95 GB E8M0 scales；其余 1.09B 参数 ≈ 2.2 GB →
**合计 ≈ 33.6 GB（≈ 31.3 GiB）**，可接受区间 32–36 GB。**若接近 63 GB = 量化静默失效，立即停。**
所有量化层 N/K 均为 32 的整数倍且 N ≥ 1856，不触发 `mm_mxfp8 N>=128` 雷。

## 3. 五步施工与验收标准

### ① 结构分析与量化方案 → `analysis.md`

| 子任务 | 内容 |
|---|---|
| 1.1 | 按 `templates/analysis.md` 填写：查阅清单（skills/quantize、evaluate、recipes、`docs/autoround_full_reference.md`、Qwen3.6 施工卡、NVIDIA FP8 config、模型 README），逐条记采纳/偏离 |
| 1.2 | 脚本化 header 审计 → `logs/index_header_audit.{py,json}`（13 shards、6243 keys、dtype/shape、参数占比） |
| 1.3 | 三件套探针：AutoConfig（原生 vs remote code 两条路径，记录 `trust_remote_code=True` 实际选中哪条）、vLLM registry、auto-round 对 NemotronH 专家拆解 |
| 1.4 | 定稿精度表、ignore 清单、量化命令、评估计划、体积预期 |

**验收**：查阅清单无空行；index 与 header 6243 keys 完全对齐；ignore 清单每一条在 index 中有命中、且无误伤（尤其 `layers.4` vs `layers.4x`）；
明确选用的 modeling 路径及理由；给出完整量化命令 + 评估计划 + 体积预期 33.6 GB。

### ② 环境 → `setup_venv.sh` + `requirements.txt` + `logs/environment_selfcheck.json`

| 子任务 | 内容 |
|---|---|
| 2.1 | `mamba create -p /home/xuehaosu/miniforge3/envs/nemotron3-mxfp8-20261013 -c conda-forge python=3.12 uv=0.13.0` |
| 2.2 | uv pin 安装 `torch==2.13.0`、`vllm==0.31.0`、`transformers==5.17.0`（满足 vLLM `<5.18`）、`auto-round==0.16.0`、`lm-eval[api]==0.4.13`、`-e /home/xuehaosu/auto-quant` |
| 2.3 | 自检：torch CUDA 可见 SM120；`AutoConfig` 识别 `nemotron_h`；vLLM `ModelRegistry` 命中 `NemotronHForCausalLM`（**硬门槛**）；AutoRound CLI 有 `--static_kv_dtype`、四个 `lm_eval` tasks/recipe 均存在；`local_completions` backend 已安装；确认 `ninja`/编译依赖可用（需要时通过 uv 补装并重跑自检） |
| 2.4 | `uv pip freeze > requirements.txt`；之后任何换包 → 重跑自检 + 重新冻结 + `aqa log-action` |

**验收**：`logs/environment_selfcheck.json` 全部硬门槛通过；torch CUDA=13.0、SM120；vLLM registry 命中；`uv pip check` clean；不改动 `autoround-test` / `qwen36-mxfp8` / base 环境；freeze 文件生成。

### ③ YAML 定稿 → `run.yaml`（canonical）+ `run_smoke.yaml` + baseline/candidate eval YAML

| 子任务 | 内容 |
|---|---|
| 3.1 | 从 `templates/run.yaml` 出发：`scheme: MXFP8`、`method: autoround`、`algorithm: rtn`、`iters: 0`、`export_format: llm_compressor`、`static_kv_fp8: true`、`device_map: "0,1"`、校准 `--seqlen 2048 --batch_size 1 --nsamples 128`、ignore 清单如上；canonical eval 列全量四 task 与 baseline/gate |
| 3.2 | eval：`backend: local_completions`、`benchmarks: [gsm8k, piqa, mmlu, hellaswag]`、`batch_size: "32"`、`baseline: true`、gsm8k `gen_kwargs: {max_gen_toks: 2048, until: "<|im_end|>"}`、`model_args: {tokenizer: <源模型>, max_length: 131073, tokenized_requests: true}` |
| 3.2a | 用户确认 thinking-on；模型 chat template 默认开启 thinking。本 checkout 的 `eval.thinking` helper 尚未接入，不依赖该字段；prompt/sample log 核验实际模板行为。 |
| 3.3 | `gate.max_accuracy_drop: 0.02`；`output.sinks: [local_fs]`；`agent.fix_loop/optimize: false` |
| 3.4 | 按 AGENTS.md §2.3 现状：CLI 的 local executor 不切换 `executor.env`，故 YAML 省略该块，直接用目标环境的 `$ENV_PREFIX/bin/aqa` 调用 |

**验收**：`aqa validate` 全过；`aqa run --dry-run` 确认 pipeline/config/output 路径（当前 LocalExecutor dry-run 不打印量化命令，精确 builder 命令另记 `logs/config_command_audit_retry.log`）；
builder 命令包含 `--scheme MXFP8 --algorithm rtn --iters 0 --static_kv_dtype fp8 --seqlen 2048 --batch_size 1 --nsamples 128 --format llm_compressor --device_map 0,1` 且**不含** static attention 参数；
ignore 清单与①一致；eval 命令 batch=32、max_length 与 131073/client-side / server `max_model_len=131072` 对齐。

### ④ 运行与输出（顺序：先量化+冒烟，后跑全量评估，尽早暴露最大风险）

**④a 量化 + 有界加载冒烟（GPU 0,1）** — attempt05 已完成 AutoRound MXFP8/RTN 量化：reference Mamba2 默认路径曾尝试约 64 GiB 临时分配，最终使用 `--seqlen 2048 --batch_size 1 --nsamples 128`；fused-MoE meta loader 无法物化新建 K/V scales，故按 AutoRound 提示设置 `AR_DISABLE_META_LOAD=1`。静态 K/V scales 已完成校准并通过导出审计。
以模型目录自带 remote code 运行不兼容；最终独立 smoke 使用原生 `NemotronHForCausalLM`、TP2+EP、Humming 和 FP8 KV，生成通过。`aqa run run_smoke.yaml` 的原始 pipeline 尝试未成功：remote Transformers config 报 `NemotronHConfig.layers_block_type` 无 setter；该状态保留在 `aqa_run.json` 和失败 manifest 备份中。此 checkout 没有 quant-only 模式，也不能在 pipeline 阶段间自动重启外置 vLLM 服务，因此四任务实测通过外置服务 + `aqa eval` 分阶段完成，不把原始 `aqa run` 标记为成功。

**attempt04 不作为验收结果**：量化已完成，但 smoke 因 lm_eval 同时收到 local-completions `max_length=131073` 与 vLLM `max_model_len=131072` 失败；独立 dtype 审计又发现旧 `backbone.layers.*` ignore 前缀没有匹配 AutoRound 的 `model.layers.*` 模块名，12 个应保留 BF16 的相邻 Mamba 投影实际导出为 F8。随后已修正 canonical ignore 清单，从干净输出目录完成 attempt05；最终精度边界以 safetensors 导出 dtype 为准，而不是命令或日志中的 ignore 字符串。

**验收**：
- attempt05 量化导出审计与原生 vLLM 独立生成 smoke 通过；`aqa_run.json` 中原始完整 pipeline 保持 `failed`，不得以此声称 end-to-end AQA 成功；量化命令实际含 `--static_kv_dtype fp8`；体积落在 32–36 GB；
- 头审计 `logs/quantized_header_audit_attempt05.json`：**5968 个 `F8_E4M3` 权重 + 5968 个 `U8` scale**（5888 专家 + 46 shared + 34 Mamba）；
  被排除的 attention 24 / Mamba 12 / lm_head / embeddings 仍为 BF16；F32 张量（A_log、D、e_score_correction_bias）仍为 F32；
- **导出键名与源 index 同构**（逐专家 `mixer.experts.N.up_proj` 形态，无融合残留键、无缺键）；
- `config.json` 中 `quantization_config` 为 mxfp8 / compressed，`kv_cache_scheme` 为 8-bit float static；**6 个 attention 层共 12 个 F32 k/v scale**。

**④b BF16 baseline 全量评估（GPU 0–3）** — BF16 源权重约 58.8 GiB，TP2 单副本的两卡总显存约 63.7 GiB，尚未留出 CUDA/运行缓冲与 KV cache；故直接外置服务 `vllm serve <BF16 source> --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 131072 --kv-cache-dtype auto`（PP=1）。
`$ENV_PREFIX/bin/aqa eval eval_baseline.yaml --out baseline` 经 `local_completions` 请求，AQA eval stdout JSON 捕获为 `baseline/accuracy.json`。实测 TP4×DP1+EP、PP=1、batch 32 全部完成，无 OOM；GSM8K 0.9484457923、PIQA 0.8052230686、MMLU 0.7121492665、HellaSwag 0.5981876120，mean 0.7660014348。若 TP4 cache 仍不足，按 OOM 日志处理，不静默降低 context。

**④c 候选全量评估（GPU 0–3）** — AQA run 的 `eval/` 冒烟结果先移存 `logs/pipeline_smoke_eval/`（不删除）；外置服务加载量化导出目录，同拓扑，`--kv-cache-dtype fp8`。`$ENV_PREFIX/bin/aqa eval eval_candidate.yaml --out eval` 经 `local_completions` 请求，stdout JSON 落为 `eval/accuracy.json`。候选服务启动时核对 6 层共 12 个静态 F32 k/v scales 非默认 1.0。
实测候选拓扑为 TP2×DP2+EP、PP=1、Humming；启用 MXFP8 Linear、静态 FP8 KV、`max_model_len=131072`、`max_num_batched_tokens=2048`，batch 32 全部完成，无 OOM。GSM8K 0.9507202426、PIQA 0.8084874864、MMLU 0.7100840336、HellaSwag 0.5958972316，mean 0.7662972486；服务日志确认 12 个静态 K/V scale 已加载，未见默认 1.0 fallback 告警。MMLU 冷加载约 13 分钟，保留服务等待完成；代理 README 请求返回 HTTP 200。两次评估均限定 `max_model_len=131072`，但这些 benchmark 不是 128K 压测。

**④b/④c 验收**：两份 `accuracy.json` 均有四任务，样本数 **gsm8k 1319 / piqa 1838 / mmlu 14042 / hellaswag 10042**；batch 32，未因 OOM 降档。GSM8K 日志每个 `doc_id` 因 strict-match/flexible-extract 有两行，生成文本相同；去重后 baseline/candidate 各有 1319 个唯一题目且无空回答。`Question:` 字面文本出现在 19/16 个唯一回答中，抽样确认它可属于同题推理内容，不能当作 API stop reason；按解码文本重编码得到 ≥2048 token 的保守指示数为 12/11，也不是服务端 `finish_reason`。服务终止后 GPU 0–3 均回落到 2 MiB / 0%，端口 8000 释放。

**④ 全程**：`journal.md` 随手记（命令 + 依据 + 坑），`aqa log-action` 记录环境变更/额外命令。后续发布请求已准备全英文 `README.md`；用户明确授权将当前私有仓库改为公开后，18 个模型文件已上传并完成远端清单、大小、哈希及 README 字节核验。

### ⑤ 回溯 → `report.md` + `gate.json` + `run_manifest.json`

| 子任务 | 内容 |
|---|---|
| 5.1 | `aqa compare baseline eval` + 逐任务差值与 `MXFP8/BF16×100%`；按 gate 0.02 出 GO/NO-GO，并注明 BF16-KV vs FP8-KV 为部署协议对比 |
| 5.2 | sanity check 证据（体积、dtype 计数、键完备性、kv scale） |
| 5.3 | 经验路由表（无空行）：本施工卡回填结果 → `cookbooks/`；量化 pin → `recipes/quant/nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json`；Mamba 混合架构/FP8 KV 经验 → skills/AGENTS.md；补丁（如有）→ `patches/` |
| 5.4 | 刷新 local_fs 镜像；`aqa inspect <run_dir>` + `aqa artifacts check <run_dir>` |

**实测与验收**：`report.md` 的 gate 结论与 `gate.json` 一致（`GO` / `all gate checks passed`）；`aqa inspect`、`aqa artifacts check` 均为 `OK`；recipe JSON 格式有效；经验路由表各行均填写；baseline/candidate 两个 local_fs 镜像已刷新并与源准确率文件一致；GPU 0–3 空闲、端口 8000 释放。仓库全量 pytest 未能完成收集：既有 `tests/test_export_repair.py` 导入缺失的 `aqa.quantize.repair_diffusion_export`，`tests/test_hooks.py` 导入缺失的 `aqa.registry.HOOKS`。聚焦 manifest/sink/action tests 有 23 项通过、3 项因既有 worktree 中 CLI/knowledge/local-path org API 不一致而失败；本任务没有改动相应源码或测试，也未修复无关问题。**不 git commit**（交给你）。

## 4. 风险与回退

| # | 风险 | 探测点 | 回退 | 本次结果 |
|---|---|---|---|---|
| R1 | AutoRound 静态 KV 标定新 scale 留在 meta | ④a 日志与 12 个 scale 的 header | 使用 `AR_DISABLE_META_LOAD=1`，不改依赖源码 | 已触发并绕过；12 个有限、非零 F32 scale 导出并在 vLLM 加载 |
| R2 | SM120 上 MXFP8 MoE 非门控 kernel/backend 不兼容 | ④a/④c 生成与服务日志 | 以 smoke 结果选择显式 backend | 自动 Marlin 报 `Invalid thread config`；显式 Humming 的 smoke 与全量评估通过 |
| R3 | dense MXFP8 Linear（Mamba/shared expert）计算异常 | ④c 全量任务 | 失败时回退 A 档须先征得用户同意 | 本次 TP2×DP2 服务与四任务均通过，无需回退 |
| R4 | FP8 KV scale 映射失败或静默用 1.0 | ④c scale / 启动日志 | 停止发布并修 loader 或征询 | 12 个 scale 有效；未见 1.0 fallback 告警 |
| R5 | remote code 与原生模型实现冲突导致导出缺键或加载失败 | ④a header 与服务加载 | 使用已核验的原生建模路径；不得忽略缺键 | attempt04 的 ignore 前缀错配已由 attempt05 修正；AQA remote config setter 冲突以原生服务路径绕过，失败记录保留 |
| R6 | 混合模型 prompt logprobs 与 prefix caching 不兼容 | ④b/④c loglikelihood 请求 | 必要时关闭 prefix caching 后重试 | 本次四任务完整通过，未观察到需关闭 prefix caching 的错误 |
| R7 | thinking-on 与 2048 token 生成上限影响 GSM8K | GSM8K 唯一题目抽样与日志 | baseline/候选保持同口径，并说明指标边界 | 两侧均有少量 ≥2048 token 的保守长度指示；无服务端 finish_reason，未据此虚报 stop/truncation |
| R8 | 共享机器 GPU 被占用 | 每阶段前后 `nvidia-smi` | 等待，不抢卡、不用 4–7 | 只用 GPU 0–3；GPU 4–6 保持他人负载未触碰；收尾时 GPU 0–3 均空闲 |
| R9 | INCModel4 私有 HF 存储额度不足以接收约 33.6 GB checkpoint | 首次 HF commit 返回 HTTP 400 `Private storage limit reached` | 私有发布需扩容或选择有额度的目标；公开发布必须先取得明确授权 | 用户明确授权公开；commit `68348e84fa9cb86dde6e20eb1c364e0aacff0317` 成功，18 个本地文件与远端大小/哈希一致 |

## 5. 时间与资源预估

环境 20–40 min；量化 30–60 min（含标定）；冒烟 15 min；baseline 与候选各 3–5 h（MMLU 初始化 + 1319 条 thinking 生成）。
总计约 **8–12 h**。磁盘余量 446 GB，量化输出约 34 GB，充足。
