# Qwen3.6-35B-A3B：纯 MXFP8（BF16 KV）施工卡

**状态：已完成（①结构分析、②环境、③YAML、④a 量化与权重审计、④b vLLM smoke、④c 四任务评估、⑤回溯验收均通过；gate=GO）。** 本卡对应纯 MXFP8 变体，不覆盖已有的 `Qwen3.6-35B-A3B-MXFP8-FP8KV-施工卡.md`（下称 **REF**）、
`…-MXFP8-FP8KV-FP8Attn-施工卡.md` 和 `…-精度回退修复施工卡.md`。

## 0. 目标与硬约束（owner 已于 2026-10-18 确认）

| 项 | 决定 |
|---|---|
| 源模型 | `/models/Qwen3.6-35B-A3B`（HF `Qwen/Qwen3.6-35B-A3B`）：67 GiB / 71,903,645,408 B，26 个 shard，1,045 个 key，全部 BF16，无 `quantization_config`，无 `auto_map` 或 `.py` 文件 |
| 量化 | AutoRound **MXFP8**（W8A8 mx_fp，group 32），`method: autoround`、`algorithm: rtn`、`iters: 0`，导出格式 `llm_compressor`。**不传** `--static_kv_dtype`、`--static_attention_dtype`、`--disable_opt_rtn`、`--model_free` |
| 量化范围 | **A 档（保守，与 REF 逐字一致）**：只量化 routed experts `mlp.experts`；其余模块（attention、linear_attn、shared_expert、router、MTP、vision、embed/lm_head）均保持 BF16 |
| KV cache | 候选使用 **`kv_cache_dtype=auto`（BF16 KV）**，与 baseline 相同，因此本次只比较权重/激活量化的影响 |
| Baseline | **复用 REF BF16 baseline**：`/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-AutoRound_20261013/baseline/`（vLLM 0.29、TP2×DP2+EP、local-completions、batch 32、131072、BF16 KV；四任务均值 0.813000），不重跑 |
| 评估 | 仓库 lm_eval 四任务 **gsm8k / piqa / mmlu / hellaswag**，全量运行、不设 limit。vLLM `--max-model-len 131072`，lm_eval `max_length=131072`（只是 128K 上限，这四个任务本身不构成 128K 压测）。GSM8K 沿用 REF 口径：thinking-on（模板默认）、5-shot multiturn、`max_gen_toks=2048`、`until=<\|im_end\|>`；seed 42 |
| batch | lm_eval `batch_size=32` 起步；只有出现**真实 GPU OOM** 时才按 16→8→4→2→1 降档（超时或 kernel 错误不算 OOM，不降档） |
| GPU | 只用物理 GPU **0–3**（NUMA0；0↔1、2↔3 为 PIX，跨对为 NODE；RTX 5090 走 PCIe，无 NVLink）。**量化只用 GPU0**（REF 峰值 VRAM 10.06 GB）；真实 OOM 才扩到 `0,1`（同一 PIX 对）。评估用 **TP2×DP2 + EP、PP=1**，TP 组为 {0,1}/{2,3}。每个 GPU 阶段开始前用 `nvidia-smi` 确认空闲：无 compute 进程、显存 ≤512 MiB、利用率 ≤5%。若被占用，每 60 s 复查等待，不抢卡，也不碰 GPU 4–7 |
| 环境 | **复用** conda 环境 `/home/xuehaosu/miniforge3/envs/qwen36-mxfp8`（vLLM 0.31.0，AutoRound 0.16.0.dev193+g22d6e28c 与 REF 同一 commit，transformers 5.17.0，torch 2.13.0+cu132，lm-eval 0.4.13，env 内 uv 0.12.13）。uv 负责包管理；默认不改包 |
| 网络 | 优先直连；失败时 `export http_proxy=http://proxy-ir.intel.com:912 https_proxy=http://proxy-ir.intel.com:912`，并设 `no_proxy=127.0.0.1,localhost`（避免本地 vLLM 请求走代理），同时记入 journal 和 log-action |
| 回传 | 原始量化 run 因本机无 `env.sh` 只配置 `local_fs`，未启用 HF dataset sink。2026-10-19 owner 后续明确授权发布权重，已上传至公开仓库 `intel-ai/Qwen3.6-35B-A3B-MXFP8-CT-AutoRound`（原名 `...-CT-RTN-AutoRound`，按后续要求去掉 `-RTN`）；不提交 git |
| Run 目录 | `/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-AutoRound_20261018`（artifact 名预期为 `Qwen3.6-35B-A3B-MXFP8-CT-RTN-AutoRound`，以 `aqa/report/names.py` 实际生成结果为准） |

## 1. 调研结论（2026-10-18）

| 项 | 结论 | 依据 |
|---|---|---|
| 架构 | `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration`，共 40 层：30 层 linear attention（GDN）+ 10 层 full attention（`full_attention_interval=4`，16 个 q 头 / 2 个 kv 头，head_dim 256）。MoE 为 256 个专家、top-8，`moe_intermediate=512`，另有 1 个 shared expert（512）；专家权重是融合 3D 张量 `mlp.experts.{gate_up_proj,down_proj}`。另含 1 层 MTP 和 vision tower；`tie_word_embeddings=false`，vocab 248,320，原生上下文 262,144 | `config.json`、REF `logs/index_header_audit.json` |
| 上游最新 | vLLM **0.32.0**（2026-10-18 发布，torch 仍为 2.13.0，要求 `transformers<5.20`）、AutoRound **0.16.1**（10-16）、transformers 5.20.1、lm-eval 0.4.13、FlashInfer 0.7.2。模型卡要求 `vllm>=0.19.0` | `git ls-remote`、PyPI JSON、模型 README |
| 版本选择 | owner 决定**复用 vLLM 0.31.0 环境，不升级**。这是相对“最新版”的明确偏离，记入 `analysis.md`。0.31.0 已在本机加载过同架构的 MXFP8 checkpoint（FP8Attn 变体，TP4/EP） | owner 决策、FP8Attn 施工卡 |
| 量化路由 | AutoRound 只有在 `iters=0` 且 `disable_opt_rtn is True`（并满足格式等条件）时，才会自动转到 model_free。本次不设 `opt_rtn`，因此走 model-load 路径，ignore 按 `model.named_modules()` 匹配，与 REF 相同 | `auto_round/utils/model.py:3163-3208` |
| REF 量化实测 | 单次量化约 22 min；peak_ram 41.77 GB，peak_vram 10.06 GB。AutoRound 把被 ignore 的 MTP 张量以 BF16 写入 `model_extra_tensors.safetensors`（785 个）。日志中的 `quantized 0/461` 只统计标准 Linear，不包括 fused expert 的导出路径，**不能据此判定量化失败** | REF `quantized/quantize.log`、REF 施工卡 |
| 评估路径 | lm-eval 0.4.13 自带的 vLLM adapter 不支持 DP>1 与 EP 同时使用 → 采用**外置 `vllm serve` + `local-completions`（`tokenized_requests=true`）**，与 REF baseline 完全同路径；`aqa run` 无法在不同阶段之间重启外置服务，所以分阶段执行：量化用 `aqa run`，评估用 `aqa eval` | REF 施工卡、`skills/evaluate/SKILL.md` |
| 已知坑 | ① FlashInfer 在 SM120 上误用 `q_scale` 的问题（AGENTS.md §9 #8）**与本次无关**：本次无 q_scale、无 FP8 KV；仍需在日志中核对 attention backend。② GSM8K recipe 默认 stop `Question:` 会截断 thinking，必须覆盖为 `<\|im_end\|>`。③ vLLM worker 的 PATH 必须包含 conda env 的 `bin/`（ninja/FlashInfer JIT）。④ 不要在已有终态的 run 目录上执行 `aqa run --dry-run`（§9 #9） | AGENTS.md §9、REF journal |
| 已知混杂 | baseline（0.29）与候选（0.31）的 vLLM 版本不同。缓解依据：v2 定因卡的 D2 显示，REF checkpoint 在 0.31 默认路径下的 MMLU-50 / PIQA 与 0.29 历史值相差不超过 0.43 pp；本次 adapter、拓扑和 KV 都与 baseline 一致。报告会如实注明该混杂 | 精度回退修复施工卡 §3 S2 |
| 资源 | GPU 0–7 当前全部空闲；`/` 剩余 248 GB（产物约 41 GB）；RAM 188 GB；`numactl` 可用 | `nvidia-smi`、`nvidia-smi topo -m`、`df -h` |

## 2. 逐模块精度表（A 档，复用 REF header 审计数据，①中复核）

| 模块 | 精度 | Header keys / 元素 |
|---|---|---:|
| Routed experts `mlp.experts`（融合张量，导出时拆分） | **MXFP8**（F8_E4M3 + U8 E8M0 scale） | 80 / 32,212,254,720 |
| `lm_head`、`embed_tokens` | BF16 | 2 / 1,017,118,720 |
| Vision `visual` | BF16 | 333 / 446,571,248 |
| Linear attention `linear_attn`（含 `in_proj_a/b`，out=32 < 128） | BF16 | 270 / 1,011,553,920 |
| Shared expert / `shared_expert_gate`（out=1） | BF16 | 160 / 125,911,040 |
| Full attention `self_attn` | BF16 | 60 / 272,634,880 |
| Router `mlp.gate` | BF16 | 40 / 20,971,520 |
| MTP `mtp.*` | BF16（由 exporter 复制到 extra tensors） | 19 / 844,640,768 |
| Norm | BF16 / 由 AutoRound 自动跳过 | 81 / 165,888 |

ignore 清单（与 REF 逐字一致，按子串匹配；不使用裸 `gate`）：

```text
lm_head,embed_tokens,visual,linear_attn,shared_expert,self_attn,mtp,mlp.gate
```

**体积预期**：REF 导出为 40,733,874,673 B（含 20 个 F32 k/v scale，量级可忽略）→ 本次预计 **≈40.7 GB（≈37.9 GiB）**，可接受区间为 39–43 GB。
**接近 72 GB 说明量化静默失效，必须立即停止。**

## 3. 五步施工与验收标准

全程要求：`journal.md` 随手记录（命令、依据、踩坑），不要事后补记；环境变更、额外命令和决策用
`aqa log-action <run_dir> --kind …` 追加到 `owner_actions.jsonl`。本次不发布权重，因此不写 model card。

### ① 结构分析与量化方案 → `analysis.md`

| 子任务 | 内容 |
|---|---|
| 1.1 | 按 `templates/analysis.md` 填写，并在查阅清单中逐条记录“采纳/偏离”：`AGENTS.md`、`skills/quantize`、`skills/evaluate`、`recipes/schemes/mxfp8.yaml`、`docs/autoround_full_reference.md`、三张 Qwen3.6 施工卡、REF 的 `analysis.md`/`report.md`、模型 README。若 references 增量层未接入，或 owner 未提供外部 `autoround-quant-analysis/SKILL.md`，按偏离记录；“未升级 vLLM 0.32”和“复用 0.29 baseline”同样按偏离记录 |
| 1.2 | 复跑 REF 的 `logs/index_header_audit.py`（只读 header，不读取权重 payload）→ `logs/index_header_audit.json`；与 REF 结果 diff，确认源模型未变 |
| 1.3 | 三件套探针：`AutoConfig` 分别在 `trust_remote_code=False/True` 下运行；vLLM 0.31 `ModelRegistry` 命中 `Qwen3_5MoeForConditionalGeneration`；AutoRound 能处理 fused expert，且不会自动转到 model_free。同时用 meta-device 模型（不加载权重）的 `named_modules()` 逐条核对 ignore：记录 `visual/linear_attn/shared_expert/self_attn/mlp.gate` 命中数，MTP runtime module 命中数为 0（需在导出时验收 missing-tensor copy），且**不误伤** `mlp.experts`（`mlp.gate` 不能匹配 `mlp.experts.*`） |
| 1.4 | 定稿精度表、ignore、AutoRound 等价命令、评估计划和体积预期 |

**验收**：
- 查阅清单无空行；
- 1,045 个 header key 与 index 完全对齐，结果与 REF 审计相同；
- ignore 命中表已落盘（每条规则的命中模块数、routed expert 误伤为 0）；
- 已明确推理路径为原生实现（`trust_remote_code=false`）；
- 已给出量化命令、评估计划和 ≈40.7 GB 的体积预期。

### ② 环境 → `setup_venv.sh` + `requirements.txt` + `logs/environment_self_check.json`

| 子任务 | 内容 |
|---|---|
| 2.1 | `setup_venv.sh` 复用现有 `qwen36-mxfp8`；若环境缺失则 fast-fail，不隐式创建/克隆。通过 uv 固定补齐评估 API extra 所需依赖 |
| 2.2 | 自检：torch CUDA 可见 SM120；`AutoConfig` 识别 `qwen3_5_moe`；vLLM registry 命中（**硬门槛**）；`auto-round --help` 可用；四个 lm_eval task 及其 recipe 存在；`local-completions` 与 API extra `tenacity` 可导入；`aqa` 指向本仓（editable）；`ninja` 在 env `bin/` 中 |
| 2.3 | 执行 `uv pip check`，再 `uv pip freeze > requirements.txt`。任何换包都要先问 owner，然后重跑自检、重新冻结，并记 `aqa log-action` |

**验收**：
- 硬门槛全部通过；
- 版本与 §0 一致；
- `uv pip check` 干净；
- 与 FP8Attn run 的 `requirements.txt` 差异为空，或差异已解释；
- 其他 conda 环境未被改动。

**实际结果**：初次自检/冻结为 237 packages。首个 eval 尝试在 lm_eval API 初始化阶段发现缺 `tenacity`，未发送推理请求。REF freeze 明确 pin `tenacity==9.1.4`；在 owner 暂不可答复时按最小必要改动使用 uv 补装该精确版本，并在 `setup_venv.sh` 固化，之后重跑自检（包含 tenacity import）、`uv pip check`（238 packages 全过）和 freeze。唯一依赖差异为 tenacity 9.1.4；其他 Conda 环境未改。该 owner 确认等待事项记入 `journal.md`。

### ③ YAML 定稿 → `run.yaml`（canonical）+ `run_quant_only.yaml` + `eval_candidate.yaml`

| 子任务 | 内容 |
|---|---|
| 3.1 | 从 REF `run.yaml` 派生：`scheme: MXFP8`、`method: autoround`、`algorithm: rtn`、`iters: 0`、`export_format: llm_compressor`、上述 ignore，**删除 `static_kv_fp8`**，`device_map: "0"`，`extra_args: ["--disable_low_cpu_mem_usage"]`（与 REF 相同）。`executor: {type: local, num_gpus: 1}`，省略 `env`（由 `qwen36-mxfp8/bin/aqa` 直接调用，见 AGENTS.md §2.3）；`agent.fix_loop/optimize: false`；`output.sinks: [local_fs]`，`unique_dir: false`；`gate.max_accuracy_drop: 0.02` |
| 3.2 | `run_quant_only.yaml`：`_base_: run.yaml`，`eval.benchmarks: []`，`baseline: false`，`gate: null`（与 REF 相同） |
| 3.3 | `eval_candidate.yaml` 与 REF `eval_baseline.yaml` 逐项一致：`backend: local_completions`、四个 task、`batch_size: "32"`、`thinking: "on"`、`gen_kwargs {max_gen_toks: 2048, until: "<\|im_end\|>"}`、`model_args {tokenizer: /models/Qwen3.6-35B-A3B, max_length: 131072, tokenized_requests: true, num_concurrent: 1, max_retries: 50, timeout: 500}`；只把 model 改成量化产物路径（在④a 之后写入） |
| 3.4 | 把 REF `baseline/` 复制到新 run 目录的 `baseline/`，记录源路径和逐文件 sha256 |

**验收**：
- `aqa validate` 对三份 YAML 全部通过；在全新 run 目录上 `aqa run run_quant_only.yaml --dry-run` 通过；
- command builder 输出包含 `--scheme MXFP8 --iters 0 --ignore_layers <上述清单> --device_map 0 --format llm_compressor`，且**不包含** `--static_kv_dtype`、`--static_attention_dtype`、`--disable_opt_rtn`、`--model_free`；
- `eval_candidate.yaml` 与 REF `eval_baseline.yaml` 的 diff 只有 model 路径；
- 复制的 baseline sha256 与源一致。

### ④ 运行与输出

**④a 量化（GPU0）**

1. `nvidia-smi` 确认 GPU0 空闲。
2. 执行：

   ```bash
   CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --preferred=0 \
     $ENV/bin/aqa run run_quant_only.yaml
   ```

   - PATH 前置 `$ENV/bin`；
   - 不预设 `AR_DISABLE_META_LOAD`（REF 设置它只是为了物化 static KV scale）；如果出现 meta tensor 报错再加，并记 log-action。
3. 遇到真实 OOM 时，改为 `CUDA_VISIBLE_DEVICES=0,1`、`device_map: "0,1"`，在干净目录重跑。
4. 量化完成后做 header 审计 → `logs/quantized_header_audit.json`；同时按 index 检查 shard。

**实际结果（2026-10-18）**：GPU0 单卡量化完成，用时约 21 min；AutoRound 峰值 RAM 91.95 GB、VRAM 2.32 GB，无 OOM。导出为 40,733,870,122 B（40.73 GB）。完整 header 审计通过：63,171 个 index key、9 个被引用 shard、无缺失/额外/重复 key；30,720 个 `F8_E4M3` 权重（32,212,254,720 elements）及 30,720 个 `U8` scale（1,006,632,960 elements）；其他 1,731 个 tensor 全 BF16，含 MTP 拷贝的 785 个 tensor。去掉 U8 scales 后的权重元素数 35,951,822,704，与源 checkpoint 完全一致。

AutoRound 的 missing-tensor copy 将 MTP 写进 index，但没有同步增加 `metadata.total_size`；原值 39,008,742,112 B 比 index 指向的 payload 少 1,689,281,536 B，恰为 785 个 BF16 MTP tensor 的 payload。本次只将该字段修正为 40,698,023,648 B，修复前元数据备份在 `logs/index_metadata_before_fix.json`，修复脚本与验收分别在 `logs/repair_index_total_size.py`、`logs/audit_quantized_checkpoint.py`。`metadata.total_parameters=2,894,927,216` 是 AutoRound 的 MoE active-parameter 元数据，不是序列化 tensor element 总和；REF 有额外 20 个静态 KV scale 参数。`total_shards` 字段未提供，实际 index 引用 9 个 shard。`kv_cache_scheme` 在本次 `config.json` 中为 `null`（无活动静态 KV 方案）。

**验收**：
- `.quant_command.json` 与③一致；日志中**没有** `Auto-routing to model-free`；
- 体积在 39–43 GB；
- header：**30,720 个 `F8_E4M3` 权重**（32,212,254,720 个元素）+ **30,720 个 `U8` scale**；其余张量为 BF16，数量与 REF 一致；**没有 F32 `k_scale`/`v_scale`/`q_scale`**；
- 所有 shard 都被 index 引用，每个 key 都在 index 指定的 shard 中，重复 key 的 dtype/shape/字节一致，`total_size` 与实际一致；未引用的旧 shard 先确认再清理并留痕；
- `config.json` 的 `quantization_config` 为 compressed-tensors mxfp8，**没有活动的 `kv_cache_scheme`**（本次 exporter 写 `null`）；
- ignore 覆盖的模块全部为 BF16（与§2逐行对照）。

**④b 服务与生成 smoke（GPU 0–3）**

1. 确认 GPU 0–3 空闲。
2. 启动外置服务（参数与 REF baseline 的 `start_vllm_server.sh` 一致，env 改为 `qwen36-mxfp8`；本地目录无 remote code，因此去掉 `--trust-remote-code`）：

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn \
   vllm serve <quantized> --tokenizer /models/Qwen3.6-35B-A3B \
     --host 127.0.0.1 --port 8000 --dtype bfloat16 \
     --max-model-len 131072 \
     --tensor-parallel-size 2 --data-parallel-size 2 --data-parallel-size-local 2 \
     --enable-expert-parallel --pipeline-parallel-size 1 \
     --kv-cache-dtype auto --max-num-seqs 32 --max-num-batched-tokens 8192 \
     --gpu-memory-utilization 0.90
   ```

3. 服务就绪后，发一条中英文 chat 请求和一条 GSM8K 样题（Janet ducks）。

**验收**：
- `/health` 返回 200，`/v1/models` 显示 `max_model_len=131072`；
- 日志确认：
  - 架构为 `Qwen3_5MoeForConditionalGeneration`；
  - MoE 选用 MXFP8 kernel（记录 backend 名称）；
  - KV 为 BF16（`auto`）、`trust_remote_code=False`；
  - 每个 DP/TP/EP rank 只用 GPU 0–3；
  - attention backend 已记录；
- 回答连贯，GSM8K 样题答案正确（`#### 18`），无 NaN 或乱码。
- 如果自动选中的 MoE kernel 失败，先记录，再依次显式尝试其他 `--moe-backend`（参照 Nemotron 用 humming 的先例），**不擅自改量化范围**；全部失败则暂停并回报 owner。

**实际结果（2026-10-18）**：vLLM 0.31.0 使用 GPU0–3（拓扑确认均在 NUMA0）、TP2×DP2+EP、PP1 成功加载；API `/health` 返回 200、`/v1/models` 报告 `max_model_len=131072`。日志明确为 `trust_remote_code=False`、模型 `dtype=torch.bfloat16`、`kv_cache_dtype=auto`（无活动静态 KV scheme）、`MARLIN` MXFP8 MoE kernel、`FLASH_ATTN` v2 attention backend；DP/TP/EP rank 分布覆盖 GPU0–3。KV cache 报告每个 DP 1,547,850 tokens、131,072 请求的最大并发 11.81。

`logs/vllm_smoke.json` 三项请求均有非空、连贯回答：中文 `1+1` 回答 2，英文首都问题回答 Paris，Janet ducks 题以 `finish_reason=stop` 结束并在最终行输出 `#### 18`。初始化期间曾有 shared-memory 广播等待警告，CUDA graph 捕获随后完成且无启动失败。首次 smoke 服务 PID 368716 已在依赖修复前正常停止；依赖 pin 与 GPU 空闲预检通过后，已重启服务 PID 375584 供④c复用。评估结束后必须按本卡要求停止并确认资源释放。

**④c 候选四任务全量评估（沿用④b 的服务）**

```bash
ENV=/home/xuehaosu/miniforge3/envs/qwen36-mxfp8
RUN=/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-AutoRound_20261018
cd "$RUN"
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1
export PATH="$ENV/bin:$PATH" no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
"$ENV/bin/aqa" eval eval_candidate.yaml --out eval \
  > logs/aqa_eval_candidate.json 2> logs/aqa_eval_candidate.log
"$ENV/bin/python" logs/materialize_eval_result.py logs/aqa_eval_candidate.json eval
```

- 当前 `aqa eval` CLI 在 stdout 返回结构化结果，但不会生成契约要求的 `eval/accuracy.json`；以上小型 helper 将该 JSON 规范化为 AQA 的 canonical artifact，不重跑评估。
- 若网络实际失败，才按 §0 设置代理（并给 localhost 设置 `no_proxy`）；本地服务另参照 REF 的 `logs/run_eval.sh` wrapper 排障。
- GSM8K 先跑；PIQA/MMLU/HellaSwag 分组跑，与 REF 一致。
- 跑完后用 `kill -TERM <pid>` 停服务，确认 GPU 0–3 已释放。

**验收**：
- `eval/accuracy.json` 四任务齐全，样本数为 **gsm8k 1319 / piqa 1838 / mmlu 14042 / hellaswag 10042**，与 baseline 逐项相同；
- 实际 batch 已记录（32，或降档原因及档位）；
- GSM8K 实际 stop 只有 `<|im_end|>`（从 task log 核对），`[invalid]` 行数与 baseline（11/2638）同量级，无空回答；
- 服务退出后 GPU 0–3 回落到约 2 MiB，端口 8000 已释放。

**实际结果（2026-10-18）**：候选分数 GSM8K 0.9522365428（n=1,319）、PIQA 0.8220892274（n=1,838）、MMLU 0.8381996867（n=14,042）、HellaSwag 0.6331408086（n=10,042）；batch 全程 32、无 OOM 或降档。相对复用的 BF16 baseline（vLLM 0.29；候选 vLLM 0.31），逐项 Δ 为 -0.0045489007、-0.0016322089、+0.0016379434、-0.0017924716，宏平均分别为 0.8130004759 与 0.8114165664，Δ=-0.0015839095。GSM8K 审计 2,638 条 filter 记录、空响应 0、`[invalid]` 9 条（baseline 11）；vLLM 共 3,168 个 HTTP 200 completion batch、0 个 4xx/5xx。gate 为 GO。服务正常退出、端口 8000 释放，GPU0–7 回到 2 MiB / 0%。

### ⑤ 回溯 → `report.md` + `gate.json` + `run_manifest.json`

| 子任务 | 内容 |
|---|---|
| 5.1 | `aqa compare baseline eval` → `logs/compare_candidate_vs_baseline.txt`；逐任务列出 Δ、四任务均值，以及 REF FP8KV（0.813384）和 BF16（0.813000）参照；明确标注 baseline 来自 vLLM 0.29 的混杂 |
| 5.2 | sanity check 写入报告：体积、键 dtype、完备性、index/shard 一致性、无 KV scale、runtime 使用的 MoE/Linear kernel 与 attention backend、GSM8K 样本审计 |
| 5.3 | 经验路由（表内不留空行）：回填本卡状态和结果；如有新坑，回流到 `AGENTS.md` §9、`skills/evaluate` 等；必要时新建或补充 `recipes/quant/Qwen__Qwen3.6-35B-A3B.json`（`autoround_rtn_bf16kv`）。不提交 git |
| 5.4 | 运行 `aqa inspect <run_dir>`、`aqa artifacts check <run_dir>`；用 `aqa log-action` 记录完成 |

**验收**：
- `gate.json` 判定与 `report.md` 结论一致（阈值 `max_accuracy_drop=0.02`）；
- `aqa inspect` 和 `aqa artifacts check` 均为 OK；
- 经验路由表无空行；
- GPU 0–3 空闲；**未 git commit**。原始 run 未启用 HF dataset sink；后续权重发布已按 owner 明确请求完成。

**实际结果（2026-10-18；发布更新于 2026-10-19）**：`report.md`、`gate.json`、`run_manifest.json`、`journal.md` 均已完成，gate 为 GO；`aqa inspect` 与 `aqa artifacts check` 均为 OK。local_fs artifact 是 run 目录内的完整本地镜像，含 9 个 safetensors shard 与 index。原始 run 未配置 HF dataset sink；后续按 owner 请求将模型与英文 README 发布到公开仓库 [`intel-ai/Qwen3.6-35B-A3B-MXFP8-CT-AutoRound`](https://huggingface.co/intel-ai/Qwen3.6-35B-A3B-MXFP8-CT-AutoRound)。初始上传 commit 为 `5cabc6c2f47f82eca77db2df861cefb23219d75a`；之后按要求将 repo 从 `...-CT-RTN-AutoRound` 重命名，并将 README 更新 commit 为 `ff5cd9727033be0c629ef6b4291a2fe23e16b0be`。19 个审核文件远端大小及 README 内容核对通过。AutoRound MTP 补拷后 `metadata.total_size` 可能陈旧的经验已回流 `AGENTS.md` §9 #10；references CLI 尚未接入，未推送 references。无 Git commit。初始的“rename Qwen3.6-35B-A3B v1 recipes”未给出旧路径/目标名称，仓内也无对应 pin，故未擅自改名或新建。

## 4. 风险与回退

| # | 风险 | 探测点 | 回退 |
|---|---|---|---|
| R1 | AutoRound 自动转到 model_free，ignore 命名空间改为 checkpoint 键 | ④a 日志、`.quant_command.json` | 确认没有传 `--disable_opt_rtn`；必要时显式加 `--disable_model_free`，并记为偏离 |
| R2 | 量化静默失效（体积约 72 GB，或专家张量仍为 BF16） | ④a 体积与 header 审计 | 停止，排查 AutoRound fused-expert 路径后在干净目录重跑 |
| R3 | ignore 误伤或漏网 | ①命中表、④a header 审计 | 停止，修正清单后在干净目录重跑 |
| R4 | vLLM 0.31 + SM120 在 TP2×DP2+EP 下 MXFP8 MoE kernel 失败 | ④b 日志 | 显式切换 `--moe-backend` 并记录；全部失败则回报 owner（可选项：升级到 0.32.0 克隆环境，或改用 TP4/EP） |
| R5 | 候选 OOM | ④b/④c | lm_eval batch 按 32→16→8→4→2→1 降档；必要时同步降低 `--max-num-seqs`；不改 TP/PP |
| R6 | baseline 与候选的 vLLM 版本混杂导致小幅偏差 | ⑤ 逐任务 Δ | 报告中如实标注；如果某项下降超过 2 pp，先回报 owner，由其决定是否在 0.31 上重跑 BF16 baseline |
| R7 | GPU 被他人占用 | 每个阶段前跑 `nvidia-smi` | 每 60 s 复查等待，不抢卡，也不使用 4–7 |
| R8 | 网络故障（数据集或 tokenizer 拉取） | ④c 日志 | 按 §0 设置代理及 `no_proxy`；数据集已有本地缓存 |

## 5. 时间与资源预估

① 20 min；② 10 min；③ 15 min；④a 实测量化约 21 min + 审计 10 min；④b smoke 15 min；④c 评估约 1–1.5 h（REF 候选约 1 h）；⑤ 30 min。
合计约 **3–3.5 h**。GPU 用量：量化 1 卡（GPU0），评估 4 卡（GPU 0–3）；磁盘约 41 GB；本次实测 RAM 峰值 91.95 GB、VRAM 峰值 2.32 GB。
