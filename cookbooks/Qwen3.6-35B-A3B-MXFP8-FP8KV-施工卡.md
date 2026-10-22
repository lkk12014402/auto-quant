# Qwen3.6-35B-A3B MXFP8 + FP8 KV 施工卡

**状态：完成（未提交 Git）。** ①结构分析、②环境、③配置、④量化与四项评估、⑤回溯验收均已完成；MXFP8 导出验收通过，候选 gate 为 **GO**；local_fs 镜像已刷新，`aqa inspect` 与 `aqa artifacts check` 均通过。

## 目标与硬约束

- 源模型：`/models/Qwen3.6-35B-A3B`。
- AutoRound：MXFP8、RTN（`method=autoround`, `algorithm=rtn`, `iters=0`）、llm_compressor；必须实际调用 `--static_kv_dtype fp8`，不启用 static FP8 attention。
- 精度范围：仅量化 fused routed experts；保留 attention、linear attention、shared expert/router、MTP、vision、embedding/head。
- 评估：仓库 `lm_eval` 的 GSM8K、PIQA、MMLU、HellaSwag；batch 32 起步，只有 OOM 才按 16→8→4→2→1 递减；vLLM `max-model-len=131072`。128K 是上限，不代表这四项任务构成 128K 压测。
- 设备：只能使用物理 GPU 0–3。量化显式 `device_map=0,1,2`；评估服务用 0–3、TP2/DP2/EP、PP1；每个 GPU 阶段开始前 `nvidia-smi` 检查，忙则等待，禁止使用 GPU 4+。
- 环境：复用 Conda `autoround-test`，uv 负责安装/冻结；本次不发布模型、不上传 HF、不提交 Git。

## 结构分析与精度范围

`analysis.md` 与 `logs/index_header_audit.json` 是本次分析真源。配置为 `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration`，与 Qwen3.5-35B-A3B 同架构；26 个 shard、1045 个 index/header key 完全匹配，全部 BF16，共 35,951,822,704 个元素、71,903,645,408 bytes。

| 模块 | 精度 | Header keys / 元素 |
|---|---|---:|
| Routed experts `mlp.experts`（融合张量） | **MXFP8** | 80 / 32,212,254,720 |
| `lm_head`、`embed_tokens` | BF16 | 2 / 1,017,118,720 |
| Vision | BF16 | 333 / 446,571,248 |
| Linear attention | BF16 | 270 / 1,011,553,920 |
| Shared experts/gate | BF16 | 160 / 125,911,040 |
| Full attention | BF16 | 60 / 272,634,880 |
| Router | BF16 | 40 / 20,971,520 |
| MTP（含其 attention/expert） | BF16 | 19 / 844,640,768 |
| Norm 参数 | BF16 / AutoRound 自动跳过 | 81 / 165,888 |

ignore 清单（按 index 子串核对；不使用裸 `gate` 或反斜杠）：

```text
lm_head,embed_tokens,visual,linear_attn,shared_expert,self_attn,mtp,mlp.gate
```

此方案较旧 Qwen3.5 v2 recipe 更保守，依 `AGENTS.md` §3.1 保护敏感模块。只量化 32.21B routed-expert 参数，预计导出约 40–43 GB 十进制（约 37–40 GiB）；若接近 BF16 源体积或 expert dtype 未量化，停止并排查静默失效。

## 环境、版本与执行路径

- 复用环境：`/home/xuehaosu/miniforge3/envs/autoround-test`；自检/冻结文件为 `setup_venv.sh`、`logs/environment_self_check.json`、`requirements.txt`。当前版本：Python 3.12.14、PyTorch 2.13.0+cu130、Transformers 5.17.0、vLLM 0.29.0、AutoRound 0.16.0.dev193+g22d6e28c、lm-eval 0.4.13、ninja 1.13.2、uv 0.13.0。AutoConfig、vLLM 主模型/MTP registry 与四个 task recipe 自检通过。
- GitHub 最新 vLLM 查询为 0.31.0；本次保留已获准复用且已命中该架构的 0.29.0，不升级共享环境或改变 torch 栈。模型卡未提供专属 pin；此为明确记录的版本偏离。
- lm-eval 自带 vLLM adapter 不支持 DP>1 与 EP 同用；按当前 evaluate skill 使用独立 vLLM OpenAI-compatible server + AQA `local_completions`。AQA full pipeline 无法在 baseline/candidate 之间重启外部 server，因此：`aqa run run_quant_only.yaml` 负责量化，独立 server/AQA eval wrapper 分别产生 baseline 与 candidate 结果，最后用 AQA gate/manifest API 汇总。主 `run.yaml` 是完整意图配置，阶段派生配置与实际命令均保留。
- local-completions 设 `tokenized_requests=true`：lm-eval 0.4.13 非 tokenized chat 请求要求 batch=1且不检查长度；tokenized 请求支持 batch 32，并在客户端限制 prompt+generation 不超过 131072，vLLM server 另设相同上限。
- baseline 首次启动在 FlashInfer JIT warmup 时失败：环境中的 `ninja` 1.13.2 已安装，但 worker 的 `PATH` 未包含 Conda env `bin/`。`logs/start_vllm_server.sh` 已导出该路径，`logs/run_quantization.sh` 与 `logs/run_eval.sh` 会激活同一 Conda 环境；首次服务故障与修复经过见 run 目录 `journal.md`、`logs/vllm_baseline.log`。
- GSM8K 首轮 baseline 的样本审计发现 recipe 默认 `Question:` stop 截断 thinking 输出（API `stop_reason=Question:`）；同 prompt 去掉该 stop 后返回 `#### 18`。canonical `run.yaml` 与 baseline YAML 对生成任务仅覆盖 `gen_kwargs.until: "<|im_end|>"`，保留 5-shot、thinking-on、2048-token 预算与其他 recipe 设置；baseline/candidate 共用该覆盖。首轮部分结果已存 `logs/baseline_attempt1_invalid/`。修正后 GSM8K baseline 完成：1319 样本，strict exact-match 0.9568、flexible 0.7998，batch32 无 OOM。
- 四项 BF16 baseline 均已完成（`baseline/accuracy.json`）：GSM8K strict exact-match **0.9568** / flexible **0.7998**（1319 样本），PIQA **0.8237**（1838），MMLU **0.8366**（14042），HellaSwag **0.6349**（10042）；batch32、上下文上限 131072，无 OOM。PIQA/MMLU/HellaSwag 同组评估前的 MMLU 57-subject 初始化耗时约 13 分 43 秒；缓存已落本机。BF16 server 已停止，GPU 0–3 当前空闲。逐步记录见 run 目录 `journal.md`。
- AutoRound 量化已完成，AQA 状态 `done`。实际导出目录：`quantized/Qwen3.6-35B-A3B-mxfp-w8g32/`；候选 YAML 已改指该目录并通过 `aqa validate`。导出目录总文件 40,733,874,673 bytes（40.73 GB / 37.94 GiB），源 BF16 权重 71,903,645,408 bytes；详细 header 审计见 `logs/quantized_header_audit.json`。
- 量化验收通过：63,191 个 safetensors index/header keys 完全匹配；30,720 个 routed-expert `F8_E4M3` 权重（32,212,254,720 elements）及 30,720 个 `U8` scales（1,006,632,960 elements），其余 1,731 个 BF16 tensors（3,739,567,984 elements）和 20 个 F32 tensors。`quantization_config.json` 为 `mxfp8-quantized`/`compressed`，KV scheme 为 8-bit float、static；10 个 full-attention 层有 20 个 F32 k/v scales。AutoRound 日志 `quantized 0/461` 仅统计并报告被 ignore 的标准 Linear 模块，不包含 fused routed-expert 专用导出路径；实际 FP8 expert headers 证明量化成功。
- 候选 vLLM 已就绪：`/health`=200，`/v1/models` 确认 `max_model_len=131072`；log 识别 `Qwen3_5MoeForConditionalGeneration` 并报告使用 FP8 KV cache。GPU 0–3 为本次服务使用（约 30 GiB/卡），未使用其他 GPU。`logs/run_eval.sh --config eval_candidate.yaml --out candidate` 以 batch32 执行，未降 batch。候选 GSM8K：strict exact-match **0.9575436**（baseline 0.9568），flexible **0.7930250**（baseline 0.7998），1319 样本；2638 个 filter rows 中 10 个 invalid（baseline 11）。
- 四项候选评估已完成，batch32、无 OOM；与 baseline 样本数逐项相同：GSM8K strict **0.957544**（差 **+0.000758**，flexible 0.793025 vs 0.799848），PIQA **0.824810**（**+0.001088**），MMLU **0.837844**（**+0.001282**），HellaSwag **0.633340**（**-0.001593**）。四任务 mean_score **0.813384** vs **0.813000**（+0.000384）；最大主指标下降 0.001593，小于 0.02 gate，`gate.json` 判定 **GO**。完整解释、限制与路由表见 run 目录 `report.md`。
- 评估服务已通过 `kill -TERM` 退出，GPU 0–3 均恢复空闲（显存约 2 MiB）。
- 已刷新 local_fs baseline/candidate artifact 镜像；两侧 accuracy 与 canonical `baseline/`、`eval/` 一致，候选 artifact status=`Finished`。`aqa inspect` 与 `aqa artifacts check` 均为 OK；最终回溯报告、manifest、gate 与路由表见 run 目录。
- Baseline 使用 BF16 KV（`auto`）；候选使用 FP8 KV 并读 AutoRound 静态 scales。Gate 表示完整部署配置对 BF16 reference 的变化，不是单独隔离权重量化误差。
- `thinking.py` 的 `auto/both` helper 当前未接入 eval runner/pipeline；本次明确使用 Qwen chat template 默认 thinking-on 模式，不声称执行了双模式。四任务范围不变。
- 直连网络当前可用；若发生网络故障，按用户指定启用 `http_proxy` 与 `https_proxy`，并记入 journal/action log。

## 五步施工与验收

| 步骤 | 状态 | 产物 | 验收标准 |
|---|---|---|---|
| ① 结构分析与量化方案 | **完成** | `analysis.md`、`logs/index_header_audit.py`、`logs/index_header_audit.json` | 26 shards 全部 header/index 对齐；模块表、ignore 命中、精度范围、体积预期和偏离均记录。 |
| ② 环境 | **完成** | `setup_venv.sh`、`requirements.txt`、`logs/environment_self_check.json` | Conda 环境、uv 可用；Transformers config 与 vLLM 主模型/MTP registry 命中；四 task 存在；自检后 freeze，未升级推理栈。 |
| ③ YAML 定稿 | **完成** | `run.yaml`、`run_quant_only.yaml`、`eval_baseline.yaml`；候选 eval YAML 在量化后写入 | 三份现有 YAML 的 `aqa validate` 通过；AutoRound 命令构造器确认 `MXFP8`、RTN、`--static_kv_dtype fp8`、`device_map=0,1,2`、不含 static-attention 参数；`aqa run --dry-run` 通过。 |
| ④ 量化与评估 | **完成** | `quantized/`、baseline/eval 结果目录、vLLM 与 lm-eval 日志、`journal.md`、`owner_actions.jsonl` | GPU 范围、实际 MXFP8 dtype/静态 KV scales、完整 header、baseline 与候选四项样本数/指标均已验收；FP8 KV、上下文上限 131072、batch32 无 OOM。首次服务故障、GSM8K stop 诊断、量化验收与逐任务结果见 journal/report。 |
| ⑤ 回溯与知识回流 | **完成** | `report.md`、`gate.json`、`run_manifest.json`、cookbook 更新 | 逐任务对比与 GO gate 一致；`aqa inspect` 和 `aqa artifacts check <run_dir>` 通过；local_fs 镜像刷新、经验路由表无空行；未提交 Git。 |

**运行目录：** `/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-AutoRound_20261013`
