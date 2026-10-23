# Gemma-4-26B-A4B：MXFP8 + FP8 KV + FP8 Attention 施工卡

**状态：部分完成，完整评估受运行时兼容性阻断。** 用户后续明确要求执行本卡；结构分析、环境准备、MXFP8 量化和 AQA 单样本 smoke 已完成。vLLM 最新版不支持 AutoRound 独立静态 attention 元数据；本次 vLLM 0.29.0 smoke 还显示导出的 K/V 与 Q scale 未加载。未运行完整四任务、baseline 或 gate，不把单样本分数作为质量结论。

## 目标与硬约束

- 源模型：`/models/gemma-4-26B-A4B`（HF id：`google/gemma-4-26B-A4B`）。
- 量化：AutoRound `MXFP8`，同时传入 `--static_kv_dtype fp8` 与 `--static_attention_dtype fp8`；经 AQA 的 `run.yaml` 执行，不徒手绕过 AQA 拼接实际量化命令。
- 评估：仓库 `lm_eval` 的 `piqa`、`mmlu`、`hellaswag`、`gsm8k` 四项；BF16 baseline 和量化模型评估都配置 `kv_cache_dtype=fp8`；vLLM `max_model_len=131072`。
- 设备：NVIDIA RTX 5090 PCIe 卡，无 NVLink。只使用完成任务所需的空闲卡；卡忙则等待，不与其他用户争抢。
- 默认只落本地 `local_fs`，不上传权重或评估结果；不提交 Git。

## 调研结果与施工依据

### 模型与量化范围

- 本机模型目录存在，源 safetensors 为两个分片，总计 `51,611,872,412` 字节。
- `config.json`：`model_type=gemma4`、`Gemma4ForConditionalGeneration`、BF16、`tie_word_embeddings=true`；文本部分 30 层、128 个 routed experts、top-8、25 个 sliding-attention 层和 5 个 full-attention 层；模型配置支持 256K。
- 仅读取 safetensors 文件头，未读取权重 payload：index 的 1,013 个 key 全部与两个分片 header 对上，全部为 BF16；含 60 个融合 expert 张量、90 个 router 张量、337 个 self-attention 张量、355 个 vision tower 张量。
- 既有 Gemma recipe 的保守范围是量化 routed experts 与 checkpoint 中存在的 self-attention 投影；保留 `router,vision_tower,embed_vision,embed_tokens,mlp`。`multi_modal_projector` 不在此 checkpoint 中，不加入 ignore 清单。执行前仍须再次将最终清单逐项与 index/header 核对。
- 参考 MXFP8 + FP8 KV 模型卡的导出体积为 `28,412,087,908` 字节；本次加 FP8 attention 后以约 `28–30 GB` 作为体积 sanity-check 参考，不把 KV-only 产物的精确体积当作新方案的硬等值。

### 环境、版本和硬件

| 项目 | 已核实情况 | 本计划选择 |
|---|---|---|
| 环境管理 | `conda 26.7.2`、`mamba 2.9.0` 可用；已有 `autoround-test` Conda 环境 | 优先复用该环境；不新建、不升级，除非后续自检证明有缺项 |
| 包管理 | 目标环境内 `uv 0.12.13` 可执行；base 环境未激活时 `uv` 不在 PATH | 激活/指定 `autoround-test` 后用 `uv pip` 管理并冻结依赖 |
| 已装量化/评估栈 | AutoRound `0.16.0.dev193+g22d6e28c`、Transformers `5.17.0`、PyTorch `2.13.0+cu130`、vLLM `0.29.0`、FlashInfer `0.7.0`、lm-eval `0.4.13`、compressed-tensors `0.17.0` | vLLM `0.29.0` 是所给模型卡实际使用的模型指定版本；当前环境已满足，不盲目替换为最新版 |
| 最新 vLLM 对照 | 上游最新 release 为 `v0.30.0`；所给 Gemma 量化卡记录的是 `v0.29.0` | 先按模型卡版本施工；任何换版须先在隔离/克隆环境验证兼容性并重新冻结 |
| GPU | 8 张 RTX 5090，每张约 32 GiB；检查时 GPU 0–7 均为 0% 利用率、约 2 MiB 显存占用，compute-app 查询为空 | 首选物理 GPU `1,2,3`，同属 NUMA 0；进程内映射为 `0,1,2`，量化 `device_map=0,1,2`，评估 TP=1/PP=3。启动前重查；若占用则重选同 NUMA 空闲组合，否则等待 |
| 主机资源 | 约 188 GiB RAM、654 GiB 可用磁盘空间 | 满足既有参考运行的资源量级；仍在开跑前复核 |
| AQA / 回传 | 目标环境中的 `aqa` 指向当前仓库代码；`env.sh` 不存在，`LOCAL_RESULTS_DATASET` 和 `LOCAL_RESULTS_HF_TOKEN` 未设置 | 默认 `local_fs`，不配 HF sink；`provider.upload` 不配置 |

AutoRound 当前 CLI 已显示两个静态 FP8 参数；AQA 命令构建器也分别映射出这两个参数。仓库 schema 明确规定 `static_attention_fp8` **必须使用 `method: autoround`**，不能用 `model_free`。因此计划采用校准加载路径、`iters: 0`、`algorithm: rtn`，避免不必要的 tuning，同时保留静态 scale 校准。

### 关键兼容风险与门禁

- 所给 Hugging Face 参考卡只验证了 **MXFP8 + 静态 FP8 KV**，没有验证 `--static_attention_dtype fp8`。它使用 vLLM 0.29.0、3 张 RTX 5090、TP=1/PP=3；两卡布局曾在 CUDA graph warmup 时遇到显存压力。
- AutoRound 导出代码会为静态 attention 写入独立顶层 `attention_input_activations` 配置。vLLM 0.30.0（最新 release）和 main 的 compressed-tensors loader 明确写明独立 Attention quantization 不受支持，只处理 KV-cache scheme；因此当前/最新 vLLM 不会应用 AutoRound 的静态 attention 标定。
- 实际 AQA smoke 日志确认 vLLM 0.29.0 使用 FP8 cache dtype，但同时报告导出的 `k_scale`、`v_scale`、`q_scale` 均因目标参数名多一层 `.attn` 而未加载。不能据此宣称校准 scale 生效；PIQA `limit=1` 的 `1/1` 仅证明请求路径能返回，不是质量分数。
- 因此只运行限量 AQA smoke；完整四任务评估、BF16 baseline 与 gate 须暂停，直到 vLLM 消费静态 attention 元数据并正确加载 Gemma4 的 KV scales。不得静默去掉 AutoRound flag，也不得把 KV-only 行为宣称为 FP8 attention。
- `max_model_len=131072` 是引擎上下文上限；四个标准 task 的样本本身不是 128K 长上下文压力测试。本卡按“上下文上限设为 128K”执行，不额外增加长上下文 benchmark。
- 本地 `tokenizer_config.json` 没有 `chat_template`。GSM8K 必须覆盖 recipe 默认值，使用 `apply_chat_template: false`、`fewshot_as_multiturn: false`；仍保留仓库 recipe 的 5-shot、strict-match 与 2,048-token generation budget。

## 分步施工与验收标准

| 步骤 | 工作与产物 | 验收标准 |
|---|---|---|
| ① 结构分析与量化方案 | 完成运行目录内 `analysis.md`；复核 config、index 与 safetensors header；查阅 quantize/evaluate skills、MXFP8 scheme、同模型 recipe/pin、AutoRound 知识及上游资料；形成逐模块精度表、ignore 清单、预期体积、量化与评估计划。 | 不加载权重 payload；所有 index key 与 header 对齐；逐模块名单和 ignore 均逐项可追溯；Transformers/vLLM/AutoRound 探针及静态 attention 风险有结论；知识 references 未接入等偏离明确记录。仓库要求的外部 `autoround-quant-analysis/SKILL.md` 当前不在 checkout，执行完整分析前需用户提供其路径，或明确允许记录此项偏离。 |
| ② 环境准备 | 生成 `setup_venv.sh` 与 `requirements.txt`；自检并复用 `autoround-test`，用环境内 uv 冻结依赖；若因兼容性必须变更依赖，变更写入 pin 后重跑自检和 freeze。 | Conda 环境可复现；`aqa`、AutoRound、Transformers、vLLM、lm-eval、CUDA/5090 可导入/命中；vLLM 按计划使用模型卡指定的 `0.29.0`，AutoRound 保留模型卡验证的构建；`uv pip freeze` 产物与实际环境一致。 |
| ③ YAML 定稿 | 从 `templates/run.yaml` 生成独立 `run.yaml`，包含 baseline、quant、eval、gate、executor、output；建议新输出目录 `/home/xuehaosu/aqa_results/Gemma-4-26B-A4B-MXFP8-FP8KV-FP8Attn-AutoRound` 并启用 unique dir。 | `aqa validate` 通过；`aqa run --dry-run` 显示 MXFP8、`--static_kv_dtype fp8`、`--static_attention_dtype fp8`、4 个 task 与 131072 上限；baseline 和量化评估均含 `kv_cache_dtype=fp8`；输出为 local_fs，不覆盖既有 KV-only run、不上传权重。 |
| ④ 量化、冒烟与评估 | 用 AQA 执行；设置 `AR_DISABLE_META_LOAD=1`；开跑前核验所选物理卡，卡忙就等待。先量化并核验文件，再做静态 attention/FP8 KV 单样本 smoke，通过后才跑 BF16 baseline 与量化模型的四项 lm_eval。 | 全程只使用经核验的空闲卡，优先同 NUMA，不使用全部 8 卡；AutoRound 收到两项静态 FP8 参数；输出权重成功写出且无未处理错误；vLLM smoke 通过且运行时 KV dtype 为 FP8；静态 attention 使用状态有证据，否则暂停；完整 baseline/candidate 评估按相同后端、参数、任务口径完成。 |
| ⑤ 回溯与归档 | 生成 `report.md`、`journal.md`、`owner_actions.jsonl`；执行 `aqa inspect`、`aqa compare`、`aqa artifacts check`；将新验证经验记录为独立的 FP8-attention 变体，不覆盖现有 KV-only 记录。 | 导出大小约符合 MXFP8 预期（KV-only 参考 28.41 GB；接近 BF16 源体积视为静默量化失败）；header 中检查 F8_E4M3/U8 scale 与应保留 BF16 层；静态 KV scale 合法；四项 task 分数、gate 与报告一致；`aqa artifacts check` 全过；不提交 Git。 |

## 运行配置草案（供确认，非可直接运行配置）

```yaml
model: /models/gemma-4-26B-A4B
quant:
  scheme: MXFP8
  method: autoround
  algorithm: rtn
  iters: 0
  export_format: llm_compressor
  static_kv_fp8: true
  static_attention_fp8: true
  ignore_layers: "router,vision_tower,embed_vision,embed_tokens,mlp"
  device_map: "0,1,2"  # CUDA_VISIBLE_DEVICES=1,2,3 后的进程内编号
eval:
  backend: vllm
  harness: lm_eval
  benchmarks: [piqa, mmlu, hellaswag, gsm8k]
  baseline: true
  thinking: off
  lm_eval:
    apply_chat_template: false
    fewshot_as_multiturn: false
    model_args:
      kv_cache_dtype: fp8
      max_model_len: 131072
      tensor_parallel_size: 1
      pipeline_parallel_size: 3
      gpu_memory_utilization: 0.85
      max_num_seqs: 1
      attention_backend: FLASHINFER
      language_model_only: true
gate:
  max_accuracy_drop: 0.02
```

完整 YAML 还需按模板补齐 executor、output、必要的 agent 开关和 sink，并以 `aqa validate` 为准。gate 的 `0.02` 是相对同次 BF16 baseline 的单项绝对分数下降上限；参考卡上的 KV-only 分数仅供对照，不是本次 FP8-attention 变体的承诺值。

## 查阅资料

- 仓库：`AGENTS.md`、`templates/analysis.md`、`templates/run.yaml`、`skills/quantize/SKILL.md`、`skills/evaluate/SKILL.md`、`recipes/schemes/mxfp8.yaml`、四个 `recipes/benchmarks/{piqa,mmlu,hellaswag,gsm8k}.yaml`。
- 同模型既有资料：`cookbooks/Gemma-4-26B-A4B-quantization-recipe.md`、`recipes/quant/google__gemma-4-26B-A4B.json`、`cookbooks/quant_gemma-4-26B-A4B.sh`。这些是当前工作区已有未提交内容；其结果是 KV-only，不覆盖、不当作本次 FP8-attention 已验证证据。
- 模型卡：[google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B)；用户提供的 [MXFP8 + FP8 KV 参考产物](https://huggingface.co/intel-ai/gemma-4-26B-A4B-MXFP8-FP8KV-CT-AutoRound)。
- 上游：AutoRound [CLI 参数](https://github.com/intel/auto-round/blob/main/auto_round/cli/parser.py)、[llm-compressor 导出](https://github.com/intel/auto-round/blob/main/auto_round/export/export_to_llmcompressor/export_to_fp.py)；[Transformers Gemma4 注册](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py)；[vLLM 支持模型说明](https://docs.vllm.ai/en/latest/models/supported_models.html)、[vLLM 0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)。

网络资料读取正常；若后续包或校准数据下载遇到网络问题，再按用户要求使用 `http_proxy` / `https_proxy`：`http://proxy-ir.intel.com:912`。
