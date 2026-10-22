# Qwen3.6-35B-A3B：MXFP8 + FP8 KV + FP8 Attention 施工卡

**状态：原始 run 的步骤①–⑤已完成并验收。** 本段只描述 2026-10-13 默认 FlashInfer 候选 run；后续切换到 TRITON_ATTN 的 S1–S4 结果与当前诊断见本卡末尾及 v2 定因卡，不覆盖原始分数。按用户最新指令仅使用物理 GPU0–3，并跳过 BF16 baseline。AutoRound 已导出 36.64 GiB MXFP8 checkpoint，header 包含 30,970 个 F8_E4M3 权重、30,970 个 U8 scales、1,481 个 BF16 与 30 个 F32 tensors。PIQA/MMLU/HellaSwag 完成；首次 GSM8K 输出因 thinking 未结束不可作为能力分数，thinking-aware 补跑得到 strict-match 0.16376 ±0.01019、flexible-extract 0.65049 ±0.01313（n=1,319）。主目录原始 `accuracy.json` 保留未覆盖。原始 FlashInfer run 确认 MXFP8 kernel、FP8 KV cache、131072 context、TP4/EP4；该 run 中 static Attention scale 是否实际消费未证实，故只标为 MXFP8 + FP8 KV。`aqa inspect` 与 `aqa artifacts check` 均为 `status: OK`；无 baseline/gate，无 Git commit。

此前两次尝试保留为历史：初次 TP4 run 完成 GSM8K BF16 baseline 后，其他 baseline tasks 卡住；TP1/PP3 资源回退虽通过 PIQA loglikelihood smoke，但完整 GSM8K generate 在 PP2 rank 报 CUDA illegal memory access。该失败 run 位于
`/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-FP8Attn_20261013-1420-PP3/`，没有开始量化。其间 4–6 卡出现 100% utilization、高频高功耗且无可见 PID；按最新指令不再检查或使用 4–7 卡。此前 BF16 GSM8K 的 0.0091 仅留档，本次不再运行或用作 baseline/gate。

当前完整 run 输出目录为
`/home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-FP8Attn_20261013-1600-GPU0-3-no-baseline/`；
本次不运行 baseline，也不声称有 baseline 对比或 gate 结论。
本卡是静态 FP8 Attention 变体，不覆盖既有的
`cookbooks/Qwen3.6-35B-A3B-MXFP8-FP8KV-施工卡.md`（KV-only 方案）。
用户已选择：若 vLLM 不消费静态 Attention scale，仍允许完成四任务评估，但结果必须标注为
MXFP8 + FP8 KV 的 serving 结果（static Attention 未生效）；评估采用 TP=4/DP=1/EP。
外部分析 skill 路径未提供，已在 `analysis.md` 中记录偏离。

## 目标与执行边界

- 源模型：`/models/Qwen3.6-35B-A3B`（HF ID：`Qwen/Qwen3.6-35B-A3B`）。
- 量化：AutoRound `MXFP8`，`method: autoround`、RTN/`iters: 0`，同时传入
  `--static_kv_dtype fp8` 和 `--static_attention_dtype fp8`，导出 `llm_compressor`。
- 评估：仓库既有 lm_eval 四任务 `gsm8k`、`piqa`、`mmlu`、`hellaswag`；
  vLLM `max_model_len=131072` 且 MXFP8 serving 显式设 `kv_cache_dtype=fp8`。
  128K 是引擎上下文上限；这四个标准任务本身不是 128K 长上下文压力测试。
- 环境：Conda/mamba 管理环境，uv 管理 Python 包；vLLM 采用调研时最新稳定版
  `0.31.0`，在克隆环境中安装，避免改动现有 `autoround-test`。
- GPU：按用户最新指令仅使用物理 GPU0–3；开跑前复核空闲，不触碰 GPU4–7。
  量化 device map 0,1,2 使用三卡，候选评估 TP=4/DP=1/EP 使用四卡。
- 不发布权重、不上传结果；默认 `local_fs`，不提交 Git。若后续下载/安装遇到网络问题，
  按用户指定设置 `http_proxy` 与 `https_proxy` 为 `http://proxy-ir.intel.com:912`。

## 只读预研与资源核验

### 模型结构与量化范围

- 本地 config：`architectures=["Qwen3_5MoeForConditionalGeneration"]`、
  `model_type=qwen3_5_moe`；Transformers 当前环境能识别该 config。
- 文本侧 40 层、hidden size 2048、256 个 routed experts/top-8、一个 shared expert；
  30 层 linear attention、10 层 full attention；另有 vision tower 与 1 层 MTP。
- safetensors index 共 26 个 shard、1,045 个 tensor key、总字节数
  `71,903,645,408`；读取各 shard header 后确认全部为 BF16，未读取权重 payload。
  专家权重包含融合 3D 张量 `mlp.experts.{gate_up_proj,down_proj}`。
- 初始 ignore 清单沿用相同 MoE 架构的 Qwen3.5-35B 配方，执行前仍需逐项对照本模型
  index 复核，避免子串误伤：
  `lm_head,embed_tokens,visual,in_proj_a,in_proj_b,conv1d,shared_expert_gate,mlp.gate`
  - 保留输入/输出 embedding、视觉塔、linear-attention 门控投影；
    `shared_expert_gate` 避免 model-load 导出多出 vLLM 无法映射的 scale。
  - 支持的 `mlp.experts.*`、`mlp.shared_expert.*` 及线性投影为 MXFP8 主体。`mtp.*`
    初始计划同策略量化，但 AutoRound 导出时将缺失的 MTP 张量复制为 BF16
    `model_extra_tensors.safetensors`；header 核验到785个BF16 MTP tensors，需在报告披露，
    不声称 MTP 已量化。
- 参考 BF16 源权重 71.9 GB，MXFP8 产物预期约 36–40 GB（估算，不作精确等值）；
  若产物接近源体积，按静默量化失败处理并停止后续评估。

### 环境与硬件

| 项目 | 当前只读核验 | 计划 |
|---|---|---|
| 环境管理 | Conda 26.7.2、mamba 2.9.0 | ✅ 已克隆 `autoround-test` 到 `/home/xuehaosu/miniforge3/envs/qwen36-mxfp8`，未修改原环境 |
| 包管理 | 源环境 uv 0.13.0；目标 Conda 环境由该 uv 调用 | ✅ 已用 `uv pip` 安装并 freeze 到 run 目录 `requirements.txt` |
| 量化/评估栈 | Python 3.12.14、AutoRound `0.16.0.dev193+g22d6e28c`（commit `22d6e28c`）、Transformers 5.17.0、PyTorch 2.13.0+cu132、vLLM 0.31.0、lm-eval 0.4.13 | ✅ 目标环境版本核验通过；mamba clone 未携带 pip 包，已在 `setup_venv.sh` 中补为显式 uv 安装 |
| AQA | 本仓 editable 安装到目标环境 | ✅ 目标环境 `aqa validate` 与 `aqa run --dry-run` 通过 |
| vLLM 版本 | vLLM 0.31.0 的 registry 已注册 `Qwen3_5MoeForConditionalGeneration`；Transformers `AutoConfig` 识别本地 config | ✅ 环境探针通过；量化产物实际加载仍须运行时验证 |
| GPU | 8 张 RTX 5090，PCIe、无 NVLink；16:09 preflight GPU0–3 均 0%/2 MiB、无 compute process，SM 180 MHz/低功耗；四卡 nearest CPU NUMA node 均为0；后续不查询或使用 GPU4–7 | 仅使用物理 GPU0–3；量化 device map `"0,1,2"` 映射到可见 GPU0–2，评估 TP4/DP1/EP；正式 workload 前重查0–3 |
| CUDA / 存储 | 目标环境 `torch.cuda.is_available()=True`；`CUDA_VISIBLE_DEVICES=0,1,2,3` 下可见四张 RTX 5090；磁盘可用约 523 GB | 目标环境探针通过；每个正式阶段开跑前只检查 GPU0–3 |

网络只读访问正常，尚未使用代理。外部结构分析 skill
`autoround-quant-analysis/SKILL.md` 不在本仓库；`AGENTS.md` 要求向 owner 获取路径。
若未提供路径，必须在 `analysis.md` 中把“未提供外部 skill”记为偏离，不得假称已查阅。

## 关键兼容性门禁：静态 FP8 Attention

- 当前 AutoRound CLI 确认存在两个静态 FP8 参数，AQA schema 也要求
  `static_attention_fp8` 使用 `method: autoround`；这只证明命令能生成参数，不证明 serving
  runtime 使用了导出的 attention scale。
- vLLM `v0.31.0` 的
  [`CompressedTensorsConfig.from_config`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)
  明确过滤独立 Attention quantization config group，并注明 attention quantization
  需要与 KV-cache quantization 耦合。尚未确认该版本的 Qwen3.6 路径会消费 AutoRound
  导出的 `attention_input_activations` / `q_scale`。
- 仓库 Gemma 的 FP8 Attention 先例在 vLLM 0.29 smoke 中观察到 K/V/Q scale 未加载；
  这是兼容性警示，不直接等同于 Qwen3.6 的实测结论。
- **结果标注门禁：**必须在 smoke 中确认 FP8 KV cache 的运行状态。若 Attention scale 被
  忽略或未加载，按 owner 已选策略仍可完成四任务评估，但报告必须明确标注为
  MXFP8 + FP8 KV 的 serving 结果、静态 Attention 未生效；不得静默删掉 AutoRound flag，
  不得声称 FP8 Attention 已运行。若 vLLM 无法启动或 FP8 KV 本身未生效，则暂停评估并回报。

## 五步施工与验收标准

| 步骤 | 工作与产物 | 验收标准 |
|---|---|---|
| **① 结构分析与量化方案（已完成）** | run 输出目录 `analysis.md`；只读 config、index、tensor header；查阅 `AGENTS.md`、quantize/evaluate skill、MXFP8 scheme、Qwen3.5 同架构 recipe、本卡及上游资料；生成逐模块精度表、最终 ignore 清单、预期体积与执行命令。 | 不加载权重 payload；1,045 个 index key 与 shard header 一致；ignore pattern 逐项匹配且无意外吞掉专家层；逐模块精度/风险有证据；references 未接入与外部 skill 缺失等偏离明记。 |
| **② 环境（已完成）** | 生成 `setup_venv.sh`、`requirements.txt`；以 mamba 克隆 `autoround-test` 到独立环境，uv 安装 vLLM 0.31.0、AutoRound 固定 commit、lm-eval 0.4.13 和 editable AQA；不升级原环境。 | 目标环境导入 AQA、AutoRound、Transformers、lm-eval、vLLM；AutoConfig 与 vLLM registry 命中 Qwen3.6 架构；AutoRound CLI 两个静态参数可用；CUDA 探针可见目标四卡；`uv pip freeze` 已落盘。 |
| **③ YAML 定稿（已完成）** | 最新 `run.yaml` 为 candidate-only：MXFP8/autoround RTN、两个静态 FP8 flags、四任务、128K、FP8 KV、batch32、TP4/DP1/EP、`baseline:false`、无 gate；量化 device map 0,1,2。 | 目标环境 `aqa validate` 与 `aqa run --dry-run` 通过；AQA command builder 预览包含两个 static FP8 flags 和 device map；CUDA probe 确认运行仅暴露物理 GPU0–3；baseline/gate 均关闭。 |
| **④ 运行与输出（已完成）** | candidate-only MXFP8 量化与四任务评估完成。结果：PIQA `acc` 0.80196、MMLU `acc` 0.41383、HellaSwag `acc` 0.57947；GSM8K thinking-aware 补跑 strict-match 0.16376、flexible-extract 0.65049。首轮 GSM8K 原始输出因未完成 thinking 留档但不计作最终分数；主 `accuracy.json` 不覆盖。AutoRound 导出 36.64 GiB；含 785 个 BF16 MTP exporter-copy tensors；static Attention scale runtime 未证实。 | 仅使用物理 GPU0–3；量化 device map 0,1,2，评估 TP4/DP1/EP4、FP8 KV、131072、batch32；权重体积/key/dtype 与 runtime MXFP8/FP8 KV 均核验；static Attention 未宣称生效；baseline/gate 未运行。 |
| **⑤ 回溯（已完成）** | 已生成 `report.md`，更新 `journal.md` 与 `owner_actions.jsonl`，记录四任务结果与各自来源；无 baseline 故未运行 `aqa compare`。 | `aqa inspect <run_dir>` 和 `aqa artifacts check <run_dir>` 均为 `status: OK`；报告含结果总览、限制及完整经验沉淀路由表；无 Git commit。 |

## YAML 关键字段草案（非可直接运行配置）

```yaml
model: /models/Qwen3.6-35B-A3B
provider:
  type: quantize
quant:
  scheme: MXFP8
  method: autoround
  algorithm: rtn
  iters: 0
  export_format: llm_compressor
  static_kv_fp8: true
  static_attention_fp8: true
  extra_args: ["--disable_low_cpu_mem_usage"]
  ignore_layers: "lm_head,embed_tokens,visual,in_proj_a,in_proj_b,conv1d,shared_expert_gate,mlp.gate"
  device_map: "0,1,2"  # CUDA_VISIBLE_DEVICES=0,1,2,3；量化仅用物理 GPU0–2
eval:
  backend: vllm
  harness: lm_eval
  benchmarks: [gsm8k, piqa, mmlu, hellaswag]
  batch_size: "32"
  baseline: false
  thinking: auto
  lm_eval:
    model_args:
      max_model_len: 131072
      kv_cache_dtype: fp8
      tensor_parallel_size: 4
      data_parallel_size: 1
      pipeline_parallel_size: 1
      enable_expert_parallel: true
      max_num_seqs: 32
executor:
  type: local
  num_gpus: 4
# 当前 checkout 的 LocalExecutor 在当前进程运行；用克隆环境执行 aqa run，
# 不依赖尚未接线的 executor.env。
output:
  dir: /home/xuehaosu/aqa_results/Qwen3.6-35B-A3B-MXFP8-FP8KV-FP8Attn_20261013-1600-GPU0-3-no-baseline
  unique_dir: false  # 路径含本次 run timestamp，避免污染未忽略的仓库 results/
  sinks:
    - type: local_fs
      path: /home/xuehaosu/aqa_results/local_fs_gpu0_3_no_baseline  # 显式放在仓库外
agent:
  fix_loop: false
```

`lm-eval 0.4.13` 的 direct vLLM adapter 不接受 `data_parallel_size>1` 与
`enable_expert_parallel` 同时启用，因此当前使用 TP=4/DP=1/EP。按 owner 最新指令，
baseline 已关闭且 gate 已删除；此前 BF16 GSM8K 结果不参与本次结论。候选 AutoRound KV
scales 仍须核对，并确认 vLLM 是否加载。`thinking: auto` 字段和 helper 已存在，但当前
pipeline/runner 未接线，不会自动按模板跑 on/off 双模式；实际行为由 chat template 默认值决定。

## 查阅资料

- 本仓库：`AGENTS.md`、`templates/analysis.md`、`templates/run.yaml`、
  `skills/quantize/SKILL.md`、`skills/evaluate/SKILL.md`、
  `recipes/schemes/mxfp8.yaml`、Qwen3.5-35B MXFP8 v2 配置、
  既有 Qwen3.6 KV-only 施工卡、Gemma FP8 Attention 兼容性施工卡。
- 模型：[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)；
  其模型卡推荐的 vLLM 最低版本为 0.19.0。
- vLLM：[v0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)、
  [支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)、
  [compressed-tensors loader 源码](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)。
- AutoRound：[CLI 参数定义](https://github.com/intel/auto-round/blob/main/auto_round/cli/parser.py)、
  [llm-compressor 导出实现](https://github.com/intel/auto-round/blob/main/auto_round/export/export_to_llmcompressor/export_to_fp.py)。

---
**Owner 最新指令：停止使用 GPU4–7，仅用 GPU0–3；无需测试 BF16 baseline。当前配置关闭
baseline 与 gate，保持 TP=4/DP=1/EP，AutoRound 量化仅用 GPU0–2。此前 PP3 失败现场保留
但不再续跑。若 static Attention scale 未被 vLLM 消费，仍允许完成四项评估，但报告必须
标注为 MXFP8 + FP8 KV runtime。外部分析 skill 路径未提供，已记录偏离。**

## 后续低分回查（2026-10-14；后续结论以 v2 定因卡为准）

本节早期“量化范围是主要嫌疑”“Qwen static Attention scales 未经 probe 证实”及“待 owner 确认”
均为 S0 只读阶段的历史判断，已被后续 runtime probe、D1/D2 A/B 和 S4 完整候选评估取代；
保留本段只为说明诊断迭代过程，不应当作当前结论。

**最终诊断与评估状态：**

- S1 逐层确认 vLLM 0.31/SM120 默认 FlashInfer 加载非单位 `q_scale`，但 query 保持 BF16；
  同一 checkpoint 使用 `TRITON_ATTN` 后启用 `QuantFP8` query，q/k/v scale 与 checkpoint 一致。
- S2 D1（RUN+TRITON_ATTN）MMLU-50=0.8484、PIQA=0.8183；D2（REF+默认 FlashInfer）
  分别为0.8474、0.8205。S4 路线 A 完整评估后，PIQA/MMLU/HellaSwag 相对历史 REF
  分别 −0.381/+0.043/+0.110 pp，支持原始 loglikelihood 严重回退主要由 attention backend/
  `q_scale` 运行时路径引起，而不是额外量化范围导致。
- 修正 `Question:` stop 后 GSM8K strict-match 为0.899166，仍比历史 REF 低5.838 pp；
  flexible-extract 接近参考，样本 hashes 全同，但 `[invalid]` 数量和 strict 配对不一致。
  REF 与 S4 的 vLLM 版本、adapter、并行拓扑及 thinking 后处理不同，残差尚未归因；
  不宣称四任务精度修复全部完成。
- 详见 `cookbooks/Qwen3.6-35B-A3B-MXFP8-FP8KV-FP8Attn-精度回退修复施工卡.md`
  与本 run `report.md`。未运行 BF16 baseline、未重做量化、未启动后续 GSM8K A/B；
  后续生成长度/adapter 实验须另获 owner 批准。
