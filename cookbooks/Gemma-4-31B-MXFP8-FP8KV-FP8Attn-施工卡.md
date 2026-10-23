# 施工卡（终稿）：google/gemma-4-31B MXFP8 静态 KV/Attention 量化 + 评估

> 版本：2026-10-19 v1.0（等待 owner 确认后开工）
> 执行者：aqa（autoquant-agent）+ owner 把关
> 模型：本地 `/models/gemma-4-31B`（= google/gemma-4-31B，pretrained 版）

## 0. 任务定义与用户拍板记录

| 决策点 | 结论（owner 拍板） |
|---|---|
| 量化 scheme | **MXFP8** + `--static_kv_dtype fp8 --static_attention_dtype fp8`（硬约束） |
| 量化路径 | auto-round **model-load（method: autoround）**——static attention 必须走校准路径（aqa config 硬校验，auto-round 官方亦要求） |
| 校准档位 | **RTN iters=0**（先跑通；不跑 iters=200 tuning） |
| 评估 | aqa lm_eval harness，backend vllm，**4 task = piqa / hellaswag / mmlu / gsm8k** |
| 评估长度 | **max_model_len=131072（128k 硬上限）** |
| batch_size | **32**（OOM 时 aqa heal 自动降 16→8→4→2→1，再降 gpu_memory_utilization，内置阶梯已确认 `aqa/eval/heal.py`） |
| thinking | **off**（lm_eval model_args `enable_thinking=false`；aqa eval.thinking 未接线，不依赖） |
| bf16 baseline | **不跑**，复用 run_5000 时期上传到 `lvkaokao/lb_local` 的历史 baseline 结果；gate 用 `aqa compare` 在报告阶段手工判定 |
| attention backend | eval 直接 **TRITON_ATTN**（owner 判断 FlashInfer 在 SM120 + static attention 大概率不兼容，先例见 AGENTS 坑 #8） |
| **SM120 QDQ 激活量化栈**（2026-10-21 新增） | **安装 intel/neural-compressor 的 `benchmark/vllm-qdq-plugin`**，评估侧开启 `VLLM_QDQ=1 VLLM_QDQ_CUTE=1`（owner 指定；机制核实与生效点见 §1.4 QDQ 链路、§3 风险 #9/#10） |
| GPU | **量化与评估一体 run，均用卡 0-3（4×5090，同属 NUMA 0）**；不占 4-7；开工前 nvidia-smi 复查，忙则等待 |
| 回传 | **仅本地落盘**（output.sinks 只配 local_fs；暂不配 hf_dataset——owner 明确接受，此为铁律偏离，已记 journal） |
| 权重发布 | 不发布（不配 provider.upload，不产 model_card） |
| 包管理 | conda 建环境 + **uv** 装包；vllm 装**最新支持版 0.32.0**（模型 README 未指定版本，vllm 0.32.0 起 registry 含 Gemma4ForConditionalGeneration） |

## 1. 结构分析结论（第①步已完成）

### 1.1 架构事实（来源：/models/gemma-4-31B/config.json + index 遍历）
- `Gemma4ForConditionalGeneration` 多模态（vision tower 0.55B + 60 层文本解码器），**稠密 MLP**（无 MoE/MTP/PLE/线性注意力）。
- 混合注意力：50 层 `sliding_attention`（窗口 1024，16 KV heads，head_dim 256）+ 10 层 `full_attention`（index 5,11,…,59；4 个全局 KV heads、head_dim 512、p-RoPE、k=v 无独立 v_proj——index 中这 10 层确无 v_proj，逐一核对过）。
- `tie_word_embeddings: true`，vocabulary 262,144；`thinking` 模型（`<|think|>` 系统提示触发；tokenizer_config 带 `think_token`）。
- 权重：1,188 张量，bf16 62.5 GB（58.3 GiB）；`transformers_version: 5.5.0.dev0`。
- 本地 preload 版 tokenizer_config **无 chat_template**（google/gemma-4-31B 与 -it 版均无，已抓取 hub 原件核实）。

### 1.2 逐模块精度表（owner 已复核）

| 模块 | 处理 | 参数量 | 依据 |
|---|---|---|---|
| text 60 层 `self_attn.q/k/v/o_proj`（50/60/50/60 个映射） | MXFP8 | ≈7.70B | 主力（AGENTS §3.1 决策表：稠密 MLP+attn linear 量化） |
| text 60 层 `mlp.gate/up/down` | MXFP8 | ≈20.81B | 主力 |
| `model.language_model.embed_tokens`（tied lm_head） | 保 bf16 | 1.41B | 默认（quant_lm_head=false；tied 权重不动） |
| `model.vision_tower.*`（0.55B） | 保 bf16 | ≈0.55B | auto-round VLM 默认 `quant_nontext_module=False` 跳过 vision（源码 `utils/common.py:758 VISION_MM_KEYS` + `compressors/base.py:795-799` 已核） |
| norm / `layer_scalar` / `std_scale/std_bias` 等 | 自动跳过 | 微小 | 导出审计确认 dtype 不降级 |
| **ignore_layers 显式清单** | 不需要 | — | 无 MoE/shared_expert/indexer；vision 由 auto-round 前缀规则默认排除（运行时以导出 dtype 审计为准，不凭日志断言） |

### 1.3 体积预期（sanity 判据）
MXFP8 ≈ 8.25 bit/param（8bit 权重 + E8M0 块 scale 0.25bit/32elem）：28.5B×1.03B ≈ 29.4GB；+ embed bf16 2.82GB + vision bf16 1.1GB + scale 幵销 → **预期 33~35GB**。
> 体积≈62.5GB 原始 = 量化静默失效（AGENTS 坑 #1），硬性验收线。

### 1.4 vllm 消费链（已逐点核实源码 v0.32.0）
- 权重 scheme：`CompressedTensorsW8A8Mxfp8`（`compressed_tensors.py:492-525, 814-815`）✓
- KV：导出 `quantization_config.kv_cache_scheme` → `CompressedTensorsKVCacheMethod` + `_k/_v_scale` 装载 ✓
- static attention：checkpoint `q_scale` → `layer.register_buffer("_q_scale", ...)`（`compressed_tensors.py:1258`）；`QuantFP8` 查询量化在 kv fp8 且 `impl.supports_quant_query_input` 时启用（`attention.py:468-521`）——**TRITON_ATTN 侧 `supports_quant_query_input = current_platform.is_cuda()` 为 True**（`triton_attn.py:529`），SM120 计算能力 12.0 fp8e4nv 原生可用（`triton_attn.py:490-498`）✓
- hybrid sliding + 全局共享 KV：`gemma4_mm.py` 用 `mm_prefix_clamp_sliding_window` + `kv_sharing_target_layer_name` 处理 ✓
- 强制 backend：engine arg `attention_backend`（`config/attention.py` AttentionConfig.backend，`AttentionBackendEnum.TRITON_ATTN`）经 lm_eval model_args 透传 ✓
- **QDQ 插件链路（owner 指定，针对 SM120）**：vllm 0.32.0 的 MXFP8 linear kernel 候选链（`kernels/linear/__init__.py:552-561`）中若前端 FlashInfer/CuTe 命不了则落 `MarlinMxfp8LinearKernel`（W8A16 marlin，`marlin.py` 注明 SM80+，K%32==0 即可，我们 K=5376/21504 均满足）。vllm-qdq-plugin 在 general_plugins entrypoint 注册，monkey-patch `marlin_gemm` 与 `torch.ops._C.marlin_gemm`（vllm≥0.31 dense 路径已覆盖，`patch.py:95-122`）：`b_q_type=float8_e4m3fn` → 注入 `mxfp8_qdq` 激活 QDQ（E8M0 group32 量化-反量化），使 marlin 路径等价 W8A8 MXFP8 真实量化语义；`VLLM_QDQ_CUTE=1` 强制 CuTe 加速 QDQ 本身（README 核实）。跟 TRITON_ATTN 无冲突（patch 的是 GEMM ops，不碰 attention）。

### 1.5 软件栈 pin（vllm 0.32.0 PyPI metadata 实测）
| 包 | 版本 | 原因 |
|---|---|---|
| python | 3.12 | vllm requires_python <3.15 → 3.12 安全 |
| torch | 2.13.0（PyPI 默认 cu128+，覆盖 sm120） | vllm 硬 pin |
| vllm | 0.32.0 | 最新 release；Gemma4 registry 命中 |
| transformers | 5.19.0 | vllm 0.32.0 要求 ≥5.16.1,<5.20.0；≥5.5.0 认 gemma4 |
| auto-round | 0.16.1 | 最新；含 gemma4 monkey-patch（special_model_handler:379-386）与 static flags（cli/parser.py:244-265） |
| lm-eval | 0.4.13 | aqa 默认 harness 匹配；vllm 后端支持 `enable_thinking`/`attention_backend` 透传 |
| compressed-tensors | 0.17.0 | vllm pin |
| flashinfer | 0.7.0.post1 | vllm 依赖装上備用；eval 用 TRITON_ATTN，FlashInfer 不参与 attention（如装失败不阻塞，报 owner 后 pin 掉） |
| **vllm-qdq-plugin** | git clone `intel/neural-compressor`（shallow）+ `pip install -e benchmark/vllm-qdq-plugin/`（**PyPI 无此包**，已核） | **owner 针对SM120 的指定需求**：激活侧 QDQ（见 §1.4 QDQ 链路） |
| **nvidia-cutlass-dsl** | ≥4.6.0（PyPI 最新 4.8.1） | `VLLM_QDQ_CUTE=1` 强制 CuTe 后端时是硬依赖（缺失报错，README Runtime Controls 已核） |

## 2. 施工步骤与验收标准（硬流程五步）

### 步① 结构分析与量化方案 → `analysis.md`
- 本卡 §1 即分析产物骨架；开工时转写为 run 目录 `<out>/Gemma-4-31B_quantization_analysis.md`（含查阅清单：AGENTS.md / templates / autoround_full_reference.md / docs/day0_quantization_recipe.md 逐条「采纳/偏离」记录；references 增量层未接入的偏离按 AGENTS §3.3 记录）。
- **验收标准**：精度表（1.2）、体积预期（1.3）、量化命令+评估计划（§3 yaml）齐备；探针结论（见步②）记录在案。

### 步② 环境 → `setup_venv.sh` + `requirements.txt`
- `mamba create -p ~/<env>/aqa_gemma4 python=3.12 -y` → `uv pip install` 安装 §1.5 pinned 清单 + 本仓 `pip install -e .`（aqa 可用）。
- 环境建成后 `uv pip freeze` 冻结 requirements。
- **三件套探针（验收硬门槛）**：
  1. `AutoConfig.from_pretrained('/models/gemma-4-31B')`（trust_remote_code=False）识别 `gemma4` → 过
  2. `from vllm.model_executor.models.registry import _MODELS; 'Gemma4ForConditionalGeneration' in ...` → 过
  3. auto-round `--help` 含 `--static_attention_dtype`、`--static_kv_dtype`；能 import auto-round 装载本模型（轻量探针，加载 config 层即可）
- 加一项：`AutoTokenizer/Processor + apply_chat_template` 实测（拍板 #3「先实测再定」）：可渲染 → 直接用；不可渲染 → 从 `google/gemma-4-31B-it` 下载 tokenizer 模板文件放 run 目录（eval-only，不混权重），并记 journal。
- 加一项（owner 指定，§1.5 两行）：
  ```bash
  git clone --depth 1 https://github.com/intel/neural-compressor.git $HOME/srcs/neural-compressor
  uv pip install -e $HOME/srcs/neural-compressor/benchmark/vllm-qdq-plugin/
  uv pip install 'nvidia-cutlass-dsl>=4.6.0'
  ```
- **验收标准**：三探针全过 + `requirements.txt` 冻结文件齐全；`python -c "import vllm_qdq_plugin"` 与 `pip show nvidia-cutlass-dsl` 在场；网络走 `http://proxy-ir.intel.com:912`（仅在需要外网下载时显式 export，不渗透进可复现产物字段）；任何非包管理器途径的手改环境禁止（铁律）。

### 步③ yaml 定稿 → `owner_tasks/run_gemma-4-31B_mxfp8.yaml`（草案即下）
```yaml
model: /models/gemma-4-31B

provider:
  type: quantize
  quant:
    scheme: MXFP8
    method: autoround            # static flags 必须走 model-load
    iters: 0                     # RTN（opt_rtn 默认开启，文档：iters==0 且未 disable）
    export_format: llm_compressor
    static_kv_fp8: true
    static_attention_fp8: true
    nsamples: 128                # 校准 forward 数（static KV/attn scale 统计 + opt_rtn 权重）
    # device_map 不写：>1 卡时 pipeline 自动 device_map auto

eval:
  backend: vllm
  harness: lm_eval
  benchmarks: [piqa, hellaswag, mmlu, gsm8k]
  baseline: false               # 不复跑 bf16；历史 baseline 见 §4 依赖物（本机已有）
  gen_max_toks: 2048
  batch_size: "32"              # 非 auto 时覆盖 recipe 的 per-task 64（已核实 aqa/eval/harness/lm_eval.py:94-98）
  lm_eval:
    seed: 42
    group: true                  # piqa/hellaswag/mmlu 合并一次调用；gsm8k 单开一组
    apply_chat_template: false   # 历史 5090 run 口径（同样本没 score 过的模板差异会让 gate 失真；pretrained 变体无 template）
    fewshot_as_multiturn: false
    model_args:
      max_model_len: 131072     # 用户：长度限定 128k
      attention_backend: TRITON_ATTN   # owner 拍板：FlashInfer→Triton
      linear_backend: marlin           # SM120 默认 auto 选非-marlin kernel → QDQ 不生效（10-19 QDQ RULER run 坑#1 实证）；显式 marlin 保证激活 QDQ 吃到
      language_model_only: true        # 纯文本 4 task；关 Gemma4 mm-prefix（历史口径）
      kv_cache_dtype: fp8              # 与 static KV 一致（历史量化 eval 口径）
      gpu_memory_utilization: 0.85     # 历史 5090 口径（32G 卡 + 128k KV 预算）
      dtype: bfloat16                  # aqa vllm backend 默认已注入（backends/vllm.py:14-30）；显式写出保持审计一致
    # aqa 默认注入免写：add_bos_token=True / enable_prefix_caching=False / trust_remote_code=True(模型无 auto_map，inert)
    # enable_thinking 不需要（无 chat template 口径下无模板渲染，拍板 #4 的"关思考"由 no-template 实现）

agent:
  fix_loop: true                # 局部知识 fallback 本地目录
  optimize: false               # owner 已定方案
  # knowledge 不配 repo（无 dataset 回传，拍板 #5）

executor:
  type: local
  num_gpus: 4

output:
  dir: ./results/gemma-4-31B-mxfp8-staticfp8
  unique_dir: true
  sinks:
    - type: local_fs            # 仅本地（铁律偏离，owner 确认）
```
- **验收标准**：`aqa validate` 通过；`aqa run --dry-run` 打印的命令与 §1.2/§1.5 一致（注意：dry-run 只能对全新输出目录执行一次，不得对已有终态 run 目录 dry-run，坑 #9）。

### 步④ 运行与输出（量化 → 评估）
顺序（同一 run，`aqa run`）：
1. **量化**（卡 0-3，device_map auto，auto-round `--iters 0 --scheme MXFP8 --static_kv_dtype fp8 --static_attention_dtype fp8 --format llm_compressor`）。
2. **量化产物 sanity + 导出审计**：
   - 体积 33~35GB（对照 §1.3）；体积≈62.5GB → 立即停（量化静默失效，查 auto-round 版本与 ignore 命中）
   - index↔磁盘 shard 逐 key 对账：只认 `weight_map` 引用的 shard/key；key 在对应 shard 存在；重复 key dtype/shape/payload 一致；`total_size`/`total_parameters` 按 AGENTS 坑 #10 语义分别核对与修正
   - 抽查量化键 dtype（语言层 FP8_E4M3 + mx scale；embed/vision/past-fp32 项不降级）
   - 确认导出目录含 processor/config/tokenizer（多模态 vllm 加载必需）
   - `quantization_config` 含 `kv_cache_scheme` + attention `q_scale`（静态导出证据）
3. **量化模型 vllm smoke**（评估前）：TP4 加载（评估侧环境变量 `VLLM_QDQ=1 VLLM_QDQ_CUTE=1`）→ 温度 0 短生成 → 日志确认：attention impl = Triton、KV cache dtype=fp8（或 checkpoint scheme 生效证据）、**本层 MXFP8 linear 实际选中 kernel**（候选链走 native 还是 Marlin）——
   - 若走 Marlin：追加一次诊断短跑 `VLLM_QDQ_TRACE=1` + `--enforce-eager`，确认日志出现 `[QDQ] backend=CuTe format=MXFP8 op=...`（CuTe JIT 编译进 smoke 首跑完成）；**正式评估不带 trace 不强制 eager**
   - 若走 native FlashInfer/CuTe kernel：QDQ 插件不触发是预期行为（activation 侧由 kernel 原生处理，语义仍真 8bit），将日志证据报 owner 记录后继续
   - 异常按 §3 风险预案处理。
4. **4 task 评估**（启动命令显式 `VLLM_QDQ=1 VLLM_QDQ_CUTE=1 CUDA_VISIBLE_DEVICES=0,1,2,3`，env 记入 `aqa log-action`；group=true 一轮 piqa/hellaswag/mmlu + 一轮 gsm8k；量评估同 num_gpus=4，卡 0-3）
5. gate/accuracy 产物 + manifest（gate 因无 baseline eval 而不出 aqa 判定）。
- **验收标准**：`aqa inspect <run_dir>` 对照 `aqa/report/layout.yaml` 全过；审计清单逐项打勾；smoke 日志存 `logs/`；四个 task accuracy.json 全产出。

### 步⑤ 回溯 → `report.md` + journal 全量整理
- **baseline 复用**：从 `hf://datasets/lvkaokao/lb_local/results/google/**`（或 run_5000 时期的 gemma-4-31B baseline 目录，具体路径开工后检索）下载原 bf16 baseline accuracy.json → `aqa compare <baseline_dir> <eval_dir>` 写入报告；**既是铁律也是验收**——找不到历史 baseline 时报告 gate 判 INCONCLUSIVE 并明确缺据（不许装饰成 GO）。
- gate 结论（手工）：`max_accuracy_drop=0.01` 口径（与 run_5000 先例一致），逐 task 对照落表。
- 经验路由（§6 照表填满，`aqa artifacts check <run_dir>` 全过为准）：
  - cookbooks 实例（本机惊艳/掉坑实录）→ `cookbooks/`
  - auto-round/vllm patch（如有修复）→ `patches/`（`aqa capture-patch`）
  - 打包层知识/AGENTS 更新 → 对应文件
  - 本卡与 journal 的偏差清单逐条回收
- **验收标准**：report 完整（量化命令、评估命令、版本、坑与对策、gate 表）；journal 每步命令+依据无空行；artifacts check 0 ERROR。

## 3. 风险预案（跟踪表）

| # | 风险 | 触发信号 | 对策 |
|---|---|---|---|
| 1 | auto-round gemma4 patch 与 transformers 5.19 组合（源码注明 "limited transformers versions validated"，<5.6 走 monkey patch 分支） | 校准 forward 阶段报错/异常输出 | 回退 transformers 5.16.1（vllm 允许下限）重探；仍不行→冻结 + 沉淀 patch 到 `patches/` |
| 2 | SM120 + static attention 量化精度异常（坑 #8 同类） | accuracy 异常低（如 <0.1） | 核实 query_quant `_q_scale` 被 vllm 装载；必要时尝试 FlashInfer 或 `disable_flashinfer_q_quantization` 变提取证据；记录对比 |
| 3 | gsm8k 无可渲染 chat template | 探针失败 | 按 §2 拍板路径（-it 模板文件），若仍失败报告 owner 不改 recipe 口径 |
| 4 | 量化 OOM（3 卡/4 卡校准余量紧张） | CUDA OOM 日志 | 降 `nsamples`（128→64）或 `--low_gpu_mem_usage` 已默认；仍 OOM→ owner 加卡决策（4 卡全占已够用） |
| 5 | vllm 加载 compressed-tensors 静态 fp8 细节（per-layer `_q_scale` 装载路径） | smoke 报 assert/shape 错 | 降级评估语义：kv fp8 保留、attention fp8 的 q_scale 问题单独验证；必要时手工单层探针收集证据 |
| 6 | 磁盘仅 220G（94% 已用） | run 目录 > 100G | run 目录建在 `/models` 分区以外最富盘点位；量化失败重跑前清理 |
| 7 | lm-eval 与 aqa 包装层 batch 参数边界 | batch_size 非 auto 生效与 per-task recipe 64 冲突 | 已核：非 auto 时全局 batch wins（lm_eval.py:94-98）→ gsm8k 也 32 |
| 8 | GPU 被他人占用 | nvidia-smi 有显存/进程 | 等待，自动复查询；**不抢** |
| 9 | **QDQ 插件与 vllm 0.32.0 版本耦合**（插件零源码改动靠 marlin wrapper 解包；vllm 标记/签名随版本演进） | smoke 里 marlin 路径无 `[QDQ]` trace 或 patch 解包告警 | 先确认是否走了 native kernel（见步④ 3 分支）；确属 marlin 但 QDQ 失效 → 停评估报 owner（精度语义偏离 W8A8；不许静默带病出分）。必要时 pin 插件Specific commit 对齐 vllm tag 重验 |
| 10 | CuTe DSL 与 vllm/torch 组合在 SM120 的 JIT 编译（cutlass-dsl 4.8.1 对 sm120 支持） | 首跑 JIT 报错/未编译 | 回退 `VLLM_QDQ_CUTE=0`（参考实现慢但语义同）跑通评估并记录；逻辑版本问题报 owner |

## 4. 运行依赖物（开工前清单）

- [ ] `~/.cache/huggingface` 足够（或 proxy 正常可下载 pile-10k / dataset）
- [x] **历史 baseline accuracy.json 定位（已就位，2026-10-21）**：`lvkaokao/lb_local` dataset 已 404（INC4AI token 验证过，非路径问题）→ **改用本机 5090 BF16 baseline**（同硬件、更优）：`/home/xuehaosu/aqa_results/Gemma-4-31B-BF16-FlashInfer-Eval-5090_20260928-1045/eval/accuracy.json`（piqa .8281 / hellaswag .6596 / mmlu .8117 / gsm8k .8628，state=done）。另有 MXFP8+FP8KV（无 static attn）5090 参考：piqa .8270 / hellaswag .6556 / mmlu .8113 / gsm8k .8635 —— 先例证明 gate 0.01 轻松过。同 handoff 文档 `aqa_results/gemma-4-31B-5090-handoff.md`。
- [ ] run 目录所在盘 ≥ 120G
- [ ] nvidia-smi：卡 0-3 空闲（每卡 <100MiB 显存占用）
- [ ] proxy 环境变量在下载步骤生效
- [ ] `vllm_qdq_plugin`/`nvidia-cutlass-dsl` 已装（步骤② 验收行）；评估启动前 `VLLM_QDQ=1 VLLM_QDQ_CUTE=1` 在 shell 内（每个里程碑复述一遍）

## 5. 时间预算（RTN 档）

baseline 0（复用）· 量化 ≈2~4h · 审计+smoke（含 QDQ 首跑 JIT）≈0.7h · 评估 ≈1h · 回溯 0.5h → **总计 ≈4.5~6.5h**

## 6. 完成定义（DoD）

①~⑤ 产物齐全 + `aqa validate`（yaml）与 `aqa inspect`（run 目录）双过 + 导出审计单全勾 + 4 task 成绩落表 + 历史基线 gate 结论（手工/INCONCLUSIVE 明示）+ journal 经验路由无空行 + `patches/`/`cookbooks/` 知识回流到位（如有）。

---

## 附：终态结论与证据指针（2026-10-21 施工完毕补记）

- **结果**：piqa .8248 / hellaswag .6573 / mmlu .8116 / gsm8k .8567（mean .7876）；对照本机 5090 BF16 baseline（.8281/.6596/.8117/.8628）全部 Δ≤0.0061 → **手工 gate GO（阈值 0.01）**。
- **量化**：RTN（iters=0）6 分钟完成；导出 34.17 GB（源 62.546GB 的 54.6%）；审计 410/410/778/261 dtype 全账闭合；static attention 实证 = 相对旧 checkpoint 新增 87 个 `self_attn.q_scale`（F32 261 vs 174）。
- **SM120/QDQ 栈（本 run 实证）**：engine 日志 `Using MarlinMxfp8LinearKernel for MXFP8 GEMM`（linear_backend=marlin 显式）；`QDQ patches applied: ... (MXFP4, MXFP8)` + `CUTE QDQ backend requested: VLLM_QDQ_CUTE=1`；KV/attention 静态 scale 以 `...self_attn.{q,k,v}_scale` Parameter 名从 checkpoint 装载（vllm 0.32 compressed_tensors.py:1159-1269）。
- **关键坑（新沉淀）**：① 无 chat template 口径必须与 baseline 同门（gsm8k recipe 默认模板需 `apply_chat_template: false` 覆盖）；② `linear_backend=auto` 时 QDQ 不生效（10-19 RULER run 实证）；③ 本 checkout `unique_dir` 未接线且 dry-run 会写终态污染（坑 #9 根因），dry-run 用一次性 yaml；④ auto-round 0.16.1 导出 scale 命名不带下划线前缀。
- **证据**：run 目录 `/home/xuehaosu/aqa_results/Gemma-4-31B-MXFP8-FP8KV-FP8Attn-AutoRound/`（report.md / analysis.md / journal.md / run_manifest / quantize.log）；recipe pin `recipes/quant/google__gemma-4-31B.json`；baseline `/home/xuehaosu/aqa_results/Gemma-4-31B-BF16-FlashInfer-Eval-5090_20260928-1045/`。
