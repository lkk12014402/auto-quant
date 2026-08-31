# GLM-5.3-Flash W4A16 day-0 量化过程日志

> 任务：zai-org/GLM-5.3-Flash 的 W4A16 RTN + W4A16 Tuning 量化（day-0 playbook 流程）。
> agent backend：由 Kimi（我）承担（本机未配置 openclaw/copilot）。
> 工作区：`/home/kaokaolv/aqa_results/glm53_flash_day0/`。
> 本文档记录每一步的命令、推理依据、结果。开始：2026-08-31T01:16Z。

## 环境基线

- GPU：8×H200（143GB），开始时全部空闲。
- venv：`~/.venv-aqa-gpu`（torch 2.11.0+cu130, vllm 0.26.0, transformers 5.16.1,
  auto-round 本地 0.15.0-dev+补丁, aqa editable）。
- 代理：proxy-dmz.intel.com:912。

## §0 判定（三件套探针，不下载权重）——已完成

**模型身份**：`model_type: glm5_next`，`Glm5NextForConditionalGeneration`。
320B 总参 / 18B 激活的多模态 MoE：**45 层 = 34 KDA 线性注意力 + 11 稀疏 MLA
（indexer）**；MLP 3 dense + 42 sparse；**288 routed experts + 1 shared，top-8**；
hyper-connection（mhc, hc_mult=4, sinkhorn）；MTP 1 层；1M context。
**原生 FP8 权重**（e4m3，dynamic activation；attn/hyper_connection/embed/lm_head 保 bf16）。

**三件套探针结果**：

| 件 | 探针 | 结果 |
|---|---|---|
| transformers 5.16.1 | `glm5_next` 在 `auto_mappings` + 完整 modeling | ✅ 认（调优路径可加载） |
| vllm 0.26.0（本地）/ 0.28.0（最新 release）/ main | registry 无 `Glm5NextForConditionalGeneration`；PR #53906 open 且含 CUDA kernel+scheduler 核心改动（不可 patch） | ❌ release 不认 → 走官方定制镜像 |
| 官方定制镜像 | docker hub `vllm/vllm-openai:glm53-flash-x86_64-cu130`（2026-08-26 更新，活跃） | ✅ 推理环境 |
| auto-round | GLM 有预置 ignore 规则（Glm4MoeLite 系）；FP8 源有 `_detect_fp8_source` 路径 | ✅ 通用路径可行 |

**model card 要点**（zai-org/GLM-5.3-Flash README + recipes.vllm.ai）：serve 惯例
TP4 + `--reasoning-parser glm45`（+ `--tool-call-parser glm47` + MTP speculative）；
**Hopper 不支持 FP8 KV cache（--kv-cache-dtype fp8 仅 Blackwell）**——H200 上
KV 用 bf16；FlashInfer ≥0.6.17（NoPE sparse MLA）。

**两条量化线的源**：存在官方 BF16 变体（`zai-org/GLM-5.3-Flash-BF16`）。
决策：RTN 和 Tuning 都从 **BF16 变体**量化（bf16→W4 保真度最高；FP8→W4 是二次压缩），
baseline 用 BF16（金标准参照）；FP8 原版作为生产参照值另行记录。

## §3 量化配置双重检查——已通过

ignore 清单对全索引（76108 键）模拟匹配：

```
--ignore_layers lm_head,embed_tokens,visual,hc_,indexer,self_attn,mlp.gate,
  shared_experts,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,b_proj,
  f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d,mlp.gate_proj,mlp.up_proj,mlp.down_proj
```

- 量化对象只剩 routed experts（12384×3 权重 + 对应 scale_inv），**全部命中**
  `mlp.experts.*.{gate,up,down}_proj`，零误伤；
- 两个实证过的陷阱在清单里显式规避：`mlp.gate` 子串会误伤 `mlp.gate_proj`
  （dense 层）→ 三个 dense MLP 投影显式 ignore（也合 auto-round GLM 预置规则：
  dense 层全保——vllm 有已知问题）；`shared_experts` 必保（每 token 都过）；
- 保护项：lm_head/embed/visual/hc_*/indexer/self_attn/attention 小件/MTP
  （eh_proj,enorm,hnorm,shared_head）/norm/conv1d/KDA 小投影。

## 进行中的下载

- FP8 原版 → `/home/kaokaolv/models/GLM-5.3-Flash`（328GB，参照/可选）
- BF16 变体 → `/home/kaokaolv/models/GLM-5.3-Flash-BF16`（~640GB，量化源 + baseline）
- docker 镜像 `vllm/vllm-openai:glm53-flash-x86_64-cu130`（评估环境）

## §2 量化执行（待下载完成）

**环境验证（镜像落盘即过）**：`vllm/vllm-openai:glm53-flash-x86_64-cu130`（28.8GB）——
vllm 定制构建注册 `Glm5NextForConditionalGeneration` ✅；transformers 5.15.1（不认
glm5_next，调优路径 packages 升 5.16.1）；torch 2.13.0（含 compile 重入 bug →
调优关 compile）；FlashInfer 0.6.17（满足 NoPE sparse MLA 要求）✅。

**两个 yaml 已备**：
- RTN：`examples/run_glm53_flash_w4a16_rtn.yaml`（model_free + baseline TP8 + gate 1%）
- Tuning：`examples/run_glm53_flash_w4a16_tuning.yaml`（iters=200 + 补丁 wheel +
  transformers 5.16.1 + **compile 开**——上游修复 pytorch#186659 的一行补丁
  （`_cc.run` → `_cc.copy().run`）以单文件 bind-mount 盖进容器 torch，
  补丁文件 `aqa_gpu_logs/patches/runtime_wrappers.py`，容器内已验证生效）。

生成的 RTN 量化命令（确定性 builder 产出，已验证）：
```
auto-round --model_name /home/kaokaolv/models/GLM-5.3-Flash-BF16 --scheme W4A16 --model_free \
  --ignore_layers lm_head,embed_tokens,visual,hc_,indexer,self_attn,mlp.gate,shared_experts,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --device_map auto --format auto_round --output_dir <out>
```

## 下载状态（01:35Z）

- FP8 原版 75%（54/72 文件）
- BF16 变体 29%（38/130 文件）——RTN 的量化源 + baseline，等它

## §2 量化执行——RTN 已起跑（01:57Z）

下载完成（FP8 62 个分片 + BF16 120 个分片）。RTN run 启动：
`aqa run examples/run_glm53_flash_w4a16_rtn.yaml`（setsid 持久，
日志 `logs/rtn_run.log`，产物 `rtn/`）。流水线顺序：baseline(bf16 TP8) 评估
→ model_free RTN 量化 → 量化评估 → gate。当前 baseline 评估中（vllm TP8
工人已起；CUDA multicast 警告是容器内已知良性噪音）。

**坑①（02:20Z，已修）**：baseline 引擎初始化炸
`max_num_seqs (1024) exceeds available Mamba cache blocks (512)`——KDA 线性
注意力的 Mamba 状态缓存每序列占一块，vllm 默认 max_num_seqs=1024 超了。
修法：yaml 的 `lm_eval.model_args` 加 `max_num_seqs: 256`（留余量），
RTN/Tuning 两份同步；kill 重跑（run2）。

**坑②（03:10Z，已修并验证）**：RTN 量化成功但量化模型评估时 vllm 炸
`KeyError: layers.0.mlp.gate_up_proj.weight`——auto-round 导出的
`extra_config` 里是**未融合名**（`mlp.gate_proj`/`mlp.up_proj`），而 vllm 的
glm5next 把 dense MLP 建成**融合的** `gate_up_proj`（MergedColumnParallelLinear）；
`inc.py` 的 extra_config 跳过检查是**精确名匹配**，融合名不在表里 → dense 层被
误建成量化层（参数是 qweight），加载纯 bf16 的 `gate_proj.weight` 时映射到
`gate_up_proj.weight` 在 params_dict 里找不到。修法：导出 config.json 的
extra_config 补融合名（凡 `X.gate_proj`+`X.up_proj` 都在表里的，补
`X.gate_up_proj` bits:16——dense 3 层 + shared_experts 68 个）。
**教训沉淀**：day-0 的量化产物加载冒烟不只是"能不能加载"，还要防
"量化范围的名称映射"在非融合/融合两端的歧义。

## §2 结果

**RTN 线完成 ✅（gate GO）**：

| benchmark | BF16 baseline | W4A16 RTN | Δ |
|---|---|---|---|
| piqa | 0.8292 | 0.8357 | +0.65pp |
| hellaswag | 0.8614 | 0.8597 | -0.17pp |
| mmlu | 0.8666 | 0.8629 | -0.37pp |
| gsm8k | 0.9735 | 0.9621 | -1.14pp |
| **mean** | **0.8827** | **0.8801** | **-0.26pp** |

**Tuning 线**（compile 开 + torch 补丁）：run3 起跑中（03:40Z），关键观察点 =
block 0 首个 tuning backward（glm5_next 的 KDA+indexer 断图比 qwen4_exp 更多，
是补丁效力的更强检验）。

**坑③（04:05Z，修复中）**：run3 过了 block0（torch 补丁生效 ✅），但炸在
layer 3（首个稀疏层）的 indexer：`masked_fill(~attention_mask[..., None], -1)`
——eager 下 attention_mask 是 bool 没事，**dynamo fake tensor 下它是 bf16**，
`~` 反转浮点张量非法（TorchRuntimeError: invert on bf16）。查了
`modeling_glm5_next.py`：同文件 176 行的写法是规范的 `~score_mask.bool()`，
而 875/959/963/1022 四处漏了 `.bool()`——建模文件自身不一致。修法：打了
transformers 补丁 wheel（wheel 里四处 `~` 反转补 `.bool()`，与 176 行对齐；
eager 下无操作），放在 `aqa_gpu_logs/wheels/transformers-5.16.1-py3-none-any.whl`，
tuning yaml 的 packages 指向它（不再从 PyPI 装）。
**教训**：day-0 新架构的建模代码里 dtype 归一常不一致；`~mask` 类反转在
compile/fake-tensor 下是照妖镜。run4 重跑中。

**run4→run5（05:45Z）**：run4 验证了两个补丁（过 block0 + layer3），但发现
8 卡纯属"装得下"的浪费——调优逐 block 串行、多卡不加速。改 `num_gpus: 2`
在 6/7 双卡重跑（run5），释放 0-5 号卡。实测双卡装得下（GPU6/7 各 ~85GB）。
进度参考：稀疏块 ~8 分钟/块。

## §2.5 激进 RTN 线（FP8 范围对齐版）——已起跑（05:57Z，GPU 0-3）

`examples/run_glm53_flash_w4a16_rtn_fp8aligned.yaml`：ignore 清单精确对齐官方
FP8 的量化范围——放开 shared_experts + dense MLP 三项 + 稀疏 MLA 的
q_a_proj/q_b_proj/kv_a_proj_with_mqa；保住 o_proj（KDA 与稀疏层同名分不开，
FP8 在 KDA 层也没量化它）和全部 KDA 小件。两个子串陷阱的修法实证：
`mlp.gate.`（尾点）保护路由器但不挡 dense gate_proj；`self_attn.b_proj`（锚定）
保护 KDA 的 b_proj 但不挡稀疏层的 q_b_proj。索引模拟确认量化对象 =
FP8 范围逐项一致（37326 键）。附带确认：`mapping_proj` 不在 checkpoint 里
（FP8 config 的占位项，本模型无此模块）。

**坑④（07:00Z，修复中）**：激进版跑通了量化但评估炸了两轮，暴露出"可部署范围
是三方交集"——
1. `layers.0.mlp.gate_up_proj.weight` KeyError：dense MLP 被 auto-round 的 GLM
   预置规则强制保留 bf16（"vllm issue"——该架构 dense 层量化后 vllm 不加载），
   而 extra_config 用的是未融合名、vllm 需要融合名 gate_up_proj（同坑②）→
   补融合名修复；
2. `layers.11.self_attn.kv_a_proj_with_mqa.qweight` KeyError（**反方向**）：
   我们量化了稀疏 MLA 大投影（q_a/q_b/kv_a，对齐 FP8），但 vllm 的 glm5next+INC
   把它们建成**普通层**（INC 的量化支持不覆盖这类注意力投影）→ 模型侧没有
   qweight 槽、checkpoint 里却是量化权重。**结论：可部署范围 = auto-round 能量化
   ∩ vllm 能量化加载**——FP8 官方的量化范围超出这个交集的部分（MLA 大投影）
   我们守不住，回收到 ignore（experts + shared_experts 是实际可部署的最大范围）。
   run2 重跑中。

**激进 RTN 结果 ✅（08:05Z，gate 判 NO-GO 但为口径误报）**：

| benchmark | 保守 RTN | 激进 RTN（+shared_experts） | 口径 |
|---|---|---|---|
| piqa | 0.8357 | 0.8341 | acc_norm/acc 差 ≤0.16pp，噪声级 |
| mmlu | 0.8629 | 0.8620 | acc（唯一口径） |
| gsm8k | 0.9621 | 0.9644 | exact_match（唯一口径） |
| hellaswag | 0.8597 | 0.6876 | **acc_norm → acc，口径变了，不可比** |

- 三项同口径对比：piqa -0.16pp / mmlu -0.09pp / gsm8k +0.23pp —— **shared_experts
  纳入量化后精度持平**，激进版实际收益 = 多量化 3×42 层 shared_experts 权重；
- gate NO-GO（mean 0.8370 < 0.87）是 **hellaswag 口径切换**（acc_norm→acc，
   recipes/benchmarks/hellaswag.yaml 改了 metric 优先序）拉低 mean 的误报，
  不是量化掉点。教训：**gate 阈值是口径绑定的，改 benchmark 口径要同步改阈值**；
  后续三方对比统一用 acc 口径重测补齐。

（待续：tuning run5 出分后三方对比）
