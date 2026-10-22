# Day-0 量化支持 Recipe：新模型架构的通用打法

> 适用场景：新发布的模型架构，HF transformers / 官方 vllm release / auto-round
> 三者之一或全部尚未支持。本文是**通用方法论**（怎么查、怎么判、怎么做、怎么沉淀），
> 附录 A/B 是两个实证案例（Qwen3.8-Flash-Next、GLM-5.3-Flash）的完整记录。
> 与正常量化路径的区别：正常路径（如 `examples/ssh_docker/run_5000_qwen3.8_27B.yaml`）假设
> 三件套都认这个模型，直接进量化+评估；day-0 要先花小成本确认"谁不认、怎么补"。

## 0. 先判定：到底谁不支持（分钟级，不下载权重）

**模型侧锚点**（只需 config.json）：
1. 读 `architectures` + `model_type`——这是唯一锚点，**模型名字会骗人，config 不会**
   （例：Flash-Next 名字像 qwen3_next，实为 `qwen4_exp`）。
2. 顺便记录结构特征，后面量化要用：dense vs MoE（experts 数、是否共享专家）、
   混合注意力（`layer_types` 里有没有 full_attention 以外的东西）、巨型附加件
   （超大 embedding 表、视觉塔、MTP 模块）——`safetensors.index.json` 的文件大小
   分布能直接看出哪个组件是显存怪兽。

**三套件的注册表探针**（各有明确判定物）：

| 件 | 探针 | 不过的含义 |
|---|---|---|
| transformers | `AutoConfig.for_model('<model_type>')` 或直接查 `CONFIG_MAPPING` | 真加载/调优路径不可用；model_free 仍可 |
| vllm | `vllm/model_executor/models/registry.py` 有无该架构类名 | 官方 release 不能推理；见 §1 找替代 |
| auto-round | 通用 Linear 路径天然兼容大多新架构；但 MoE 融合模块解耦、特殊层型要过坑清单（§3） | 可能要打补丁 |

transformers 的版本判定要精确到**版本号**：新架构常在最新 release 或 main 里
（例：`qwen4_exp` 在 5.16.1 注册、5.15.1 没有）。查询顺序：PyPI 最新版的
`CONFIG_MAPPING_NAMES` → GitHub main 源码 → 相关 PR。

## 1. 推理路径（vllm 未支持时）怎么查、怎么选

**信息源优先级**：
1. **HF model card** 部署段（作者意图的最直接表达）；
2. **recipes.vllm.ai/<org>/<model>** 的 per-model recipe 页——含经验证的 serve
   参数、TP 约束、已知坑；一句 "PyPI installation is not supported" 就能直接
   排除 release 路线；
3. **vllm PR 搜索**：`api.github.com/search/issues?q=repo:vllm-project/vllm+<model_type>`；
4. **docker tag 元数据**的 last_updated = 新鲜度信号。

**决策矩阵**：

| PR diff 面 | 判定 | 做法 |
|---|---|---|
| 只加模型文件 + 注册表 + 纯 Python | 可 patch | diff 打到本地 site-packages（留 `.bak`，记 patch 文件） |
| 含 scheduler/KV-cache/custom-ops 核心改动或新增 `.cu` kernel | **不可 patch** | 从 PR 分支源码构建专用 venv，或走官方定制镜像 |
| 官方提供定制镜像且有 docker | 最省 | 直接 `docker run vllm/vllm-openai:<tag>` |
| 以上都不行 | 最后手段 | HF transformers 路径评估（慢但通用）或等合并 |

**实证**：Flash-Next 的 PR 含新 CUDA kernel（`fused_gdn_decode_kernel.cu`）→
patch 否决；先尝试源码构建专用 venv（失败过，编译坑多），最终走官方定制镜像
`vllm/vllm-openai:qwen38-flash-next`——**有官方定制镜像时优先用它，别自己编译**。

## 2. 量化路径选型：model_free 还是调优

| | model_free（RTN） | 调优（`method: autoround`, iters>0） |
|---|---|---|
| 是否加载模型 | 不加载（流式权重替换） | 真加载 + 校准前向 + block 级调优 |
| transformers 支持 | **不需要** | **必须**（AutoConfig 要认 model_type） |
| 耗时 | 分钟~小时级 | 大模型可达 10 小时+ |
| 精度 | 已证实在 Flash-Next 上几乎无损（-0.0007） | 不一定更好（见附录 A 的反直觉结论） |
| day-0 适用性 | **首选下限** | 上限探索 |

**纪律：先跑 model_free RTN 建立精度下限，再决定要不要为调优付 10 小时。**
RTN 已经接近无损时，调优没有可恢复的空间。

**model_free 的静默回退陷阱（实证）**：INT 系（W4A16/W8A16）的 model_free 只认
`auto_round`（auto_gptq 打包）格式；`llm_compressor` 只对 MXFP/NVFP4 系开放。
组合不认时**静默回退到全量加载流程**（日志出现 `falling back to the regular
AutoRound flow`）——老架构无感，day-0 架构直接炸（AutoConfig 不认识 model_type）。
**判定纪律：grep 日志确认没有这行 fallback，才算真免加载。**

**调优路径的 day-0 附加检查**：
1. transformers 版本必须认架构（§0 探针）——不够就 packages 里升版；
2. torch.compile × 多卡 hook 派发可能撞上游 bug（pytorch#185372，见附录坑 3）
   ——先加 `--disable_torch_compile` 保底，确认 torch 版本含修复再开；
3. 巨型 embedding 表（如 102GB 的 PLE）会压垮多卡兜底的 device 0——
   确认 auto-round 侧有散布处理（我们的补丁2）或单卡放得下。

## 3. 量化配置的双重检查（day-0 必须做，正常路径可以不做）

ignore_layers / layer_config 写错了不会报错，只会静默产出坏模型。纪律：

1. **模拟匹配全索引**：把 ignore 清单 + 自动附加模式（lm_head/embed/conv/gate）
   逐条对 `model.safetensors.index.json` 的全部键做匹配，产出每个模块的精度
   分配表——任何模块的命运都不许靠猜。
2. **子串陷阱**：`mlp.gate` 会子串匹配到 `mlp.gate_proj`（dense 模型上会把
   gate_proj 全部误跳过）；要精确匹配路由用锚定形式 `mlp.gate$`。
3. **MoE 保护规则**：lm_head + embedding + router/gate + **shared_expert**
   （每个 token 都过它）必保高精度；day-0 再叠加引擎侧融合层（hyper_connection
   这类）+ 不整除 group_size 的小投影。
4. **导出后核对**：导出的 `quantization_config` 里的 ignore 清单必须覆盖每一个
   实际存为明文的 Linear——不一致时 vllm 会按量化层构建然后 `KeyError`。

## 4. 环境策略（local venv vs docker vs ssh docker）

**三条路径是同一条流水线的三种载体**（实证对齐过）：包集合推导共用
`container_packages()`；docker 系（local/ssh）共用 `build_container_bootstrap`
（装栈 + agent backend 供给 + 内层 `aqa run`）；venv 走等价实现（pip 装栈 +
`_venv_agent_provision`：node 装进 venv 前缀、npm --prefix 装 openclaw/copilot、
`aqa.agent.provision` 同步 skills/密钥）。结果/模型/日志的上传（sinks、
HF dataset）在内层 run 里统一发生，与载体无关；secrets 经 `collect_secret_env`
同一份白名单 + sink token 引用注入。所以从 ssh docker 切到 local docker/venv
只是 yaml 里 `executor` 块的事，流程语义不变。

| | local venv | local docker（day-0 推荐） | ssh docker（生产验证路径） |
|---|---|---|---|
| 安装速度 | 慢（源码编译 vllm 可达数小时，还可能失败） | 快（官方镜像直接 pull） | 同左（镜像拉在远端） |
| 栈完整性 | 自己凑 | 镜像自带定制 vllm；`executor.env.packages` 显式补齐 auto-round/lm-eval/transformers，**排除 vllm**（别用 PyPI 覆盖镜像定制版） | 同左 |
| 适用 | 无 docker 权限时 | 有官方/定制镜像时 | 远端算力 |

**docker 派发的实证坑（aqa 已全修，yaml 注释里有记录）**：
- 本地源码版 auto-round 要在宿主机**先打 wheel** 再挂载安装——容器里没 git，
  源码直装会挂在 setup.py 的 `git describe`；
- 容器内 transformers 版本不够时，packages 里加 `transformers==X.Y.Z`，
  但要先验证与镜像内 vllm 的兼容（import + ModelConfig 路径冒烟）；
- 代理要显式转发（host 网络不共享环境变量）；多卡注意 `CUDA_VISIBLE_DEVICES`
  的容器内重映射；NCCL 在容器里要 `NCCL_NVLS_ENABLE=0`；
- 长任务用 `setsid nohup` 脱离 session 启动（session 关闭会杀启动器，
  容器本身由 daemon 托管可活，但日志流会断）。

## 5. 验证顺序（成本递增，任何一步不过就不往下走）

1. config 解析 + registry 注册确认（秒级）；
2. bf16 单请求冒烟（`vllm serve` 打一条 chat）——推理路径通；
3. **量化产物加载冒烟**（`scripts/smoke_vllm.py`：TP 加载 + 3 条确定性生成）——
   风险最高的一步放最前；
4. 小 limit 评估（piqa limit 16 之类）；
5. 全量基准 + gate。

**Attention backend × KV dtype 必须做设备/维度组合冒烟，静态能力声明不够**：
混合 head size 模型的某一层可能直到 KV-cache 初始化或首个 prefill JIT 才触发限制。
实测边界（vLLM 0.30.0、FlashInfer 0.6.18.post1、Gemma 4、A100 SM80）：

- 默认 `TRITON_ATTN` 明确拒绝 SM80 上的 FP8 KV（其原生 FP8 路径要求 SM89+）。
- 显式 `FLASHINFER` 虽然 `supports_combination` 静态检查可接受 head size 512 + FP8，
  但真实 FA2 prefill JIT 对 `head_dim>256` 的一字节 KV 只生成 CUDA major
  `[10, 11, 12]`（SM100+），所以 Gemma 4 的 512-dim full-attention 层在 SM80 仍会失败。
  FlashInfer 上游 `tests/attention/test_batch_prefill_kernels.py` 也将该 FP8 large-head
  路径在 pre-SM100 上 gate 掉。
- 对纯文本 task，`language_model_only=true` 可关闭 Gemma 4 的 multimodal-prefix attention，
  但不会放宽 head-dimension / compute-capability kernel 限制；不得把它用于图像/视频评估。
- 先用真实量化 checkpoint、目标 GPU、真实 FP8 cache 做单样本 smoke，并确认日志已
  选定 backend、进入 prefill/cache 初始化且生成结果；未出样本前不得启动全量任务。
  不得仅为让 smoke 通过而伪造架构、放宽未验证的 JIT gate 或静默改用 BF16/跳过层。

## 6. 导出保真检查（tuning 路径特有，实证新增）

调优路径经 transformers 重序列化导出，可能引入三类静默损坏：

1. **config 结构漂移**：字段被改名/规范化到安装版 transformers 的 schema
   （例：`full_attention` → `qwen_sparse_attention`，vllm 校验拒收）。
   aqa 已在导出后自动 `restore_config_structure`（源 config + quantization_config
   嫁接，原文件留 `.bak`）；
2. **字面量分片键**：加载时拼接的张量被按源索引的通配符模式字面写回
   （`shard_*.weight` 进 safetensors）——索引里出现带 `*` 的键即为此坑，
   从源模型按原始分片边界字节级拆回即可（脚本见附录 A）；
3. **孤儿文件**：同一巨型 tensor 被写两遍，index 只引用其一——
   用「index 引用集 vs 目录文件集」的差集检测，删掉孤儿。
   （量化产物大小的 sanity：只量化 experts 时总量 ≈ experts/4 + 其余 bf16。）

## 7. 沉淀义务（每次 day-0 都要还）

- 踩坑进 `aqa/agent/recipes/fix_playbook.md`（机制级，带错误签名）；
- 量化命令进 `recipes/quant/<model>.json`（pinned recipe）；
- 方法论更新进本文；
- patch/构建记录随 run 的 manifest 归档。

---

# 附录 A：Qwen3.8-Flash-Next 实证全记录（2026-08-26 ~ 08-28）

## A.1 判定与决策

- `qwen4_exp` 不在 vllm 0.26.0 注册表；recipes.vllm.ai 明示 PyPI 不支持；
  PR #53896（模型支持）open 未合，且含新 CUDA kernel → patch 否决；
- 最终路线：官方定制镜像 `vllm/vllm-openai:qwen38-flash-next` + docker 派发。

## A.2 RTN（model_free）线 ✅

- bf16 baseline：piqa 0.8194 / hellaswag 0.8767 / mmlu 0.8652 / gsm8k 0.9689，**mean 0.8825**。
- W4A16 model_free RTN（ignore 含 shared_expert 保护）：mean **0.8819**（掉点 0.0007，gate GO）。

## A.3 调优（iters=200）线 ✅——六个坑全记录

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| 1 | 镜像 transformers 5.15.1 不认 `qwen4_exp` | 调优路径走 AutoConfig 真加载（model_free 绕得过，调优绕不过） | packages 升 `transformers==5.16.1`（已验证与镜像定制 vllm 兼容） |
| 2 | `Qwen4ExpTextRMSNorm has no attribute 'group_size'` | auto-round `apply_plan_to_model` 对所有模块无差别剥离与量化方案同名属性；该 norm 的 `group_size` 是分组 RMSNorm 的结构超参数 | auto-round 补丁1：剥离范围限定为可量化层/计划内模块（`resolver.py`） |
| 3 | compiled backward `cannot enter context ... already entered` | 上游 torch bug（[pytorch#185372](https://github.com/pytorch/pytorch/issues/185372)）：嵌套 compiled backward 重入 TLS 暂存的同一 Context 对象。嵌套来自 graph break：qwen4_exp 的 decoder 在 indexer 稀疏注意力处断图 → 编译续体嵌套内层编译调用。实证矩阵：27B(dense)/M3(128-expert MoE) 多卡+compile 正常，27B 消融（去 GDN+PLE、experts 减到 2）仍炸、64 个纯 hook Linear 玩具不炸 → 判据是「编译区内 graph break 嵌套」 | 修复 [pytorch#186659](https://github.com/pytorch/pytorch/pull/186659) 已合入 torch main（2026-06-22）；容器 torch 2.13.0 早于它。本 run `--disable_torch_compile` 绕过；打修复行后原必炸配置实证通过 |
| 4 | block 1 OOM（139GB 卡爆） | PLE n-gram embedding 加载时拼接成单个 102.4GB nn.Embedding，多卡兜底逻辑把非量化模块全堆 device 0 | auto-round 补丁2：被忽略的 nn.Embedding 叶子散布到预算最空闲的卡；非 Embedding 叶子保持原行为（权重可能被父模块函数式引用，hook 拦不到） |
| 5 | vllm 拒载量化产物 | config.json 重序列化把 `full_attention` 改名 `qwen_sparse_attention`（vllm 拒收）；PLE 被存成字面量 `shard_*.weight` 单 blob | aqa 导出后 `restore_config_structure`；PLE 从源模型按 128 分片字节级拆回（逐字节校验一致） |
| 6 | 产物比预期大 ~100GB | auto-round 把 PLE 大 tensor 写两遍：block 级即时保存一份 + 最终保存一份，前者成孤儿 | 删孤儿（删除前验证 index 无引用）；「孤儿 safetensors 检测」列入 aqa 导出后检查 |

非代码事故：run5 在 block7 被外部租户抢占 GPU 显存 OOM（OOM 信息里我方进程仅占
16GB——**共享机器上 OOM 归因先看我方进程实际占用**）；run4 启动器随 session 关闭
被杀但容器存活——长任务一律 setsid。

- 调优运行：48 block × ~15 分钟 ≈ 10.5 小时（compile 关），导出 181GB（bf16 的 50%）。
- **三方对比（W4A16）**：

| benchmark | bf16 baseline | RTN | tuning iters=200 |
|---|---|---|---|
| piqa | 0.8194 | 0.8199 | 0.8215 |
| hellaswag | 0.8767 | 0.8755 | 0.8744 |
| mmlu | 0.8652 | 0.8655 | 0.8560 |
| gsm8k | 0.9689 | 0.9666 | 0.9682 |
| **mean** | **0.8825** | **0.8819** | **0.8800** |

- **结论（反直觉）**：RTN 已近乎无损时，调优没有可恢复空间，反而 mmlu -0.95pp——
  128 条通用校准数据的 block 级 MSE 最优 ≠ 下游任务最优。

## A.4 产物与补丁归档

- 量化命令（run6 实际执行）：
  ```
  auto-round --model_name /home/kaokaolv/models/Qwen3.8-Flash-Next --scheme W4A16 --iters 200 \
    --ignore_layers lm_head,embed_tokens,visual,linear_attn,self_attn,hyper_connection,mlp.gate,shared_expert,in_proj_a,in_proj_b,ple,mtp,indexer \
    --device_map auto --format auto_round --disable_torch_compile --output_dir <out>
  ```
- auto-round 补丁：`patches/autoround_qwen4exp_fixes.patch`（resolver group_size + device.py Embedding 散布；应用方式见 `patches/README.md`）。
- aqa 改动：本仓 `multi-modal` 分支的三笔 commit（provider 嵌套解析、config 结构恢复、导出修复、测试）。
- PLE 分片修复脚本：`/tmp/repair_ple_shards.py`；推理冒烟脚本：`scripts/smoke_vllm.py`。
- 运行 yaml：`examples/local_docker/run_qwen38_flash_next_docker.yaml`（RTN）、`run_qwen38_flash_next_tuning.yaml`（调优）、`eval_qwen38_flash_next_tuned.yaml`（纯评估）。
- 产物：`aqa_results/qwen38_flash_next_w4a16_docker/`（RTN）、`qwen38_flash_next_w4a16_tuning/`（调优量化）、`qwen38_flash_next_tuning_eval/`（调优评估）。

---

# 附录 B：GLM-5.3-Flash 实证全记录（2026-08-31）

模型：`zai-org/GLM-5.3-Flash`（`glm5_next`；320B/18B 激活；45 层 = 34 KDA 线性
注意力 + 11 稀疏 MLA(indexer)；MLP 3 dense + 42 sparse(288 experts top-8 + 1 shared)；
hyper-connection；MTP；多模态；原生 FP8）。量化源 = **官方 BF16 变体**
（`GLM-5.3-Flash-BF16`，bf16→W4 保真最高，勿从 FP8 二次压缩）。
过程全记录（逐命令逐推理）：`aqa_results/glm53_flash_day0/RUN_JOURNAL.md`。

## B.1 判定与决策

- transformers 5.16.1 认 `glm5_next`（调优路径可用，需 packages 升版——镜像自带
  5.15.1 不认）；vllm 0.26.0/0.28.0 注册表均无，PR #53906 open 且含 CUDA kernel +
  scheduler 核心改动 → **patch 否决**；
- 最终路线：官方定制镜像 `vllm/vllm-openai:glm53-flash-x86_64-cu130` + docker 派发
  （同附录 A 结论：有官方定制镜像就别自己编译）；
- auto-round 通用路径可用，但 **GLM 预置规则会强制 dense MLP 保 bf16**——ignore 里
  放开也无效，配置双重检查时要把"预置规则"也算进命中模拟。

## B.2 RTN（model_free，保守范围）线 ✅

- bf16 baseline（TP8）：piqa 0.8292 / hellaswag 0.8614(acc_norm) / mmlu 0.8666 /
  gsm8k 0.9735，**mean 0.8827**；
- W4A16 RTN（只量化 routed experts）：piqa 0.8357 / hellaswag 0.8597 / mmlu 0.8629 /
  gsm8k 0.9621，**mean 0.8801**（掉点 0.26pp，gate GO）。

## B.3 坑全记录

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| 1 | 引擎初始化炸 `max_num_seqs (1024) exceeds available Mamba cache blocks (512)` | KDA 线性注意力的 Mamba 状态缓存每序列占一块，512 块 < vllm 默认 1024 | `model_args` 加 `max_num_seqs: 256`。**线性注意力架构的固有约束，serve/评估都要带** |
| 2 | 量化产物加载炸 `KeyError: layers.0.mlp.gate_up_proj.weight` | auto-round 导出 extra_config 用**未融合名**（gate_proj/up_proj），vllm 把 dense/shared MLP 建成**融合** `gate_up_proj`（MergedColumnParallelLinear），INC 跳过检查是精确名匹配 → 融合层被误建成量化层 | aqa 导出后自动 `repair_fused_name_gaps`（凡 `X.gate_proj`+`X.up_proj` 成对在表，补 `X.gate_up_proj` bits:16）+2 测试。**通用化沉淀，不止 GLM** |
| 3 | 调优 compile 下 layer 3 炸 `TorchRuntimeError: invert on bf16` | `modeling_glm5_next.py` 875/959/963/1022 四处 `~mask` 漏 `.bool()`（同文件 176 行是规范写法）——eager 下 mask 是 bool 无感，**dynamo fake tensor 下是 bf16**，反转非法 | transformers 补丁 wheel（四处补 `.bool()`），`patches/transformers_glm5_next_mask_bool.patch`。**新架构建模代码 dtype 归一常不一致，`~mask` 在 compile/fake-tensor 下是照妖镜** |
| 4 | 激进版量化后加载炸 `KeyError: layers.11.self_attn.kv_a_proj_with_mqa.qweight`（**反方向**） | 对齐官方 FP8 范围量化了稀疏 MLA 大投影（q_a/q_b/kv_a），但 vllm glm5next+INC 把它们建成普通层——**checkpoint 有量化权重，模型侧没有 qweight 槽** | 回收 ignore。**结论：可部署量化范围 = auto-round 能量化 ∩ vllm 能加载**，官方 FP8 范围超出交集的部分守不住 |
| 5 | tuning 产物评估全随机（piqa 0.49/hellaswag 0.25/mmlu 0.24），加载无报错 | transformers `conversion_mapping` 把 q/k/v_conv1d 加载时 Concatenate 成融合 `conv1d`；auto-round ShardWriter 即时保存只做**名级**反转 → 融合张量错名写进 `q_conv1d`（24576 维），k/v 丢失；vllm 缺权重**静默随机初始化** | aqa 侧：102 个 conv 权重从源拷回（`fix_tuning_export.py`）。根修：auto-round `fix/shard-writer-converter-reverse`（7026d988，张量级反转，+单测；未提 PR）。**调优权重本身无损，不用重跑** |

另复用了附录 A 的两个修复：torch `runtime_wrappers.py` 的 `_cc.run`→`_cc.copy().run`
（pytorch#186659 移植，开 compile 必须，`patches/torch_186659_compiled_backward_context.patch`）；
调优逐 block 串行 → **2 卡够，多卡只解决装得下不加速**（实测双卡 ~100+81GB vram、
峰值 RAM ~800GB，45 块 × ~6 分钟）。

## B.4 激进 RTN（FP8 范围对齐）线

动机：官方 FP8 版量化了 attn 以外的更多模块，验证"跟着 FP8 范围量化 W4"是否可行。
yaml：`examples/local_docker/run_glm53_flash_w4a16_rtn_fp8aligned.yaml`。

- 子串陷阱的两个锚定修法（实证）：`mlp.gate.`（尾点）保护路由器但不挡 dense
  `gate_proj`；`self_attn.b_proj`（锚定）保护 KDA 的 b_proj 但不挡稀疏层 `q_b_proj`。
  **坑④回收 MLA 大投影后，逐项展开收敛回 `self_attn` 整段 + 非锚定 `b_proj`
  的短形式**（全索引模拟验证两份清单命中集一致，量化集 37290 键）——锚定只在
  "要量化同前缀下的部分投影"时才必要；
- 三方交集实测：dense MLP 被 auto-round GLM 预置规则强制保留（放开无效）；
  稀疏 MLA 大投影被 vllm INC 拒载（坑 4）→ **实际量化范围 = experts + shared_experts，
  相比保守版的唯一增量是 shared_experts**；
- `mapping_proj` 是 FP8 config 的占位项，checkpoint 里不存在（对齐前先验证模块存在性）；
- **结果：与保守版同口径持平**（piqa 0.8341 / mmlu 0.8620 / gsm8k 0.9644，
  Δ ≤0.23pp）——shared_experts 纳入量化不掉点，放心扩大范围；
- 插曲：run 内 gate 报 NO-GO 是 hellaswag 口径从 acc_norm 切 acc 拉低 mean 的误报——
  **gate 阈值是口径绑定的，改 benchmark 口径必须同步改阈值**。

## B.5 Tuning（iters=200，compile 开）线与五方对比

- 环境：容器 torch 2.13.0 + `torch_186659` 补丁 + transformers 5.16.1 补丁 wheel
  （坑 3）；2×H200 足够（逐 block 串行，多卡不加速），45 块 × ~6 分钟 ≈ 4.5 小时；
- 坑⑤（见 B.3 表）修复后出分。**五方对比（口径统一：hellaswag 取 acc——历史
  results json 同时存 acc/acc_norm，直接取无需补测）**：

| benchmark（口径） | BF16 baseline | 保守 RTN | 激进 RTN(+shared) | 保守 Tuning | 激进 Tuning(+shared) |
|---|---|---|---|---|---|
| piqa (acc) | 0.8292 | 0.8357 | 0.8341 | 0.8319 | 0.8303 |
| hellaswag (acc) | 0.6903 | 0.6893 | 0.6876 | 0.6893 | 0.6879 |
| mmlu (acc) | 0.8666 | 0.8629 | 0.8620 | 0.8620 | 0.8652 |
| gsm8k (em) | 0.9735 | 0.9621 | 0.9644 | 0.9712 | 0.9719 |
| **mean** | **0.8399** | **0.8375** | **0.8370** | **0.8386** | **0.8388** |

- 结论：全部接近无损（≤0.29pp）；**激进 Tuning 最佳**（0.8388，-0.11pp），
  调优收益主要在 gsm8k（0.9719 逼近 baseline 0.9735）——与
  Qwen3.8-Flash-Next「RTN 已无损则调优无空间」不同，是否调优取决于 RTN 余量，
  day-0 先 RTN 探底的纪律不变；
- shared_experts 纳入量化是免费的（激进 vs 保守持平甚至略好）；
- 坑⑤根修 E2E 验证：激进 tuning 用含 ShardWriter 修复（7026d988）的 wheel，
  导出直接正确分拆 q/k/v_conv1d（dtype 继承内存 fp32，vllm cast 无碍）。

## B.6 产物与补丁归档

- pinned recipe：`recipes/quant/zai-org__GLM-5.3-Flash.json`；
- 运行 yaml：`examples/local_docker/run_glm53_flash_w4a16_rtn.yaml`（保守 RTN）、
  `run_glm53_flash_w4a16_rtn_fp8aligned.yaml`（激进 RTN）、
  `run_glm53_flash_w4a16_tuning.yaml`（调优 iters=200）、
  `eval_glm53_rtn_verify.yaml` / `eval_glm53_fp8aligned_verify.yaml` /
  `eval_glm53_tuning_fixed.yaml`（纯评估复跑）；
- 补丁：`patches/torch_186659_compiled_backward_context.patch`、
  `patches/transformers_glm5_next_mask_bool.patch`、
  `patches/autoround_shard_writer_converter_reverse.patch`（坑⑤根修，对应
  auto-round 分支 `fix/shard-writer-converter-reverse`；应用方式见
  `patches/README.md`；补丁版全文件分别在 `aqa_gpu_logs/patches/runtime_wrappers.py`、
  `aqa_gpu_logs/wheels/transformers-5.16.1-py3-none-any.whl`）；
- aqa 改动：`repair_fused_name_gaps` 进 `aqa/quantize/__init__.py` 的导出后链（+2 测试）；
- 修复脚本：`aqa_results/glm53_flash_day0/logs/fix_tuning_export.py`（conv 拷回，幂等）、
  `logs/cmp_tun_rtn.py`（RTN/tuning 权重对比排查）；
- 产物：`aqa_results/glm53_flash_day0/`（rtn/ rtn_fp8aligned/ tuning/ logs/ RUN_JOURNAL.md）。

## C. Gemma-4-31B 稠密 · SM120 · static KV/Attn + QDQ 线 ✅（2026-10-21）

- 模型：`/models/gemma-4-31B`（Gemma4ForConditionalGeneration 多模态轉 dense 文本塔，60 层 =
  50 sliding(窗口 1024·16KV·head256) + 10 full(5,11,…,59·4KV·head512·p-RoPE·k=v 无 v_proj)）。
- 方案：MXFP8 · model-load(autoround) · RTN iters=0 · `--static_kv_dtype fp8 --static_attention_dtype fp8`
  · llm_compressor；vision 保 bf16（quant_nontext_module=false），embed tied 保 bf16。
- **三件套探针**（vllm 0.32.0 / transformers 5.19.0 / auto-round 0.16.1 全命中）：
  GPUs 4×5090(SM120)；`linear_backend=marlin` 显式 → `Using MarlinMxfp8LinearKernel`；
  vllm-qdq-plugin@123f1d4（`VLLM_QDQ=1 VLLM_QDQ_CUTE=1`）激活侧 QDQ 达成 W8A8 语义。
- ** Scores**：四 task（piqa/hswag/mmlu/gsm8k@131072·TP4·batch32·无模板口径）对同机 BF16
  baseline Δ≤0.0061 → 手工 gate GO（这款是 5090 上首次 dense 全家四 task 过线）。
- **审计口径**：静态 scale 导出命名是 `...self_attn.{q,k,v}_scale`（**不带下划线前缀**，区别于
  vllm 内部 `_*_scale` buffer）；体积 accounting：F8 410（6·60+50 v_proj）+ U8 410 + BF16 778
  + F32 261（文本 180 + vision 81）——相对旧 checkpoint +87 q_scale 即 static attention 证据。
- **坑**：① 评估必须与 baseline 同门（本模型无 chat_template，gsm8k recipe 默认模板要用
  `apply_chat_template: false` 压掉）；② `linear_backend=auto` 时不一定命中 Marlin（QDQ 失效，
  10-19 RULER run 实证）；③ 本 checkout `unique_dir` 无消费者 + dry-run 写终态（坑 #9 根因）。
- pinned recipe：`recipes/quant/google__gemma-4-31B.json`；施工卡+终态：
  `cookbooks/Gemma-4-31B-MXFP8-FP8KV-FP8Attn-施工卡.md`；run 证据：
  `aqa_results/Gemma-4-31B-MXFP8-FP8KV-FP8Attn-AutoRound/`（run_id `...102331_c085b8`）。
