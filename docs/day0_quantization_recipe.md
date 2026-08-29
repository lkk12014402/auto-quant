# Day-0 量化支持 Recipe：新模型架构的通用打法

> 适用场景：新发布的模型架构，HF transformers / 官方 vllm release / auto-round
> 三者之一或全部尚未支持。本文是**通用方法论**（怎么查、怎么判、怎么做、怎么沉淀），
> 附录 A 是首个实证案例（Qwen3.8-Flash-Next）的完整记录。
> 与正常量化路径的区别：正常路径（如 `examples/run_5000_qwen3.8_27B.yaml`）假设
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
- 运行 yaml：`examples/run_qwen38_flash_next_docker.yaml`（RTN）、`run_qwen38_flash_next_tuning.yaml`（调优）、`eval_qwen38_flash_next_tuned.yaml`（纯评估）。
- 产物：`aqa_results/qwen38_flash_next_w4a16_docker/`（RTN）、`qwen38_flash_next_w4a16_tuning/`（调优量化）、`qwen38_flash_next_tuning_eval/`（调优评估）。
