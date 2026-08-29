# aqa 借鉴 NeMo Evaluator / NeMo Skills 的设计文档

> 日期：2026-08-25。状态：待评审。
> 上游分析：`/home/kaokaolv/auto_agent/auto-quant_设计分析与NeMo对标.md`（差异与 19 条借鉴清单）；本文把其中可落地的条目展开为**设计**（现状 → 借鉴对象 → 改动 → 验收）。三目录（inference/evaluation/dataset）的逐文件深挖见 `docs/nemo_skills_deep_dive.md`（2026-08-26），D3–D8 的契约细节以它为准。
> 代码基线：`/home/kaokaolv/auto_agent/auto-quant`，405 测试全绿。所有 file:line 以该基线为准。

## 1. 目标与原则

**目标**：把 aqa 打造成「全面且文档完善的自动评估 benchmark 库 + 强大的自动量化框架」。

**借鉴对象**：
- **NEL**（NVIDIA-NeMo/Evaluator 新架构）："evaluator owns the loop"、per-problem 结果 schema、失败分类学、suite checkpoint/resume、透明分片 + merge、interceptor proxy 链、deploy/evaluate 正交。
- **Skills**（NVIDIA-NeMo/Skills）：benchmark 目录级自描述、外部 benchmark 插件（`benchmark_map.json`）、generation/scoring 分离 + 离线重算、`:N` 多次采样聚合（pass@k/majority@k/方差）、CI 守 benchmark 配置、文档即产品。

**aqa 侧不可破坏的原则**（借鉴的边界）：
1. kernel 保持确定性、零重依赖——**不引入** Hydra / NeMo-Run / 容器化 harness 全家桶；
2. 一切新机制走现有插件面（entry-point group / registry / recipe YAML），不开后门；
3. agent 只能帮忙不能挡路：新机制全部有确定性 fallback；
4. 每个里程碑独立可用、独立验收，测试先行（当前 405 测试全绿是底线）。

## 2. 设计地图

| # | 设计项 | 借鉴 | 落地模块 | 里程碑 |
|---|---|---|---|---|
| D1 | recipe_store 闭环（量化 recipe 沉淀复用） | Skills「pinned 配置资产化」思想 | `recipe_store.py` / `optimize.py` / `pipeline.py` / `config.py` | M-A |
| D2 | scheme registry 真化 | Skills `test_configs.py` 守配置 | `config.py` / `recipes/schemes/` | M-A |
| D3 | 外部 recipe 目录注入 | Skills `benchmark_map.json` | `eval/registry.py` / `config.py` | M-B |
| D4 | benchmark 自描述扩展（prepare/known_issues） | Skills 目录级自描述 | `eval/registry.py` schema / `recipes/benchmarks/` | M-B |
| D5 | per-problem raw 落盘 + `aqa rescore` | NEL `results.jsonl` + Skills 离线重算 | `eval/registry.py` / harness 三件套 / `cli.py` | M-B |
| D6 | 失败分类学 `failure_analysis.json` | NEL failure taxonomy | 新 `eval/failure.py` / harness / `report/` | M-B |
| D7 | 多次采样 `repeats` + pass@k/方差 | Skills `:N` 语法 | `config.py` / `eval/runner.py` / `report/stats.py` | M-B |
| D8 | suite 断点续跑 + 分片 merge | NEL checkpoint/`--resume`/shard+merge | `pipeline.py` / `executors/` / `cli.py` | M-B |
| D9 | recipe 配置 CI 校验测试 | Skills `test_configs.py` | `tests/test_recipes.py`（新） | M-A |
| D10 | gate 指标方向标志 | NEL metric 方向语义 | `eval/registry.py` ScoringSpec / `report/__init__.py` | M-C |
| D11 | 部署冒烟 stage（deploy/evaluate 分离） | NEL services 生命周期 | `pipeline.py` / `backends/` | M-C |
| D12 | P2 草图：proxy 拦截链 / 观测 sink / docs 站 | NEL interceptor / export、Skills mkdocs | `agent_eval/proxy.py` / `report/sinks.py` / `docs-site/` | P2 |

P0 卫生债（死字段、文档漂移、`--resume` 假实现等）不在本设计内展开，按对标文档 §6.1–6.3 先行清理；D8 会**真实现** `--resume`，与该清理天然合并。

---

## 3. 详细设计

### D1 recipe_store 闭环（量化 recipe 沉淀复用）

**现状**：`recipe_store.py` 完整实现了 `RepoDirStore.load/pin`，但**全仓无生产调用方**；`HfDatasetStore` 是 stub（`recipe_store.py:59-63`）；`QuantConfig.recipe` 字段无人消费（`config.py:39`）；`recipes/quant/` 目录只有 README。

**借鉴**：Skills 把 benchmark 默认配置钉在 `dataset/<name>/__init__.py` 作为单一事实源；aqa 的 SKILL.md 早已承诺「pin the recipe so later runs reuse a validated recipe」——兑现它。

**设计**：
- **pin 时机**：pipeline 在 **gate 判 GO** 之后，把本次实际生效的量化配置（取自 manifest 已记录的 `quant_command` + `QuantConfig.model_dump()`）写入 `recipes/quant/<org>__<model>.json`。gate 未过不 pin——pin 的语义是「已验证」，不是「跑过」。
- **load 时机**：`QuantConfig.recipe` 取值语义——
  - `"auto"`（默认）：先查 pinned recipe（命中且 scheme 一致 → 直接复用其命令，跳过 agent-optimize）；未命中才走 `agent.optimize` / 确定性 builder；
  - 其余值：作为 key（或 repo_dir 相对路径）显式加载，找不到报错。
- **接线点**：`pipeline.py` quantize 相位前加 `_resolve_pinned_recipe(model, cfg.quant)`；命中时走 `run_quant_command(recipe["command"], ...)`（`quantize/__init__.py:73-77` 已支持预构建命令）。
- pinned JSON schema：`{model, scheme, method, algorithm, command, export_format, source: agent_optimized|deterministic, gate: {verdict, at}, aqa_version}`。
- `HfDatasetStore`：本里程碑只把 stub 换成「download/upload `recipes/` 目录到配套 HF dataset」的薄实现（复用 `sinks.py` 已有的 token 解析与 Xet fallback），团队共享 recipe；失败降级为本地。

**验收**：同一模型同一 scheme 第二次 run 不再调用 agent-optimize（manifest 记录 `recipe_source: pinned`）；单测覆盖 pin/load/复用全链路。

### D2 scheme registry 真化

**现状**：`recipes/schemes/{mxfp4,mxfp8,nvfp4,w4a16}.yaml` 不被任何代码加载；`QuantConfig.scheme` 自由字符串直通 CLI（`config.py:36`）。

**借鉴**：Skills 用 CI 守住全部 benchmark 配置正确性——前提是配置被代码真正加载。

**设计**：
- `config.py` 加载 run.yaml 时读取 schemes 目录：未知 scheme → **warning 而非 error**（auto-round 支持的 scheme 多于仓内 4 个，硬报错会误伤）；已知 scheme → 用 YAML 里的约束（如 MX 系 `group_size=32` 固定）做交叉校验，冲突报 `RecipeError`。
- scheme YAML 增加可选字段供 optimize/validate 消费：`fixed_group_size`、`export_format_hint`、`notes`（已有）。
- `agent/optimize.py` 的 `validate_command` 读取同一 YAML 约束，拒绝违反 scheme 硬约束的 agent 命令。

**验收**：`scheme: MXFP4` + 显式 `layer_config` 里 group_size≠32 的配置在 load 时报错；未知 scheme 只警告。`tests/test_recipes.py`（D9）覆盖 schemes YAML schema。

### D3 外部 recipe 目录注入

**现状**：`BenchmarkRegistry` 根目录写死 `_RECIPE_ROOT`（仓内 `recipes/benchmarks`，`eval/registry.py:21`）；runner 有注入口（`runner.py:63`）但 config/CLI 没暴露。

**借鉴**：Skills 的 external benchmarks——`benchmark_map.json` + 环境变量，私有评测集不进主仓也能跑。

**设计**：
- `BenchmarkRegistry(roots: list[Path])`：多目录扫描，**后面的目录覆盖同名 recipe**（团队目录覆盖仓内默认值）；单 root 行为不变。
- `EvalConfig.extra_recipe_dirs: list[str]` + 环境变量 `AQA_RECIPE_DIRS`（冒号分隔），runner 统一装配。
- 文档化目录约定：`README.md`（目录说明）+ `<name>.yaml`（recipe）。不引入 `benchmark_map.json` 新格式——aqa 的 recipe 本身就是自包含 YAML，目录即 map。

**验收**：仓外临时目录放一个 `mybench.yaml`，`AQA_RECIPE_DIRS=... aqa run` 直接跑通；同名覆盖顺序有测试。

### D4 benchmark 自描述扩展

**现状**：`BenchmarkRecipe` 已有 name/harness/type/dataset/prompt/scoring/description（`eval/registry.py:83-110`），无数据准备与已知问题字段。

**借鉴**：Skills 的五步新增流程 + 每 benchmark 记录已知问题（如 SWE-bench gold patch 假阴性）的诚实传统。

**设计**：
- schema 加两个可选字段：`prepare: str`（数据准备命令，如 `python -m aqa.data.prepare xxx` 或任意 shell；run 前若 dataset 不可达则提示执行）、`known_issues: str`（已知坑，注入 fix-loop 的 operator context 与评估报告）。
- 新增 benchmark 五步流程写进 `docs/`（建 yaml → 选 harness → 选 scorer → 冒烟 → CI 校验），与 D9 的测试联动。
- `aqa list` 输出附带 description/known_issues 摘要。

**验收**：新增一个带 `prepare`/`known_issues` 的 recipe，`aqa list` 与 fix prompt 中可见；schema 严格模式（`extra="forbid"`）不破坏既有 16 个 recipe。

### D5 per-problem raw 落盘 + `aqa rescore`

**现状**：`BenchmarkResult.per_sample` 只有 0/1 数组（`eval/registry.py:119`）；lm_eval 的 `--log_samples` 输出不持久化；想换 scorer/换阈值只能整跑重推。

**借鉴**：NEL 的 `results.jsonl`（每 problem 一行：problem_idx/repeat/reward/expected_answer/model_response/metadata.category）+ Skills 的「raw 落盘 → 判分追加字段 → 离线重算」。

**设计**：
- **统一 raw schema**（NEL 对齐）：`out_dir/eval/<bench>/raw.jsonl`，每行
  ```json
  {"problem_idx": 0, "repeat": 0, "prompt": "...", "model_response": "...",
   "expected_answer": "...", "reward": 1.0, "failure_class": null,
   "category": "...", "latency_s": 1.23}
  ```
- **harness 侧落盘**：native/diffusion 直接写（生成在自己手里）；lm_eval 用 `--log_samples` 输出目录转存为同一 schema（response/expected 从 lm_eval samples 提取，缺失字段置 null，不强造）。
- **`aqa rescore <run_dir> [--scorer X] [--threshold T]`**：从 raw.jsonl 离线重算分数并改写 `accuracy.json`（native/diffusion 全支持；lm_eval 路径在字段完整时支持换 metric 重聚合，否则报清晰错误）。重算不重推。
- raw.jsonl 计入 manifest artifacts；hf_dataset sink 上传时经 scrub 洗密（`report/scrub.py` 现有机制）。

**验收**：跑 gsm8k_native + drawbench_clip 后 raw.jsonl 完整；改 threshold 后 `aqa rescore` 秒级出新分数且与重跑一致；lm_eval 路径字段缺失时错误信息可读。

### D6 失败分类学

**现状**：agentic 路径有干净分母分类（`agent_eval/verifier.py:30-43`：ok/model_wrong/infra_error/parse_error/timeout）；lm_eval/native 路径只有「全部 0 分 ⇒ 判失败」（`pipeline.py:278-288`），无法区分「模型坏了」与「基建坏了」。

**借鉴**：NEL 的 6 类失败分类（refusal/format_error/timeout/rate_limit/empty/infra_error）+ `failure_analysis.json`。

**设计**：
- 新 `aqa/eval/failure.py`：`FailureClass` 枚举 = NEL 6 类 ∪ aqa 现有 5 类（ok/model_wrong/parse_error/infra_error/timeout），harness 共用一套正则/启发式分类器（refusal 词表、空输出、JSON/markdown 格式校验等）；agentic verifier 迁移到该模块，行为不变。
- 各 harness 逐样本填 `failure_class`（写入 D5 的 raw.jsonl）；`report/` 聚合产出 `failure_analysis.json`（per benchmark × class 计数）。
- **量化语义**：gate reasons 中附失败构成（例：`format_error 从 0.1% → 12%` 是典型的量化退化信号，区别于 `infra_error` 全集）。先进报告，不进 gate 判定逻辑（保持 gate 简单）。

**验收**：人为制造空输出/拒绝回答的 fixture，分类正确；`failure_analysis.json` 进 manifest；agentic 既有测试不破。

### D7 多次采样 repeats + 聚合

**现状**：每个 benchmark 跑一次；gate 的可选显著性（McNemar，`report/__init__.py:51-70`）只在有 baseline 对比时生效；无方差概念。

**借鉴**：Skills 的 `--benchmarks=gsm8k:4`（高温重复采样 N 次）→ pass@1[avg-of-k]/majority@k/pass@k + 跨 run std/stderr；`robust_eval` 的 prompt_sensitivity 留作后续。

**设计**：
- `EvalConfig.repeats: int = 1`；benchmark 列表项支持 `"gsm8k:4"` 语法覆盖全局值（解析放在 config 层，保持 runner 干净）。
- runner 对 repeat i 以 `seed+i` 跑，raw.jsonl 按 `repeat` 字段区分（D5 schema 已预留）；结果聚合：
  - **pass@1[avg-of-N] ± std**：只需 per-sample 0/1，一期就做；
  - **majority@k / pass@k**：需要 raw 答案文本，依赖 D5 落盘后二期做。
- `accuracy.json` 的 results 行扩展 `{score, metric, num_samples, repeats, std}`；gate 默认对聚合均值判定，显著性检验输入改用「均值 ± 配对样本」。

**验收**：`repeats: 3` 跑 gsm8k_native，std 正确落盘；gate 在均值判定下行为与现有一致（repeats=1 时完全不变）。

### D8 suite 断点续跑 + 分片 merge

**现状**：`--resume` 从 CLI 一路传到两个 executor 然后被静默忽略；diffusion harness 内部有文件级断点续跑（`diffusion.py:184-230`），suite 级没有；无分片。

**借鉴**：NEL 的 per-benchmark checkpoint + failure isolation + `--resume`；env-var 透明分片（`NEL_SHARD_IDX/NEL_TOTAL_SHARDS`）+ `nel eval merge` 重算全局指标。

**设计**：
- **续跑**：evaluate 相位每个 benchmark（bucket）完成后写 `out_dir/eval/<bench>/.done`（含 recipe hash + model 路径 + repeats 数，三者不一致则 done 无效重跑）。`--resume` 时跳过有效 done 项，从 raw.jsonl 复原 `BenchmarkResult`。pipeline 级 resume 只覆盖 evaluate（量化产物存在性检查先行：checkpoint 在且 `.quant_command.json` 匹配配置则跳过 quantize）。
- **分片**：`AQA_SHARD_IDX` / `AQA_SHARD_TOTAL` 环境变量对 `cfg.benchmarks` 列表做确定性交错切分（benchmark 粒度，不切样本——样本级切分留给 harness 自带 limit/seed）；新增 `aqa merge <dir...> -o <dir>` 合并多分片的 accuracy.json + raw.jsonl 并重算聚合指标（数学上正确的前提是同 recipe hash，merge 时校验）。
- executor 两端的 `--resume` 形参真正生效（local 直接传、ssh 透传到远端 `aqa run --resume`）。

**验收**：kill 掉一个 3-benchmark run 的中段，`--resume` 后只重跑未完成项且 accuracy.json 与一次性跑完一致；2 分片 + merge 结果与单跑一致。

### D9 recipe 配置 CI 校验

**现状**：无 CI；16 个 benchmark recipe 与 4 个 scheme YAML 只有被用到时才解析。

**借鉴**：Skills 的 `tests/test_configs.py`/`test_datasets.py`/`test_metrics.py`。

**设计**：`tests/test_recipes.py`——遍历仓内全部 recipe/scheme YAML：schema 校验通过、harness/scorer 在 registry 可解析、`prompt` 占位符与 `target_field` 不自相矛盾、diffusion recipe 必须显式声明 harness（裸任务名会 fallback lm_eval 的坑已在 plan 里记过）。同时把 D2 的 schemes 校验、D3 的多目录覆盖语义钉死。

**验收**：故意改坏一个 recipe 字段，测试精确报错；全量 pytest 保持 <10s。

### D10 gate 指标方向标志

**现状**：`evaluate_gate` 只有「越高越好」语义（`report/__init__.py:45` `drop = b.score - q.score`）；diffusion plan 的 M2+ 已列此需求（FID/LPIPS 越低越好）。

**借鉴**：NEL 的 metric 元数据（score 方向是指标定义的一部分）。

**设计**：`ScoringSpec.higher_is_better: bool = True`（recipe 声明，默认不破坏现有行为）；`evaluate_gate` 按方向取反 drop；`accuracy.json` results 行带 `higher_is_better`。与 diffusion harness 的 FID/ImageReward 指标（M3）配套。

**验收**：一个 `higher_is_better: false` 的 fake recipe，量化变差时 gate 正确 NO-GO。

### D11 部署冒烟 stage

**现状**：量化产物加载失败目前混在 evaluate 相位暴露（全 0 分启发式）；vllm_omni 的 `block_name_to_quantize` 缺失这类问题要等生图才发现。

**借鉴**：NEL 的 deploy/evaluate 正交——`services` 统一 `start/health_wait/stop`，加载健康检查先于评估。

**设计**：quantize → evaluate 之间插入轻量 **deploy smoke**（可关）：backend=vllm → `vllm serve` 拉起 + `/health` + 一条 1-token 生成即停；backend=vllm_omni → `Omni(model=...)` 加载 + 2 步生图即停；hf → 跳过。失败分类为 `infra_error`（接 D6），直接终止 run 并把错误喂 fix-loop——比「跑完整套评估再发现全 0」省时且语义正确。checkpoint 结构校验（如 transformer `config.json` 的 `quantization_config.block_name_to_quantize`）作为 smoke 的静态预检。

**验收**：人为删掉 checkpoint 的 `block_name_to_quantize`，run 在 smoke 阶段以 infra_error 终止，fix prompt 含明确证据。

### D12 P2 草图（本阶段不实现，只定方向）

- **Interceptor proxy 链**（NEL）：扩展 `agent_eval/proxy.py` 的 ServingProxy 为拦截链：`drop_params`（屏蔽量化后端不支持的采样参数）、`reasoning`（`<think>` 归一化）、SQLite 缓存（baseline 重跑零成本）。先服务 agentic 评测，再考虑接到 lm_eval 的 vllm 后端。
- **观测 sink**：按 `aqa.sinks` entry-point 加 `wandb`/`mlflow`（NEL export 的对应物）；hf_dataset leaderboard 写回是 aqa 独有优势，继续强化。
- **docs 站**（Skills mkdocs Material + `strict: true`）：README 已 610 行近极限；拆 Getting started / Quantization cookbook（直接吸收 `/home/kaokaolv/auto_agent/recipes/` 的 9 份 per-model 实战）/ Evaluation 分类页 / Agent 自愈。
- **远期**：`nel serve` 式 environment 服务化（仅当接训练侧）、XSTest/garak 安全评估（以 recipe 收录即可，无需架构改动）。

---

## 4. 里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M-A 卫生与基础**（约 1 周，可立即开始） | P0 卫生债（对标文档 §6.1–6.3）+ D1 recipe_store 闭环 + D2 scheme 校验 + D9 recipe CI | 全量测试绿；同模型二次 run 命中 pinned recipe；坏 recipe 被测试拦截 |
| **M-B 评估规模化**（2–3 周） | D3 外部 recipe 注入 + D4 自描述扩展 + D5 raw 落盘/rescore + D6 失败分类 + D7 repeats + D8 resume/分片 | 仓外 recipe 直接可跑；`aqa rescore` 离线重算一致；3-benchmark run 可 kill→resume；2 分片 merge 一致 |
| **M-C 量化深化**（与 M-B 并行，联动 diffusion plan M2/M3） | D10 gate 方向 + D11 部署冒烟 +（diffusion plan 已定项：Z-Image/FLUX.2、MXFP8、optimize DiT 分支） | 加载失败在 smoke 阶段以 infra_error 终止；diffusion 全链路含失败分类报告 |
| **M-D benchmark 迁移流水线**（M-B 地基之后；契约细节见 `nemo_skills_deep_dive.md`） | ① `module::Class` 动态定位（scorer/evaluator/metrics 三处注册表支持路径形式）；② 聚合算术移植（无偏 pass@k 组合数、majority@k tie-break、std_dev/std_err_across_runs → `report/stats.py`）；③ `math_verify` 判分接入 scorer 库；④ 按模型需求逐类迁移 benchmark：**竞赛数学（AIME24/25/26、OMNI-MATH）→ 代码（LiveCodeBench 时间切片/BigCodeBench，code_exec 沙箱承载）→ 长上下文（RULER 动态生成）→ 多语（mmlu-prox 等纯数据）→ check_contamination**；lm_eval 已覆盖的不迁 | 每个迁入的 benchmark：recipe yaml + scorer/evaluator + CI 校验测试 + 与上游公布口径的数值对齐记录 |
| **M-E 轨迹数据化**（借鉴 Harbor：ATIF 格式 + traces export，见 [harbor training-workflows](https://github.com/harbor-framework/harbor/tree/main/docs/content/docs/training-workflows)） | ① `agent_eval` Trajectory → ATIF-v1.7 导出（`agent_eval/atif.py`）+ run 落盘 `trajectories.jsonl`；② `aqa traces export`：agent_eval 轨迹 → OpenAI messages / ShareGPT 数据集（success 过滤、parquet/HF push 可选）；③ fix-loop 修复轨迹（TrajectoryStore + KnowledgeStore patch）→ 正样本 SFT 数据集 | 单测覆盖转换/过滤/格式；真实 run 产出数据集并可用 Harbor `traces export` 交叉消费 |
| **M-F 可插拔相位 hooks（capability 组件层）** | 新增 `aqa.hooks` entry-point group + 有序 hook 链（`aqa/hooks/`）：`after_quantize` / `before_benchmark` / `after_benchmark` 等相位；validator 类 hook 可判死相位，其余 hook 出错只记日志。种子实现：`quant_sanity`（导出带 quantization_config 但无打包权重 → 判失败，实跑教训）、`eval_sanity`（全零分甄别，从 pipeline 内联逻辑迁入）。判断类能力走 skill 包（skills/、fix_playbook），机制类能力走 hooks——后续新能力（trajectory 捕获、soft-fail 等）一律以 hook 形式注册。**借鉴范围收窄为 Codex-only**（`docs/codex_harness_deep_dive.md`）：① hook 模型对齐 codex hooks crate（matcher + handler、`async` 标志、hook 输出体积上限防污染 fix-loop 签名）；② 配套借用 execpolicy（声明式命令规则 + strictest-wins + 内联单测 → `sandbox/policy.py`）与 bwrap 沙箱参数模板；③ rollout 三件套（append-only JSONL + durable/transient 持久化白名单 + log 即 resume 状态）进 `agent_eval` 轨迹链 | hooks 协议 + 注册 + 两个种子 hook 的测试；`quant_sanity` 用今晚的 0 层事故做回归 fixture |
| **P2 平台化**（按需） | D12 各项 | 逐项立项时再定 |

## 5. 风险与取舍

1. **raw 落盘的体积**：`log_samples` 全量存 prompt+response，大套件下 GB 级——sink 上传默认排除 raw（manifest 记路径），本地保留。
2. **repeats 的成本**：N 倍推理成本，默认 1 不变；pass@k/majority@k 依赖 D5，分两期避免一次性大改。
3. **lm_eval 路径的能力上限**：raw 转存依赖 lm_eval 的 samples 输出格式，换版本可能破损——转存器做版本哨兵（字段缺失时降级为「无 raw」并 warning，不炸 run）。
4. **不引入 Hydra/NeMo-Run/Slurm**：aqa 的零依赖 config 组合器与 ssh+docker executor 已够现状；执行后端的规模化等真有集群需求时再立项。
5. **与 NeMo-Gym 的迁徙保持观察**：Skills 的 benchmark 正迁往 NeMo-Gym，NEL 用 `gym://`/`skills://` URI 接入——aqa 的 D3 目录注入已等价覆盖私有场景，URI scheme 集成（`lm-eval://` 等）暂不需要，因为 aqa 的 lm_eval harness 已是深度集成而非容器外壳。

## 6. 附：NeMo 源机制映射

| aqa 落点 | 借鉴源 |
|---|---|
| D3/D4 | Skills `nemo_skills/dataset/<name>/` 目录约定 + [external-benchmarks 文档](https://nvidia-nemo.github.io/Skills/evaluation/external-benchmarks/) |
| D5 | NEL [result-format](https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/main/docs/result-format.mdx) + Skills [generation/scoring 分离](https://nvidia-nemo.github.io/Skills/pipelines/generation/) |
| D6 | NEL [scoring/failure 分类](https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/main/docs/evaluation/scoring.mdx) |
| D7 | Skills [evaluation `:N` 聚合](https://nvidia-nemo.github.io/Skills/evaluation/) |
| D8 | NEL [deployment-patterns（checkpoint/resume/shard/merge）](https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/main/docs/deployment-patterns.mdx) |
| D9 | Skills `tests/test_configs.py` |
| D11 | NEL [architecture: Model Deployment / services](https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/main/docs/architecture/index.mdx) |
| D12 | NEL interceptor proxy 链 / export；Skills [mkdocs.yml](https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/main/mkdocs.yml) |
