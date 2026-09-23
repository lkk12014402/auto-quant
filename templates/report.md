# <Model> <Scheme> 量化回溯报告（第⑤步产物）

> 模板使用：run 结束后对照真实产物填写，**不凭记忆**——每个数字指回
> run 目录里的文件（accuracy.json / gate.json / manifest / 日志）。
> 章节不许删。收尾完成的定义 = 本报告填完 + 经验路由表无空行 +
> `aqa artifacts check <run_dir>` 全过。

- 模型 / 方案：`<model>` / `<scheme + method + algorithm>`
- run 目录：`<output.dir>`；run_id：`<>`
- 日期 / 机器：`<>` / `<主机 + GPU>`

## 1. 结果总览

| benchmark（口径） | baseline | 量化 | Δ | 备注 |
|---|---|---|---|---|
| <piqa (acc)> | | | | |
| **mean** | | | | |

- gate 结论：**<GO / NO-GO>**（与 `gate.json` 一致；阈值 `<>`，口径配对说明）
- 结论一句话：`<协议内不可区分 / 掉点 X pp——注意措辞限定在已测协议内>`
- 未覆盖面声明：`<长上下文 / agentic / 吞吐 / 多模态 / thinking-on...>`

## 2. sanity check 证据

- 产物体积：`<实测> GB` vs 位宽预期 `<analysis.md §4 的估算> GB` → <符合/不符及解释>
- 量化键抽查：`<dtype / 层数统计（实测命令）>`
- 导出完备性：`<源权重完备性扫描 / 冒烟加载 / 出分验证结论>`
- model_free 专项检查：`<日志无 "falling back to the regular AutoRound flow" 等>`

## 3. 回传核对

- hf_dataset sink：`<repo 路径>`，文件数 `<>`，与本地 run 目录一致 ✅/❌
- 状态回写 / 聚合记录：`<status/*.json 是否生成>`
- knowledge / trajectories：`<是否随库回传>`
- 权重发布（如配 provider.upload）：`<model repo 地址；model_card.md 已作为 README ✅>`

## 4. 问题清单（本次遇到的坑，逐条）

| # | 现象 | 根因 | 修法 | 状态 |
|---|---|---|---|---|
| ① | | | | <已修/绕过/未决> |

## 5. 经验沉淀路由表（硬要求：每行必填，不允许空白行）

每行两个状态列：**已落地（附路径/链接）** 或 **明确不沉淀（附理由）**。
本次没有该类经验就写「本次无」，不许删行。落点分**本仓**（文件改/拷到
对应位置并在状态列标明路径即可；**git 提交交给 user，agent 不擅自提交**——
但只回传 HF 不回流本仓 = 知识只存了一半）与 HF/外部。

| 经验类型 | 落点 | 机制 | 本次内容（一句话） | 状态 |
|---|---|---|---|---|
| 本模型实例文档与脚本（analysis 成稿、quant/eval/setup 脚本） | **本仓** `cookbooks/` | 成稿拷成 `cookbooks/<Model>-quantization-recipe.md` + 对应 `*.sh`（参照现有实例） | | |
| 报错→修法（可复用 fix pattern） | HF knowledge store `knowledge/` | 随 run 回传（`agent.knowledge.repo`） | | |
| auto-round 行为与文档不符 | HF references 增量层 | `docs/knowledge_refresh.md` 复核后 `aqa knowledge push-references` | | |
| 应随 aqa 发布的方案性知识 | **本仓** `aqa/agent/recipes/` / `docs/autoround_full_reference.md` | 直接改打包层文档（与增量层的分工见 knowledge_refresh.md） | | |
| 代码修复（依赖/上游 bug） | **本仓** `patches/` | `aqa capture-patch`；值得上游的 `aqa publish-patch` | | |
| 本模型可复用配方 | **本仓** `recipes/quant/<org>__<model>.json` | RecipeStore pin（quantize skill 第 4 步） | | |
| 对外 recipe 成果 | INC-Quantization-Recipes 站 | `skills/recipe-page/SKILL.md` | | |
| owner/aqa 排错轨迹 | HF `trajectories/`（全量）＋ **本仓** `demo/trajectory_export/`（代表性，可选） | `skills/session-sediment/SKILL.md`；代表性轨迹 `aqa traces convert` 验证后拷入 | | |
| 通用坑（跨模型适用） | **本仓** AGENTS.md §9 / day0 playbook | 直接改文档 | | |
| skill 本身有缺口 | **本仓** `skills/*/SKILL.md` | 直接改 skill | | |

## 6. 后续行动

- `<未决坑的跟进项 / 建议补测的协议 / 值得上游的 issue·PR（含链接）>`
