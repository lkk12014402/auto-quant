---
name: recipe-page
description: >
  After an aqa quantization+eval run, write or update the model's pages in the
  INC-Quantization-Recipes site (scheme recipe pages with repro commands,
  resources tables, and the multi-table accuracy page). Use when the user asks
  to 更新/回刷 recipe 文档, publish quantization results, or sync aqa run
  results into the recipe site. The recipe repo may live on another machine
  (specify its path); never commit.
---

# Recipe-page skill（aqa run → INC-Quantization-Recipes 回写）

把一次 aqa run 的产物变成 recipe 站（INC-Quantization-Recipes）的正式页面。
**样式基准（唯一权威样例）**：`key-model-recipes/qwen3-235b-a22b/` 与
`key-model-recipes/qwen3-30b-a3b-instruct-2507/`（经人工评审定稿）——
scheme 页（mxfp8/mxfp4/nvfp4）+ **按 scheme 拆分的 benchmark 页** + 模型页，
不要自由发挥结构。

## 铁律

- **数字必须来自 run 产物**：分数读 `eval/accuracy.json` + `baseline/accuracy.json`；
  命令读 `.quant_command.json` / `run_manifest.json` 的 commands 段；版本读
  manifest `environment.packages`；体积读 `du -sh` 实测。**不允许凭印象填数**；
  没有 run 背书的段落写 `_待补_` 并注明缺什么 run——**估计值必须标"估计"**，
  禁止把估计写成测量值（占位符混入正式页面是被用户抓过的真事故）。
- **HF 链接必须验证存在**（`HF API /api/models/<id>` 返回 200 才放上页面）；
  拿不准 org（INCModel/INCModel2.../INCModelSharing 并存）就逐个验证。
- **不 commit**（改动留在工作区由人审查）；**不跑 `tools/build.py`**（会覆盖手改）；
  不直推 main——走 PR。
- front matter 只沿用现有约定（title/parent/grand_parent/nav_order/permalink），
  不新增键；Recipe 概念表（V1/V2/V3 块）原样保留，新增内容全部在其后追加。
- **导航排序按 scheme 交错**：`mxfp8-recipe(1) → mxfp8-benchmark(2) →
  mxfp4-recipe(3) → mxfp4-benchmark(4) → nvfp4-recipe(5) → nvfp4-benchmark(6)`——
  每个 scheme 的 Recipe 与它的 Benchmark 紧邻（两个样例模型已统一成此序）。
- recipe 站不在本机时：先确认路径/主机（ssh + scp/rsync 或直接远端编辑），
  不要假设路径存在。

## 页面结构与内容来源

### `{scheme}-recipe.md`（固定六段）

| 段 | 内容 | 数据来源 |
|---|---|---|
| Recipe | 现有概念表（V1/V2/V3 块） | **原样保留**，一个字不动 |
| Implementation | V1/V2/V3 ↔ auto-round 参数（`--scheme X --iters N` / `--static_kv_dtype fp8` / `--static_attention_dtype fp8`）与 vllm 侧（`kv_cache_dtype=fp8`）对照表 + 量化模型命名 | 命名按 aqa 规范：`<Model>-<Scheme>[-Mixed][-FP8KV][-FP8Attn][-CT][-RTN]-AutoRound` |
| Quantization | 可复制的命令块（首行注释 env 版本）+ 校准数据 + 导出格式 | `.quant_command.json` + manifest `environment.packages` |
| Resources | **双表**：量化资源（tuning/RTN × 硬件）+ 推理资源（V1/V2/V3 × 硬件 TP 配置）；硬件档固定：Intel B60 24G / CRI 160G / RTX 5090 32G / RTX 6000 48G / H200 141G / B200 192G | 可行性由 checkpoint 体积 vs 显存确定性推导；峰值显存/时长由 run 实测回填 |
| Evaluation | lm_eval / evalscope 命令块 + 环境版本 + v2/v3 的 `kv_cache_dtype=fp8` 差异 | manifest commands.evaluate |
| Deployment notes | vllm 版本要求、硬件特殊项（如 FlashInfer N≥128、NVFP4 的 sm100 原生/marlin 回退） | run 日志 + 已验证事实 |

### `{scheme}-benchmark.md`（按 scheme 拆分，页内自带复现命令）

不用单页 `accuracy-result.md`——按 scheme 拆：`mxfp8-benchmark.md` /
`mxfp4-benchmark.md`（/ `nvfp4-benchmark.md`），title 用 "MXFP8 Benchmark" 这类
（不叫 "Accuracy Result"）。每页内容：

- 本 scheme 的 lm_eval 表（V1/V2/V3 行，保留 Ratio + notes）
- 长上下文表（RULER，有则保留并转 MD 表）
- **复现命令块**（该 scheme 的实测 lm_eval/evalscope 命令 + 环境版本行）——
  页面自包含，不用跳 recipe 页
- 脚注：量化模型 HF 链接（V1/V2/V3 全列）+ 回链对应 recipe 页
- **口径纪律**：不同口径的数字不拼一张表（如 128k RULER 口径的 gsm8k 不能
  借进 8k 的 lm_eval 主表）；缺数据的行写 `_待补_` 并注明缺什么口径
- 评估口径的特殊说明要保留在页面（如"gsm8k 不套 chat template"这类模型级经验）

### 模型二级页（原空桩）

简介（2-3 句）/ 原模型 + 量化模型 HF 链接 / 硬件需求总表（BF16 vs 各 scheme
的体积、最低/推荐配置）/ scheme 页导航。`overall-status.md` 大表对应行同步
（recipe ✓ 链接 + 模型链接）。

## 流程

1. 确认目标模型目录与现有页面（`ls key-model-recipes/<model>/`）；
   新模型先按 README 的三级约定建桩页 + 更新 `overall-status.md` 和
   `index.md` 的 Revision History
2. 对每个 scheme：有 run 产物就填实，没有就 `_待补_` 并写明缺哪个 run
   （**不要把别的 scheme 的数字搬过来**）
3. `bundle exec jekyll build` 验证（在能 build 的机器/容器上；本仓库有
   `_config.yml` 即可），给用户本地预览地址
4. 输出改动清单（哪些文件、每页加了什么段、哪些数字待补）交人审查

## 常见错误（不要犯）

- 凭 recipe 表"脑补"命令参数——命令必须从 `.quant_command.json`/manifest 来
- 把 tuning 的耗时写在 RTN 行（量化方式不同资源完全不同）
- 给没有 V2/V3 的 scheme 硬凑 V2/V3 段（如 NVFP4 的 V1-only 要写明原因）
- 改 front matter 的 nav_order/permalink（会破坏导航与外链）
