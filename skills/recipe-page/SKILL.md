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
**样式基准（唯一权威样例）**：`key-model-recipes/qwen3-235b-a22b/`（mxfp4/mxfp8/
nvfp4 三个 scheme 页 + accuracy-result.md + 模型页），不要自由发挥结构。

## 铁律

- **数字必须来自 run 产物**：分数读 `eval/accuracy.json` + `baseline/accuracy.json`；
  命令读 `.quant_command.json` / `run_manifest.json` 的 commands 段；版本读
  manifest `environment.packages`；体积读 `du -sh` 实测。**不允许凭印象填数**；
  没有 run 背书的段落写 `_待补_` 并注明缺什么 run。
- **不 commit**（改动留在工作区由人审查）；**不跑 `tools/build.py`**（会覆盖手改）；
  不直推 main——走 PR。
- front matter 只沿用现有约定（title/parent/grand_parent/nav_order/permalink），
  不新增键；Recipe 概念表（V1/V2/V3 块）原样保留，新增内容全部在其后追加。
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

### `accuracy-result.md`（分表制）

lm_eval 表（保留现有数字与 Ratio 列；V1/V2/V3 行命名）/ evalscope 表
（aime/gpqa_diamond，标 repeats）/ agentic 表（terminal-bench 等）/
长上下文表（RULER，有则保留并转 MD 表）。新行附硬件与版本 notes；
旧数字保留并标注来源。

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
