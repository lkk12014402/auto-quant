# auto-round 知识刷新流程（references refresh）

auto-round 迭代快，aqa 内置的量化知识（`aqa/agent/recipes/*.md`、
`docs/autoround_full_reference.md`）会随版本过期。本流程是更新机制：
**增量同步的存储**（knowledge store 的 `references/` 子树）+ **确定性圈定变更范围**
（`aqa knowledge diff`）+ **owner 复核刷新**（agent 读 PR/源码，人/agent 写文档）
+ **推送共享**（`aqa knowledge push-references`）。

## 角色与存储

- `references/` 子树与 fix-knowledge 共用同一个 HF dataset（`agent.knowledge.repo`），
  每次 run 开始 `sync_down` 拉最新；optimize agent 的 prompt 知识
  （`load_recipe_knowledge`）**优先读同步副本**，读不到回退 aqa 打包版（离线兜底）。
- `references/MANIFEST.json` 是新鲜度锚点：`auto_round_ref`（文档验证时对应的
  auto-round 版本/commit）、`updated_at`、`note`、`files`。
- **整个 knowledge store 的结构（lessons/patches/trajectories/stats/references）
  与 MANIFEST 字段、更新语义（整树替换 vs 单文件增量）都有显式定义**：
  `aqa/agent/knowledge_layout.yaml`——改这里的任何行为前先读它。

## 触发时机

- auto-round 发新 release（`git tag` / GitHub Releases 页），或
- run 中踩到"文档说的和代码做的不一致"（如 CLI flag 不存在、行为变更）。

## 刷新步骤（owner 执行，agent 可驱动）

```bash
# 0. 看新鲜度：记录的 ref 是什么
aqa knowledge show

# 1. 圈定变更范围：自上次验证以来，知识相关路径的提交清单
aqa knowledge diff --auto-round-dir /home/kaokaolv/auto_agent/auto-round
```

2. **逐个变更定性**（agent 读 PR/commit diff + 源码，确定性判断）：
   - 影响面分类：新 scheme / CLI flag 增删 / 行为变更（model_free、matcher、
     导出、fused MoE）/ 新模型支持 / 仅重构（不影响知识）
   - 只读 `diff` 列出的文件，不要全仓重读
3. **复核受影响的知识条目**：参考文档里的论断都带 `文件:行号` 引用——
   逐个回去看那些行是否仍然成立（引用就是验证锚点）
4. **改文档**：在工作副本（如 `aqa/agent/recipes/` 的拷贝）里更新
5. **推送共享**：

```bash
aqa knowledge push-references <更新后的目录> \
    --repo <knowledge dataset，如 lvkaokao/lb_local> \
    --auto-round-ref <新验证版本，如 v0.15.1> \
    --note "同步 #PR号: <变更摘要>"
```

6. 在 run journal 里记一笔：刷新了什么、依据的 PR/版本。

## 纪律

- **文档论断必须可验证**：改知识文档时保持 `文件:行号` 引用习惯——
  这是下次刷新的验证锚点；没有引用的论断刷新时无法快速证伪
- **宁可标注过期，不要留错误**：来不及逐条复核的章节，在文档头部标注
  "本节验证于 <旧版本>"，比静默过期安全
- **不要在 run 中途顺手刷新**：刷新是独立任务，做完推送后再开新 run，
  避免一次 run 里知识前后不一致
