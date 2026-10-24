# RFC 0002 — Recipe Manifest 导出（INC-Recipes 兼容）+ 逐模块精度覆盖 + 复现路径

状态：Draft（§2–§4 已落地，§5 复现为下一步）· 作者：owner agent
对标：[intel-sandbox/INC-Quantization-Recipes](https://github.com/intel-sandbox/INC-Quantization-Recipes)（PR #5 `suyue/reorg`，`model-recipe.schema.json` v1.0）
依赖：RFC 0001（Session 状态机 + 阶段门 + Recipe KB）

> 本 RFC 回答一个新痛点：**一次 aqa 量化 run 结束后，产物里没有一份「可被 recipe 站直接吸收、
> 且把最终模型逐模块精度讲清楚」的清单。** `manifest.json` 是 aqa 内部复现清单（面向自己），
> 不是 INC-Recipes 的 model-recipe 格式（面向站点 + 人审 + 跨模型对比）。结论是：在 **CLOSE**
> 阶段**多产一份 `recipe.yaml`**——核心块严格对齐 INC schema 1.0，再叠两个**可剥离的 aqa 扩展**
> （`quantization_details` 逐模块精度覆盖 + `provenance` 轨迹）。并给出**用这份 yaml 反向复现**
> 的设计（量化结构复现 + 评估数值复现，下一步实现）。

---

## 0. TL;DR

| 支柱 | 现状问题 | 目标 | 落地 |
|---|---|---|---|
| E1 清单导出 | CLOSE 只产 aqa 私有 `manifest.json`；无 recipe 站可吸收的格式 | CLOSE 多产 `recipe.yaml`（INC model-recipe v1.0 兼容） | ✅ `aqa/report/recipe_manifest.py` |
| E2 逐模块精度 | 没有「最终模型哪些 module 落在哪个精度、参数占比多少」的结构化产物 | 新增 `quantization_details`：全模块分组 → scheme → 参数占比 → effective_bits | ✅ `aqa/report/quant_details.py` |
| E3 诚实可审 | 导出的 recipe 不能谎称「已验证」 | INC 协议 1.0 的 `issues` 完整性规则：缺命令/无 fingerprint/未 GO → 列 `issues`，不写非 schema 的 `status` | ✅ emitter 内建 |
| E4 可回灌 | aqa 扩展字段会违反 INC `additionalProperties:false` | 扩展只在**顶层**且**可剥离**；`to_inc_strict()` 投影后过 INC schema | ✅ `to_inc_strict()` |
| E5 反向复现 | 有了 recipe.yaml 却不能「拿它重放一遍并校验」 | `aqa reproduce <manifest> --variant <id>`：复用相位机 + VERIFY 三面校验 | ✅ `aqa/reproduce.py` + CLI + 外部 agent API |

落地顺序：**先 `quantization_details` 计算器 + CLOSE `recipe.yaml` emitter（本次）→ 再 `aqa reproduce`（下一步）**。

---

## 0.1 实现状态（本次提交）

| 项 | 状态 | 落地文件 |
|---|---|---|
| `quantization_details` 计算器（**首选**权重 header 地面真相：逐 tensor 分组 + 导出 artifact 判量化 + 参数精确；**回退**解析式 dense/MoE 估算 + effective_bits） | ✅ | `aqa/report/quant_details.py` + 新增 `aqa/report/weight_map.py` |
| ANALYZE 真实结构图（`model_structure` 读本地权重 header；权重不在本地则降级为 note，绝不下载） | ✅ | `aqa/report/weight_map.py`、`aqa/report/phase_artifacts.py`（`write_analysis`） |
| `recipe.yaml` emitter（INC 核心块 + aqa 扩展 + `to_inc_strict()` 投影 + 字面块 YAML dump）+ CLOSE 同时写 `<model.id>.yaml`（strict，可直接提交 INC） | ✅ | 新增 `aqa/report/recipe_manifest.py` |
| CLOSE 接线（`write_manifest` 后 best-effort 调用，仅 quantize run；读 `resolve_export_dir` 的导出 config） | ✅ | `aqa/pipeline.py`（`_write_recipe_manifest`） |
| 产物契约：`layout.yaml` 增 `recipe.yaml` 条目（phase CLOSE，when `session_quantize`，sink） | ✅ | `aqa/report/layout.yaml` |
| 单测：计算器（dense/MoE/未知架构/两种导出 config）+ **weight_shapes 地面真相（hybrid 分组拆分 + 导出 artifact 判量化 + MTP 保 BF16 + ANALYZE 结构图 + 离线降级）** + emitter（documented/draft 降级/无命令降级/非 quantize/round-trip/KV-attn）+ 真 INC schema jsonschema 校验 | ✅ | `tests/test_quant_details.py`、新增 `tests/test_weight_map.py`、`tests/test_recipe_manifest.py` |
| `aqa reproduce` + VERIFY 相位 + `reproduction_report.json` | ✅ | 新增 `aqa/reproduce.py`、`aqa/cli.py`（CLI）、`aqa/pipeline.py`（VERIFY 接线）、`aqa/session/{state,actions}.py`（Phase.VERIFY）、`tests/test_reproduce.py`；外部 agent 可调 `aqa.reproduce.reproduce_manifest` / `plan_reproduction` |
| **协议 1.0 模板对齐（2026-10-24）**：命令改用 `MODEL`/`OUTPUT_DIR`/`MODEL_PATH` 变量（无本机绝对路径）；删除非 schema 字段（`status`/`working_directory`/`tasks`/column `metric`）；operations 从真实逐模块 disposition 派生（支持 mixed）；variant id 改 `<family>-v1`；结果列只保留 `task`/`unit`；`inc_schema_errors()` + vendored schema 单测 | ✅ | `aqa/report/recipe_manifest.py`、`tests/test_recipe_manifest.py`、`tests/data/inc-model-recipe.schema.json` |

### 真实复现验证（2026-10-24）

- **Qwen3.8-27B（真实 checkpoint，eval-only）**：用已发布的 AWQ MXFP4 checkpoint
  在三张空闲 GPU 之一上跑 PIQA 全量；`input_equivalence=match`（9 个包版本逐一匹配）、
  `evaluation_reproduction=pass`（expected 0.81012 → got 0.808487，abs 0.001633，
  rel 0.002015，容差 0.03）、`verdict=reproduced`；`aqa inspect` status OK。
- **Qwen3.8-27B（真实 full：重新量化 + 评估）**：双卡 GPU 2/3 上按 manifest 冻结命令
  重跑 AWQ MXFP4，量化耗时 33.5 min，导出 19GB / 8 shards（compressed-tensors，
  `weight_scale` 齐全）；`quantization_reproduction=pass`（10 个 module group，
  0 mismatch，MXFP4 参数占比 87.6%）；PIQA expected 0.81012 → got 0.812296
  （abs 0.002176，rel 0.002686）；`verdict=reproduced`，`aqa inspect` status OK。
- **tiny GPT-2（full 模式）**：`QUANT→EVAL→CLOSE→VERIFY` 全链路（量化脚本为可复现的
  复制脚本，用于验证 phase 机与三面校验）、`quantization_reproduction=pass`、
  `evaluation_reproduction=pass`、`verdict=reproduced`。
- **tiny GPT-2（eval-only）**：`verdict=reproduced`，`aqa inspect` status OK。

> 注：`recipe.yaml` 为 **best-effort**——emitter 任何失败只 `log.warning` 不中断 run（与
> `phase_artifacts.py` 一致）。因此 `layout.yaml` 把它列为 `session_quantize` 契约项：完成的
> session 量化 run **应当**有它，缺失即 `aqa inspect` 报 contract violation（与 `env_lock.json`
> 等机读阶段产物同级）。

---

## 1. INC recipe 格式 review（PR #5 ground truth）

PR #5 把 recipe 站重组为**每模型一份 `_data/recipes/<model>.yaml`** + 一份
`model-recipe.schema.json`（JSON Schema draft 2020-12，**顶层/variant/operation/env/table 全部
`additionalProperties:false`**）。顶层六个必填块：

```
schema_version: "1.0"
model:        {id, name, source, revision?, notes?}
environments: {<env-id>: {packages[name==ver], python?, accelerator?, ...}}
operations:   {<op-id>: {label, settings[{key,label,value,unit?}]}}   # 按 ID 被 variant 引用
variants:     [{id, family, label, status, recipe, quantization, evaluation, avg_bits?, hf_url?, issues?}]
baselines?:   [{id, label, evaluation}]
result_tables:[{id, title, notes, columns[...], rows[...]}]
```

关键约束（emitter 必须满足）：

- **id 模式** `^[a-z0-9]+(?:-[a-z0-9]+)*$`（严格 kebab）——`model.id`/`variant.id`/`op-id`/
  列 id/行 id 全受限。benchmark 名如 `mmlu_pro` 带下划线：**列 id 必须 slug 成 `mmlu-pro`**，
  但 `task` 字段（`text` 类型）保留原名 `mmlu_pro`。
- **operation setting key** `^[a-z][a-z0-9_.-]*$`（**允许点号**，故 `weight.dtype` 合法）。
- **env packages** 必须 `name==version` 且 `minItems:1`、`uniqueItems`。
- **command 块**（quantization/evaluation）必填 `environment/shell:bash/working_directory:"."/variables/script`；
  evaluation 额外必填 `tasks[{id,metric}]`（`minItems:1`）。
- **status 机**：`draft` 必须带 `issues`（`minItems:1`），`quantization`/`evaluation` 可为 `null`；
  `documented` **禁止** `issues`，且 `quantization`/`evaluation` 必须是完整 command 块。
- **result row**：`variant` 行必须有 `variant` 且禁止 `baseline`；`baseline` 行反之；
  `comparison` 行**禁止** `variant`/`baseline`/`input_fingerprint`。
- **`avg_bits`** = 规范化平均权重位宽，**不含 scale bits**；**`input_fingerprint`**（`sha256:...`）
  把一行结果绑定到输入——**没有 fingerprint 的行 = 「仅上报数字（reported numbers only）」，是合法的**。

### 1.1 `operations:` 的问题 + 更好的方案

PR #5 现状的 `operations:` 有三处可改进：

1. **纯展示自由串、与执行真相脱节**：`{key: scheme, value: "Weight MXFP8 / Activation MXFP8"}`
   是给人看的，和实际 `--scheme MXFP8` 命令无结构化绑定——人审能看懂，机器无法据此复算位宽/排除层。
2. **W/A 混进一个串**：weight 与 activation 精度被塞进一个 scheme 字符串，不利于「只量权重不量激活」
   （如 `W4A16`）这类场景的机读。
3. **拼写漂移**：`lm-head`（v1）vs `lm_head`（schema 内 example）——`exclude_layers` 的值没有受控词表。

**aqa 的方案**：`operations:` 照样产出（保持与站点兼容、给人读），但**真相下沉到
`quantization_details`**（§3）——后者**首选读本地权重 header 的真实 tensor 结构**（并以导出的
`weight_scale` artifact 判定「哪个 module 被量化」），权重不在本地时回退到**导出的
`quantization_config`** + 解析式估算，是可复算、可审计的。`operations.<linear>.settings.exclude_layers`
的值由 emitter 从 `quantization_details` 里 scheme==BF16 的**可量化**分组 id 反推（稳定的 kebab id，
无拼写漂移）：只列**整族保留**的量化候选族（如 `lm-head`/`visual`/`mtp`），跳过永久保留族
（`embed`/`router`/`norms`）与**部分保留**族的 `-bf16` 分裂余项（后者的细节在 `module_groups`）。
例：Qwen3.8-27B（`--ignore_layers lm_head,embed_tokens,visual`）→ `exclude_layers: [lm-head, visual, mtp]`
（`embed` 为非 Linear 永久 BF16，不列；`mtp` 整族保留，补列）。

---

## 2. `recipe.yaml` 导出设计

### 2.1 一次 run = 一个 variant；扩展可剥离

一次 aqa run 产出**一个 variant**（本 run 的 scheme/method）。把多个 variant 并入「每模型一份
catalogue」是**另一件事**（recipe-page skill / 合并工具），不在 emitter 职责内。

emitter 产出的是 **aqa-extended manifest** = INC 核心六块 + **两个顶层可剥离扩展**：

```yaml
# ... INC 核心六块 ...
quantization_details: {<variant-id>: {...}}   # §3，aqa 扩展
provenance: {tool: aqa, run_id, created_at, gate, note}   # aqa 扩展
```

因为 INC schema **顶层 `additionalProperties:false`**，这两块放在顶层就会违反 schema。故 emitter
配一个 **`to_inc_strict(manifest)` 投影**：深拷贝后 `pop` 掉这两个顶层键，**剩余核心过 INC schema**
（已用 PR #5 的 `model-recipe.schema.json` + `jsonschema` 实证 VALID）。回灌 recipe 站前先 strip。
CLOSE 现在**同时**把该投影写成 `<out_dir>/<model.id>.yaml`（文件名等于 `model.id`，可直接
放进 `_data/recipes/` 过 INC 的 `bundle exec ruby tools/recipe_data.rb validate`）；`recipe.yaml`
保留 aqa 扩展版供本仓审计。

> 为什么不塞进 variant/recipe 的 `extensions`？variant 无 `additionalProperties`，recipe.extensions
> 只接受 `^[a-z][a-z0-9_-]*\.[a-z...]` 的**命名空间化 setting 条目**，无法承载 `module_groups` 这种
> 嵌套结构。顶层可剥离扩展是**唯一**既不违反 schema、又保留完整结构的落点。

### 2.2 auto-quant 产物 → yaml 字段映射

emitter 的输入是**已在 pipeline 早先落盘的结构化产物** + 活的 `RunConfig`/`EvalResult`，不额外采集：

| INC 块 | 来源 | 说明 |
|---|---|---|
| `model.{id,name,source}` | `run_cfg.model` | `source`=原始 id；`name`=basename；`id`=slug(name) |
| `environments.{quantization,evaluation}` | `env_lock.json`（`packages`/`accelerator`/`python`） | 按 allowlist 拆 quant/eval 两环境；`auto_round→auto-round`/`lm_eval→lm-eval` 规范化去重；`name==ver` 格式 |
| `operations.<family>-linear` | `quant.scheme` + `quantization_details` | settings 合成（data_type/scheme/element_format/exclude_layers）；命令带 `--static_kv_dtype`/`--static_attention_dtype` 时追加 `fp8-kv-cache`/`fp8-attention` op |
| `variant.recipe` | `quant.method`/`iters` | `approach:PTQ`；`algorithm:RTN`（model_free 或 iters=0）否则 `AutoRound` |
| `variant.quantization` | `quant_plan.json.command`（即 `.quant_command.json` 冻结命令） | script=真实执行命令；缺则 `null`+issue |
| `variant.evaluation` | `result.commands` + `result.benchmarks[].metric` | tasks 用**真实 metric**（无则 `unspecified`+issue）；缺命令则 `null`+issue |
| `variant.avg_bits` | `quantization_details.effective_bits` | 参数加权平均位宽（不含 scale bits） |
| `variant.hf_url` | `provider.upload.{repo,repo_name}` | 两者齐全才产；否则省略 |
| `baselines[bf16]` | `baseline` EvalResult | 仅当 baseline 有命令+benchmark 才产 |
| `result_tables[accuracy]` | `result`/`baseline` 分数 | metric 列 + `average`/`ratio` reported_summary 列；行**不带 fingerprint**（诚实标注「reported numbers only」） |
| `provenance` | `run_id`/`gate_verdict` | aqa 扩展 |

### 2.3 诚实的 status 机

emitter **默认 `draft`**，只在**全部满足**时升 `documented`：

```
documented ⟺ gate=GO  ∧  quantization 命令已录  ∧  evaluation 命令已录  ∧  所有 benchmark metric 具名
```

否则 `draft` 并收集 issues（缺量化命令 / 缺评估命令 / metric 未录 / 结果无 fingerprint / gate 非 GO）。
这确保导出的 recipe **不会谎称已验证**——与 PR #5 里 chensuyue 手标的 draft+issues 同义，但自动化。

---

## 3. `quantization_details` — 逐模块精度覆盖（新增块）

### 3.1 需求

把「最终量化模型」讲清楚：**从 module（可归类但要全面）、量化的 scheme（MXFP8？还是没量化保持
bf16？）、参数量占比**。这是 `operations:` 做不到的（见 §1.1），也是 aqa 相对 PR #5 的增量价值。

### 3.2 结构

```yaml
quantization_details:              # 顶层、按 variant-id 索引（aqa 扩展）
  <variant-id>:
    derived_from: weight_shapes | exported_quant_config | declared_scope
    params_basis: safetensors_headers          # 仅 weight_shapes 路径
    quantized_basis: export_artifacts | ignore_heuristic   # 仅 weight_shapes 路径
    total_params / quantized_params / quantized_ratio / effective_bits   # 不可算时省略
    module_groups:
      - {id, label, modules[展示 pattern], operation: <op-id>|null, scheme, layers?, params?, params_ratio?}
    kv_cache:  {scheme, element_format} | 省略
    attention: {scheme, element_format} | 省略
    note?: str                     # 参数不可算时的降级说明
```

### 3.3 首选真相来源：权重 header 地面真相（weight_shapes）

**不猜结构、也不猜参数量**——首选路径直接读**本地 safetensors header**（只读开头 8 字节
uint64 header 长度 + 该长度的 JSON，拿到每个 tensor 的 `shape`，**不加载权重、不占卡、不联网**）。
共享实现 `aqa/report/weight_map.py`：

- `read_param_shapes(model_source)`：`name → shape`。非本地目录 / 无 `*.safetensors` → `None`
  （权重未下载 / 远程 id / 其它序列化），调用方回退到 §3.4 解析式路径。
- `classify_module(name)`：**架构无关**地把每个 tensor 归入粗粒度 family（`self-attn` / `linear-attn` /
  `mlp` / `moe-experts` / `shared-expert` / `router` / `visual` / `mtp` / `embed` / `lm-head` /
  `norms` / `other`），容器优先（`visual`/`mtp` 先于其内部 `attn`/`mlp`）。新架构降级为 `other`
  而非出错——**coverage 永不缺**。

**量化 vs 保留的权威判定**（`quantized_basis`）：

- `export_artifacts`（首选）：`read_quantized_modules(quant_dir)` 读导出目录 header，凡带
  `.weight_scale` / `.weight_packed` / `.weight_zero_point` 的 module stem 即「**被量化**」。
  这比 `ignore` 列表更权威——`ignore` 不会列出**量化图之外**的 module（如 MTP：auto-round 原样复制
  为 BF16「额外 tensor」，不在 `ignore` 里，但也没有 `weight_scale`）。
- `ignore_heuristic`（无导出 artifact 时）：回退到「`.weight` 二维 且 不在导出/声明 keep 集」的子串
  启发式。
- **防错守卫**：若导出 artifact 集非空却与 base 模型 module 名零交集（命名错位 → 0 量化参数），
  自动退回 `ignore_heuristic`。

一个 family 同时含量化与保留 tensor（如 hybrid 线性注意力：`in_proj_qkv`/`out_proj` 量化、
`in_proj_a`/`conv1d`/`A_log` 保留 BF16）时，**精确拆成** `<family>-quant` 与 `<family>-bf16` 两组。
参数量 = `Σ prod(shape)`，`derived_from: weight_shapes` / `params_basis: safetensors_headers`。

> **ANALYZE 复用同一读取器**：相位 ANALYZE 的 `analysis.json` 用 `model_structure()`
> （= `read_param_shapes` + `summarize_structure`）给出**真实结构图**（family → params/tensors/层数，
> 按参数占比降序）。权重在本地则写 `model_structure`，否则写 `model_structure_note`（显式说明
> 只读 header、绝不下载）。ANALYZE 给「量化前的真实结构」，CLOSE 给「量化后的精度覆盖」，同源同口径。

### 3.4 回退真相来源：导出 config + 解析式参数

当权重不在本地（远程 id / 未下载）时，回退到**导出的 `quantization_config`** + **config dims 解析式**
参数估算（不加载权重）。计算器 `read_applied_quant()` 规范化两种真实形态：

- **auto_round**：`{quant_method:"auto-round", bits:N, extra_config:{<module>:{bits,data_type}}}`。
  `extra_config` 里 `{bits>=16}` 或 `{data_type:"float"}` = 该 module **被保留高精度（bf16）**；
  key 是**精确 module 路径**。
- **compressed-tensors**：`{quant_method:"compressed-tensors", ignore:[...], config_groups:{...}}`。
  `ignore[]` 列出保留层。

归一为 `kept_modules:set`（+ 声明 `ignore_layers` 拆分 + 永久保留项），对每个解析式分组用**代表路径
子串匹配**判保留。`derived_from` 标 `exported_quant_config`（读到导出 config）或 `declared_scope`
（仅凭声明 scope 推断）。

分组**架构感知**（dense + MoE），参数量从 config dims 解析式计算：

| 分组 id | 覆盖 | 代表路径 | 默认精度 |
|---|---|---|---|
| `embed` | 词嵌入 | `model.embed_tokens` | 永久 BF16 |
| `attn-proj` | q/k/v/o 投影 | `model.layers.0.self_attn.q_proj` | 量化 |
| `mlp`（dense） | gate/up/down | `model.layers.0.mlp.down_proj` | 量化 |
| `moe-experts`（MoE） | 专家 gate/up/down | `...mlp.experts.0.down_proj` | 量化 |
| `shared-expert`（MoE） | 共享专家 | `...mlp.shared_expert.down_proj` | 量化 |
| `router`（MoE） | 路由 gate | `model.layers.0.mlp.gate` | 永久 BF16 |
| `lm-head` | 输出头 | `lm_head` | `quant_lm_head` 才量化，否则 BF16 |
| `norms` | 各层 norm + 末 norm | `model.layers.0.input_layernorm` | 永久 BF16 |

- attn per layer = `2*H*nh*hd + 2*H*nkv*hd`；dense MLP = `L*3*H*I`；MoE experts = `L*ne*3*H*moe_i`；
  router = `L*H*ne`（保留 bf16）；embed = `V*H`；lm_head = `0`（tie）否则 `V*H`；norms = `2*L*H+H`。
- **优雅降级**：未知架构/缺 dims → `params=null`、省略 `total_params`/`effective_bits` + 写 `note`，
  但**仍给全分组 + scheme 覆盖**（coverage 永不缺）。

> **两路一致性**：解析式路径按标准 dense transformer 近似，对 hybrid/多模态模型会**系统性漏算**
> （如 Qwen3.8-27B 把 64 层全当 self-attn、漏掉 linear_attn/visual/mtp → total 24.35B，真值 27.78B，
> 偏 ~12%）。因此本地权重可读时**一律走 §3.3 地面真相**，解析式仅作离线回退。

### 3.5 位宽与 effective_bits

`_nominal_weight_bits(scheme)`：`W(\d+)A\d+`→首组；`MXFP4/NVFP4/FP4`→4；`MXFP8/FP8`→8；`FP6`→6；
`BF16/FP16`→16；否则 `None`。`effective_bits = Σ(params_g * bits_g) / Σ params_g`（不含 scale bits，
对齐 INC `avg_bits` 约定）。KV/attention 静态量化从冻结命令解析 `--static_kv_dtype`/
`--static_attention_dtype`，产 `kv_cache`/`attention` 块。

### 3.6 实例（Llama-3.3-70B，MXFP8，ignore lm_head）

```yaml
quantization_details:
  mxfp8-rtn:
    derived_from: exported_quant_config
    module_groups:
      - {id: embed,     scheme: BF16,  operation: null,         params_ratio: 0.0149}
      - {id: attn-proj, scheme: MXFP8, operation: mxfp8-linear, layers: 80, params_ratio: 0.1712}
      - {id: mlp,       scheme: MXFP8, operation: mxfp8-linear, layers: 80, params_ratio: 0.7990}
      - {id: lm-head,   scheme: BF16,  operation: null,         params_ratio: 0.0149}
      - {id: norms,     scheme: BF16,  operation: null,         params_ratio: 0.0000}
    total_params: 70553706496
    quantized_params: 68451041280
    quantized_ratio: 0.9702
    effective_bits: 8.238
```

### 3.7 实例（Qwen3.8-27B，W4A16，hybrid 多模态 · weight_shapes 地面真相）

真实验证产物（`Qwen3_5ForConditionalGeneration`：16 全注意力 + 48 线性注意力 + 视觉塔 + MTP）。
解析式路径会漏算到 24.35B，地面真相给出全貌 **27.78B**、量化 400/607 module：

```yaml
quantization_details:
  w4a16-rtn:
    derived_from: weight_shapes
    params_basis: safetensors_headers
    quantized_basis: export_artifacts
    module_groups:
      - {id: mlp,               scheme: W4A16, operation: w4a16-linear, layers: 64, params_ratio: 0.6160}
      - {id: linear-attn-quant, scheme: W4A16, operation: w4a16-linear, layers: 48, params_ratio: 0.1993}
      - {id: self-attn-quant,   scheme: W4A16, operation: w4a16-linear, layers: 16, params_ratio: 0.0604}
      - {id: embed,             scheme: BF16,  operation: null,         params_ratio: 0.0458}
      - {id: lm-head,           scheme: BF16,  operation: null,         params_ratio: 0.0458}
      - {id: visual,            scheme: BF16,  operation: null, layers: 27, params_ratio: 0.0166}
      - {id: mtp,               scheme: BF16,  operation: null, layers: 1,  params_ratio: 0.0153}
      - {id: linear-attn-bf16,  scheme: BF16,  operation: null, layers: 48, params_ratio: 0.0009}
    total_params: 27781492736
    quantized_params: 24326963200
    quantized_ratio: 0.8757
    effective_bits: 5.492
```

---

## 4. YAML 渲染

`write_recipe_manifest()` 用 `yaml.SafeDumper` 子类 + str 代表器：**多行字符串（script）渲染为 `|`
字面块**，`sort_keys=False` 保持块顺序，便于人读与 diff（见 §3.6 / §2 实例）。

---

## 5. 反向复现 `aqa reproduce`（已实现）

> 实现入口：`aqa reproduce <manifest> --variant <id> [--mode full|eval-only]`；
> 外部 agent 调 `aqa.reproduce.reproduce_manifest(...)`（返回同一个 report dict）。
> `VERIFY` 是 `Phase` 枚举中的只读相位，产物 `reproduction_report.json` 由
> `layout.yaml`（`when: reproduce`）纳入 `aqa inspect` 契约。`eval-only` 跳过量化面。

### 5.1 不新增 pipeline——复用相位机

复现**不另起一条 pipeline**。`recipe.yaml` 已经把「环境 + 量化命令 + 评估命令 + 期望结果」讲全了，
所以复现 = **前端适配器 + 相位后置条件上的复现叠加**：

```
aqa reproduce <manifest.yaml> --variant <id> [--mode full|eval-only] [--tolerance ...]
      │
      ├─ 适配器：recipe.yaml(+variant) → RunConfig（model.source→model，quantization.script→quant 命令，
      │          evaluation→eval 配置，environments→executor/env 期望）
      ├─ full 模式：复用现有 QUANT→EVAL→GATE 相位机跑一遍
      └─ eval-only 模式：跳过 QUANT，对既有 checkpoint/hf_url 只复跑 EVAL
```

### 5.2 三面校验（VERIFY 相位）

复现**不只是重跑**，要**校验三件事**，产 `reproduction_report.json`：

| 面 | 校验什么 | 判定 |
|---|---|---|
| ① 输入等价 | pre-flight 指纹：model.source / 量化命令规范化 / 环境关键包版本 | mismatch → 记录但不致命（可 `--on-input-mismatch warn\|fail`） |
| ② 量化结构复现 | 新导出的 `quantization_config` vs `quantization_details`：逐分组 scheme/ignore 是否一致 | 结构不符 → `fail` |
| ③ 评估数值复现 | 复跑分数 vs recipe `result_tables` 期望：**abs + rel 双容差** | 超容差 → `fail`；**环境差异致不可比** → `inconclusive`（非 `fail`） |

> 为什么 QUANT 做完、EVAL 发现 recipe 不合适要重做 QUANT 的场景**不在这里**处理：那是 RFC 0001 §S3
> 的**有界 `GATE→QUANT` re-QUANT 回环**（`EVAL→QUANT` 静默重量化非法）。复现路径是**只读校验**，
> 发现不符就如实报 `fail`/`inconclusive`，**不自动改配方**——改配方是正向 run 的 Coordinator 职责。

### 5.3 `reproduction_report.json`（草案）

```json
{
  "schema_version": 1,
  "manifest": "<path>", "variant": "<id>", "mode": "full|eval-only",
  "input_equivalence": {"status": "match|mismatch", "diffs": [...]},
  "quantization_reproduction": {"status": "pass|fail|skipped",
    "group_mismatches": [{"id": "...", "expected": "MXFP8", "got": "BF16"}]},
  "evaluation_reproduction": {"status": "pass|fail|inconclusive",
    "tolerance": {"abs": 0.01, "rel": 0.02},
    "per_benchmark": [{"id": "piqa", "expected": 0.83, "got": 0.828, "within": true}]},
  "verdict": "reproduced|not_reproduced|inconclusive"
}
```

### 5.4 config

```yaml
reproduce:
  mode: full | eval-only          # 默认 full
  tolerance: {abs: 0.01, rel: 0.02}
  on_input_mismatch: warn | fail  # 默认 warn
```

GATE 不变（沿用 RFC 0001）；VERIFY 是**新增轻量只读相位**，产 `reproduction_report.json`，
与 CLOSE 的 `recipe.yaml` 对称。

---

## 6. 测试

- `tests/test_quant_details.py`：dense + MoE 合成 config；auto_round + compressed-tensors 两种导出
  config；断言**全分组覆盖**、scheme 分配、`params_ratio` 求和→1、`effective_bits` 区间、
  `quant_lm_head` 翻转、未知架构降级（params 全 null + note）、`derived_from` 由导出 config 驱动。
- `tests/test_weight_map.py`：合成 safetensors fixture（hybrid 多模态 base + 带 `weight_scale` 的导出）
  验证 `read_param_shapes`/`classify_module`/`summarize_structure`/`read_quantized_modules`；**地面真相
  路径**断言 `derived_from=weight_shapes`、`quantized_basis=export_artifacts`、MTP/视觉/embed/lm_head
  保 BF16、mixed family 拆 `-quant`/`-bf16`、`params_ratio` 求和→1、离线（无权重）回退解析式。
- `tests/test_recipe_manifest.py`：documented（GO+命令+具名 metric）/ draft 降级（NO-GO）/ 无评估命令
  降级（evaluation=null）/ 非 quantize 返回 None / YAML round-trip（字面块 script 存活）/
  KV+attention op 由命令派生；`to_inc_strict()` 后**结构断言**镜像 INC schema 硬约束。
- 开发期已用 **PR #5 真 `model-recipe.schema.json` + `jsonschema`** 实证 `to_inc_strict()` 核心
  **VALID**（documented，avg_bits 8.238，operations `[mxfp8-linear, fp8-kv-cache]`，列 id kebab 化）。
- **真机验证**：Qwen3.8-27B W4A16 全程 run（piqa 0.8090→0.8058，gate GO），CLOSE 重新导出
  `recipe.yaml` 的 `quantization_details` 由地面真相给出 total 27.78B / quantized_ratio 0.8757 /
  effective_bits 5.492，`aqa inspect` CLOSE/recipe.yaml present。

运行：`~/.venv-aqa-gpu/bin/python -m pytest tests/test_quant_details.py tests/test_weight_map.py tests/test_recipe_manifest.py -q`。

---

## 7. 不做什么（MVP 边界）

- **不重实现 INC 的 `input_fingerprint`**（Ruby canonical-JSON SHA256）——结果行**不带 fingerprint**
  （= 「reported numbers only」，schema 合法）。待 §5 复现落地时再按需补 fingerprint 以支撑「可比」判定。
- **不从 `num_bits+type` 反推 compressed-tensors 的 scheme 串**——scheme **label 一律用声明的
  `run_cfg.quant.scheme`**（稳定、人读），导出 config 只用来判「量化 vs 保留」。
- **MoE 参数数学为尽力**——已对 DeepSeek 式（`n_routed_experts`+`shared_expert`）合成 config 验证；
  遇未建模架构则 params 降级为 null，coverage 不缺。
