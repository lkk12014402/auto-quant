# 评估（evaluate）用法

> 内核：`aqa/eval/`（harness / scorer / runner）+ `aqa/report/`（gate / sinks）。
> 判断层：`skills/evaluate/SKILL.md`。

## 最小配置

```yaml
model: Qwen/Qwen3-0.6B
eval:
  benchmarks: [piqa, hellaswag, gsm8k]
  backend: vllm              # hf | vllm | vllm_omni（diffusion）| vllm_xpu（插件）
  harness: lm_eval           # lm_eval | native | diffusion | evalscope（插件）
  baseline: true             # 同时评 bf16 原版（gate 的对照组）
gate:
  max_accuracy_drop: 0.02    # 量化版 vs 基线掉超 2 个百分点 = NO-GO
```

## harness × scorer 怎么选

| 任务族 | harness | scorer（recipe 里 `scoring.scorer`） |
|---|---|---|
| 标准选择题/生成（piqa/mmlu/gsm8k…） | `lm_eval` | 从 lm_eval 结果读 metric |
| 开放问答/偏好打分 | `native` | `llm_judge`（OpenAI 兼容端点评分） |
| 代码生成（HumanEval/LiveCodeBench…） | `native` | `code_exec`（沙箱跑测试） |
| 文生图 | `diffusion` | `clip_score`（默认）/ `image_metrics`（三指标）/ `geneval` / `dpgbench` |
| 端点驱动/agentic | `agent_eval`（库内调用） | verify 命令或 rubric |

## per-task 参数在哪

任务的正确配置钉在 `recipes/benchmarks/<task>.yaml`（type/apply_chat_template/
num_fewshot/batch_size/scoring），run.yaml 只放全局策略（backend/seed/batch_size）。
**别把任务级参数写进 run.yaml。** 已带：mmlu/gsm8k/hellaswag/arc_challenge/piag…
gpqa_diamond/mmlu_pro/ifeval/bbh/truthfulqa/winogrande/openbookqa/openqa_judge/
humaneval/drawbench_clip/drawbench_metrics/geneval/dpgbench。没食谱的任务也能跑
（lm_eval 直通）。

## 任务分组

`eval.lm_eval.group: true`（默认）把生成参数兼容的任务并进一次 `lm_eval` 调用
（一次模型加载），不兼容的（如 gsm8k 要 chat template + fewshot）自动拆开单独跑。

## gate 判定

- `max_accuracy_drop`：量化版 vs 基线逐 benchmark 比，掉超阈值 → NO-GO；
- `min_accuracy`：无基线时的绝对下限（baseline: false 时用）；
- `significance: true`：要求掉点统计显著才判 NO-GO（配对检验）；
- 判定写进 `gate.json`（`verdict` + `reasons`），`aqa compare A B` 看两次 run 的分差。

## 相位钩子

`hooks: [quant_sanity, eval_sanity]`（默认开）：量化产物没有打包张量、某 benchmark
恰好 0 分这类"假成功"会被钩子判死，喂给 fix-loop 做签名。`hooks: []` 全关。

## 结果与归档

- 本地：`output.dir/{eval,baseline}/accuracy.json` + `run_manifest.json`（完整复现清单）+ `gate.json`；
- HF dataset 归档：`output.sinks: [{type: hf_dataset, repo: org/ds, token_env: HF_TOKEN}]`
  ——整树上传（日志/patch/根因/逐样本），失败 run 也会带状态写回（不会静默消失）；
- 量化模型上传：`provider.upload: {repo: MyOrg, token_env: HF_TOKEN}`。
