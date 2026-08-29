# NeMo-Skills 深度调研报告：inference / evaluation / dataset 三目录

调研对象：`NVIDIA-NeMo/Skills@main`（截至 2026-08）。方法：GitHub Contents API 列目录 + raw.githubusercontent.com 读文件，未 clone。核心结论先行：**这套框架对 aqa 最有价值的不是代码本身，而是三份契约**——(1) benchmark 目录即插件的声明式约定，(2) evaluator 原地富化 output jsonl 的 key 约定，(3) `output.jsonl` / `output-rs*.jsonl` 多 seed 文件布局与 pass@k/majority@k/std 聚合算法。这些都可以剥离 NeMo-Run/Slurm/Hydra 单独移植。

---

## 1. `nemo_skills/inference/` — generation 如何组织为可恢复、可分片任务

### 1.1 核心：`GenerationTaskConfig` + `GenerationTask`（generate.py，45KB）

来源：[generate.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/generate.py)

配置是一个巨大的 kw-only dataclass（Hydra 托管），关键字段：

```python
input_file: str
output_file: str
prompt_config: Any = None          # yaml 路径（"generic/math"）或 dict
prompt_format: str = "ns"          # "ns" 模板 or "openai"（messages 直接来自数据）
generation_key: str = "generation" # 输出 key 可换（judge 用 "judgement"）
max_samples: int = -1
skip_filled: bool = False          # 断点续跑开关
max_concurrent_requests: int = 512 # asyncio 信号量并发
num_chunks: int | None = None      # 数据分片
chunk_id: int | None = None
add_generation_stats: bool = True
dry_run: bool = False              # 只打印第一条 prompt
eval_type: str | None = None       # 生成后/交织评测
eval_config: dict = ...
parse_reasoning: bool = False      # 把 reasoning 拆到 _full_generation
enable_litellm_cache: bool = False # 磁盘 cache 防 job 超时长任务丢中间结果
```

**可恢复性（resumable）协议**——这是最可借的部分，纯文件协议、零外部依赖：

- async loop 中每条结果**立即 append** 到 `output_file + "-async"`（`buffering=1` 行缓冲），每条带 `_async_position` 序号；
- `skip_filled=True` 重启时读 `-async` 文件里已有的 position 集合，跳过已完成样本（`skip_completed_samples`）；
- 全部完成后 `restore_async_order()` 按 position 重排、写回正式 `output_file` 并删除 `-async`；
- 若非 skip_filled 重跑，开头直接 unlink 旧文件，保证无脏数据。

**可分片（shardable）协议**：

- `chunk_data(data, output_file, chunk_id, num_chunks)` 在 `load_data()` 里把数据均分，输出文件名变为 `..._chunk{id}.jsonl`；base 文件已存在则整个跳过；
- [merge_chunks.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/merge_chunks.py)：检查每个分片旁的 `.done` 标记文件，齐了才 `cat` 合并、写 `output.done`、删分片，可附挂 post-merge 命令。这是为 Slurm 多 job 设计的，但协议本身（分片文件 + `.done` + cat 合并）对 aqa 的单机多 worker 也适用。

**每 datapoint 的输出 schema**：`process_single_datapoint` 返回 `{"generation", "num_generated_tokens", "num_input_tokens", "finish_reason", ...}`，`postprocess_single_output` 把 `generation` 挪到 `generation_key`、附加 `generation_start_time/end_time/generation_time`，再把**原始输入 datapoint 的所有 key 合并进输出**（输出文件自包含，下游 evaluator/metrics 不需要再 join 原始数据集——对 aqa 很重要的一条约定）。

**判错/重试（三层，逐个说）**：

1. **litellm 层**：`max_retries=3` 传给 litellm（[model/base.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/model/base.py) `BaseModel.__init__`）；另有针对 `openai.BadRequestError` 中含 "output messages (reasoning and final)" 的硬编码重试 `max_retries = 2`（gpt-oss workaround）。
2. **context-limit soft-fail**（[model/context_retry.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/model/context_retry.py)，21KB，**独立可移植**）：
   ```python
   @dataclass
   class ContextLimitRetryConfig:
       enable_soft_fail: bool = False
       strategy: str | None = None  # reduce_generation / reduce_prompt_from_start / reduce_prompt_from_end
       num_special_tokens_budget: int = 50
   ```
   `@with_context_retry` 装饰 `generate_async`；用 5 套 regex 从各 server 的报错文本解析 `max_context_length`/`message_tokens`（OpenAI/vLLM/SGLang 各家文案不同），然后按策略缩减 `tokens_to_generate` 或裁 prompt（string prompt 直接 encode/decode 截断；message list prompt 从头或从尾逐条保留，最后一条可部分截断）。**裁不动或二次失败时的 soft-fail 输出**：
   ```python
   def return_empty_generation_with_error(detailed_error, error_reason="See detailed_error"):
       return {"generation": "", "num_generated_tokens": 0,
               "error": error_reason, "detailed_error": detailed_error, "finish_reason": "error"}
   ```
   即**失败变成一个合法 datapoint**（空 generation + error 字段）而不是炸掉整个 job——这正好喂给 aqa 的 gate：失败样本自然计入 pass@1 分母。非 context 类错误在 soft-fail 开启时也同样降级为空生成。
3. **job 级**：`enable_litellm_cache` 把 LLM 响应落盘（`StableLiteLLMCache`，per output_file+chunk 一份 cache.pkl），Slurm job 超时杀掉后重跑可命中缓存。

### 1.2 prompt → fill → answer extraction 管线

- prompt 是 yaml 配置（`nemo_skills/prompt/config/...`），schema 极简。[generic/math.yaml](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/prompt/config/generic/math.yaml) 全文：

```yaml
few_shot_examples:
  prefix: "Here are some examples of problems and solutions you can refer to.\n\n"
  template: "Problem:\n{problem}\n\nSolution:\n{solution}\n\n\n\n\n\n"
  suffix: "Here is the problem you need to solve:\n"
user: |-
  Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{{}}.

  {examples}{problem}
```

- `fill_prompt(data_point, data)`：`prompt.fill()` 用数据 dict 做模板插值（`{problem}` 等），支持 `start_assistant_response_key`（prefill assistant 开头，RULER base 模式用）、`prompt_suffix`（如 ` /no_think`）、`system_message`/`user_message` 覆盖、`format_as_string`（text completions endpoint 时）。`prompt_format="openai"` 时数据里直接带 `messages`。
- **answer extraction 不在 generate.py**——它是 evaluator 的职责（见 §2.2 `MathEvaluator` 写 `predicted_answer`），generate 只产出裸文本。`parse_reasoning=True` 时把 `</think>` 之前内容挪走，避免 boxed 抽取抽到 thinking 里的垃圾。

### 1.3 judge 管线如何挂接

两层机制：

1. **数据侧声明**：benchmark 的 `__init__.py` 里放 `JUDGE_PIPELINE_ARGS` + `JUDGE_ARGS`（详见 §3）。pipeline 层读到后把主生成输出放 `tmp-eval-results/`，再对同一文件跑第二个 generation job。
2. **judge 本身也是 GenerationTask 子类**，复用同一套 resumable/shardable 机制。[llm_math_judge.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/llm_math_judge.py) 全文要点：

```python
class LlmMathJudgeConfig(GenerationTaskConfig):
    prompt_config: str = "judge/math"
    generation_key: str = "judgement"
    add_generation_stats: bool = False   # 不覆盖主生成的统计

class LLMMathJudgeTask(GenerationTask):
    def preprocess_data(self, data):
        for data_point in data:
            if "predicted_answer" not in data_point:
                data_point["predicted_answer"] = extract_answer(data_point["generation"])
        return data
    def prefill_generation(self, data_point):   # 短路：无需 LLM 时直接预填
        judgement = prefill_judgement(data_point)   # 答案抽不出或精确匹配时跳过 judge
        return None if judgement is None else {"generation": judgement}
```

`prefill_generation` 是基类钩子（默认返回 None），judge/contamination 都靠它短路 LLM 调用。[check_contamination.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/check_contamination.py) 同款模式：一个 datapoint 展开成 top_k 个 `problem1/problem2` 查询并行打 judge，字符串精确匹配直接 prefill `True`，`postprocess()` 统计污染比例。Arena judge（[inference/eval/arena_judge.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/eval/arena_judge.py)）则展示了**一次 datapoint 发两个方向请求**（gen-base / base-gen 消除位置偏置），输出 `judgement-gen-base` / `judgement-base-gen` 两个 key。

### 1.4 server 层（基本不用看）

- [serve_vllm.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/server/serve_vllm.py)：60 行的 argparse→`subprocess.run("python3 -m vllm.entrypoints.openai.api_server --tensor-parallel-size=... ")` 薄壳，多节点加 `--distributed-executor-backend=ray`。
- [serve_unified.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/server/serve_unified.py)：NeMo Magpie TTS 专用 unified server 启动器（uvicorn + `recipes.multimodal.server`），与 LLM eval 无关。
- 真正client在 `inference/model/`：`get_model(server_type=...)` 注册表（[model/__init__.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/inference/model/__init__.py)：`vllm/sglang/openai/azureopenai/gemini/megatron/trtllm/vllm_multimodal/tts_nim/asr_nim`，`trtllm` 直接映射到 `VLLMModel`），统一走 OpenAI-compatible endpoint + litellm。注意 `inference/factory.py` 不是 get_model 工厂，而是 `GENERATION_MODULE_MAP = {generate, math_judge, check_contamination}` 的模块路由。

---

## 2. `nemo_skills/evaluation/` — evaluator 输出与 metrics 聚合的精确契约

### 2.1 evaluator 注册表与基类契约

[evaluator/__init__.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/__init__.py)：**懒加载双注册表**，函数式（batch-only）与类式分开：

```python
_EVALUATOR_MAP_PATHS = {            # 函数式 eval_type -> "module:function"
    "evalplus": "...code:eval_evalplus", "if": "...ifeval:eval_if",
    "bfcl": "...bfcl:eval_bfcl", "multichoice": "...mcq:eval_mcq",
    "ruler": "...ruler:eval_ruler", "livecodebench": "...livecodebench:eval_livecodebench",
    ... 共 18 个
}
_EVALUATOR_CLASS_MAP_PATHS = {      # 类式 eval_type -> "module:Class"
    "math": "...math:MathEvaluator", "lean4-proof": "...math:Lean4ProofEvaluator",
    "code_exec": "...code:CodeExecEvaluator", "ioi": ..., "icpc": ..., "ccc": ...,
    ... 共 13 个
}
```

两个 map 交集为空有断言。`eval_type` 还支持 `module.path::ClassName` 或 `/abs/path/file.py::ClassName` 动态路径（`locate()`）——**这是 aqa 最值得抄的插件点**：scorer 库之外允许任意自定义 evaluator 不改主仓代码。`register_evaluator()` 支持运行时注册。

[evaluator/base.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/base.py) 基类契约全文（极短，值得逐字）：

```python
class BaseEvaluator(ABC):
    def __init__(self, config: Dict[str, Any], num_parallel_requests=10): ...

    async def eval_full(self) -> None:
        """Evaluate full dataset in batch mode."""
        # 读 input_file 全部行 → 每行 eval_single → merged.update(updates)
        # 写 input_file + "-tmp" → os.replace 原子替换原文件

    async def eval_single(self, data_point) -> Dict[str, Any]:
        """Returns: Dict with evaluation results to merge into data_point"""
        raise NotImplementedError(...)

    def supports_single_eval(self) -> bool:
        return self.__class__.eval_single is not BaseEvaluator.eval_single
```

**契约要点**：
- evaluator **原地富化 output jsonl**（`os.replace(tmp, input_file)` 原子重写），不是另写结果文件；
- 返回 dict 会被 `merged.update(updates)` 合进该行——新增 key 约定为 `predicted_answer` / `symbolic_correct` / `judgement` / `is_correct` / `loose_eval`/`strict_eval`（ifeval）等；
- 类式 evaluator 若实现 `eval_single`，generate.py 会**与生成交织执行**（`evaluate_single_datapoint` 在 `_generate_and_save_datapoint` 里调用，还记录 `interleaved_eval_single_time_s`）；否则生成完后 `evaluate(eval_type, eval_config)` 整批跑。aqa 可以直接借这个"交织 or 批后"双模式。

三个代表性实现：
- [evaluator/math.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/math.py)：`eval_single` 里 `extract_answer(generation)` → `predicted_answer`，`math_equal(expected_answer, predicted_answer)` → `symbolic_correct`。config 有 `use_predicted_answer_key`（上游已抽好就不重抽）、`extract_from_boxed`/`extract_regex`/`relaxed_extraction`/`take_modulo`。
- [evaluator/mcq.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/mcq.py)：boxed/regex 抽字母 + 多语言字母归一化（阿拉伯语/孟加拉语/全角日文 A-D）+ Artificial-Analysis 风格 `Answer: X` 兜底 regex，写 `predicted_answer` + `symbolic_correct`；per-sample 的 `extract_from_boxed`/`extract_regex` 可覆盖 config 默认（数据里可带 per-row 评测参数——aqa 可学）。
- [evaluator/ifeval.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/ifeval.py)：函数式，shell 调官方 `instruction_following_eval` 仓库，把 loose/strict 结果 zip 回原文件。

### 2.2 metrics：`BaseMetrics` 的 pass@k / majority@k / std 机制

[metrics/base.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/metrics/base.py)。关键签名与语义：

```python
class BaseMetrics(abc.ABC):
    def __init__(self, compute_no_answer: bool = True)
    def update(self, predictions)            # predictions = 同一题跨 k 个 seed 的 list[dict]
    def _get_score_dict(self, prediction) -> dict[str, bool|int|float]
        # 例: {'symbolic_correct': True, 'judge_correct': False} —— 多评分法并存
    def _compute_pass_at_k(self, predictions, predicted_answers=None, eval_dict=None)
    def _compute_majority_at_k(self, predictions, predicted_answers, eval_dict=None)
    def get_incorrect_sample(self, predictions)  # 超长过滤时强制判错
```

- **pass@k**：二分分数时用无偏估计 `1 - C(total_incorrect, k)/C(total, k)`（组合数，非 `any()`）；非二分分数退化为 `max(scores[:k])`。同时累加 `pass@1[avg-of-k] = sum(scores[:k])/k`。聚合 key 为 `eval_dict[f"pass@{k}"][score_method]`，多 score_method（symbolic/judge/both/any）各自独立成列。
- **majority@k**：对 `predicted_answers[:k]` 里非 None 的 `(answer, score)` 做 `Counter.most_common`，并列时取并列答案的 score 均值、答案取排序后第一个（确定性 tie-break）；`no_answer` 单列统计（`compute_no_answer`）。
- **std（`_add_std_metrics`）**：`self.all_scores[score_method]` 存每题的 k 次分数，对 k=2..max_k 计算并挂到 `pass@1[avg-of-{k}]` 下：
  - `{method}_statistics.avg` —— 跨 run 均值；
  - `std_dev_across_runs` —— 把数据转置（第 i 次 attempt 组成 run i），对 run 均值取 `np.std(ddof=1)`；
  - `std_err_across_runs = std / sqrt(k)`；
  - `avg_sample_std_dev` —— 每题内部 k 次 std 的平均。
  打印层 `as_percentage` 把 `std_dev_across_runs` 显示为 `±x.xx%`。**这就是 aqa gate 统计显著性可直接对齐的口径**。
- 还有钩子 `_update_score_metrics_for_pass/_majority`、`_update_metrics_for_pass/_majority` 供子类挂自定义指标；`subset_for_metrics` 字段让 ComputeMetrics 自动按子集（如 mmlu-pro category、livecodebench difficulty）拆出 `{benchmark}-{subset}` 报表。

### 2.3 `map_metrics.py` 路由表

[map_metrics.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/metrics/map_metrics.py)：`METRICS_MAP` 约 40 条，**多处复用 + functools.partial 参数化**：

```python
METRICS_MAP = {
    "math": MathMetrics,
    "hle": functools.partial(MathMetrics, compute_no_answer=False, answer_key="generation"),
    "multichoice": MathMetrics,                      # MCQ 复用 MathMetrics
    "multichoice_multilingual": MCQMultilingualMetrics,
    "livecodebench": LiveCodeBenchMetrics, "livecodebench_pro": LiveCodeBenchMetrics,
    "swe-bench": SweBenchMetrics, "bfcl": BFCLMetrics, "arena": ArenaMetrics,
    "ruler": RulerMetrics, "ruler2": Ruler2Metrics, ...
}
get_metrics(metric_type)  # 同样支持 "module::Class" / "/path.py::Class" 动态路径
```

注意 **eval_type 与 metrics_type 是两个独立路由**（evaluator 负责判对错写 key，metrics 负责聚合），很多 benchmark 共用 `MathMetrics`。aqa 的 recipe YAML 里 scorer 和 metrics aggregation 也应解耦。

### 2.4 `math_grader.py` —— 不是 vendored qwen grader

[math_grader.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/math_grader.py)（5.6KB）：依赖的是**第三方库 `math_verify` + `latex2sympy2_extended`**（HuggingFace 系），不是内嵌的 qwen math grader / 不是裸 sympy。流程：

1. MCQ 快路径：A-J 字母用 `math_verify.parse`+`StringExtractionConfig`+`verify`；
2. `_additional_normalization`（去 `%`/`\%`、去句尾 `.`）+ `normalize_latex(NormalizationConfig)` 字符串规范化相等即 True（快路径）;
3. 纯文本字面量直接判 False；
4. 兜底：包 `$...$` 后 `parse(..., [LatexExtractionConfig()])` + `verify(...)`（支持 `numeric_precision`、`timeout_seconds` 等 kwargs）做符号等价。

`extract_answer`：`search_boxed`（手写花括号配平扫描 `rfind("\\boxed")`）与 `search_regex`（默认 `r"The final answer is (.+)$"`），`relaxed=True` 时先 regex 后 boxed。`is_correct_judgement`（[metrics/utils.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/metrics/utils.py)）解析 judge 文本的三种格式：`Judgement: Yes/No`（兼容 markdown 加粗）、`\boxed{Correct/Incorrect}`、`<points>N out of 7</points>`（≥6 算对）。aqa 的 numeric_match/fuzzy_match 可以考虑直接依赖 `math_verify` 而不是自维护。

### 2.5 多 seed 文件 → metrics.json 的聚合链（完整契约）

1. **生成侧**（pipeline 层 [pipeline/utils/eval.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/utils/eval.py)）：`--benchmarks gsm8k:4` → `random_seeds = range(starting_seed, starting_seed + num_samples)`，文件名由 `get_chunked_rs_filename()` 生成 `output-rs{seed}.jsonl`；贪心单跑为 `output.jsonl`。
2. **聚合侧** [pipeline/summarize_results.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/summarize_results.py)：
   ```python
   has_greedy = Path(f"{benchmark_path}/output.jsonl").exists()
   output_rs_pattern = re.compile(r"^output-rs\d+\.jsonl$")
   input_files = sorted([... glob output-rs*.jsonl 且匹配 ...])
   if has_greedy and has_sampling:
       raise ValueError("Both output.jsonl and output-rs*.jsonl found ... not clear how to process")
   metrics = metrics_calculator.compute_metrics(input_files=input_files)
   ```
   即 **greedy 与 sampling 互斥**；rs 文件按行 `zip_longest` 对齐（`read_predictions`），同一行号 = 同一题，组成 `predictions: list[dict]` 喂给 `BaseMetrics.update`。
3. **METRICS_TYPE 来源**：`ComputeMetrics.__init__` 里 `metric_type=None` 时 `get_dataset_module(benchmark).METRICS_TYPE`——回到 dataset 目录的 `__init__.py`，闭环。
4. **group 聚合**两条路：
   - `add_benchmark_groups()`：按 `.` 前缀（`ruler.llama_128k.niah_single_1` → `ruler.llama_128k`）把同组 benchmark 的同名 metric **算术平均**，`num_entries` 换成 `num_benchmarks_in_group`，且强制校验同组所有 benchmark 的 eval_mode/metric key 集合一致；
   - `SCORE_MODULE`（group `__init__.py` 里声明，如 `nemo_skills.dataset.bfcl_v4.bfcl_score`）：[compute_group_score.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/compute_group_score.py) 把多个 metrics.json 合并后调 `score_module.compute_score(combined_metrics)` 出官方口径总分（BFCL 的类别加权等）。
5. 结果写 `<output_dir>/metrics.json`，打印表格列由 `metrics_to_print()`/`evaluations_to_print()`（默认 `[pass@1[avg-of-max_k], majority@max_k, pass@max_k]`）控制。

### 2.6 `aggregate_answers.py` —— majority 投票回填

[aggregate_answers.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/aggregate_answers.py)：两种 mode——`fill`（把多数答案当 `expected_answer` 回填所有 rs 文件，用于自蒸馏造数据）与 `extract`（把多数答案当 `predicted_answer` 写 `output-agg.jsonl`，即 majority@k 的"答卷"）。支持 RM 加权（`use_highest_rm_score`/`use_majority_rm_score`）、`require_num_files` 防缺文件、`majority_votes`/`total_votes` 记录。对 aqa：如果想要"majority 答案本身"而不仅是 majority@k 指标，这个文件就是参考。

---

## 3. `nemo_skills/dataset/` — 118 个 benchmark 的可维护性解剖

### 3.1 目录解剖：一个 benchmark = 一个自包含文件夹

8 个代表目录的文件清单（[Contents API](https://api.github.com/repos/NVIDIA-NeMo/Skills/contents/nemo_skills/dataset)）：

| benchmark | 文件 | 特点 |
|---|---|---|
| `gsm8k` | `__init__.py`, `prepare.py` | 最简形态；prepare 内含**已知错误答案 `fixes` 字典**（11 条训练集错题修正） |
| `aime25` | `__init__.py`, `prepare.py`, `test.txt` | **原始数据直接 vendored 进仓**（test.txt 62KB），prepare.py 仅 `shutil.copy` 改名 test.jsonl |
| `livecodebench` | `__init__.py`, `prepare.py` | `EVAL_SPLIT = "test_v6_2408_2505"`（版本+日期区间命名 split）；`DEFAULT_SPLITS` 一次备 4 个时间切片；文件尾注释记录各 split 期望样本数（`test_v6_2408_2502.jsonl: 374 samples`）作回归校验 |
| `bfcl_v4` | `__init__.py`, `prepare.py`, `bfcl_score.py`, 22 个子目录 | **benchmark group**：`IS_BENCHMARK_GROUP=True` + `BENCHMARKS = {f"bfcl_v4.{split}": {}}` + `SCORE_MODULE`；prepare 复用 bfcl_v3 的下载函数 + gorilla 官方 repo 的 `ALL_SCORING_CATEGORIES` |
| `ruler` | `__init__.py`（仅 `REQUIRES_DATA_DIR = True`）, `prepare.py`, `ruler_score.py` | **数据按模型/长度动态生成**（`--setup llama_128k --max_seq_length 131072`），prepare.py 还会**程序化生成子 benchmark 的 `__init__.py`**（每 task 一个，填好 `METRICS_TYPE="ruler"`、`eval_config.match_type`、text-completion 参数），并生成 group 级 `__init__.py` |
| `mmlu-pro` | `__init__.py`, `prepare.py`, `subsets/` | `subsets/10pct_opt_v1.txt`（问题 hash 清单）→ `--split 10pct_opt_v1` 从 test 集过滤出子集，用于量化回归的快速 smoke 集 |
| `swe-bench` | `__init__.py`, `prepare.py`, `dump_images.py`, `dump_repos.py` | `GENERATION_MODULE = "nemo_skills.inference.eval.swebench"`（评测与生成融合的 50KB 自定义模块）；数据行内嵌 `container_formatter`/`repo_formatter` |
| `arena-hard` | `__init__.py`, `prepare.py` | 无 eval_type——judge 直接在 metrics 层消费；`JUDGE_PIPELINE_ARGS` 声明 judge 模型/服务器；prepare 里注释记录 upstream 字段怪癖的 workaround（drop category 字段以走默认 prompt 分支） |

### 3.2 `__init__.py` 声明式协议（verbatim 示例）

gsm8k（[来源](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/gsm8k/__init__.py)），全文即最小配置：

```python
# settings that define how evaluation should be done by default (all can be changed from cmdline)
METRICS_TYPE = "math"
GENERATION_ARGS = "++prompt_config=generic/math ++eval_type=math"
```

hle（[来源](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/hle/__init__.py)），带 judge 的完整形态：

```python
METRICS_TYPE = "hle"  # This uses the MathMetrics class, but with compute_no_answer=False
GENERATION_ARGS = "++prompt_config=generic/hle ++eval_type=math"
EVAL_SPLIT = "text"

# Some answers are not possible to compare symbolically, so have to use a judge model
# Currently using o3-mini-2025-01-31 which is used by the official leaderboard - https://agi.safe.ai/
JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-2025-01-31",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/hle ++generation_key=judgement ++add_generation_stats=False"
```

arena-hard（[来源](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/arena-hard/__init__.py)）——judge 走自定义 generation module：

```python
METRICS_TYPE = "arena"
GENERATION_ARGS = "++prompt_config=generic/default"

JUDGE_PIPELINE_ARGS = {
    "generation_module": "nemo_skills.inference.eval.arena_judge",
    "model": "gpt-4.1",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
```

omni-math（[来源](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/omni-math/__init__.py)）用 `"generation_type": "math_judge"` 走 `GENERATION_MODULE_MAP` 路由。bfcl_v4 的 group 声明（[来源](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/bfcl_v4/__init__.py)）：

```python
SPLITS = ["simple_python", "parallel", ..., "memory_rec_sum"]  # 22 个
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.bfcl_v4.bfcl_score"
BENCHMARKS = {f"bfcl_v4.{split}": {} for split in SPLITS}   # value 是 per-split override dict
```

全部声明字段（pipeline/utils/eval.py `get_benchmark_args_from_module` 读取处）：`METRICS_TYPE / GENERATION_ARGS / EVAL_SPLIT / GENERATION_MODULE / JUDGE_PIPELINE_ARGS / JUDGE_ARGS / NUM_SAMPLES / NUM_CHUNKS / REQUIRES_SANDBOX / KEEP_MOUNTS_FOR_SANDBOX / SANDBOX_ENV_VARS / REQUIRES_DATA_DIR / IS_BENCHMARK_GROUP / BENCHMARKS / SCORE_MODULE`，另有 deprecated 的 `PROMPT_CONFIG`/`EVAL_ARGS`。

### 3.3 prepare.py 与 jsonl schema

- 统一入口 [dataset/prepare.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/prepare.py)：`ns prepare_data <names...>`，`--parallelism 20 --retries 3`，不指定则遍历**所有子目录**——**没有单独的 dataset_groups 注册表，"bulk group"就是目录 glob**；真正的 group 概念是 §3.2 的 `IS_BENCHMARK_GROUP`（运行时展开成 `bfcl_v4.simple_python` 等点分子 benchmark）。
- 输出 schema 约定：`problem`/`question`、`expected_answer` 是基础 pair；可选 `reference_solution`、`subset_for_metrics`、`options`+`A/B/C/...`（MCQ，`get_mcq_fields` 生成）、`task_id`、`category`、`baseline_answer`（arena）、`similar_items`（contamination）。**原则：prepare 把上游异构数据映射到统一 key，prompt 模板和 evaluator 只认统一 key**。
- 下载方式三档：urllib 直拉 GitHub raw（gsm8k/arena-hard）、`datasets.load_dataset`（mmlu-pro/swe-bench/livecodebench，livecodebench 还 pin `revision="refs/pr/7"` 绕开 datasets 版本限制）、clone 官方 repo 跑其生成脚本（ruler/bfcl）。`download_with_retries` 通用重试。ruler 这类大数据集用 `REQUIRES_DATA_DIR=True` 强制用户走共享 data_dir 而非随 job 上传。

### 3.4 新增 benchmark 的最小 checklist（官方 5 步，[docs](https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/evaluation/index.md) 逐字）

1. 在 `nemo_skills/dataset` 建新文件夹；
2. 写 `prepare.py`，运行后产出 `<split>.jsonl`（可带额外 CLI 参数）；
3. 写 `__init__.py` 声明默认 `GENERATION_ARGS`（prompt config + eval_type）与 METRICS_TYPE；生成逻辑要定制就给 `GENERATION_MODULE`；
4. 不能复用就新写 evaluator（注册进 `EVALUATOR_MAP`/`EVALUATOR_CLASS_MAP`，或用 `module::Class` 路径）；
5. 不能复用就新写 metrics class（注册进 `METRICS_MAP` 或用路径）。

**且支持完全放在仓外**：[external-benchmarks 文档](https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/evaluation/external-benchmarks.md)——外部 repo 提供 `benchmark_map.json`（name→目录）+ `NEMO_SKILLS_EXTRA_BENCHMARK_MAP` 环境变量，`METRICS_TYPE`/`eval_type` 都接受 `my_pkg.metrics.foo::MyMetrics` 形式。最小外部 benchmark 真的只要 2 个文件（复用内建 math evaluator 时）。

### 3.5 118 个目录可维护的约定

- **命名即版本**：`bfcl_v3`/`bfcl_v4`、`arena-hard`/`arena-hard-v2`、`aime25`/`aime26`、`hle`/`hle_verified` 并列存在，旧版本不删——版本升级 = 新目录 + 新 `__init__.py`，不重写旧的，保证历史结果可复现。
- **split 即切片**：`EVAL_SPLIT` + 多 split 文件（livecodebench 的 `test_v6_2408_2505`），一个目录承载多时间点切片防污染对比。
- **已知问题就地记录**：gsm8k prepare 里的 `fixes` 字典、livecodebench 文件尾的期望样本数注释、arena-hard prepare 里对上游 category 字段的 workaround 注释、bfcl_v4 的 `TODO: format_sensitivity not graded` 注释——**全部内嵌在 prepare.py 里，不另建 tracker**。
- **去重工具沉淀在 `dataset/utils.py`**：[utils.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/utils.py) 提供 `locate()`、`get_dataset_module()`（内建/extra map/绝对路径三级解析，冲突报错）、`save_data_from_qwen()`（一批数学数据集共用 Qwen2.5-Math 的 jsonl）、`get_mcq_fields()`、`get_question_hash()`+`filter_by_subset()`（子集清单机制）、`add_rounding_instruction()`。新 benchmark 的 prepare 大量复用这些，所以单个 prepare.py 普遍 <100 行。
- **每个 benchmark 在 docs 的类别页面有一节条目**（定义链接 + 原始出处链接），judge 默认开哪些 benchmark 也在文档里显式列出。

---

## 4. aqa 不该抄的部分（与 NeMo-Run/Slurm/Hydra/server 栈耦合）

- **Hydra/omegaconf 配置机**：`@nested_dataclass(kw_only=True)` + `cs.store` + `@hydra.main` + `++key=value` CLI 语法贯穿所有任务。aqa 已有 recipe YAML，不要引入第二套配置系统；把 `GENERATION_ARGS` 字符串这种"CLI fragment as config"改成结构化 YAML 字段即可。
- **pipeline 层**（`nemo_skills/pipeline/*`）：cluster configs、容器打包（`/nemo_run/code` 硬编码路径到处都是）、`executor: local/slurm/none`、sshtunnel、 NeMo-Run experiment 依赖链。`get_benchmark_args_from_module` 里大量 mount/unmount 路径换算对单机 aqa 是无谓复杂度。
- **server 启动层**：`serve_vllm.py`/`serve_sglang.py`/`serve_vllm_dp_ray.py`/`serve_unified.py` 是 Slurm+容器语境下的 subprocess 壳；aqa 的 harness（in-process HF / lm_eval CLI / vllm-omni）已经是不同抽象，最多借鉴"统一 OpenAI endpoint + `get_model(server_type)` 注册表"这一个想法。
- **litellm 全家桶**：BaseModel 里对 litellm 的三处 TTL patch、logging patch、disk cache——这是为了统一调几十家 API provider 付出的代价；aqa 主要自托管 vLLM，直接用 openai SDK/httpx 即可。但 `context_retry.py`（soft-fail + 三种裁剪策略 + 各 server 报错 regex 解析）**逻辑独立、应该抄**，只需把 litellm 异常换成自己 harness 的异常。
- **自定义 GENERATION_MODULE 子类化范式**：swebench.py（50KB）、bfcl.py（27KB）这种"整个生成循环重写"对大 agentic benchmark 是必要恶，但不作为默认扩展点——aqa 保持 recipe 声明式，只在确实需要时留一个 `module::Class` 逃生门。
- **代码执行 sandbox / tool_calling / parallel_thinking wrapper**：与量化评测 gate 的主线无关。

## 5. 建议 aqa 直接采纳的清单（按性价比排序）

1. **输出文件自包含 + evaluator 原地富化**：输出行 = 输入行 + `generation` + `predicted_answer`/`symbolic_correct`/`judgement`/`is_correct`，`os.replace` 原子重写。aqa 的 scorer 输出 key 直接对齐这套命名，metrics 聚合就能零胶水复用。
2. **`output.jsonl` vs `output-rs{seed}.jsonl` 布局 + 互斥校验 + 按行 zip 对齐**，配合 `BaseMetrics._compute_pass_at_k` 的无偏 pass@k（组合数公式）、`pass@1[avg-of-k]`、majority@k 确定性 tie-break、`_add_std_metrics` 的 std_dev/std_err_across_runs + avg_sample_std_dev——这正是 aqa gate 统计显著性需要的全部算术，且 `subset_for_metrics` 一行数据字段就能自动出分维度报表。
3. **recipe 声明字段对齐 `__init__.py` 协议**：aqa 的 `recipes/benchmarks/*.yaml` 增加 `metrics_type`、`eval_split`、`judge: {model, server..., generation_type}`、`is_benchmark_group + benchmarks + score_module`、`num_samples`（即 `:N` repeats）、`requires_data_dir` 这些键，就把 NeMo-Skills 的目录协议翻译成了 YAML 协议。
4. **`module::Class` / `/path.py::Class` 动态定位**（`locate()`，20 行）：scorer/evaluator/metrics 三处注册表都支持路径形式，外部 benchmark 无需改主仓——配合 aqa 的 scorer 库（exact_match/…/llm_judge/code_exec）作为内建注册表。
5. **judge 即第二个 generation task**：`generation_key="judgement"`、`add_generation_stats=False`、`prefill_judgement` 短路（抽不出答案或精确匹配时不调 judge，省钱）、arena 式双向请求消位置偏置。
6. **soft-fail error 字段**：失败样本落成 `{"generation":"", "finish_reason":"error", "error": ...}` 计入分母，而不是 job 崩溃 + 缺数据。
7. **math_verify 依赖**替换/补强 numeric_match；`is_correct_judgement` 的多格式 judge 解析。
8. **断点协议**：`-async` 行缓冲追加 + `_async_position` + skip_filled + 完成后重排删临时文件；分片 `_chunk{i}.jsonl` + `.done` + cat 合并。
9. **prepare 侧约定**：统一 key schema、已知错题 `fixes` 内嵌、期望样本数注释做回归、subsets/*.txt hash 清单做快速 smoke 子集（对量化回归的"小集快筛→大集确认"工作流直接可用）、`ns prepare_data` 式并行批量准备（线程池 20 + retries 3）。

Sources 汇总：[inference 目录](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/inference) · [evaluation 目录](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/evaluation) · [dataset 目录](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/dataset) · [官方 evaluation 文档](https://nvidia-nemo.github.io/Skills/evaluation/) · [Adding new benchmarks](https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/evaluation/index.md#adding-new-benchmarks) · [External benchmarks](https://github.com/NVIDIA-NeMo/Skills/blob/main/docs/evaluation/external-benchmarks.md)