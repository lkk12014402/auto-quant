调研完成。以下是结构化报告。

---

# DeepSeek 开源评测/Harness 工具调研报告（面向 aqa）

## 0. 核心结论（先说重点）

**DeepSeek 没有开源过一个统一的评测框架。** 其内部评测框架（集成在 HAI-LLM 训练框架内，见 [V3 报告 §5.3.1](https://arxiv.org/html/2412.19437v1)）从未开源。真正开源的是**分散在各模型 repo 里的 benchmark-specific 评测脚本**，外加 2026 年 8 月新开源的 `deepseek-harness`——但那是**agent harness（agent 运行时）**，不是 eval harness。V3/V3.2/R1 论文里的主表数字大多来自内部框架 + 公开 harness（simple-evals prompts、agentless、Claude Code/Terminus 等），复现靠的是论文里写得非常详细的**评测协议参数**，而不是代码。

⚠️ 注意：`deepseek-harness` 创建于 2026-08-13、两周内 197k stars，数据异常（V3 一年多才 100k），我对其真实性持保留态度，下面如实记录其内容。

## 1. GitHub org 盘点（36 个 repo，[API](https://api.github.com/orgs/deepseek-ai/repos?per_page=100)）

eval/harness 相关候选：

| Repo | 性质 | 评测代码内容 | 框架基础 |
|---|---|---|---|
| [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) | **Agent harness，非 eval**（"Everything is a Plugin"，TypeScript + [Cordis](https://github.com/cordiverse/cordis) 插件体系，Web UI，`npx @deepseek-ai/dsh web`） | [BENCHMARK.md](https://raw.githubusercontent.com/deepseek-ai/deepseek-harness/master/BENCHMARK.md) 仅一页：用 Python SDK 的 `jsonrpc-agent` minimal variant 跑独立 benchmark task（session/workspace 隔离）。packages/ 下有 `sandbox/`, `mcp/`, `hooks/`, `compaction/`, `goal/`, `plan/` 等 agent 运行时包 | 自研 Cordis 插件架构 |
| [DeepSeek-Math/evaluation](https://github.com/deepseek-ai/DeepSeek-Math/tree/main/evaluation) | 数学评测脚本 | 多 GPU 并行（`submit_eval_jobs.py` + `run_subset_parallel.py` + `summarize_results.py`），vLLM 推理，JSON config 驱动；CoT/TIR 两类 prompt 模板（英文 `Please reason step by step, and put your final answer within \boxed{}` + 中文对应）；scorer 是 `eval_script.py` 的 `math_equal` 符号等价 + 数值容差 1e-3；miniF2F 走外部 [PISA/Isabelle server](https://github.com/wellecks/lm-evaluation-harness/blob/minif2f-isabelle/docs/isabelle_setup.md)；附全部模型输出 `outputs.zip` | **纯自研**，非 lm-eval/simple-evals fork |
| [DeepSeek-Math-V2/inference](https://github.com/deepseek-ai/DeepSeek-Math-V2/tree/main/inference) | proof-based 评测 pipeline | 最完整的开源评测代码：`main.py` 实现 generate → self-eval（证明尾部输出 `\boxed{score}` 自评）→ verification（每证明 N=4~64 次独立打分）→ meta-verification → refinement（proof pool 采样 best-k 组合重写）的多轮循环，参数化：temp 1.0、proof_gen 128K max_len、verify 64K max_len、max_rounds 16~20；底层是 async OpenAI API client（`generate.py`）打自建服务 | 自研 + OpenAI API 协议 |
| [DeepSeek-Coder/Evaluation](https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/Evaluation) | 代码评测 | 每 benchmark 一个目录：HumanEval（fork 自 [openai/human-eval](https://github.com/openai/human-eval)，`evaluate_functional_correctness`）、DS-1000、LeetCode、MBPP、PAL-Math（程序辅助）；HF transformers 推理，自定义 instruction 模板 | fork openai/human-eval + 自研 |
| [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) / [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) | OCR 评测 | 见 §3 | vLLM 自定义模型注册 |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) | speculative decoding 训练+评测 | [eval.py](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/main/eval.py)：测 draft 模型接受率/加速比，任务列表 gsm8k(500)/math500(500)/aime25(30)/humaneval/mbpp/livecodebench/mt-bench/alpaca/arena-hard-v2，temp 1.0，max_new_tokens 2048，带 confidence-head early-stop 校准指标 | 自研 deepspec 包 |
| DeepSeek-V3 / R1 / V3.2-Exp / VL2 / Janus / Prover-V2 | **无评测代码** | 仅 README 表格 + 推理 demo（V3.2-Exp 的 [inference/](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference) 是 FP8 权重转换 + 交互 demo） | — |

## 2. 技术报告中的评测协议（这才是最有价值的部分）

**DeepSeek-V3**（[arXiv 2412.19437 §5.3.1](https://arxiv.org/html/2412.19437v1)）：
- 自评："evaluation is based on our internal evaluation framework integrated in our HAI-LLM framework"——**内部框架，未开源**
- MMLU/DROP/GPQA/SimpleQA 用 [openai/simple-evals](https://github.com/openai/simple-evals) 的 prompt；MMLU-Redux 用 [Zero-Eval](https://arxiv.org/abs/2406.15039) prompt 格式
- AIME/CNMO 2024：**temp 0.7，16 次平均**；MATH-500：**greedy decoding**；所有 benchmark max output **8192 tokens**
- SWE-bench 用 [agentless](https://arxiv.org/abs/2407.01489) 框架；Aider 用 "diff" 格式；RewardBench 报 maj@6

**DeepSeek-R1**（[arXiv 2501.12948 §3 Evaluation Setup](https://arxiv.org/html/2501.12948v1) + [README](https://github.com/deepseek-ai/DeepSeek-R1)）：
- max generation **32,768 tokens**；**temp 0.6 / top-p 0.95**
- 明确说明为何不用 greedy："greedy decoding … higher repetition rates and significant variability across different checkpoints"
- **avg pass@k**：每题采 k 个回答（按测试集大小 4~64 个），`pass@1 = (1/k)Σ p_i`；AIME 另报 **cons@64**（majority vote）
- SWE-Verified 走 agentless；官方建议：不加 system prompt；数学 prompt 强制 `put your final answer within \boxed{}`；**强制输出以 `<think>\n` 开头**防止 bypass thinking；建议多次测试取平均

**DeepSeek-V3.2**（[arXiv 2512.02556 §4 + Appendix C/D](https://arxiv.org/html/2512.02556v1)）：
- Tool-use benchmark 统一用 standard function-call 格式 + thinking mode；MCP-Universe/MCP-Mark 用**内部环境**复跑所有模型，**temp 1.0，context 128K**
- 数学模板与 R1 相同；HLE 另用官方模板对照（25.1 vs 23.9）
- **Table 3 双报告制：accuracy + output token 数（千）**——效率与精度并列评估
- thinking vs non-thinking 系统对比（Appendix C Table 9）；**框架鲁棒性协议**：Terminal Bench 2.0 因 thinking context 管理与 Terminus 不兼容，改用 Claude Code 框架（46.4 vs 39.3）；SWE-Verified 用内部框架主跑、Claude Code/RooCode/non-thinking 交叉验证（72~74 区间）
- 长上下文处理：search agent 超 128K 的 case 用 context management 技术（带 * 标注）
- Appendix D（竞赛评测）：max gen 128K、无工具、先生成大量候选 → 过滤无法解/拒答 → **选 thinking trace 最长的 50 条提交**（IOI）；ICPC 每题 32 候选同法筛选
- 合成 agentic 任务报 **pass@k，k=1/2/4**

## 3. DeepSeek-OCR / OCR-2

- Benchmark：**OCR-1 用 [Fox](https://github.com/ucaslcl/Fox) + [OmniDocBench](https://github.com/opendatalab/OmniDocBench)；OCR-2 用 OmniDocBench v1.5**（README 明确致谢，[OCR-1 README](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-OCR/main/README.md) / [OCR-2 README](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-OCR-2/main/README.md)）
- 开源的只有**推理/生成侧**：[run_dpsk_ocr_eval_batch.py](https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py)（OCR-2 同构）：vLLM `ModelRegistry.register_model` 注册自定义架构，**temp 0.0、max_tokens 8192**、`NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90)` 防重复、prompt `<image>\n<|grounding|>Convert the document to markdown.`、`max_num_seqs=100` 并发
- **打分代码未开源**，需用 OmniDocBench/Fox 官方评测代码对输出后处理。olmOCR 未使用

## 4. 对 aqa 的移植清单（verdict）

**值得移植（按优先级）：**

1. **评测协议参数包（recipes/benchmark packs）**：R1 系（temp 0.6 / top_p 0.95 / max_gen 32768 / 禁用 greedy 的理由）、V3 系（AIME temp 0.7×16 次平均、MATH-500 greedy、max 8192）、V3.2 agentic 系（temp 1.0、128K context）。来源：上述论文章节，直接做成 aqa 的 preset。
2. **avg pass@k + cons@k（maj@k）scorer 插件**：`pass@1 = (1/k)Σp_i` + majority voting，k 按数据集大小 4~64（R1 §3）；V3 的 RewardBench maj@6 也是同一模式。aqa 的 lm_eval/native harness 都应支持。
3. **数学符号等价 scorer**：DeepSeek-Math 的 [`math_equal` + boxed 提取 + 数值容差](https://github.com/deepseek-ai/DeepSeek-Math/tree/main/evaluation/eval)，自包含、无重依赖，可直接抄成 scorer plugin。
4. **accuracy + output-token 双指标报告**（V3.2 Table 3 模式）：对量化评估天然契合（量化影响输出长度/重复），建议做成 aqa 的标配 metric。
5. **thinking-mode phase hooks**：强制 `<think>\n` 前缀（R1 README）、thinking content 在 tool-use 多轮中的保留/丢弃策略（V3.2 §3.2.1）——正好落在 aqa 的 phase-hook 层。
6. **verifier-in-the-loop 评测循环**（[Math-V2 inference](https://github.com/deepseek-ai/DeepSeek-Math-V2/tree/main/inference)）：generate → self-eval → verify → refine 的 proof-pool 多轮结构，参数化完整，可作为 E0-E4 agentic eval 的高级 recipe 模板。
7. **框架鲁棒性协议**：SWE-bench/terminal-bench 适配器上，主框架 + 备用框架交叉验证并报告区间（V3.2 做法），aqa 已有这两类 adapter，加这个协议成本低价值高。
8. **`NoRepeatNGramLogitsProcessor`**（OCR 系）：防长文档生成重复的 logits processor，可进 diffusion/生成类评测的 sampling 配置。

**不必移植 / 太内部：**

- **HAI-LLM 内部评测框架**——未开源，无从移植；V3.2 的内部 MCP 环境、内部 SWE 框架同理。
- **deepseek-harness（dsh）**：是 agent 运行时（Cordis 插件 + Web UI + Python SDK），不是 eval harness；仅当 aqa 想把 dsh 作为 E0-E4 的**被测 agent target** 时才有意义，其 BENCHMARK.md（jsonrpc-agent 跑独立 task）提供了一点 session 隔离思路，但其余与评测无关。
- **DeepSpec eval.py**：只服务 speculative decoding 接受率测量，任务列表可参考，评测器本身与量化无关。
- **竞赛提交流程**（Appendix D 的最长-thinking-trace 筛选）：过于特定，不通用。

**诚实的总评**：DeepSeek 开源的评测代码是"每个模型 repo 各带一套脚本"的形态，质量参差但协议文档极其详尽。对 aqa 最有价值的不是代码本身，而是**他们论文里沉淀的评测协议**（采样参数、pass@k 方法学、thinking/非 thinking 双模式、框架交叉验证、效率-精度双报告），这些适合以 recipes + scorer plugins + phase hooks 的形式固化；代码层面只有 DeepSeek-Math 的数学 scorer 和 Math-V2 的 verify-refine 循环值得直接借鉴。