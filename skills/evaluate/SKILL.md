---
name: evaluate
description: >
  Evaluate a model on one or more benchmarks. Use to pick benchmarks, select a
  harness (native / lm_eval / evalscope) and infra backend (hf / vllm / vllm_xpu),
  handle generative vs loglikelihood tasks, and merge scores.
---

# Evaluate skill

Run benchmarks and produce a normalized `accuracy.json`, choosing the right
**harness** and **backend** per benchmark recipe.

## When to use
- The user asks to evaluate/benchmark a (quantized or bf16) model.
- A benchmark needs special handling: generative (chat template + fewshot as
  multiturn + generation budget) vs loglikelihood multiple-choice.

## Knowledge source
- `recipes/benchmarks/*.yaml` — each declares `harness`, `type`, scoring, and any
  chat-template / gen params. Adding a benchmark = add a YAML (+ a `.py` scorer
  only when correct scoring needs code: math equivalence, code sandbox, judge).

## Harness routing
- `native`   — in-repo kernel (dataset URI + prompt + scorer library). Best when
  correct scoring needs code we own (see the repo `NOTICE` for attributions).
- `lm_eval`  — EleutherAI lm-eval tasks (piqa/mmlu/hellaswag/gsm8k/arc/...).
- `evalscope`— optional plugin (VLM/embedding/RAG + perf TTFT/TPOT).

## Backend routing
- `hf` (transformers) | `vllm` (CUDA) | `local_completions` (an already-running
  OpenAI-compatible completions service) | `vllm_xpu` (plugin). The backend
  builds lm_eval `model_args`; generative tasks add chat-template flags
  automatically. Keep service startup/parallelism in a reproducible launcher
  and record the command with `aqa log-action`.
- Use `local_completions` when the serving topology must be controlled by a
  separate vLLM server or cannot be expressed by the installed lm_eval vLLM
  adapter. For lm_eval 0.4.13, its direct vLLM adapter rejects
  `data_parallel_size > 1` together with `enable_expert_parallel`; a local
  completions endpoint can still use vLLM TP/DP/EP while AQA runs and records
  the lm_eval tasks.
- Probe both Transformers config paths when `config.json` has an `auto_map`.
  A registered native vLLM architecture does not require `trust_remote_code`;
  enabling it can force an older/incompatible remote config implementation even
  though the installed Transformers and vLLM support the model natively. Prefer
  `trust_remote_code: false` when the native config/registry path is verified,
  and set the CLI flag only when remote code is required and independently tested.
- For quantized MoE, a registry hit or backend support probe is only a filter:
  run a real weight-load and generation smoke with the target GPU, model shape,
  and final TP/DP/EP topology, then confirm the selected backend in the log.
  If an automatically selected kernel fails for a shape/configuration reason,
  investigate or test an explicit supported backend; shrink batch/token budgets
  only for demonstrated OOM. Keep the working backend flag in the reproducible
  serving launcher and revalidate it on the full-evaluation topology.
- For AutoRound static FP8 attention scales, checkpoint presence alone does not
  prove query quantization is active. In the validated vLLM 0.31.0 / RTX 5090
  SM120 path, use `TRITON_ATTN` with FP8 KV; the runtime probe must compare each
  attention layer's loaded q/k/v scales with the checkpoint and confirm
  `supports_quant_query_input=true` plus a `QuantFP8` query quantizer. The
  tested Nemotron 3 Nano setup passed this probe on all six attention layers
  under TP2×DP2+EP. Do not infer active FP8 attention from YAML or exported
  scales alone.
- Gemma-4-31B on vLLM 0.32.0 / SM120 (2026-10-21): `linear_backend=auto` may pick
  a kernel outside the vllm-qdq-plugin's marlin patch coverage, silently
  dropping activation QDQ (2026-10-19 QDQ RULER run pitfall #1). For
  QDQ-enabled evals pass `linear_backend: marlin` explicitly and accept the run
  only when the engine log shows `Using MarlinMxfp8LinearKernel for MXFP8 GEMM`
  plus `QDQ patches applied: ... torch.ops._C.marlin_gemm (MXFP4, MXFP8)` and
  `CUTE QDQ backend requested: VLLM_QDQ_CUTE=1`. vLLM's "Unknown vLLM
  environment variable" warning for `VLLM_QDQ*` is benign; also note AutoRound
  0.16.1 exports scales as `...self_attn.{q,k,v}_scale` (no underscore prefix;
  vLLM maps them onto `_*_scale` buffers via compressed_tensors.py:1159-1269).
- Regression evidence for Qwen3.6-35B-A3B on vLLM 0.31 / SM120: default
  FlashInfer kept the query in BF16 (`supports_quant_query_input=false`,
  `query_quant=null`) while loading a non-unit checkpoint `q_scale`; MMLU-50
  fell to 0.418. Explicit `TRITON_ATTN` enabled `QuantFP8` and recovered MMLU-50
  to 0.848. Pin and probe the backend before evaluation. AQA's vLLM lm_eval
  harness forwards custom `gen_kwargs` (such as `until`) via `--gen_kwargs`;
  inspect the effective task parameters in the run log to verify the override.
- `enable_thinking` is a model-level argument and is incompatible with
  loglikelihood tasks in the tested lm_eval/vLLM path. When a suite mixes
  generative and loglikelihood tasks, run them with separate AQA configs so
  generation-only thinking arguments do not invalidate the likelihood tasks.

## Long-context RULER

### Suite and task recipes
- The lm_eval 0.4.13 suite has 13 standalone IDs: `niah_single_1/2/3`,
  `niah_multikey_1/2/3`, `niah_multiquery`, `niah_multivalue`, `ruler_vt`,
  `ruler_cwe`, `ruler_fwe`, `ruler_qa_squad`, and `ruler_qa_hotpot`. Pass the
  explicit task IDs when the aggregate group could select a different context
  bucket; use the repository benchmark recipes as the source of task names,
  scorers, and native generation caps.
- For the current repository recipes, generation caps are 128 tokens for the
  eight NIAH tasks, 30 for VT, 120 for CWE, 50 for FWE, and 32 for each QA
  task. Keep RULER generative: for models without a chat template set
  `eval.lm_eval.apply_chat_template: false` and
  `fewshot_as_multiturn: false`, without changing the recipe's scoring or
  generation budget.
- RULER synthesizes prompts with a tokenizer. Set both
  `metadata.max_seq_lengths` and `metadata.tokenizer` (or `pretrained`) so data
  construction uses the intended model tokenizer. For a 128K run, select
  `[131072]` and confirm the emitted primary metric is `131072,none`.
  `--limit 100` applies **per task**; it is a capped comparison, not an
  official full-set score. In lm_eval 0.4.13, the task builder may synthesize
  500 records per task/bucket before the limit is applied, so dataset
  construction can take appreciable time even when only 100 samples are
  scored.

### AQA configuration and serving topology
- Declare the task list, `limit`, harness, endpoint, tokenizer, context bucket,
  and generation behavior in the AQA YAML before starting the server/run.
  Preserve run YAMLs, launch scripts, environment pins, and results under the
  run output directory; use `aqa validate` and `aqa run --dry-run` before
  consuming accelerators.
- If a MoE run needs vLLM TP+DP+EP, check the installed lm_eval adapter first.
  lm_eval 0.4.13's direct vLLM adapter rejects `data_parallel_size > 1`
  together with `enable_expert_parallel`; use AQA `local_completions` to call
  an already-running vLLM server when that topology is required. For the
  validated same-NUMA setup, TP=2/DP=2/EP with PP=1 fit on four GPUs; PP is a
  fallback only after confirming that the preferred layout cannot fit.
- Long prompts consume substantial KV memory. Start with conservative server
  `max_num_seqs` and client concurrency/batch settings, then increase only
  after checking free memory and throughput. A validated setup used
  `max_num_seqs=1` per DP worker and client `batch_size=2`,
  `num_concurrent=2`; these values are hardware-specific, not universal
  defaults. Do not allocate all visible GPUs by default.
- A concrete `eval.batch_size` in the AQA YAML overrides the benchmark recipe's
  batch-size default; `"auto"` retains the recipe default. Verify the emitted
  `[CMD]` before launch, especially for RULER recipes that declare a default of
  2. If a task specifies a required starting batch size, reduce it only after
  observing GPU OOM; a client timeout alone is not evidence of OOM.
- A model's `dtype=bfloat16` does not prove its KV cache is BF16. Set the
  intended `kv_cache_dtype` explicitly where needed and verify the actual
  vLLM startup/engine logs (including the resolved cache dtype), especially
  when comparing BF16 and quantized checkpoints.
- On SM120 with the Intel vLLM QDQ plugin, vLLM 0.31 may report
  `VLLM_QDQ*` variables as unknown even when the plugin's startup patches are
  applied. Patch-registration messages and a "CUTE backend requested" notice
  prove initialization/configuration only; confirm per-op QDQ/CUTE execution
  from runtime traces before claiming kernel coverage or acceleration.

### Tokenized context-boundary handling
- RULER appends `gen_prefix` after its synthetic-data length calculation.
  Consequently, a nominally full-length document may exceed the engine limit
  after the prefix is added; raw-text re-tokenization can add further
  uncertainty. Reducing the task's native generation cap by one token is not a
  reliable fix and can still overflow.
- For a vLLM completions endpoint, use `tokenized_requests: true` and
  `add_bos_token: true` so request IDs match the tokenizer used to build the
  synthetic data. The lm_eval API adapter subtracts one from `max_length`
  internally: with `max_model_len=131072`, configure `max_length: 131073`.
  The adapter then left-truncates any excess prompt prefix as needed to keep
  prompt plus completion within the 131072-token limit. Preserve task-native
  generation budgets and record this boundary behavior; do not raise the
  server's `max_model_len` to compensate.

### Environment and data-source pitfalls
- The RULER custom task dependencies can include `wonderwords` and NLTK
  `punkt_tab`. Check them during environment preflight, install/pin missing
  dependencies through the declared environment, then rerun self-checks and
  refresh the frozen package list. Keep downloaded NLTK data in an explicit,
  private data directory. Only set `NLTK_ALLOW_PROXIED_URLOPEN=1` when using a
  trusted proxy and only for the process that performs the download.
- With `datasets==5.0.1` and `huggingface-hub==1.32.0`, set `HF_ENDPOINT` to
  the official Hub host **without a trailing slash**; otherwise the `hf://`
  resolver can construct a malformed API path for remote RULER data. Verify
  the remote source is reachable before launching a long evaluation.

### Result checks and interpretation
- After each run, verify every requested task appears in `eval/accuracy.json`;
  check the framework result/`n-samples` data for the effective per-task count
  and confirm the primary metric key matches the selected bucket. Read the
  successful task logs; do not infer success from process exit alone.
- Compute a RULER mean from the RULER task scores only. Keep it separate from
  other benchmark-suite means. For comparisons, report per-task scores and
  `MXFP8 / BF16 × 100%` ratios; if the two runs use different KV-cache
  precision, label the result a serving-protocol comparison, not a weights-only
  ablation. A capped `--limit` run should not be described as an official
  full-set result.
- Finish through the normal AQA flow: inspect the run artifacts, complete the
  report and experience-routing table, and run `aqa artifacts check`. Do not
  stop a shared vLLM server or begin the next model run until AQA has finished
  and its GPUs are confirmed free.

## Procedure
1. Resolve each benchmark to its recipe; split generative vs loglikelihood.
2. Build & run the harness command; parse the framework's own results.
3. Merge per-benchmark scores into `accuracy.json`; optionally `compare` vs a
   bf16 baseline and apply a `gate` (GO/NO-GO on accuracy drop).

## Accuracy rules
- Never apply chat templates to loglikelihood tasks (corrupts scores).
- Generative recipes use chat templates and few-shot-as-multiturn by default. If a model has no chat template, explicitly set both `eval.lm_eval.apply_chat_template: false` and `eval.lm_eval.fewshot_as_multiturn: false`; the task remains generative and retains its recipe's shots and generation budget.
- Let each framework score its own benchmark; the adapter only invokes + parses.
