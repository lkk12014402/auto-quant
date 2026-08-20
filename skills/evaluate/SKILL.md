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
  correct scoring needs code we own (migrated from NeMo Evaluator, Apache-2.0).
- `lm_eval`  — EleutherAI lm-eval tasks (piqa/mmlu/hellaswag/gsm8k/arc/...).
- `evalscope`— optional plugin (VLM/embedding/RAG + perf TTFT/TPOT).

## Backend routing
- `hf` (transformers) | `vllm` (CUDA) | `vllm_xpu` (plugin). The backend builds
  lm_eval `model_args`; generative tasks add chat-template flags automatically.

## Procedure
1. Resolve each benchmark to its recipe; split generative vs loglikelihood.
2. Build & run the harness command; parse the framework's own results.
3. Merge per-benchmark scores into `accuracy.json`; optionally `compare` vs a
   bf16 baseline and apply a `gate` (GO/NO-GO on accuracy drop).

## Accuracy rules
- Never apply chat templates to loglikelihood tasks (corrupts scores).
- Let each framework score its own benchmark; the adapter only invokes + parses.
