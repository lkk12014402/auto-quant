# autoquant-agent

Agent-driven **automatic model quantization + evaluation + self-healing**.

A declarative, extensible framework that turns a single `run.yaml` into a
`quantize → evaluate → gate` pipeline, with an agent fix/patch loop that
root-causes and repairs environment/code failures and captures reviewable
patches. Built to be **open-source** and to eventually **replace** the historic
`local_dispatch`/`lb_eval` flow.

Design is grounded in the patterns proven by
[NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) (Apache-2.0):
declarative recipes, a plugin registry, harness/backend/executor adapters, and
`compare`/`gate`. The native evaluation kernel (scorer library + loop + dataset
loading) is adapted from NeMo Evaluator — see `NOTICE`.

## Install

```bash
pip install -e .                 # core (config + registry + CLI + native scoring)
pip install -e ".[lm-eval]"      # + lm-evaluation-harness adapter
pip install -e ".[hf]"           # + transformers/torch for the native HF runner
pip install -e ".[dev]"          # + pytest/ruff
```

## Quick start

```bash
aqa list                          # benchmarks + registered plugins
aqa validate examples/run.yaml    # type-check a run config, no execution
aqa run examples/run.yaml         # quantize -> evaluate -> gate (local executor)
aqa run examples/run.yaml --dry-run
aqa eval examples/eval_only.yaml --out ./results     # evaluate only
aqa compare ./results/base ./results/cand            # score deltas
```

## One file describes a run (`run.yaml`)

```yaml
model: Qwen/Qwen3-0.6B
provider: { type: quantize, quant: { scheme: MXFP4, method: model_free } }
eval: { benchmarks: [piqa, hellaswag, gsm8k], backend: vllm, baseline: true }
gate: { max_accuracy_drop: 0.02 }          # NO-GO if quantized drops > 2%
agent: { fix_loop: true, optimize: true, tiers: [openclaw, minimax, opus] }
executor: { type: local }                  # local | ssh | intel_reserve (plugin)
output: { dir: ./results/run, sinks: [ { type: local_fs } ] }
```

Composition: `_base_: other.yaml` (inherit + deep-merge, `null` deletes),
`${VAR}` / `${VAR:-default}` env expansion. **Secrets are only referenced via
`*_env` variables, never inlined** (open-source hard constraint).

## Model providers: evaluation is the core, quantization is a plugin

The framework is **eval-centric**: the primary job is to *evaluate a model*; how
that model is produced/served is a pluggable **`provider`** (mirroring NeMo
Evaluator's separation of the *served target* from the *benchmark*).

| `provider.type` | What it evaluates |
|---|---|
| `pretrained` (default) | a HF repo id / local path, as-is |
| `quantize` | runs autoquant (optionally agent-optimize + fix-loop), then evaluates the quantized checkpoint |
| `endpoint` | an already-served OpenAI-compatible endpoint (agentic path) |

```yaml
# eval-only (default provider is `pretrained`):
model: Qwen/Qwen3-0.6B
eval: { benchmarks: [piqa, gsm8k], backend: vllm }

# quantize-then-eval (quantization is now just a provider):
provider: { type: quantize, quant: { scheme: MXFP4, method: model_free } }

# evaluate a served endpoint:
provider: { type: endpoint, base_url: http://host:8000/v1, served_model: org/m,
            api_key_env: MY_API_KEY }
```

**Backward compatible.** Older quant-centric configs that put `quant:` at the top
level keep working unchanged — the loader folds a top-level `quant:` into
`provider: {type: quantize, quant: {...}}`, and `run_cfg.quant` still returns the
quantize provider's config. `examples/run_5000.yaml` runs as before. Add your own
provider via the `aqa.providers` entry-point group.


### Remote dispatch: NVIDIA CUDA or Intel XPU

The `ssh` executor runs the whole pipeline inside a container on a remote host.
`executor.ssh.accelerator` (`cuda` default, or `xpu`) switches the device flags
and default base image — no other config changes:

```yaml
executor:
  type: ssh
  image: { xpu: intel/llm-scaler-vllm:0.21.0-b1 }   # or `gpu:` for CUDA
  ssh:
    host: sdp@192.168.11.2
    gpus: "0,1"                 # cuda -> --gpus device=; xpu -> ZE_AFFINITY_MASK
    accelerator: xpu            # cuda | xpu
    auth: { method: password, password_env: B60_SSH_PASS }
    jump: [ ... ]               # optional multi-hop chain
```

- **cuda** → NGC base + `--gpus '"device=<ids>"'`, installs matching `vllm`.
- **xpu** → Intel `llm-scaler-vllm` base + `/dev/dri` passthrough, `--privileged`,
  `ZE_AFFINITY_MASK=<ids>`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`. The base image
  already ships `vllm-xpu`, so aqa does **not** reinstall CUDA `vllm` (only
  `auto-round` + `lm-eval` on top). See `examples/run_b60_xpu.yaml`.

`image.dockerfile: local` builds the accelerator-appropriate shipped Dockerfile —
`docker/gpu.dockerfile` (NGC) for CUDA, `docker/xpu.dockerfile` (Intel base) for
XPU — each adding tooling + Node + OpenClaw for the fix-loop. Omit `dockerfile` to
run the base image directly (Node/OpenClaw are then installed at runtime).

## Evaluation: harnesses, task grouping, and model_args

`eval` selects a **harness** (`lm_eval` default, `native`, or `evalscope` plugin)
and an infra **backend** (`hf` | `vllm`). Harness-specific knobs live in per-method
blocks so the eval methods stay cleanly separated:

```yaml
eval:
  benchmarks: [piqa, hellaswag, mmlu, gsm8k]
  backend: vllm
  harness: lm_eval
  lm_eval:
    model_args: { enable_thinking: false, reasoning_parser: qwen3, max_num_seqs: 128 }
    gen_kwargs: { max_gen_toks: 2048 }   # default --gen_kwargs for generative tasks
    group: true                          # merge compatible tasks into one call
    seed: 42
  evalscope: { }                         # options forwarded to the evalscope plugin
```

- **`model_args`** are appended to the backend's `--model_args` (user keys override
  defaults) — e.g. Qwen3 reasoning models set `enable_thinking=False,reasoning_parser=qwen3`.
- **Task grouping** (`group: true`) runs `piqa,hellaswag,mmlu` in **one** `lm_eval`
  call (one model load), while `gsm8k` — which needs `--apply_chat_template
  --fewshot_as_multiturn` — runs **separately**. Tasks are grouped by matching
  generation params; incompatible tasks each get their own invocation.

**Where per-task params live.** A benchmark's own knobs — `type` (generative vs
loglikelihood), `apply_chat_template`, `fewshot_as_multiturn`, `num_fewshot`,
`gen_kwargs`, `batch_size`, `scoring` — belong in its recipe file
`recipes/benchmarks/<task>.yaml` (single source of truth), **not** in every
`run.yaml`. For example `recipes/benchmarks/gsm8k.yaml` pins gsm8k to
`generative + apply_chat_template + fewshot_as_multiturn + num_fewshot:5 +
batch_size:"64"`, so any run that lists `gsm8k` gets the correct settings and the
runner automatically isolates it from the loglikelihood group. Use the `eval.*`
blocks only for run-wide policy (backend, global batch size, seed, model_args).

**Breadth via lm_eval passthrough.** Any lm_eval task name works even without a
recipe file — `benchmarks: [<task>]` synthesises a bare `lm_eval` passthrough, so
lm-eval-harness's full catalogue is reachable. Curated recipes just pin the
*correct* per-task settings so a task "just works": shipped ones now include the
static instruct/base suite that NEL runs — `gpqa_diamond`, `mmlu_pro`, `ifeval`,
`bbh`, `truthfulqa`, `winogrande`, `openbookqa` (alongside `mmlu`, `gsm8k`,
`hellaswag`, `arc_challenge`, `piqa`). Add one file to onboard the next task; you
never reimplement the benchmark.

**LLM-as-judge scoring** (`scorer: llm_judge`) covers the open-ended / rubric
family (NEL's Gemini-judge AA-Omniscience, GDPVal pairwise/rubric). The judge is
any OpenAI-compatible endpoint, configured per recipe under `scoring.judge`:

```yaml
scoring:
  scorer: llm_judge
  judge: { model: gpt-4o-mini, api_key_env: OPENAI_API_KEY, scale: 10, threshold: 7 }
```

The judge rates each answer `0..scale`, normalised to `[0, 1]` (or binarised by
`threshold`). See `recipes/benchmarks/openqa_judge.yaml`. The judge client is
injected, so the scorer is unit-tested offline with a fake judge.

**Code-execution scoring** (`scorer: code_exec`) covers the functional-code family
(LiveCodeBench / HumanEval / MBPP / SciCode): the model writes a solution, the
scorer assembles `[completion + test + check(entry_point)]` into one program, runs
it in a sandbox, and passes iff it exits cleanly.

```yaml
scoring:
  scorer: code_exec
  code_exec: { include_prompt: true, timeout_s: 30, test_field: test, entry_point_field: entry_point }
```

See `recipes/benchmarks/humaneval.yaml`. The executor (`CodeRunner`) is injected —
tests run real python via `LocalCodeRunner`; for untrusted generations inject a
container-backed runner (the default `LocalCodeRunner` runs on the host).

## Uploading results to a HF dataset

The `hf_dataset` sink writes the **whole run** back to a Hugging Face dataset repo
— matching what `local_dispatch` uploaded: `eval/` results **plus** agent fix
records (`logs/agent_fixes/*/attempt_N.log`, `prompt_N.txt`), captured patches
(`patches/*.patch`), lessons, and summaries. Model weights and caches are
excluded and secrets are scrubbed from every text file.

```yaml
output:
  sinks:
    - type: hf_dataset
      repo: lvkaokao/lb_local     # must exist; token needs WRITE access
      path_in_repo: results
      token_env: HF_TOKEN         # env var holding the write token (default: HF_TOKEN)
      request_filename: true      # embed the source request filename in each record
      # full_run: true            # upload the whole run (default; false = eval/ only)
      # sanitize: true            # scrub secrets from text artifacts (default)
      # status: true              # write status/<org>/<request_filename>.json (default)
      # disable_xet: true         # fall back to plain LFS if you hit a Xet 403
```

Layout written (same shape as the leaderboard consumed):

```
results/<org>/<artifact>/run_<id>/...     full run (logs, fix records, patches, eval)
results/<org>/<artifact>/results_<id>.json aggregate (status, results, request_filename)
status/<org>/<request_filename>.json       status write-back
```

**Status** is derived automatically so a forgotten write-back can't masquerade as
success: an empty result set **or any benchmark scoring exactly 0 → `Eval Failed`**;
a failing gate → `Gate Failed`; otherwise `Finished`.

The token is resolved as **explicit `token`** > **`token_env`** > `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`, and is forwarded into the
remote container automatically (see the secret allowlist). Common failure:
a `403 Forbidden` on `.../xet-write-token` means the token **reached HF but lacks
write permission** on that dataset (wrong account, read-only/fine-grained token
without `write`, or no token at all → anonymous). aqa now pre-flights the token
(`whoami`) and turns a 403 into an actionable error instead of a raw stack trace.
If the token is a valid *write* token but Xet still 403s, set `disable_xet: true`.

## Uploading the quantized model to a HF model repo

When the `quantize` provider succeeds, aqa can push the resulting checkpoint to a
Hugging Face **model** repo (a compact model card is generated automatically):

```yaml
provider:
  type: quantize
  quant:
    scheme: W4A16
  upload:
    repo: MyOrg                  # org/owner to create the repo under
    # repo_name: Base-w4a16      # optional; defaults to the run artifact name
    token_env: HF_TOKENS         # env var holding the WRITE token (default: HF_TOKEN)
    private: false
    # disable_xet: true          # fall back to plain LFS if you hit a Xet 403
    # model_card: false          # skip README.md generation
```

The repo id becomes `<repo>/<repo_name>`; `create_repo(exist_ok=True)` +
`upload_folder` run with an ignore list so eval logs, `accuracy.json`,
`request.json`, session files and the quantize script never leak into the public
model repo. Token resolution matches the dataset sink (explicit `token` >
`token_env` > standard HF env vars). A failed upload is **warned** about but never
aborts an otherwise-successful run. This is a minimal single-token / single-org
port — multi-account quota failover is intentionally out of scope for now.

## Architecture

```
aqa/
  config.py        typed RunConfig + run.yaml composer (single parse point)
  registry.py      entry-point plugin registries (7 extension points)
  pipeline.py      provide-model -> evaluate (+ baseline) -> gate -> sinks
  cli.py           run / eval / swebench / validate / list / compare
  providers/       pretrained | quantize | endpoint  (model-under-eval source)
  eval/
    scoring.py     native scorer library (adapted from NeMo Evaluator)
    dataset.py     hf://…?split= URI loading + prompt templating
    registry.py    benchmark recipe schema + loader
    runner.py      orchestrate benchmarks across harnesses/backends
    harness/       native | lm_eval  (evalscope via optional plugin)
  backends/        hf | vllm  (vllm_xpu via plugin)
  executors/       local | ssh  (intel_reserve via plugin)
  report/          sinks (local_fs | hf_dataset) + gate/compare
  quantize/        auto-round command builder + runner
  agent/           fix-loop / patch / optimize (Python orchestration)
  agent_eval/      E0-E3: client / endpoint / solver / ReAct / environment
  sandbox/         local | docker  (tool-use / terminal execution)
  terminal_bench/  E4 self-contained terminal tasks (YAML + verifier)
  swebench/        SWE-bench Verified adapter (dataset/spec/env/report/run)
recipes/           benchmarks/*.yaml, schemes/*.yaml, quant/*.json (pinned)
skills/            quantize / evaluate / fix-and-patch  (SKILL.md for agents)
```

### Extension points (add a plugin = pip package, no fork)

| entry-point group | add a new… |
|---|---|
| `aqa.providers`     | model-under-eval source (e.g. `distill`, `lora_merge`) |
| `aqa.harnesses`     | evaluation framework (e.g. `evalscope`) |
| `aqa.backends`      | inference infra (e.g. `vllm_xpu`) |
| `aqa.executors`     | machine access (e.g. `intel_reserve`) |
| `aqa.sinks`         | result destination (e.g. `s3`, `wandb`) |
| `aqa.recipe_stores` | recipe source/sink |
| `aqa.scorers`       | scoring function |

New benchmark = drop a `recipes/benchmarks/<name>.yaml` (+ a `.py` scorer only
when correct scoring needs code: math equivalence, code sandbox, judge).

## Skills

Three agent skills (knowledge layer on top of the kernel):
`skills/quantize`, `skills/evaluate`, `skills/fix-and-patch`. Skills provide the
*judgment* (scheme choice, benchmark config, root-cause/fix strategy); the `aqa/`
kernel provides the deterministic, testable execution.

## Agent fix-loop

When a phase (quantize / evaluate) fails, `aqa/agent/` drives an escalating,
self-healing retry loop (enabled by `agent.fix_loop`, on by default):

- **Tiers** (`agent.tiers`, cheapest first, e.g. `[openclaw, minimax, opus]`):
  each rung maps to an agent backend + model. The loop escalates a tier when the
  agent declares `VERDICT: UNFIXABLE`/`ESCALATE`, when it *drifts* (same
  `ERROR_CLASS` `drift_threshold` times in a row), or when a tier burns its
  `tier_attempt_budget`. Deeper quant progress (a later `layers.N`) resets drift.
- **Structured protocol**: the agent must emit `COMPONENT / ERROR_CLASS /
  ROOT_CAUSE / EVIDENCE / VERDICT / FIX_PLAN / SMOKE_TEST`; `verdict.py` parses it
  strictly (prose/placeholder lines are ignored).
- **Lessons** (`lessons.py`): each attempt's signature + resolution is recorded to
  per-phase JSONL and fed back into the next prompt.
- **Patch capture** (`agent.patch.capture`): the installed `auto_round` / `vllm`
  source and HF model custom-code are snapshotted pristine before any edit, then
  the agent's changes are captured as unified-diff patches next to the results.
- **Agent-optimize** (`agent.optimize`): before quantizing, an agent analyses the
  model (download-free: config + safetensors index names) and proposes one optimal
  `auto-round` command; it is validated (never FP-protects routed experts) and used
  in place of the deterministic recipe, or silently ignored on any miss.

Everything above is pure orchestration/state-machine in Python (unit-tested); only
the in-container command execution stays in shell.

### Backend provisioning (remote runs)

For the `ssh` executor, both fix-loop backends are installed and configured inside
the container by default (parity with the local dispatch pipeline):

- The **`local` GPU image** (`docker/gpu.dockerfile`) bakes in Node.js + the
  **OpenClaw** CLI (the default tier-0 backend).
- The **container bootstrap** (when `agent.fix_loop`/`optimize` is on) installs
  Node if the base image lacks it (e.g. the Intel XPU image), installs OpenClaw if
  missing, and adds the **Copilot** CLI when a copilot-backed tier
  (`opus`/`sonnet`/`minimax`) is configured.
- `python -m aqa.agent.provision` then lays down `~/.openclaw` + `~/.copilot`
  config from bundled templates, syncs `MINIMAX_API_KEY` into OpenClaw's auth
  profiles, and shares the project `skills/` with both backends.

Auth is env-only: export `MINIMAX_API_KEY` (openclaw / minimax BYOK) and/or
`COPILOT_GITHUB_TOKEN` (opus/sonnet) before `aqa run`; they are forwarded into the
container. A tier whose CLI or token is missing is skipped and the loop escalates.

## Agentic / terminal evaluation (endpoint-centric, E0 base)

Some benchmarks (tool-use, terminal tasks) must drive the model as a *served
endpoint* rather than via the lm_eval subprocess path.  `aqa/agent_eval/` provides
the foundation:

- `VllmEndpoint` — launch `vllm serve <model>`, wait for `/health`, tear down on exit.
- `ModelClient` — async OpenAI-compatible chat/completion client (injectable
  transport, so solvers/loop are unit-testable without a server).
- `Solver` / `ChatSolver` / `CompletionSolver` — how a task is prompted.
- `run_agentic_eval` — bounded-concurrency async solve + score + aggregate
  (`per_sample()` feeds the same statistical gate as the lm_eval path).
- `serve_and_eval(model, tasks, score_fn, ...)` — one call: serve, evaluate, done.
- `run_sharded_eval` / `split_gpu_shards` — distribute tasks round-robin across a
  pool of endpoints (multi-GPU throughput); `ray_shard_endpoints` is an optional
  Ray cluster backend (`pip install autoquant-agent[ray]`).

Install the endpoint extra with `pip install autoquant-agent[agent]` (httpx).

## Sandboxes (tool-use / terminal execution, E2)

Agentic and terminal benchmarks drive the model to *act* — run shell commands,
edit files, inspect output — inside an isolated workspace.  `aqa/sandbox/` is
that workspace, a uniform `exec` / `write_file` / `read_file` surface:

```python
from aqa.sandbox import create_sandbox

with create_sandbox("docker", image="python:3.12-slim") as sb:
    sb.write_file("solve.py", "print(6 * 7)")
    result = sb.exec("python solve.py", timeout=30)
    print(result.stdout, result.ok)   # "42\n" True
```

- `local` — `LocalSandbox`, a scratch dir on the host.  Fast and dependency-free,
  but *not* isolated, so an unsafe-command guard (`rm -rf /`, fork bombs, …) is on
  by default (`allow_unsafe=True` to bypass).
- `docker` — `DockerSandbox`, a throwaway container (true isolation; the model may
  run anything).  All Docker calls go through an injectable runner, so the logic
  is unit-tested without a daemon.

Heavier infra (ECS / Slurm / Apptainer) ships as `aqa-sandbox-*` plugins via the
`aqa.sandboxes` entry-point group — no fork of core required.  `aqa list` shows
the registered backends.

## Tool-use / ReAct agents (E3)

`aqa/agent_eval/` turns the endpoint + sandbox into an agent that *acts* on a
multi-turn loop: the model emits a shell command, the environment runs it, the
observation is fed back, and it repeats until a final answer or the turn budget
is spent.

```python
from aqa.agent_eval import (
    ModelClient, ServingProxy, ToolUseSolver, SandboxEnvironment,
    Task, run_tool_use_eval,
)
from aqa.sandbox import create_sandbox

client = ServingProxy(ModelClient(base_url="http://localhost:8000", model="q"))

def env_factory(task):
    return SandboxEnvironment(
        sandbox=create_sandbox("docker", image="python:3.12-slim"),
        files=task.metadata["files"],
        verify_command=task.metadata["verify"],   # exit 0 == solved
    )

agent = ToolUseSolver(client=client, env_factory=env_factory, max_turns=15)
result = await run_tool_use_eval("terminal", tasks, agent, concurrency=4)
print(result.mean_score())        # fraction of tasks solved
```

- `ToolUseSolver` — ReAct loop; parses ```` ```bash ```` blocks as actions and
  `Final Answer:` as completion (portable text protocol, no function-calling
  schema needed).  Produces a `Trajectory` (transcript + reward).
- `Environment` / `SandboxEnvironment` — the world the agent acts in; actions run
  as shell commands in an E2 sandbox, a verify command/callable computes reward.
- `ServingProxy` — drop-in client wrapper adding a **turn counter** (cost budget
  via `max_turns`) and a **response cache** (skips re-hitting the endpoint on
  identical deterministic requests); `.stats` for observability.
- `run_tool_use_eval` — bounded-concurrency aggregation into the same
  `AgentEvalResult` (so `per_sample` feeds the statistical gate).

## Terminal benchmarks (E4)

`aqa/terminal_bench/` are concrete, self-contained task specs — instruction, seed
files, setup commands, and a verifier — that run on the E2 sandbox through the E3
agent.  A task is authored in YAML with no pytest harness required:

```yaml
id: hello-file
instruction: "Create hello.txt containing exactly: hello world"
max_turns: 6
verify:
  files:
    - path: hello.txt
      equals: "hello world"
```

The verifier combines (AND) an optional shell `command` (must exit 0) with
declarative `files` checks (`equals` / `contains` / `exists`).  Load a suite and
score a served model:

```python
from aqa.agent_eval import ModelClient, ServingProxy
from aqa.terminal_bench import load_terminal_tasks, builtin_task_pack, run_terminal_bench

tasks = load_terminal_tasks("path/to/tasks/")     # or builtin_task_pack()
client = ServingProxy(ModelClient(base_url="http://localhost:8000", model="q"))
result = await run_terminal_bench("terminal", tasks, client, default_sandbox="docker")
print(result.mean_score())                        # solve rate
```

A small runnable example pack (`hello-file`, `python-sum`, `fix-add-bug`) ships
in `aqa/terminal_bench/tasks/` and runs on the `local` sandbox out of the box.

## SWE-bench Verified (E4, official scoring)

`aqa/swebench/` evaluates a served model on real GitHub bug-fixes with the
**official** FAIL_TO_PASS / PASS_TO_PASS resolution rule, reusing the same
sandbox + ReAct stack. Each instance runs in its released eval container
(`swebench/sweb.eval.<arch>.<id>`, repo checked out at `/testbed`); the agent
edits the source, the diff is captured (`git diff`), the gold `test_patch` is
applied, the target tests run, and the log is parsed to decide *resolved*.

```bash
# dataset = swebench_verified.parquet (needs the `swebench` extra for parquet)
aqa swebench swebench_verified.parquet \
    --base-url http://localhost:8000 --model my-quantized-model \
    --limit 20 --concurrency 4 --out results/swe
# -> writes predictions.jsonl (upstream-harness compatible) + report.json
```

```python
from aqa.swebench import load_swebench, run_swebench
insts = load_swebench("swebench_verified.parquet", limit=20)
agent_result, report = run_swebench(insts, client, out_dir="results/swe")
print(report.resolved_rate)                       # == agent_result.mean_score()
```

Repo/version-specific test commands are delegated to the upstream `swebench`
package when installed (via an injectable `spec_provider`); otherwise a generic
pytest spec runs the target node ids so the adapter still works end-to-end. The
scoring rule itself (all FAIL_TO_PASS **and** PASS_TO_PASS tests pass ⇒ resolved;
a missing test never counts as a pass) is owned by `aqa.swebench.spec` and unit
-tested without Docker. `predictions.jsonl` matches the official
(`instance_id`, `model_name_or_path`, `model_patch`) shape, so results can be
re-graded by the upstream harness independently.

## Development

```bash
pytest -q          # unit tests (config, scoring, registry, harness, gate)
ruff check aqa     # lint
```

## License

Apache-2.0. See `LICENSE` and `NOTICE` for attribution of adapted code.
