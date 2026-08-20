from pathlib import Path

from aqa.backends.hf import HfBackend
from aqa.backends.vllm import VllmBackend
from aqa.eval.harness.lm_eval import LmEvalHarness, _extract_metric
from aqa.eval.harness.native import NativeHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry


def test_registry_loads_recipes():
    reg = BenchmarkRegistry()
    names = reg.names()
    assert {"piqa", "hellaswag", "mmlu", "gsm8k"} <= set(names)
    gsm8k = reg.get("gsm8k")
    assert gsm8k.type == "generative"
    assert gsm8k.apply_chat_template is True


def test_registry_unknown_falls_back_to_lm_eval():
    reg = BenchmarkRegistry()
    r = reg.get("some_unregistered_task")
    assert r.harness == "lm_eval"


def test_lm_eval_command_loglikelihood():
    recipe = BenchmarkRecipe(name="piqa", harness="lm_eval", type="loglikelihood")
    cmd = LmEvalHarness(num_gpus=2).build_command(recipe, "org/m", HfBackend(), Path("/tmp/o"))
    assert "--model" in cmd and "hf" in cmd
    assert "--apply_chat_template" not in cmd
    assert "pretrained=org/m,dtype=bfloat16,trust_remote_code=True,parallelize=True" in cmd


def test_lm_eval_command_generative_vllm():
    recipe = BenchmarkRecipe(name="gsm8k", harness="lm_eval", type="generative")
    cmd = LmEvalHarness(num_gpus=4).build_command(recipe, "org/m", VllmBackend(), Path("/tmp/o"))
    assert "--apply_chat_template" in cmd and "--fewshot_as_multiturn" in cmd
    # vllm carries max_gen_toks inside model_args, so no --gen_kwargs
    assert "--gen_kwargs" not in cmd
    joined = " ".join(cmd)
    assert "tensor_parallel_size=4" in joined and "max_gen_toks=2048" in joined


def test_extract_metric_prefers_recipe_metric():
    recipe = BenchmarkRecipe(name="gsm8k", scoring={"metric": "exact_match,strict-match"})
    data = {"exact_match,strict-match": 0.73, "exact_match,flexible-extract": 0.70}
    score, key = _extract_metric(data, recipe)
    assert score == 0.73 and key == "exact_match,strict-match"


def test_native_evaluate_samples_pure():
    recipe = BenchmarkRecipe(
        name="toy", harness="native", type="generative",
        dataset="hf://x?split=test", prompt="Q: {q}\nA:", target_field="a",
        scoring={"scorer": "numeric_match"},
    )
    samples = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}]
    # fake generate_fn: echoes the correct answer for the first, wrong for second
    outputs = iter(["the answer is 4", "the answer is 5"])
    result = NativeHarness().evaluate_samples(recipe, samples, lambda p: next(outputs))
    assert result.num_samples == 2
    assert result.score == 0.5
    assert result.metric == "numeric_match"


def test_lm_eval_model_args_extra_and_seed():
    from aqa.eval.harness.lm_eval import LmEvalHarness
    recipe = BenchmarkRecipe(name="piqa", harness="lm_eval", type="loglikelihood")
    h = LmEvalHarness(model_args={"enable_thinking": False, "reasoning_parser": "qwen3"},
                      batch_size="64", seed=7)
    cmd = h.build_command(recipe, "org/m", VllmBackend(), Path("/tmp/o"))
    joined = " ".join(cmd)
    assert "enable_thinking=False" in joined  # python-bool rendering for lm_eval
    assert "reasoning_parser=qwen3" in joined
    assert "--seed 7" in joined
    assert "--batch_size 64" in joined


def test_lm_eval_group_command_merges_tasks():
    from aqa.eval.harness.lm_eval import LmEvalHarness
    h = LmEvalHarness()
    recipes = [BenchmarkRecipe(name=n, type="loglikelihood") for n in ("piqa", "hellaswag", "mmlu")]
    cmd = h.build_command(recipes, "org/m", VllmBackend(), Path("/tmp/o"))
    i = cmd.index("--tasks")
    assert cmd[i + 1] == "piqa,hellaswag,mmlu"
    assert "--apply_chat_template" not in cmd


def test_lm_eval_group_key_separates_generative():
    from aqa.eval.harness.lm_eval import LmEvalHarness
    h = LmEvalHarness()
    ll = BenchmarkRecipe(name="piqa", type="loglikelihood")
    gen = BenchmarkRecipe(name="gsm8k", type="generative", apply_chat_template=True)
    assert h.group_key(ll) != h.group_key(gen)
    # loglikelihood tasks share a key
    assert h.group_key(ll) == h.group_key(BenchmarkRecipe(name="mmlu", type="loglikelihood"))


class _RecordingHarness:
    name = "lm_eval"

    def __init__(self):
        self.groups = []
        self.singles = []

    def available(self):
        return True

    def group_key(self, recipe):
        gen = recipe.type == "generative" or recipe.apply_chat_template
        return (gen,)

    def run_group(self, recipes, model, backend, out_dir):
        from aqa.eval.registry import BenchmarkResult
        self.groups.append([r.name for r in recipes])
        return [BenchmarkResult(name=r.name, score=0.5, metric="acc") for r in recipes]

    def run(self, recipe, model, backend, out_dir):
        from aqa.eval.registry import BenchmarkResult
        self.singles.append(recipe.name)
        return BenchmarkResult(name=recipe.name, score=0.5, metric="acc")


def test_runner_groups_loglikelihood_and_isolates_gsm8k(monkeypatch, tmp_path):
    from aqa.config import EvalConfig
    from aqa.eval import runner
    rec = _RecordingHarness()
    monkeypatch.setattr(runner, "_make_harness", lambda name, cfg, num_gpus: rec)

    class _FakeBackend:
        name = "vllm"
    monkeypatch.setattr(runner.BACKENDS, "create", lambda name: _FakeBackend())

    cfg = EvalConfig(benchmarks=["piqa", "hellaswag", "mmlu", "gsm8k"], backend="vllm")
    res = runner.run_evaluation("org/m", cfg, tmp_path, num_gpus=1)
    # piqa+hellaswag+mmlu in one group; gsm8k is generative -> its own group of one
    assert ["piqa", "hellaswag", "mmlu"] in rec.groups
    assert rec.singles == ["gsm8k"]  # single-member generative group -> run()
    assert set(res.benchmarks) == {"piqa", "hellaswag", "mmlu", "gsm8k"}


def test_runner_no_grouping_when_disabled(monkeypatch, tmp_path):
    from aqa.config import EvalConfig, LmEvalOptions
    from aqa.eval import runner
    rec = _RecordingHarness()
    monkeypatch.setattr(runner, "_make_harness", lambda name, cfg, num_gpus: rec)

    class _FakeBackend:
        name = "vllm"
    monkeypatch.setattr(runner.BACKENDS, "create", lambda name: _FakeBackend())

    cfg = EvalConfig(benchmarks=["piqa", "hellaswag"], backend="vllm",
                     lm_eval=LmEvalOptions(group=False))
    runner.run_evaluation("org/m", cfg, tmp_path, num_gpus=1)
    assert rec.singles == ["piqa", "hellaswag"] and rec.groups == []


def test_xpu_dockerfile_selected_for_xpu_local():
    from aqa.config import parse_run_config
    from aqa.executors.ssh import _dockerfile_bytes
    data = {
        "model": "org/m",
        "eval": {"benchmarks": ["piqa"], "backend": "vllm"},
        "executor": {"type": "ssh", "image": {"dockerfile": "local"},
                     "ssh": {"host": "h", "accelerator": "xpu"}},
    }
    cfg = parse_run_config(data)
    df = _dockerfile_bytes(cfg).decode()
    assert "vllm-openai-xpu" in df
    assert "openclaw" in df


def test_gpu_dockerfile_selected_for_cuda_local():
    from aqa.config import parse_run_config
    from aqa.executors.ssh import _dockerfile_bytes
    data = {
        "model": "org/m",
        "eval": {"benchmarks": ["piqa"], "backend": "vllm"},
        "executor": {"type": "ssh", "image": {"dockerfile": "local"}, "ssh": {"host": "h"}},
    }
    cfg = parse_run_config(data)
    df = _dockerfile_bytes(cfg).decode()
    assert "nvcr.io/nvidia/pytorch" in df


def test_recipe_batch_size_overrides_global():
    from aqa.eval.harness.lm_eval import LmEvalHarness
    recipe = BenchmarkRecipe(name="gsm8k", type="generative", batch_size="64")
    h = LmEvalHarness(batch_size="auto")
    cmd = h.build_command(recipe, "org/m", VllmBackend(), Path("/tmp/o"))
    i = cmd.index("--batch_size")
    assert cmd[i + 1] == "64"  # per-recipe wins over the global "auto"


def test_recipe_batch_size_falls_back_to_global():
    from aqa.eval.harness.lm_eval import LmEvalHarness
    recipe = BenchmarkRecipe(name="piqa", type="loglikelihood")  # no batch_size
    h = LmEvalHarness(batch_size="32")
    cmd = h.build_command(recipe, "org/m", VllmBackend(), Path("/tmp/o"))
    i = cmd.index("--batch_size")
    assert cmd[i + 1] == "32"


def test_group_key_splits_on_batch_size():
    from aqa.eval.harness.lm_eval import LmEvalHarness
    h = LmEvalHarness(batch_size="auto")
    a = BenchmarkRecipe(name="piqa", type="loglikelihood")
    b = BenchmarkRecipe(name="mmlu", type="loglikelihood", batch_size="64")
    # different batch sizes must not be merged into one lm_eval call
    assert h.group_key(a) != h.group_key(b)


def test_registry_gsm8k_recipe_carries_all_generative_params():
    reg = BenchmarkRegistry()
    g = reg.get("gsm8k")
    assert g.apply_chat_template and g.fewshot_as_multiturn
    assert g.num_fewshot == 5
    assert g.gen_kwargs == {"max_gen_toks": 2048}
    assert g.batch_size == "64"
