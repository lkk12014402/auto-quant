from pathlib import Path

from aqa.backends.local_completions import LocalCompletionsBackend
from aqa.eval.harness.lm_eval import LmEvalHarness
from aqa.eval.registry import BenchmarkRegistry


RULER_GENERATION_BUDGETS = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "niah_multivalue": 128,
    "ruler_vt": 30,
    "ruler_cwe": 120,
    "ruler_fwe": 50,
    "ruler_qa_squad": 32,
    "ruler_qa_hotpot": 32,
}


def test_all_ruler_tasks_have_128k_recipe_metrics():
    registry = BenchmarkRegistry()

    for name, max_gen_toks in RULER_GENERATION_BUDGETS.items():
        recipe = registry.get(name)
        assert recipe.harness == "lm_eval"
        assert recipe.type == "generative"
        assert recipe.num_fewshot == 0
        assert recipe.gen_kwargs == {"max_gen_toks": max_gen_toks}
        assert recipe.batch_size == "2"
        assert recipe.scoring.metric == "131072,none"
        assert recipe.scoring.prefer == ["131072,none"]


def test_local_completions_backend_builds_ruler_command():
    registry = BenchmarkRegistry()
    recipe = registry.get("niah_multiquery")
    backend = LocalCompletionsBackend()
    harness = LmEvalHarness(
        num_gpus=4,
        batch_size="2",
        limit=100,
        gen_max_toks=128,
        model_args={
            "tokenizer": "/models/gemma-4-26B-A4B",
            "num_concurrent": 2,
            "tokenized_requests": True,
            "add_bos_token": True,
            "max_length": 131073,
        },
        metadata={
            "max_seq_lengths": [131072],
            "tokenizer": "/models/gemma-4-26B-A4B",
        },
        apply_chat_template=False,
        fewshot_as_multiturn=False,
    )

    command = harness.build_command(
        recipe, "/models/gemma-4-26B-A4B", backend, Path("/tmp/ruler")
    )
    model_args = command[command.index("--model_args") + 1]

    assert command[command.index("--model") + 1] == "local-completions"
    assert "model=/models/gemma-4-26B-A4B" in model_args
    assert "base_url=http://127.0.0.1:8000/v1/completions" in model_args
    assert "tokenizer_backend=huggingface" in model_args
    assert "num_concurrent=2" in model_args
    assert "tokenized_requests=True" in model_args
    assert "add_bos_token=True" in model_args
    assert "max_length=131073" in model_args
    assert command[command.index("--limit") + 1] == "100"
    assert command[command.index("--gen_kwargs") + 1] == "max_gen_toks=128"
    assert command[command.index("--metadata") + 1] == (
        '{"max_seq_lengths": [131072], "tokenizer": "/models/gemma-4-26B-A4B"}'
    )
    assert "--apply_chat_template" not in command
    assert "--fewshot_as_multiturn" not in command


def test_local_completions_passes_every_ruler_recipe_generation_budget():
    registry = BenchmarkRegistry()
    backend = LocalCompletionsBackend()
    harness = LmEvalHarness(
        num_gpus=4,
        batch_size="2",
        gen_max_toks=128,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
    )

    for name, max_gen_toks in RULER_GENERATION_BUDGETS.items():
        command = harness.build_command(
            registry.get(name), "/models/gemma-4-26B-A4B", backend, Path("/tmp/ruler")
        )
        assert command[command.index("--gen_kwargs") + 1] == (
            f"max_gen_toks={max_gen_toks}"
        )


def test_ruler_commands_respect_recipe_chat_and_multiturn_flags():
    registry = BenchmarkRegistry()
    backend = LocalCompletionsBackend()
    harness = LmEvalHarness()

    raw_task = harness.build_command(
        registry.get("niah_single_1"),
        "/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        backend,
        Path("/tmp/ruler"),
    )
    chat_task = harness.build_command(
        registry.get("niah_multiquery"),
        "/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        backend,
        Path("/tmp/ruler"),
    )

    assert "--apply_chat_template" not in raw_task
    assert "--fewshot_as_multiturn" not in raw_task
    assert "--apply_chat_template" in chat_task
    assert "--fewshot_as_multiturn" not in chat_task


def test_explicit_batch_size_overrides_ruler_recipe_default():
    registry = BenchmarkRegistry()
    backend = LocalCompletionsBackend()
    recipe = registry.get("niah_multiquery")

    for batch_size in ("32", "16", "8", "4", "2", "1"):
        harness = LmEvalHarness(
            num_gpus=8,
            batch_size=batch_size,
            limit=100,
            gen_max_toks=128,
            model_args={"tokenized_requests": True},
            metadata={"max_seq_lengths": [131072]},
        )
        command = harness.build_command(
            recipe, "/models/gemma-4-31B", backend, Path("/tmp/ruler")
        )
        assert command[command.index("--batch_size") + 1] == batch_size


def test_auto_batch_size_keeps_ruler_recipe_default():
    registry = BenchmarkRegistry()
    backend = LocalCompletionsBackend()
    harness = LmEvalHarness(
        num_gpus=8,
        batch_size="auto",
        limit=100,
        gen_max_toks=128,
        model_args={"tokenized_requests": True},
        metadata={"max_seq_lengths": [131072]},
    )
    command = harness.build_command(
        registry.get("niah_multiquery"),
        "/models/gemma-4-31B",
        backend,
        Path("/tmp/ruler"),
    )
    assert command[command.index("--batch_size") + 1] == "2"
