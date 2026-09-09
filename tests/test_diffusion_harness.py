"""Diffusion harness tests -- no GPU / network / vllm-omni required.

The vllm-omni API surface (``Omni`` + ``OmniDiffusionSamplingParams``) is faked
via ``sys.modules`` so the real lazy-import paths in the harness are exercised;
the CLIP scorer is monkeypatched out (it needs torch/transformers weights).
"""

import dataclasses
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from aqa.config import parse_run_config
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.logging_util import AqaError
from aqa.registry import BACKENDS, HARNESSES

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


# ── registration / config ────────────────────────────────────────────────────
def test_recipe_loads_drawbench_clip():
    recipe = BenchmarkRegistry().get("drawbench_clip")
    assert recipe.harness == "diffusion"
    assert recipe.type == "generative"
    assert recipe.dataset == "sayakpaul/drawbench"
    assert recipe.prompt_field == "Prompts"


def test_registries_create_diffusion_plugins():
    harness = HARNESSES.create("diffusion")
    assert harness.name == "diffusion"
    backend = BACKENDS.create("vllm_omni")
    assert backend.name == "vllm_omni"
    assert backend.build_model_args("org/m") == ""
    assert backend.env_overrides() == {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}


def test_diffusion_options_defaults_in_run_config():
    cfg = parse_run_config({"model": "org/model"})
    opts = cfg.eval.diffusion
    assert opts.num_inference_steps == 28
    assert opts.guidance_scale == 3.5
    assert (opts.height, opts.width) == (1024, 1024)
    assert opts.scorer_model == "openai/clip-vit-base-patch32"
    assert cfg.eval.backend == "hf"  # existing defaults untouched


def test_example_run_diffusion_flux1_parses():
    from aqa.config import load_run_config

    cfg = load_run_config(EXAMPLES / "run_diffusion_flux1.yaml")
    assert cfg.quant.scheme == "W4A16"
    assert cfg.quant.export_format == "auto_round"
    assert cfg.quant.iters == 0
    assert "--disable_opt_rtn" in cfg.quant.extra_args
    assert cfg.eval.backend == "vllm_omni"
    assert cfg.eval.harness == "diffusion"
    assert cfg.eval.benchmarks == ["drawbench_clip"]
    assert cfg.eval.baseline is True
    assert cfg.eval.diffusion.num_inference_steps == 28
    assert cfg.gate.max_accuracy_drop == 2.0
    assert cfg.agent.optimize is False


def test_container_packages_vllm_omni():
    from aqa.executors._remote import container_packages

    cfg = parse_run_config({"model": "org/m", "eval": {"backend": "vllm_omni"}})
    pkgs = container_packages(cfg)
    assert "vllm-omni" in pkgs
    # vllm-omni imports vllm but doesn't pin it; without it the harness import fails
    assert "vllm" in pkgs
    assert {"diffusers", "datasets", "transformers", "pillow"} <= set(pkgs)


def test_available_false_without_vllm_omni():
    assert importlib.util.find_spec("vllm_omni") is None  # test env must not have it
    assert DiffusionHarness().available() is False


# ── fakes for the vllm-omni API surface ──────────────────────────────────────
@dataclasses.dataclass
class _FakeSamplingParams:
    height: int | None = None
    width: int | None = None
    seed: int | None = None
    guidance_scale: float = 0.0
    num_inference_steps: int = 0
    num_outputs_per_prompt: int = 1


class _FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake-png")


class _FakeOutput:
    images: ClassVar[list] = [_FakeImage()]


class _FakeOmni:
    instances: ClassVar[list] = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.default_sampling_params_list = None
        self.generate_calls = []
        self.closed = False
        _FakeOmni.instances.append(self)

    def generate(self, prompt, sampling_params_list=None):
        self.generate_calls.append((prompt, sampling_params_list))
        return [_FakeOutput()]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_vllm_omni(monkeypatch):
    _FakeOmni.instances = []
    pkg = types.ModuleType("vllm_omni")
    pkg.__spec__ = importlib.machinery.ModuleSpec("vllm_omni", loader=None)
    omni_mod = types.ModuleType("vllm_omni.entrypoints.omni")
    omni_mod.Omni = _FakeOmni
    data_mod = types.ModuleType("vllm_omni.inputs.data")
    data_mod.OmniDiffusionSamplingParams = _FakeSamplingParams
    for name, mod in {
        "vllm_omni": pkg,
        "vllm_omni.entrypoints": types.ModuleType("vllm_omni.entrypoints"),
        "vllm_omni.entrypoints.omni": omni_mod,
        "vllm_omni.inputs": types.ModuleType("vllm_omni.inputs"),
        "vllm_omni.inputs.data": data_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return _FakeOmni


@pytest.fixture
def prompts_file(tmp_path):
    rows = [{"Prompts": "a cat on a mat"}, {"Prompts": "a red cube"}, {"Prompts": "ignore"}]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _recipe(prompts_file, name="drawbench_clip"):
    return BenchmarkRecipe(
        name=name, harness="diffusion", type="generative", dataset=str(prompts_file)
    )


def _fake_scores(self, rows):
    return [30.0 + int(row["idx"]) for row in rows]


def test_run_end_to_end_with_fake_omni(fake_vllm_omni, prompts_file, tmp_path, monkeypatch):
    monkeypatch.setattr(DiffusionHarness, "_score_images", _fake_scores)
    harness = DiffusionHarness(num_inference_steps=28, limit=2)
    out_dir = tmp_path / "out"
    result = harness.run(_recipe(prompts_file), "org/flux", BACKENDS.create("vllm_omni"), out_dir)

    omni = _FakeOmni.instances[0]
    assert omni.model == "org/flux"
    assert omni.kwargs == {"mode": "text-to-image"}
    assert omni.closed is True  # released even on the success path
    # per-prompt generation with the harness's sampling params
    assert len(omni.generate_calls) == 2  # limit=2 honored
    request, params_list = omni.generate_calls[0]
    assert request == {"prompt": "a cat on a mat", "modalities": ["image"]}
    params = params_list[0]
    assert (params.height, params.width) == (1024, 1024)
    assert params.num_inference_steps == 28 and params.guidance_scale == 3.5
    assert params.seed == 42  # seed + idx
    # images + metadata on disk
    assert (out_dir / "images" / "0000.png").exists()
    assert (out_dir / "images" / "0001.png").exists()
    meta = (out_dir / "images" / "metadata.csv").read_text()
    assert meta.splitlines()[0] == "idx,prompt,image_path,seed"
    assert "a red cube" in meta
    # normalised result
    assert result.metric == "clip_score"
    assert result.num_samples == 2
    assert result.per_sample == [30.0, 31.0]
    assert result.score == 30.5
    assert result.raw["metadata"].endswith("metadata.csv")
    assert "vllm_omni" in result.command and "clip_score" in result.command


def test_run_resumes_from_existing_metadata(fake_vllm_omni, prompts_file, tmp_path, monkeypatch):
    monkeypatch.setattr(DiffusionHarness, "_score_images", _fake_scores)
    harness = DiffusionHarness()
    recipe = _recipe(prompts_file)
    out_dir = tmp_path / "out"
    harness.run(recipe, "org/flux", None, out_dir)
    first = _FakeOmni.instances[0]
    assert len(first.generate_calls) == 3

    # Second run over the same out_dir: every idx already in metadata.csv.
    result = harness.run(recipe, "org/flux", None, out_dir)
    second = _FakeOmni.instances[1]
    assert second.generate_calls == []
    assert result.num_samples == 3


def test_run_group_shares_one_omni_load(fake_vllm_omni, prompts_file, tmp_path, monkeypatch):
    monkeypatch.setattr(DiffusionHarness, "_score_images", _fake_scores)
    harness = DiffusionHarness(limit=1)
    recipes = [_recipe(prompts_file, "bench_a"), _recipe(prompts_file, "bench_b")]
    assert harness.group_key(recipes[0]) == harness.group_key(recipes[1])
    results = harness.run_group(recipes, "org/flux", None, tmp_path / "grp")
    assert len(_FakeOmni.instances) == 1  # one DiT load for the whole group
    assert [r.name for r in results] == ["bench_a", "bench_b"]
    for r in results:
        assert r.num_samples == 1
    assert (tmp_path / "grp" / "bench_a" / "images" / "0000.png").exists()
    assert (tmp_path / "grp" / "bench_b" / "images" / "0000.png").exists()


def test_missing_dataset_raises_aqa_error():
    harness = DiffusionHarness()
    recipe = BenchmarkRecipe(name="bad", harness="diffusion", type="generative")
    with pytest.raises(AqaError):
        harness.run(recipe, "org/flux", None, Path("/tmp/unused"))


def test_omni_kwargs_forwarded_to_omni(fake_vllm_omni, prompts_file, tmp_path, monkeypatch):
    """Per-model engine knobs (omni_kwargs) reach the Omni constructor verbatim."""
    monkeypatch.setattr(DiffusionHarness, "_score_images", _fake_scores)
    harness = DiffusionHarness(omni_kwargs={"enable_cpu_offload": True}, limit=1)
    result = harness.run(_recipe(prompts_file), "org/flux", None, tmp_path / "out")
    omni = _FakeOmni.instances[0]
    assert omni.kwargs == {"mode": "text-to-image", "enable_cpu_offload": True}
    assert "enable_cpu_offload" in result.command  # reproducibility string


def test_new_diffusion_examples_parse():
    from aqa.config import load_run_config

    zimage = load_run_config(EXAMPLES / "run_diffusion_zimage.yaml")
    assert zimage.model == "Tongyi-MAI/Z-Image-Turbo"
    assert zimage.quant.scheme == "W4A16" and zimage.quant.iters == 0
    assert zimage.eval.diffusion.num_inference_steps == 8
    assert zimage.eval.diffusion.guidance_scale == 0.0  # distilled: no CFG
    assert zimage.eval.backend == "vllm_omni"

    klein4b = load_run_config(EXAMPLES / "run_diffusion_flux2_klein4b.yaml")
    assert klein4b.quant.scheme == "MXFP8"
    assert klein4b.quant.export_format == "auto_round"
    assert klein4b.eval.diffusion.guidance_scale == 0.0

    klein9b = load_run_config(EXAMPLES / "run_diffusion_flux2_klein9b.yaml")
    assert klein9b.model == "black-forest-labs/FLUX.2-klein-9B"
    assert klein9b.quant.scheme == "W4A16"

    dev = load_run_config(EXAMPLES / "run_diffusion_flux2_dev.yaml")
    assert dev.eval.diffusion.num_inference_steps == 28
    assert dev.eval.diffusion.guidance_scale == 3.5
    assert dev.eval.diffusion.omni_kwargs == {}

    ocr = load_run_config(EXAMPLES / "run_quant_deepseek_ocr2.yaml")
    assert ocr.model == "deepseek-ai/DeepSeek-OCR-2"
    assert ocr.eval.backend == "hf"
    assert ocr.eval.benchmarks == ["piqa"]
    assert ocr.eval.limit == 16
    assert ocr.eval.lm_eval.model_args["trust_remote_code"] == "True"


# ── GenEval: multi-image layout + scoring dispatch ───────────────────────────
def _geneval_recipe(prompts_file, name="geneval", images_per_prompt=4):
    return BenchmarkRecipe(
        name=name, harness="diffusion", type="generative",
        dataset=str(prompts_file), prompt_field="Prompts",
        images_per_prompt=images_per_prompt,
        scoring={"scorer": "geneval"},
    )


def test_images_per_prompt_official_layout_and_resume(
    fake_vllm_omni, prompts_file, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        DiffusionHarness, "_score_images",
        lambda self, rows: [float(len(rows))] * len(rows),
    )
    recipe = BenchmarkRecipe(
        name="multi", harness="diffusion", type="generative",
        dataset=str(prompts_file), images_per_prompt=2,
    )
    harness = DiffusionHarness(limit=2)
    out_dir = tmp_path / "out"
    harness.run(recipe, "org/flux", None, out_dir)

    omni = _FakeOmni.instances[0]
    assert len(omni.generate_calls) == 4  # 2 prompts x 2 samples
    # official per-prompt layout: <idx>/metadata.jsonl + <idx>/samples/<j>.png
    for idx in range(2):
        prompt_dir = out_dir / "images" / f"{idx:04d}"
        meta = json.loads((prompt_dir / "metadata.jsonl").read_text())
        assert meta["Prompts"]  # the full dataset row is recorded
        assert (prompt_dir / "samples" / "0.png").exists()
        assert (prompt_dir / "samples" / "1.png").exists()
    # metadata.csv extended with the sample column
    lines = (out_dir / "images" / "metadata.csv").read_text().splitlines()
    assert lines[0] == "idx,sample,prompt,image_path,seed"
    assert len(lines) == 1 + 4
    # seeds follow seed + idx*1000 + j
    seeds = {str(params[0].seed) for _req, params in omni.generate_calls}
    assert seeds == {"42", "43", "1042", "1043"}

    # resume: everything in metadata.csv -> no new generation
    result = harness.run(recipe, "org/flux", None, out_dir)
    assert _FakeOmni.instances[1].generate_calls == []
    assert result.num_samples == 4


def test_group_key_separates_geneval_from_clip(prompts_file):
    harness = DiffusionHarness()
    clip_a = _recipe(prompts_file, "a")
    clip_b = _recipe(prompts_file, "b")
    assert harness.group_key(clip_a) == harness.group_key(clip_b)
    # scorer identity and images_per_prompt both split groups
    assert harness.group_key(_geneval_recipe(prompts_file)) != harness.group_key(clip_a)
    multi = BenchmarkRecipe(
        name="m", harness="diffusion", type="generative",
        dataset=str(prompts_file), images_per_prompt=2,
    )
    assert harness.group_key(multi) != harness.group_key(clip_a)


def test_geneval_scoring_dispatch(fake_vllm_omni, prompts_file, tmp_path, monkeypatch):
    from aqa.eval.geneval import GenevalScorer

    fake_rows = [
        {"filename": "0.png", "tag": "single_object", "prompt": "a", "correct": True,
         "reason": "", "metadata": "{}", "details": "{}"},
        {"filename": "1.png", "tag": "single_object", "prompt": "a", "correct": False,
         "reason": "missing", "metadata": "{}", "details": "{}"},
        {"filename": "2.png", "tag": "colors", "prompt": "b", "correct": True,
         "reason": "", "metadata": "{}", "details": "{}"},
    ]
    monkeypatch.setattr(GenevalScorer, "evaluate_directory",
                        lambda self, images_dir: fake_rows)
    recipe = _geneval_recipe(prompts_file, images_per_prompt=1)
    result = DiffusionHarness(limit=1).run(recipe, "org/flux", None, tmp_path / "out")

    assert result.metric == "geneval"
    assert result.num_samples == 3
    assert result.per_sample == [1.0, 0.0, 1.0]
    # overall = unweighted mean of per-task means: (0.5 + 1.0) / 2
    assert result.score == pytest.approx(0.75)
    assert result.raw["task_means"] == {"single_object": 0.5, "colors": 1.0}
    assert result.raw["results"].endswith("results.jsonl")
    assert result.raw["detector_model"]  # scorer config surfaced
    assert "geneval(" in result.command
    # results.jsonl keeps the official row shape for diffing against upstream
    written = (tmp_path / "out" / "images" / "results.jsonl").read_text().splitlines()
    assert len(written) == 3
    assert json.loads(written[0])["filename"] == "0.png"
