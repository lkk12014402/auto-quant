"""Image-metrics (clip / clip_iqa / imagereward) scorer tests -- no GPU/weights.

The heavy scorers (transformers CLIP, the isolated ImageReward subprocess) and
the vllm-omni API are faked; only the dispatch, result shaping, and env
provisioning logic are exercised for real.
"""

import json
import sys
import types
from pathlib import Path

import pytest

from aqa.eval import image_metrics as im
from aqa.eval.harness.diffusion import DiffusionHarness
from aqa.eval.registry import BenchmarkRecipe, BenchmarkRegistry
from aqa.logging_util import AqaError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


# ── recipe / config ──────────────────────────────────────────────────────────
def test_recipe_loads_drawbench_metrics():
    recipe = BenchmarkRegistry().get("drawbench_metrics")
    assert recipe.harness == "diffusion"
    assert recipe.scoring.scorer == "image_metrics"
    spec = recipe.scoring.image_metrics
    assert spec.metrics == ["clip", "clip_iqa", "imagereward"]
    assert spec.clip_model == "openai/clip-vit-base-patch32"
    assert spec.imagereward_model == "ImageReward-v1.0"


def test_image_metric_spec_defaults():
    from aqa.eval.registry import ImageMetricSpec

    spec = ImageMetricSpec()
    assert spec.metrics == ["clip"]


def test_unknown_metric_rejected():
    with pytest.raises(AqaError, match="unknown image metric"):
        im.score_rows([], ["clip", "bogus"], clip_model="m")


# ── score_rows dispatch (fake scorer fns) ────────────────────────────────────
def test_score_rows_dispatches_each_metric(monkeypatch):
    rows = [{"prompt": "p1", "image_path": "/x/0.png"}, {"prompt": "p2", "image_path": "/x/1.png"}]
    seen = {}
    def _fake_clip(r, m, d):
        seen["clip"] = (r, m, d)
        return [1.0, 2.0]

    monkeypatch.setattr(im, "clip_score", _fake_clip)
    monkeypatch.setattr(im, "clip_iqa", lambda r, m, d: [70.0, 80.0])
    monkeypatch.setattr(im, "imagereward_score", lambda r, m, d: [0.5, -0.5])

    out = im.score_rows(rows, ["clip", "clip_iqa", "imagereward"],
                        clip_model="clip-x", imagereward_model="ir-y", device="cpu")
    assert set(out) == {"clip", "clip_iqa", "imagereward"}
    assert out["clip"] == [1.0, 2.0]
    assert seen["clip"][1] == "clip-x"
    assert out["imagereward"] == [0.5, -0.5]


# ── harness end-to-end: one recipe -> three results ─────────────────────────
class _FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake-png")


class _FakeOmni:
    instances: list = []

    def __init__(self, model, mode, **kwargs):
        self.model, self.kwargs = model, {"mode": mode, **kwargs}
        self.closed = False
        _FakeOmni.instances.append(self)

    def generate(self, request, sampling_params_list):
        class _Out:
            images = [_FakeImage()]

        return [_Out()]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_vllm_omni(monkeypatch, tmp_path):
    _FakeOmni.instances = []
    _FakeOmni._tmp_images = str(tmp_path / "gen")
    entry = types.ModuleType("vllm_omni.entrypoints.omni")
    entry.Omni = _FakeOmni
    data = types.ModuleType("vllm_omni.inputs.data")

    class OmniDiffusionSamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    data.OmniDiffusionSamplingParams = OmniDiffusionSamplingParams
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.omni", entry)
    monkeypatch.setitem(sys.modules, "vllm_omni.inputs.data", data)
    return _FakeOmni


def _metrics_recipe(prompts_file):
    return BenchmarkRecipe(
        name="drawbench_metrics", harness="diffusion", type="generative",
        dataset=str(prompts_file),
        scoring={
            "scorer": "image_metrics",
            "image_metrics": {"metrics": ["clip", "clip_iqa", "imagereward"]},
        },
    )


def test_image_metrics_recipe_emits_one_result_per_metric(
        fake_vllm_omni, tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"Prompts": "a cat"}) + "\n")

    monkeypatch.setattr(im, "clip_score", lambda r, m, d: [30.0])
    monkeypatch.setattr(im, "clip_iqa", lambda r, m, d: [77.7])
    monkeypatch.setattr(im, "imagereward_score", lambda r, m, d: [1.01])

    harness = DiffusionHarness(limit=1)
    results = harness.run(_metrics_recipe(prompts), "org/flux", None, tmp_path / "out")
    assert isinstance(results, list)
    by_name = {r.name: r for r in results}
    assert set(by_name) == {
        "drawbench_metrics.clip", "drawbench_metrics.clip_iqa",
        "drawbench_metrics.imagereward",
    }
    assert by_name["drawbench_metrics.clip"].score == 30.0
    assert by_name["drawbench_metrics.clip"].metric == "clip_score"
    assert by_name["drawbench_metrics.clip_iqa"].score == 77.7
    assert by_name["drawbench_metrics.imagereward"].score == 1.01
    assert by_name["drawbench_metrics.imagereward"].raw["scorer_model"] == "ImageReward-v1.0"
    # one generation, shared by all three scorers
    assert len(_FakeOmni.instances) == 1


def test_image_metrics_group_flattens_results(fake_vllm_omni, tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"Prompts": "a cat"}) + "\n")
    monkeypatch.setattr(im, "clip_score", lambda r, m, d: [30.0])
    monkeypatch.setattr(im, "imagereward_score", lambda r, m, d: [1.0])

    recipe = BenchmarkRecipe(
        name="m2", harness="diffusion", type="generative", dataset=str(prompts),
        scoring={"scorer": "image_metrics",
                 "image_metrics": {"metrics": ["clip", "imagereward"]}},
    )
    harness = DiffusionHarness(limit=1)
    results = harness.run_group([recipe], "org/flux", None, tmp_path / "grp")
    assert sorted(r.name for r in results) == ["m2.clip", "m2.imagereward"]


# ── isolated env provisioning ────────────────────────────────────────────────
def test_ensure_imagereward_env_provisions_with_pins(tmp_path, monkeypatch):
    env_dir = tmp_path / "irenv"
    monkeypatch.setattr(im, "IMAGEREWARD_ENV_DIR", env_dir)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        (env_dir / "ImageReward").mkdir(parents=True, exist_ok=True)

        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(im.subprocess, "run", fake_run)
    out = im.ensure_imagereward_env()
    assert out == env_dir
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "pip", "install"]
    for pin in ("transformers==4.44.2", "diffusers==0.29.2", "image-reward"):
        assert pin in cmd
    assert any("github.com/openai/CLIP" in c for c in cmd)


def test_ensure_imagereward_env_idempotent(tmp_path, monkeypatch):
    env_dir = tmp_path / "irenv"
    (env_dir / "ImageReward").mkdir(parents=True)
    monkeypatch.setattr(im, "IMAGEREWARD_ENV_DIR", env_dir)
    monkeypatch.setattr(
        im.subprocess, "run",
        lambda *a, **k: pytest.fail("existing env must not reinstall"),
    )
    assert im.ensure_imagereward_env() == env_dir


def test_imagereward_score_runs_worker_subprocess(tmp_path, monkeypatch):
    env_dir = tmp_path / "irenv"
    (env_dir / "ImageReward").mkdir(parents=True)
    monkeypatch.setattr(im, "IMAGEREWARD_ENV_DIR", env_dir)
    seen = {}

    def fake_run(cmd, capture_output, text, env, check):
        seen["env"] = env
        payload = json.loads(cmd[-1])

        class R:
            returncode = 0
            stdout = json.dumps([0.25] * len(payload["rows"]))
            stderr = ""
        return R()

    monkeypatch.setattr(im.subprocess, "run", fake_run)
    rows = [{"prompt": "p", "image_path": "/x.png"}]
    assert im.imagereward_score(rows, device="cpu") == [0.25]
    assert str(env_dir) in seen["env"]["PYTHONPATH"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""  # cpu scorer forces GPU off


def test_imagereward_score_propagates_worker_failure(tmp_path, monkeypatch):
    env_dir = tmp_path / "irenv"
    (env_dir / "ImageReward").mkdir(parents=True)
    monkeypatch.setattr(im, "IMAGEREWARD_ENV_DIR", env_dir)

    class R:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(im.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(AqaError, match="boom"):
        im.imagereward_score([{"prompt": "p", "image_path": "/x.png"}])
