"""Diffusion text-to-image harness: vllm-omni generation + CLIP score.

Reads a prompt set from the recipe's ``dataset`` (a HF datasets uri such as
``sayakpaul/drawbench``, or a local ``.jsonl``/``.csv`` with a prompt column),
generates one image per prompt through vllm-omni's ``Omni`` API, then scores
each image against its prompt with a CLIP model (image-text cosine x 100, i.e.
CLIP's own ``logits_per_image`` scale -- higher is better, gate-compatible).

Generation and scoring are deliberately two phases: the ``Omni`` instance (the
DiT is expensive to load and holds the GPU) is released before the CLIP scorer
loads.  Benchmarks sharing generation parameters are grouped by the runner and
served by a single ``Omni`` load via :meth:`DiffusionHarness.run_group`.

Heavy dependencies (``vllm_omni``, ``torch``, ``transformers``, ``datasets``,
``PIL``) are imported lazily inside methods so the module stays importable on
machines without a GPU stack.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

from ...logging_util import AqaError, get_logger
from ..registry import BenchmarkRecipe, BenchmarkResult

_log = get_logger("harness.diffusion")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_env_scored(
    name: str, scorer_desc: str, *, num_scored: int, num_env_failed: int
) -> None:
    """Fail the run rather than report a metric produced only by a missing scorer.

    When every image's scoring failed because the judge/OCR/CLIP model is
    unavailable, the resulting ``0.0`` is an environment artifact, not a
    measurement -- reporting it would silently corrupt a baseline-vs-quantized
    comparison.
    """
    if num_scored == 0 and num_env_failed > 0:
        raise AqaError(
            f"{name}: every image failed to score because {scorer_desc} is "
            f"unavailable ({num_env_failed} image(s)); refusing to report a 0.0 "
            "that is an environment artifact. Install/enable the scorer or drop "
            "the benchmark."
        )


def _incomplete_raw(summary: dict) -> dict:
    """``num_missing``/``num_env_failed``/``complete`` block for a result's raw."""
    return {
        "complete": summary.get("complete", True),
        "num_missing": summary.get("num_missing", 0),
        "num_env_failed": summary.get("num_env_failed", 0),
    }


class DiffusionHarness:
    name = "diffusion"

    def __init__(
        self,
        *,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
        seed: int = 42,
        scorer_model: str = "openai/clip-vit-base-patch32",
        scorer_device: str = "auto",
        prompt_field: str = "Prompts",
        dataset_split: str = "train",
        omni_kwargs: dict[str, Any] | None = None,
        ar_extra_body: dict[str, Any] | None = None,
        limit: int | None = None,
    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width
        self.seed = seed
        self.scorer_model = scorer_model
        self.scorer_device = scorer_device
        self.prompt_field = prompt_field
        self.dataset_split = dataset_split
        self.omni_kwargs = dict(omni_kwargs or {})
        # Model-declared extra body for the AR-stage input builder (e.g.
        # ``{"bot_task": ..., "use_system_prompt": ...}``). Empty means "use the
        # model's own defaults", which is what the offline examples do when the
        # user passes no --use-system-prompt/--extra-body.
        self.ar_extra_body = dict(ar_extra_body or {})
        self.limit = limit
        # Set per run (see ``_load_omni``): the model path is needed to load the
        # AR tokenizer, and ``_ar_ctx`` caches the model-declared AR-input
        # builder so it is resolved once per harness, not once per prompt.
        self._model_path: str | None = None
        self._ar_ctx: tuple[Any, Any] | None = None

    def available(self) -> bool:
        return importlib.util.find_spec("vllm_omni") is not None

    def group_key(self, recipe: BenchmarkRecipe) -> tuple:
        """Params that must match for recipes to share one ``Omni`` load."""
        return (
            self.num_inference_steps,
            self.guidance_scale,
            self.height,
            self.width,
            self.seed,
            self.scorer_model,
            # Layout (seeds, dirs) and scoring differ per benchmark family --
            # a geneval recipe must never share a load with a clip_score one.
            recipe.images_per_prompt,
            recipe.scoring.scorer,
        )

    # ── prompt set ───────────────────────────────────────────────────────────
    def _load_prompt_rows(self, recipe: BenchmarkRecipe) -> list[dict]:
        """Raw dataset rows (dicts) with a prompt column, repo-relative aware."""
        if not recipe.dataset:
            raise AqaError(
                f"diffusion recipe {recipe.name} needs a `dataset` "
                "(HF datasets uri or a local .jsonl/.csv file)"
            )
        field = recipe.prompt_field or self.prompt_field
        source = recipe.dataset
        path = Path(source)
        if not path.is_absolute() and not path.exists():
            # Recipes may reference repo-vendored data relative to the repo root
            # (e.g. recipes/data/geneval/...) regardless of the caller's cwd.
            alt = _REPO_ROOT / source
            if alt.exists():
                path = alt
        if path.suffix == ".jsonl" and path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif path.suffix == ".csv" and path.exists():
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:  # pragma: no cover - optional heavy dep
                raise AqaError(
                    "diffusion harness needs `datasets` for HF prompt sets: pip install datasets"
                ) from exc
            rows = list(load_dataset(source, split=self.dataset_split))
        rows = [row for row in rows if row.get(field)]
        if self.limit is not None:
            rows = rows[: self.limit]
        if not rows:
            raise AqaError(
                f"diffusion recipe {recipe.name}: no prompts found in {source!r} "
                f"(field {field!r}, split {self.dataset_split!r})"
            )
        return rows

    def _load_prompts(self, recipe: BenchmarkRecipe) -> list[str]:
        field = recipe.prompt_field or self.prompt_field
        return [str(row[field]) for row in self._load_prompt_rows(recipe)]

    # ── generation (vllm-omni) ───────────────────────────────────────────────
    def _load_omni(self, model: str):
        try:
            from vllm_omni.entrypoints.omni import Omni
        except ImportError as exc:
            raise AqaError(
                "vllm-omni is not installed; install it to use the diffusion harness"
            ) from exc
        # Remembered for ``_ar_input_builder`` (the AR tokenizer is loaded from
        # the same path). Safe here: one harness load maps to one model.
        self._model_path = model
        self._ar_ctx = None
        return Omni(model=model, mode="text-to-image", **self.omni_kwargs)

    def _release_omni(self, omni) -> None:
        """Free the DiT / VRAM held by an ``Omni`` instance (best-effort)."""
        try:
            close = getattr(omni, "close", None)
            if callable(close):
                close()
        finally:
            # vllm-omni keeps its diffusion parallel groups (dp/cfg/sp/pp/...) alive
            # after close(); a second Omni in the same process then refuses to init
            # ("model parallel state must be empty before initialization"). Tear the
            # groups down explicitly so baseline -> quantized eval works in-process.
            try:
                from vllm_omni.diffusion.distributed import parallel_state as omni_ps

                omni_ps.destroy_distributed_env()
            except Exception as exc:  # noqa: BLE001 - version-dependent, best-effort
                _log.debug("omni parallel-state teardown skipped: %s", exc)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - CPU-only box
                pass

    def _sampling_params_list(self, omni, seed: int) -> list:
        """Build the per-stage sampling params for one generation call.

        Mirrors ``examples/offline_inference/text_to_image/text_to_image.py``:
        single-stage models (FLUX.1-dev) take our params directly; for
        multi-stage pipelines our diffusion params replace the diffusion slot
        of the engine's ``default_sampling_params_list``.
        """
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        params = OmniDiffusionSamplingParams(
            height=self.height,
            width=self.width,
            seed=seed,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_outputs_per_prompt=1,
        )
        defaults = list(getattr(omni, "default_sampling_params_list", None) or [])
        if not defaults:
            return [params]
        replaced = False
        resolved = []
        for default in defaults:
            if isinstance(default, OmniDiffusionSamplingParams):
                resolved.append(params)
                replaced = True
            else:
                resolved.append(default)
        if not replaced and len(resolved) == 1:
            resolved = [params]
        return resolved

    @staticmethod
    def _extract_image(outputs) -> Any:
        """Pull the first PIL image out of ``omni.generate()`` outputs."""
        images = None
        for output in outputs or []:
            images = getattr(output, "images", None)
            if images:
                break
        if not images:
            raise AqaError("vllm-omni returned no images for a prompt")
        image = images[0]
        if not hasattr(image, "save"):  # numpy array -> PIL
            import numpy as np
            from diffusers.utils import numpy_to_pil

            image = numpy_to_pil(np.asarray(image))[0]
        return image

    def _generate_images(
        self,
        omni,
        prompts: list[str],
        images_dir: Path,
        *,
        images_per_prompt: int = 1,
        prompt_rows: list[dict] | None = None,
    ) -> list[dict]:
        """Generate PNGs per prompt into ``images_dir`` + ``metadata.csv``.

        Default (``images_per_prompt == 1``): one flat ``<idx>.png`` per prompt,
        seeds ``seed + idx``.  With ``images_per_prompt > 1`` the official
        GenEval layout is used: ``<idx>/metadata.jsonl`` (the prompt's full
        metadata row) plus ``<idx>/samples/<j>.png``, seeds
        ``seed + idx*1000 + j``.  Rows are appended/flushed per image and
        existing rows are skipped, so an interrupted run resumes where it left
        off.
        """
        images_dir.mkdir(parents=True, exist_ok=True)
        meta_path = images_dir / "metadata.csv"
        if images_per_prompt <= 1:
            return self._generate_single(omni, prompts, images_dir, meta_path)
        return self._generate_multi(
            omni, prompts, images_dir, meta_path, images_per_prompt, prompt_rows
        )

    def _generate_single(self, omni, prompts: list[str], images_dir: Path,
                         meta_path: Path) -> list[dict]:
        """One PNG per prompt (flat layout) -- the historical behavior."""
        done: dict[int, dict] = {}
        if meta_path.exists():
            with meta_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    done[int(row["idx"])] = row

        with meta_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if not done:
                writer.writerow(["idx", "prompt", "image_path", "seed"])
                fh.flush()
            for idx, prompt in enumerate(prompts):
                if idx in done:
                    continue
                row = {"idx": str(idx),
                       **self._generate_one(omni, prompt, idx, self.seed + idx,
                                            images_dir / f"{idx:04d}.png")}
                writer.writerow([row["idx"], row["prompt"], row["image_path"], row["seed"]])
                fh.flush()
                done[idx] = row
        return [done[idx] for idx in sorted(done)]

    def _generate_one(self, omni, prompt: str, idx: int, seed: int,
                      image_path: Path) -> dict:
        """Generate one image via ``omni.generate``; return its metadata row."""
        request = {"prompt": prompt, "modalities": ["image"]}
        params_list = self._sampling_params_list(omni, seed)
        self._apply_ar_stage_inputs(omni, prompt, request, params_list)
        try:
            outputs = omni.generate(request, sampling_params_list=params_list)
            self._extract_image(outputs).save(image_path)
        except AqaError:
            raise
        except Exception as exc:
            raise AqaError(
                f"vllm-omni generation failed for prompt {idx} ({prompt[:60]!r}): {exc}"
            ) from exc
        _log.info("  generated %s (seed=%d)", image_path, seed)
        return {"prompt": prompt, "image_path": str(image_path), "seed": str(seed)}

    # ── model-declared AR-stage inputs ───────────────────────────────────────
    def _ar_input_builder(self, omni):
        """The model's AR-stage input builder, or ``None`` if it declares none.

        Models whose pipeline has a text/AR stage ahead of the diffusion stage
        (e.g. HunyuanImage3) declare an ``ar_input_builder`` through
        ``vllm_omni.model_extras``.  It supplies the AR's template-formatted
        prompt tokens and, critically, its ``stop_token_ids``.  Without them the
        AR has no terminator and decodes until ``max_tokens`` -- roughly 10x
        slower, and worse, the DiT stage is then conditioned on a runaway CoT
        instead of the recaption prefix the model hands over by design.

        Resolution follows the same declarative contract as the shared task
        examples: a model that declares nothing (FLUX, Z-Image, Qwen-Image, ...)
        gets ``None`` here and generation is untouched.  Resolved once and
        cached on the harness.
        """
        if self._ar_ctx is not None:
            return self._ar_ctx
        builder = tokenizer = None
        try:
            from vllm_omni.model_extras import get_ar_input_builder, get_model_class_name

            builder = get_ar_input_builder(get_model_class_name(omni))
        except Exception as exc:  # noqa: BLE001 - older vllm-omni: stay model-agnostic
            _log.debug("AR-input builder lookup skipped: %s", exc)
            builder = None
        if builder is not None:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_path,
                    trust_remote_code=bool(self.omni_kwargs.get("trust_remote_code", False)),
                )
            except Exception as exc:  # noqa: BLE001 - degrade to plain generation
                # Without a tokenizer the builder falls back to a string prompt
                # and cannot resolve stop tokens, so skipping is safer than a
                # silently mis-terminated AR decode.
                _log.warning(
                    "AR-stage inputs disabled: could not load the AR tokenizer from %s (%s)",
                    self._model_path, exc,
                )
                builder = None
        self._ar_ctx = (builder, tokenizer)
        if builder is not None:
            _log.info("AR-stage inputs enabled (model declares an ar_input_builder)")
        return self._ar_ctx

    def _apply_ar_stage_inputs(self, omni, prompt: str, request: dict, params_list: list) -> bool:
        """Add the model's AR-stage inputs to *request* (in place). No-op if none.

        Mirrors ``_apply_ar_stage_inputs`` in the shared task examples: prompt
        tokens, the DiT-side system-prompt prefix, and ``stop_token_ids`` on the
        stage(s) the builder names -- by explicit index, never by guessing
        "whichever stage is not a diffusion stage".
        """
        builder, tokenizer = self._ar_input_builder(omni)
        if builder is None:
            return False
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        ar = builder(
            prompt=prompt,
            tokenizer=tokenizer,
            extra_body=self.ar_extra_body or {},
            num_images=0,  # text-to-image: no reference images
            height=self.height,
            width=self.width,
            text_output=False,
        )
        if getattr(ar, "prompt_token_ids", None) is not None:
            request["prompt_token_ids"] = ar.prompt_token_ids
        elif getattr(ar, "prompt", None) is not None:
            request["prompt"] = ar.prompt
        if getattr(ar, "use_system_prompt", None):
            request["use_system_prompt"] = ar.use_system_prompt
        if getattr(ar, "modalities", None):
            request["modalities"] = ar.modalities
        for stage_index in getattr(ar, "stage_indices", ()) or ():
            if stage_index >= len(params_list):
                continue
            stage_params = params_list[stage_index]
            if not isinstance(stage_params, OmniDiffusionSamplingParams) and hasattr(
                stage_params, "stop_token_ids"
            ):
                stage_params.stop_token_ids = list(ar.stop_token_ids or [])
        return True

    def _generate_multi(
        self,
        omni,
        prompts: list[str],
        images_dir: Path,
        meta_path: Path,
        images_per_prompt: int,
        prompt_rows: list[dict] | None,
    ) -> list[dict]:
        """N images per prompt in the official GenEval per-prompt layout."""
        done: dict[tuple[int, int], dict] = {}
        if meta_path.exists():
            with meta_path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    done[(int(row["idx"]), int(row["sample"]))] = row

        with meta_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if not done:
                writer.writerow(["idx", "sample", "prompt", "image_path", "seed"])
                fh.flush()
            for idx, prompt in enumerate(prompts):
                prompt_dir = images_dir / f"{idx:04d}"
                (prompt_dir / "samples").mkdir(parents=True, exist_ok=True)
                # Official: the prompt's metadata sits beside samples/ (the full
                # GenEval row when the dataset carries one, else the bare prompt).
                meta_row = prompt_rows[idx] if prompt_rows is not None else {"prompt": prompt}
                (prompt_dir / "metadata.jsonl").write_text(
                    json.dumps(meta_row, ensure_ascii=False)
                )
                for j in range(images_per_prompt):
                    if (idx, j) in done:
                        continue
                    seed = self.seed + idx * 1000 + j
                    row = self._generate_one(
                        omni, prompt, idx, seed, prompt_dir / "samples" / f"{j}.png"
                    )
                    row = {"idx": str(idx), "sample": str(j), **row}
                    writer.writerow(
                        [row["idx"], row["sample"], row["prompt"],
                         row["image_path"], row["seed"]]
                    )
                    fh.flush()
                    done[(idx, j)] = row
        return [done[k] for k in sorted(done)]

    # ── scoring (CLIP) ───────────────────────────────────────────────────────
    def _score_images(self, rows: list[dict]) -> list[float]:
        """CLIP image-text cosine x 100 per row (CLIP's logits_per_image scale)."""
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        device = self.scorer_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(self.scorer_model).to(device)
        processor = CLIPProcessor.from_pretrained(self.scorer_model)
        scores: list[float] = []
        for row in rows:
            image = Image.open(row["image_path"])
            # DrawBench prompts can exceed CLIP's 77-token context; truncation is the
            # standard CLIP-score convention (and transformers 5.x hard-errors on
            # over-length input instead of silently truncating).
            inputs = processor(
                text=[row["prompt"]], images=[image], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # logits_per_image is cosine similarity scaled by CLIP's logit
            # scale (100), i.e. the conventional CLIP score x 100.
            scores.append(float(outputs.logits_per_image[0, 0]))
        return scores

    # ── Harness contract ─────────────────────────────────────────────────────
    def _command_str(self, model: str, *, scorer: str | None = None) -> str:
        # No CLI invocation exists (the harness drives the Omni API in-process);
        # record the exact generation+scoring setup as the reproducibility string.
        base = (
            f"vllm_omni Omni(model={model!r}, mode='text-to-image') "
            f"generate(num_inference_steps={self.num_inference_steps}, "
            f"guidance_scale={self.guidance_scale}, size={self.width}x{self.height}, "
            f"seed={self.seed}) + {scorer or f'clip_score({self.scorer_model})'}"
        )
        if self.omni_kwargs:
            base += f" omni_kwargs={self.omni_kwargs!r}"
        return base

    def _result_geneval(
        self, recipe: BenchmarkRecipe, model: str, recipe_dir: Path
    ) -> BenchmarkResult:
        """GenEval detection scoring over the official-layout image dir."""
        from ..geneval import DEFAULT_CLIP_MODEL, DEFAULT_DETECTOR, GenevalScorer, summarize

        spec = recipe.scoring.geneval
        scorer = GenevalScorer(
            detector_model=spec.detector_model if spec else DEFAULT_DETECTOR,
            clip_model=spec.clip_model if spec else DEFAULT_CLIP_MODEL,
            device=self.scorer_device,
        )
        images_dir = recipe_dir / "images"
        rows = scorer.evaluate_directory(images_dir)
        if not rows:
            raise AqaError(f"geneval: no generated images found under {images_dir}")
        results_path = scorer.write_results(rows, images_dir / "results.jsonl")
        summary = summarize(rows)
        _log.info("  geneval: overall=%.4f tasks=%s",
                  summary["overall"], summary["task_means"])
        return BenchmarkResult(
            name=recipe.name,
            score=summary["overall"],
            metric="geneval",
            num_samples=summary["num_images"],
            per_sample=[1.0 if r["correct"] else 0.0 for r in rows],
            raw={
                "task_means": summary["task_means"],
                "results": str(results_path),
                "detector_model": scorer.detector_model,
                "clip_model": scorer.clip_model,
                # Geneval loads its detector/CLIP once and raises on failure, so a
                # row that exists was fully scored; env failures surface as raises.
                **_incomplete_raw({}),
            },
            command=self._command_str(
                model,
                scorer=f"geneval({scorer.detector_model} + {scorer.clip_model})",
            ),
        )

    def _result_dpgbench(
        self, recipe: BenchmarkRecipe, model: str, recipe_dir: Path,
        rows: list[dict], prompt_rows: list[dict],
    ) -> BenchmarkResult:
        """DPG-Bench VQA judging over the flat single-image-per-prompt layout."""
        from ..dpgbench import (
            DEFAULT_JUDGE_BASE_URL,
            DEFAULT_VLLM_JUDGE_MODEL,
            DEFAULT_VQA_MODEL,
            DpgBenchScorer,
            summarize,
        )

        spec = recipe.scoring.dpgbench
        scorer = DpgBenchScorer(
            vqa_model=spec.vqa_model if spec else DEFAULT_VQA_MODEL,
            resolution=spec.resolution if spec else 1024,
            device=(spec.device if spec and spec.device != "auto" else self.scorer_device),
            # Judge is config-selectable: the official mPLUG path (default) or an
            # OpenAI-compatible vLLM VLM endpoint (the documented substitution).
            judge_kind=spec.judge if spec else "mplug",
            judge_model=spec.judge_model if spec else DEFAULT_VLLM_JUDGE_MODEL,
            base_url=spec.base_url if spec else DEFAULT_JUDGE_BASE_URL,
            api_key_env=spec.api_key_env if spec else "OPENAI_API_KEY",
            timeout=spec.timeout if spec else 300.0,
            max_tokens=spec.max_tokens if spec else 16,
        )
        images_dir = recipe_dir / "images"
        image_rows = scorer.evaluate_rows(rows, prompt_rows)
        if not image_rows:
            if scorer.num_env_failed:
                raise AqaError(
                    f"dpgbench: every image failed to score because the VQA judge "
                    f"is unavailable ({scorer.num_env_failed} image(s)); refusing to "
                    "report a 0.0 that is an environment artifact."
                )
            raise AqaError(f"dpgbench: no generated images could be scored under {images_dir}")
        jsonl, _txt = scorer.write_results(image_rows, images_dir / "dpgbench_results")
        summary = summarize(
            image_rows, num_skipped=scorer.num_skipped,
            num_env_failed=scorer.num_env_failed,
        )
        _log.info("  dpgbench: overall=%.4f categories=%s",
                  summary["overall"], summary["categories_l1"])
        return BenchmarkResult(
            name=recipe.name,
            score=summary["overall"],
            metric="dpgbench",
            num_samples=summary["num_images"],
            # dependency-adjusted per-question 0/1 (headline metric granularity)
            per_sample=[
                float(v) for row in image_rows for v in row["adj_scores"].values()
            ],
            raw={
                "categories": summary["categories"],
                "categories_l1": summary["categories_l1"],
                "results": str(jsonl),
                "vqa_model": scorer.vqa_model,
                "judge": scorer.judge_label(),
                **_incomplete_raw(summary),
            },
            command=self._command_str(model, scorer=f"dpgbench({scorer.judge_label()})"),
        )

    def _result_cvtg2k(
        self, recipe: BenchmarkRecipe, model: str, recipe_dir: Path,
        rows: list[dict], prompt_rows: list[dict],
    ) -> list[BenchmarkResult]:
        """CVTG-2K text/CLIP scoring -> one result per metric (acc / ned / clipscore)."""
        from ..cvtg2k import DEFAULT_CLIP_MODEL, Cvtg2kScorer, summarize

        spec = recipe.scoring.cvtg2k
        scorer = Cvtg2kScorer(
            ocr_lang=spec.ocr_lang if spec else "en",
            clip_model=spec.clip_model if spec else DEFAULT_CLIP_MODEL,
            clipscore_x2_5=spec.clipscore_x2_5 if spec else False,
            device=(spec.device if spec and spec.device != "auto" else self.scorer_device),
        )
        image_rows = scorer.evaluate_rows(rows, prompt_rows)
        if not image_rows:
            raise AqaError(
                f"cvtg2k: no generated images could be scored under {recipe_dir / 'images'}"
            )
        jsonl, _txt = scorer.write_results(image_rows, recipe_dir / "images" / "cvtg2k_results")
        summary = summarize(image_rows)
        _assert_env_scored(
            "cvtg2k", "PaddleOCR / the CLIP scorer",
            num_scored=summary["num_scored"], num_env_failed=summary["num_env_failed"],
        )
        scored_rows = [r for r in image_rows if r.get("scored", True)]
        overall = summary["overall"]
        _log.info("  cvtg2k: word_acc=%.4f ned=%.4f clipscore=%.4f",
                  overall["word_accuracy"], overall["ned"], overall["clipscore"])

        def _per_image(key):
            return [r[key] for r in scored_rows]

        raw = {
            "area_results": summary["area_results"],
            "total_words": overall["total_words"],
            "correct_words": overall["correct_words"],
            "results": str(jsonl),
            "clip_model": scorer.clip_model,
            "ocr_lang": scorer.ocr_lang,
            "clipscore_x2_5": scorer.clipscore_x2_5,
            **_incomplete_raw(summary),
        }
        metrics = [
            ("acc", "cvtg2k_acc", overall["word_accuracy"],
             [r["correct_words"] / r["total_words"] if r["total_words"] else 0.0
              for r in scored_rows]),
            ("ned", "cvtg2k_ned", overall["ned"],
             [sum(r["word_neds"]) / len(r["word_neds"]) if r["word_neds"] else 0.0
              for r in scored_rows]),
            ("clipscore", "cvtg2k_clipscore", overall["clipscore"], _per_image("clip_score")),
        ]
        return [
            BenchmarkResult(
                name=f"{recipe.name}.{name}",
                score=score,
                metric=metric,
                num_samples=len(scored_rows),
                per_sample=per_sample,
                raw=raw,
                command=self._command_str(
                    model,
                    scorer=(f"cvtg2k(ocr=paddleocr[{scorer.ocr_lang}] + "
                            f"clip={scorer.clip_model}, x2_5={scorer.clipscore_x2_5})"),
                ),
            )
            for name, metric, score, per_sample in metrics
        ]

    def _result_glyphdraw2(
        self, recipe: BenchmarkRecipe, model: str, recipe_dir: Path,
        rows: list[dict], prompt_rows: list[dict],
    ) -> list[BenchmarkResult]:
        """GlyphDraw2 Acc/NED via PP-OCR -> one result per metric."""
        from ..glyphdraw2 import GlyphDraw2Scorer, summarize

        spec = recipe.scoring.glyphdraw2
        scorer = GlyphDraw2Scorer(
            ocr_lang=spec.ocr_lang if spec else "en",
            device=(spec.device if spec and spec.device != "auto" else self.scorer_device),
        )
        image_rows = scorer.evaluate_rows(rows, prompt_rows)
        if not image_rows:
            raise AqaError(
                f"glyphdraw2: no generated images could be scored under {recipe_dir / 'images'}"
            )
        jsonl, _txt = scorer.write_results(
            image_rows, recipe_dir / "images" / "glyphdraw2_results"
        )
        summary = summarize(image_rows)
        _assert_env_scored(
            "glyphdraw2", "PaddleOCR",
            num_scored=summary["num_scored"], num_env_failed=summary["num_env_failed"],
        )
        scored_rows = [r for r in image_rows if r.get("scored", True)]
        overall = summary["overall"]
        _log.info("  glyphdraw2: acc=%.4f ned=%.4f", overall["acc"], overall["ned"])
        raw = {
            "breakdown": summary["breakdown"],
            "results": str(jsonl),
            "ocr_lang": scorer.ocr_lang,
            **_incomplete_raw(summary),
        }
        metrics = [
            ("acc", "glyphdraw2_acc", overall["acc"],
             [sum(r["line_acc"]) / len(r["line_acc"]) if r["line_acc"] else 0.0
              for r in scored_rows]),
            ("ned", "glyphdraw2_ned", overall["ned"],
             [sum(r["line_neds"]) / len(r["line_neds"]) if r["line_neds"] else 0.0
              for r in scored_rows]),
        ]
        return [
            BenchmarkResult(
                name=f"{recipe.name}.{name}",
                score=score,
                metric=metric,
                num_samples=len(scored_rows),
                per_sample=per_sample,
                raw=raw,
                command=self._command_str(
                    model, scorer="glyphdraw2(ocr=paddleocr)"
                ),
            )
            for name, metric, score, per_sample in metrics
        ]

    def _result_wise(
        self, recipe: BenchmarkRecipe, model: str, recipe_dir: Path,
        rows: list[dict], prompt_rows: list[dict],
    ) -> BenchmarkResult:
        """WISE_Verified binary judging over the flat one-image-per-prompt layout."""
        from ..wise import (
            DEFAULT_BASE_URL,
            DEFAULT_JUDGE_MODEL,
            WiseScorer,
            summarize,
        )

        spec = recipe.scoring.wise
        scorer = WiseScorer(
            judge_model=spec.judge_model if spec else DEFAULT_JUDGE_MODEL,
            base_url=spec.base_url if spec else DEFAULT_BASE_URL,
            api_key_env=spec.api_key_env if spec else "OPENAI_API_KEY",
            timeout=spec.timeout if spec else 300.0,
            max_retries=spec.max_retries if spec else 3,
        )
        image_rows = scorer.evaluate_rows(rows, prompt_rows)
        if not image_rows:
            raise AqaError(
                f"wise: no generated images could be scored under {recipe_dir / 'images'}"
            )
        jsonl, _txt = scorer.write_results(image_rows, recipe_dir / "images" / "wise_results")
        summary = summarize(image_rows)
        _assert_env_scored(
            "wise", "the vLLM judge",
            num_scored=summary["num_scored"], num_env_failed=summary["num_env_failed"],
        )
        _log.info("  wise: overall=%.4f categories=%s", summary["overall"],
                  summary["categories"])
        return BenchmarkResult(
            name=recipe.name,
            score=summary["overall"],
            metric="wise",
            num_samples=summary["num_images"],
            per_sample=[float(r["score"]) for r in image_rows if r["score"] is not None],
            raw={
                "categories": summary["categories"],
                "counts": summary["counts"],
                "complete": summary["complete"],
                "num_missing": summary["num_missing"],
                "num_env_failed": summary["num_env_failed"],
                "missing_prompt_ids": summary["missing_prompt_ids"],
                "results": str(jsonl),
                "judge_model": scorer.judge_model,
            },
            command=self._command_str(
                model, scorer=f"wise(judge={scorer.judge_model}, base_url={scorer.base_url})"
            ),
        )

    def _result_image_metrics(
        self, recipe: BenchmarkRecipe, model: str, rows: list[dict]
    ) -> list[BenchmarkResult]:
        """One BenchmarkResult per requested image metric (clip / clip_iqa /
        imagereward), all over the same generated rows."""
        from ..image_metrics import IMAGEREWARD_MODEL, score_rows

        spec = recipe.scoring.image_metrics
        metrics = spec.metrics if spec else ["clip"]
        clip_model = (spec.clip_model if spec else None) or self.scorer_model
        ir_model = (spec.imagereward_model if spec else None) or IMAGEREWARD_MODEL
        per_metric = score_rows(rows, metrics, clip_model=clip_model,
                                imagereward_model=ir_model, device=self.scorer_device)
        results: list[BenchmarkResult] = []
        for metric, scores in per_metric.items():
            if not scores:
                raise AqaError(f"image_metrics: metric {metric!r} produced no scores")
            metric_name = {"clip": "clip_score", "clip_iqa": "clip_iqa",
                           "imagereward": "imagereward"}[metric]
            scorer_model = ir_model if metric == "imagereward" else clip_model
            results.append(BenchmarkResult(
                name=f"{recipe.name}.{metric}",
                score=sum(scores) / len(scores),
                metric=metric_name,
                num_samples=len(scores),
                per_sample=scores,
                raw={
                    "metadata": str(Path(rows[0]["image_path"]).parent / "metadata.csv"),
                    "scorer_model": scorer_model,
                    # image_metrics loads its CLIP/ImageReward once and raises on
                    # failure, so a score that exists was fully measured.
                    **_incomplete_raw({}),
                },
                command=self._command_str(model, scorer=f"{metric_name}({scorer_model})"),
            ))
        return results

    def _result_for(
        self, recipe: BenchmarkRecipe, model: str, recipe_dir: Path, rows: list[dict],
        prompt_rows: list[dict] | None = None,
    ) -> BenchmarkResult | list[BenchmarkResult]:
        if recipe.scoring.scorer == "geneval":
            return self._result_geneval(recipe, model, recipe_dir)
        if recipe.scoring.scorer == "dpgbench":
            return self._result_dpgbench(recipe, model, recipe_dir, rows,
                                         prompt_rows or [])
        if recipe.scoring.scorer == "cvtg2k":
            return self._result_cvtg2k(recipe, model, recipe_dir, rows,
                                       prompt_rows or [])
        if recipe.scoring.scorer == "glyphdraw2":
            return self._result_glyphdraw2(recipe, model, recipe_dir, rows,
                                           prompt_rows or [])
        if recipe.scoring.scorer == "wise":
            return self._result_wise(recipe, model, recipe_dir, rows,
                                     prompt_rows or [])
        if recipe.scoring.scorer == "image_metrics":
            return self._result_image_metrics(recipe, model, rows)
        scores = self._score_images(rows)
        return BenchmarkResult(
            name=recipe.name,
            score=sum(scores) / len(scores) if scores else 0.0,
            metric="clip_score",
            num_samples=len(scores),
            per_sample=scores,
            raw={
                "metadata": str(recipe_dir / "images" / "metadata.csv"),
                "scorer_model": self.scorer_model,
                **_incomplete_raw({}),
            },
            command=self._command_str(model),
        )

    def run(self, recipe: BenchmarkRecipe, model: str, backend, out_dir: Path) -> BenchmarkResult | list[BenchmarkResult]:
        prompt_rows = self._load_prompt_rows(recipe)
        field = recipe.prompt_field or self.prompt_field
        prompts = [str(row[field]) for row in prompt_rows]
        omni = self._load_omni(model)
        try:
            rows = self._generate_images(
                omni, prompts, out_dir / "images",
                images_per_prompt=recipe.images_per_prompt, prompt_rows=prompt_rows,
            )
        finally:
            self._release_omni(omni)
        return self._result_for(recipe, model, out_dir, rows, prompt_rows)

    def run_group(
        self, recipes: list[BenchmarkRecipe], model: str, backend, out_dir: Path
    ) -> list[BenchmarkResult]:
        """Evaluate several recipes on one shared ``Omni`` load (DiT load is expensive).

        A recipe may yield several results (e.g. ``image_metrics`` emits one per
        requested metric); the list is flattened for the runner.
        """
        omni = self._load_omni(model)
        try:
            generated = []
            for recipe in recipes:
                prompt_rows = self._load_prompt_rows(recipe)
                field = recipe.prompt_field or self.prompt_field
                prompts = [str(row[field]) for row in prompt_rows]
                recipe_dir = out_dir / recipe.name
                rows = self._generate_images(
                    omni, prompts, recipe_dir / "images",
                    images_per_prompt=recipe.images_per_prompt, prompt_rows=prompt_rows,
                )
                generated.append((recipe, recipe_dir, rows, prompt_rows))
        finally:
            self._release_omni(omni)
        results: list[BenchmarkResult] = []
        for recipe, recipe_dir, rows, prows in generated:
            res = self._result_for(recipe, model, recipe_dir, rows, prows)
            results.extend(res if isinstance(res, list) else [res])
        return results
