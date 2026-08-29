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
        self.limit = limit

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
        try:
            outputs = omni.generate(
                request, sampling_params_list=self._sampling_params_list(omni, seed)
            )
            self._extract_image(outputs).save(image_path)
        except AqaError:
            raise
        except Exception as exc:
            raise AqaError(
                f"vllm-omni generation failed for prompt {idx} ({prompt[:60]!r}): {exc}"
            ) from exc
        _log.info("  generated %s (seed=%d)", image_path, seed)
        return {"prompt": prompt, "image_path": str(image_path), "seed": str(seed)}

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
            DEFAULT_VQA_MODEL,
            DpgBenchScorer,
            summarize,
        )

        spec = recipe.scoring.dpgbench
        scorer = DpgBenchScorer(
            vqa_model=spec.vqa_model if spec else DEFAULT_VQA_MODEL,
            resolution=spec.resolution if spec else 1024,
            device=(spec.device if spec and spec.device != "auto" else self.scorer_device),
        )
        images_dir = recipe_dir / "images"
        image_rows = scorer.evaluate_rows(rows, prompt_rows)
        if not image_rows:
            raise AqaError(f"dpgbench: no generated images could be scored under {images_dir}")
        jsonl, _txt = scorer.write_results(image_rows, images_dir / "dpgbench_results")
        summary = summarize(image_rows)
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
            },
            command=self._command_str(model, scorer=f"dpgbench({scorer.vqa_model})"),
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
