"""Deterministic model-card generator for a published quantized checkpoint.

The card is a **run artifact** (``<out_dir>/model_card.md``) and is uploaded
verbatim as the Hugging Face model repo's ``README.md``.  It follows
``templates/model_card.md``: fixed H2 sections, one deployment command, one
evaluation table, one quantization command, and a ``Signed-off-by`` last line.

Everything is derived from the run directory (``run_manifest.json``,
``env_lock.json``, ``eval/accuracy.json``, ``baseline/accuracy.json`` and the
exported quantization config) — never from memory or the template example.  The
operator identity is **not** inferred: it must be supplied explicitly
(``provider.upload.signoff``) because uploads normally use a bot account.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from ..config import RunConfig
from ..eval.registry import EvalResult
from ..logging_util import AqaError, get_logger
from .names import canonical_artifact_name, scheme_display
from .quant_details import compute_quantization_details

_log = get_logger("report.model_card")

CARD_FILE = "model_card.md"

_SIGNOFF_RE = re.compile(
    r"^Signed-off-by: [^<>{}]*\S <([^<>@\s{}]+@[^<>@\s{}]+\.[^<>@\s{}]+)>$")
_PLACEHOLDER_DOMAINS = ("example.com", "example.org", "example.net")
_PLACEHOLDER_LINE = "Signed-off-by: {member name} <{member email}>"

_FIXED_SECTIONS = """\
## Ethical Considerations and Limitations

The model can produce factually incorrect output, and should not be relied on to produce factually accurate information. Because of the limitations of the pretrained model and the finetuning datasets, it is possible that this model could generate lewd, biased or otherwise offensive outputs.
Therefore, before deploying any applications of the model, developers should perform safety testing.

## Caveats and Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.
Here are a couple of useful links to learn more about Intel's AI software:

- [Intel Neural Compressor](https://github.com/intel/neural-compressor)
- [AutoRound](https://github.com/intel/auto-round)

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please consult an attorney before using this model for commercial purposes.

## Cite

```
@article{cheng2023optimize,
  title={Optimize weight rounding via signed gradient descent for the quantization of llms},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```

[arxiv](https://arxiv.org/abs/2309.05516) [github](https://github.com/intel/auto-round)
"""


def signoff_problem(text: str) -> str | None:
    """Return the reason a model card fails the sign-off contract (``None`` = OK)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = _SIGNOFF_RE.match(lines[-1]) if lines else None
    if not match:
        return ("model_card.md 末行须为 `Signed-off-by: Name <email>`"
                "（身份每次向用户确认）")
    domain = match.group(1).rsplit("@", 1)[1].lower()
    if domain in _PLACEHOLDER_DOMAINS:
        return "model_card.md 的 Signed-off-by 是占位身份，须填向用户确认的真实成员"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _repo_id(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip().rstrip("/")
    return value if re.fullmatch(r"[^/\s]+/[^/\s]+", value) else None


def _base_metadata(base: str, *, local_model: str | None = None) -> dict[str, Any]:
    """Best-effort license/pipeline tag from the base model's local files.

    ``local_model`` is preferred when it is a real directory: a run may set
    ``output.report_model`` to the HF id for the card's ``base_model`` while the
    weights/README actually live on disk under ``run_cfg.model``.
    """
    model_dir: Path | None = None
    for candidate in (local_model, base):
        if not candidate:
            continue
        direct = Path(candidate)
        if direct.is_dir():
            model_dir = direct
            break
    if model_dir is None and _repo_id(base):
        org, name = base.split("/", 1)
        hf_home = Path(os.environ.get("HF_HOME") or
                       (Path.home() / ".cache" / "huggingface"))
        hub = hf_home / "hub" / f"models--{org}--{name}"
        snaps = sorted((hub / "snapshots").glob("*")) if (hub / "snapshots").is_dir() else []
        if snaps:
            model_dir = snaps[-1]
    meta: dict[str, Any] = {}
    if model_dir is not None:
        readme = model_dir / "README.md"
        if readme.is_file():
            try:
                text = readme.read_text(errors="replace")
                if text.startswith("---"):
                    end = text.find("\n---", 3)
                    if end > 0:
                        front = yaml.safe_load(text[3:end]) or {}
                        if isinstance(front, dict):
                            meta.update(front)
            except (OSError, yaml.YAMLError):
                pass
        cfg = _read_json(model_dir / "config.json")
        if "pipeline_tag" not in meta:
            arch = " ".join(str(a) for a in (cfg.get("architectures") or []))
            vision = any(k in cfg for k in (
                "vision_config", "vision_tower", "image_token_id", "mm_vision_tower"))
            if vision or "ConditionalGeneration" in arch:
                meta["pipeline_tag"] = "image-text-to-text"
            elif arch:
                meta["pipeline_tag"] = "text-generation"
    return meta


def _scheme_phrase(scheme: str, *, mixed: bool) -> str:
    s = (scheme or "").upper()
    if "MXFP4" in s:
        phrase = "MXFP4 (W4A4, group_size 32)"
    elif "MXFP8" in s:
        phrase = "MXFP8 (W8A8, group_size 32)"
    elif s.startswith("W4A16"):
        phrase = "int4 (W4A16, group_size 128)"
    elif s.startswith("W8A16"):
        phrase = "int8 (W8A16, group_size 128)"
    elif "NVFP4" in s:
        phrase = "NVFP4 (W4A4, group_size 16)"
    elif "FP8" in s:
        phrase = "FP8 (W8A8)"
    else:
        phrase = scheme or "quantized"
    return f"{phrase} with mixed-precision overrides" if mixed else phrase


def _article(phrase: str) -> str:
    first = phrase.split()[0].lower() if phrase.split() else ""
    # Acronyms are read by their letter names ("em-eks-eff-pee-four"), so they
    # take "an"; "W4A16" is read "double-u" and takes "a".
    if first.startswith(("mx", "nv", "fp", "int", "f", "s", "x", "n")):
        return "an"
    if first.startswith("w") or first.startswith(("h", "y", "u")):
        return "a"
    return "an" if first[:1] in "aeiou" else "a"


def _method_phrase(quant) -> str:
    if quant is None:
        return ""
    if quant.method == "model_free" or (quant.iters or 0) == 0:
        return " using RTN"
    return f" with {quant.iters} tuning iterations"


def _module_families(details: dict[str, Any]) -> tuple[list[str], list[str]]:
    groups = details.get("module_groups") or []
    quantized: list[str] = []
    kept: list[str] = []
    for group in groups:
        label = str(group.get("label") or group.get("id") or "").strip()
        if not label:
            continue
        target = kept if group.get("scheme") == "BF16" else quantized
        if label not in target:
            target.append(label)
    return quantized, kept


def _model_config(run_cfg: RunConfig, quant_dir: Path | None) -> dict[str, Any]:
    candidates = []
    if quant_dir is not None:
        candidates.append(Path(quant_dir) / "config.json")
    model = Path(run_cfg.model)
    if model.is_dir():
        candidates.append(model / "config.json")
    for path in candidates:
        cfg = _read_json(path)
        if cfg:
            return cfg
    return {}


def _quantization_details(run_cfg: RunConfig, out_dir: Path,
                          quant_dir: Path | None) -> dict[str, Any]:
    quant = run_cfg.quant
    if quant is None:
        return {}
    command = str(_read_json(out_dir / "quant_plan.json").get("command") or "")
    kv = "FP8" if quant.static_kv_fp8 or "--static_kv_dtype" in command else None
    attn = "FP8" if quant.static_attention_fp8 or "--static_attention_dtype" in command else None
    try:
        return compute_quantization_details(
            model_cfg=_model_config(run_cfg, quant_dir),
            declared_scheme=quant.scheme,
            quant_dir=quant_dir,
            model_source=run_cfg.model,
            declared_ignore=quant.ignore_layers,
            quant_lm_head=quant.quant_lm_head,
            layer_config=quant.layer_config,
            kv_scheme=kv,
            attn_scheme=attn,
        )
    except Exception as exc:  # noqa: BLE001 - card generation is best-effort
        _log.warning("model card: quantization details unavailable: %s", exc)
        return {}


def _read_accuracy(out_dir: Path, subdir: str) -> EvalResult | None:
    path = out_dir / subdir / "accuracy.json"
    if not path.is_file():
        return None
    try:
        from . import read_eval_result

        return read_eval_result(out_dir / subdir)
    except Exception as exc:  # noqa: BLE001
        _log.warning("model card: could not read %s: %s", path, exc)
        return None


def _fmt_score(score: float | None) -> str:
    return "—" if score is None else f"{score * 100:.2f}"


def _fmt_recovery(base: float | None, quant: float | None) -> str:
    if base is None or quant is None or base == 0:
        return "—"
    return f"{quant / base * 100:.2f}%"


def _evaluation_table(result: EvalResult | None, baseline: EvalResult | None,
                      scheme_label: str) -> list[str]:
    if result is None or not result.benchmarks:
        return ["| Benchmark | " + scheme_label + " |",
                "|---|---:|"]
    ruler_names = {"niah_multiquery", "ruler_qa_squad"}
    general = [n for n in result.benchmarks if n not in ruler_names]
    ruler = [n for n in result.benchmarks if n in ruler_names]

    if baseline is None:
        lines = ["| Benchmark | " + scheme_label + " |", "|---|---:|"]
    else:
        lines = ["| Benchmark | BF16 | " + scheme_label + " | Recovery |",
                 "|---|---:|---:|---:|"]

    def row(name: str) -> str:
        q = result.benchmarks[name].score
        b = baseline.benchmarks[name].score if baseline and name in baseline.benchmarks else None
        if baseline is None:
            return f"| {name} | {_fmt_score(q)} |"
        return f"| {name} | {_fmt_score(b)} | {_fmt_score(q)} | {_fmt_recovery(b, q)} |"

    def average(names: list[str]) -> str | None:
        if not names:
            return None
        q_mean = sum(result.benchmarks[n].score for n in names) / len(names)
        if baseline is None:
            return f"| **Average** | **{_fmt_score(q_mean)}** |"
        b_names = [n for n in names if n in baseline.benchmarks]
        if len(b_names) != len(names):
            return f"| **Average** | — | **{_fmt_score(q_mean)}** | — |"
        b_mean = sum(baseline.benchmarks[n].score for n in names) / len(names)
        return (f"| **Average** | **{_fmt_score(b_mean)}** | **{_fmt_score(q_mean)}** "
                f"| **{_fmt_recovery(b_mean, q_mean)}** |")

    for name in sorted(general):
        lines.append(row(name))
    avg = average(general)
    if avg:
        lines.append(avg)
    for name in sorted(ruler):
        lines.append(row(name))
    if ruler:
        ravg = average(ruler)
        if ravg:
            lines.append(ravg.replace("**Average**", "**RULER Average**"))
    return lines


def _manifest_commands(out_dir: Path) -> tuple[str, list[str]]:
    manifest = _read_json(out_dir / "run_manifest.json")
    commands = manifest.get("commands") or {}
    quant = str(commands.get("quantize") or "")
    evaluate = [str(c) for c in (commands.get("evaluate") or []) if c]
    if not quant:
        plan = _read_json(out_dir / "quant_plan.json")
        quant = str(plan.get("command") or "")
    return quant, evaluate


def _shell_token(token: str) -> str:
    if "${" in token:
        escaped = token.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        return f'"{escaped}"'
    return shlex.quote(token)


def _format_script(tokens: list[str]) -> str:
    """One flag per line, keeping its value on the same line."""
    if not tokens:
        return ""
    lines = [tokens[0]]
    for token in tokens[1:]:
        if token.startswith("-"):
            lines.append("    " + _shell_token(token))
        else:
            lines[-1] += " " + _shell_token(token)
    return " \\\n".join(lines)


def _rewrite_quant_command(command: str, *, base: str, repo_name: str,
                           model: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--device_map":
            i += 2
            continue
        if token.startswith("--device_map="):
            i += 1
            continue
        if token in {"--model_name", "--model", "--model_path"}:
            out.append(f"{token}={base}" if token != "--model_name" else "--model_name")
            if token == "--model_name":
                out.append(base)
            i += 2
            continue
        if token.startswith(("--model_name=", "--model=", "--model_path=")):
            flag = token.split("=", 1)[0]
            out.append(f"{flag}={base}")
            i += 1
            continue
        if token == "--output_dir":
            out.append("--output_dir")
            out.append(repo_name)
            i += 2
            continue
        if token.startswith("--output_dir="):
            out.append(f"--output_dir={repo_name}")
            i += 1
            continue
        if token == model or token.rstrip("/") == model.rstrip("/"):
            out.append(base)
            i += 1
            continue
        if token.startswith("/") or token.startswith("~/"):
            # Drop unknown local paths (template R3: no local paths).
            i += 1
            continue
        out.append(token)
        i += 1
    return _format_script(out)


def _rewrite_eval_commands(commands: list[str], quant_dir: Path | None,
                           model: str) -> str:
    if not commands:
        return ""
    replacements: list[tuple[str, str]] = []
    if quant_dir is not None:
        replacements.append((str(quant_dir.resolve()), "${MODEL_PATH}"))
        replacements.append((str(quant_dir), "${MODEL_PATH}"))
    replacements.append((str(model), "${MODEL_PATH}"))
    rendered: list[str] = []
    for command in commands:
        text = command
        for old, new in sorted(replacements, key=lambda kv: -len(kv[0])):
            if not old:
                continue
            text = text.replace(old, new)
        try:
            tokens = shlex.split(text)
        except ValueError:
            rendered.append(text)
            continue
        rendered.append(_format_script(tokens))
    return "\n".join(rendered)


def _inference_versions(out_dir: Path) -> dict[str, str]:
    manifest = _read_json(out_dir / "run_manifest.json")
    packages = (manifest.get("environment") or {}).get("packages") or {}
    return {str(k): str(v) for k, v in packages.items() if v}


def _vllm_flags(run_cfg: RunConfig) -> list[str]:
    quant = run_cfg.quant
    if quant is not None and quant.static_kv_fp8:
        return ["--kv-cache-dtype fp8", "--dtype bfloat16"]
    args = dict(run_cfg.eval.lm_eval.model_args or {})
    flags: list[str] = []
    mapping = {
        "attention_backend": "--attention-backend",
        "moe_backend": "--moe-backend",
        "linear_backend": "--linear-backend",
        "kv_cache_dtype": "--kv-cache-dtype",
        "dtype": "--dtype",
        "reasoning_parser": "--reasoning-parser",
    }
    for key, flag in mapping.items():
        value = args.get(key)
        if value is not None and not isinstance(value, bool):
            flags.append(f"{flag} {value}")
    if args.get("language_model_only") is True:
        flags.append("--language-model-only")
    if not any(f.startswith("--dtype ") for f in flags):
        flags.append("--dtype bfloat16")
    return flags


def _verdict_frontmatter(base: str, meta: dict[str, Any], *, scheme: str,
                         quant) -> list[str]:
    lines = ["---", "base_model:", f"- {base}"]
    if meta.get("license"):
        lines.append(f"license: {meta['license']}")
    if meta.get("license_name"):
        lines.append(f"license_name: {meta['license_name']}")
    if meta.get("license_link"):
        lines.append(f"license_link: {meta['license_link']}")
    lines.append(f"pipeline_tag: {meta.get('pipeline_tag') or 'text-generation'}")
    lines.append("library_name: transformers")
    lines.append("tags:")
    lines.append("- auto-round")
    lines.append(f"- {scheme.lower()}")
    if quant is not None and quant.static_kv_fp8:
        lines.append("- fp8-kv-cache")
    if quant is not None and quant.static_attention_fp8:
        lines.append("- fp8-attention")
    if quant is not None and quant.export_format == "llm_compressor":
        lines.append("- compressed-tensors")
    lines.append("---")
    return lines


def render_model_card(
    run_cfg: RunConfig,
    *,
    out_dir: str | Path,
    repo_name: str,
    result: EvalResult | None = None,
    baseline: EvalResult | None = None,
    quant_dir: str | Path | None = None,
    signoff: str | None = None,
) -> str:
    """Render the README card from a run directory.

    Missing operator identity / base repo id are allowed: the card is then a
    **draft** with the template's own placeholders (``{member name}`` /
    ``{base_repo_id}``).  A publish path still validates the sign-off and skips
    the upload rather than publishing a placeholder identity.
    """
    if signoff:
        problem = signoff_problem(f"Signed-off-by: {signoff}")
        if problem:
            raise AqaError(problem)
    base = _repo_id(run_cfg.output.report_model) or _repo_id(run_cfg.model)
    if not base:
        base = "{base_repo_id}"
    quant = run_cfg.quant
    scheme = scheme_display(quant) if quant is not None else "Unknown"
    mixed = bool(quant and quant.layer_config)
    quant_dir_path = Path(quant_dir) if quant_dir else None

    out_dir = Path(out_dir)
    if result is None:
        result = _read_accuracy(out_dir, "eval")
    if baseline is None:
        baseline = _read_accuracy(out_dir, "baseline") or _read_accuracy(
            out_dir, "baseline_thinking_off")
    if result is None:
        raise AqaError("model card requires evaluation results (eval/accuracy.json missing)")

    details = _quantization_details(run_cfg, out_dir, quant_dir_path)
    quantized_families, kept_families = _module_families(details)
    meta = _base_metadata(base, local_model=run_cfg.model)
    phrase = _scheme_phrase(quant.scheme if quant else "", mixed=mixed)

    lines = _verdict_frontmatter(base, meta, scheme=scheme, quant=quant)
    lines += [
        "",
        "## Model Details",
        "",
        (f"This model is {_article(phrase)} {phrase} model of "
         f"[{base}](https://huggingface.co/{base}) generated by "
         f"[intel/auto-round](https://github.com/intel/auto-round)"
         f"{_method_phrase(quant)}. Please follow the license of the original model."),
        "",
        f"- **Quantized ({scheme})**: "
        f"{', '.join(quantized_families) if quantized_families else 'linear layers'}",
        f"- **Kept in BF16**: "
        f"{', '.join(kept_families) if kept_families else 'embeddings, norms'}",
    ]
    static_bits = []
    if quant is not None and quant.static_kv_fp8:
        static_bits.append("KV cache")
    if quant is not None and quant.static_attention_fp8:
        static_bits.append("attention")
    if static_bits:
        lines.append(f"- **Static FP8**: {' / '.join(static_bits)}")

    versions = _inference_versions(out_dir)
    install: list[str] = []
    if versions.get("vllm"):
        install.append(f"uv pip install vllm=={versions['vllm']}")
    if versions.get("flashinfer-python"):
        install.append(f"uv pip install flashinfer-python=={versions['flashinfer-python']}")
    if not install:
        install.append("uv pip install vllm")

    org = (run_cfg.provider.upload.repo
           if run_cfg.provider.upload is not None and run_cfg.provider.upload.repo
           else "{org}")
    serve = [f"vllm serve {org}/{repo_name}"]
    flags = _vllm_flags(run_cfg)
    serve_block = serve[0]
    if flags:
        serve_block += " \\\n" + " \\\n".join(f"    {flag}" for flag in flags)

    lines += [
        "",
        "## How to Use",
        "",
        "### Environment",
        "",
        "```bash",
        *install,
        "```",
        "",
        "### vLLM Usage",
        "",
        "```bash",
        serve_block,
        "```",
        "",
        "## Evaluation",
        "",
        *_evaluation_table(result, baseline, scheme),
        "",
        "## Generate the Model",
        "",
    ]

    quant_command, _ = _manifest_commands(out_dir)
    ar_version = versions.get("auto-round") or versions.get("auto_round") or ""
    sha = None
    m = re.search(r"\+g([0-9a-f]{40})", ar_version)
    if m:
        sha = m.group(1)
    if sha:
        lines += [
            "```bash",
            ("uv pip install git+https://github.com/intel/auto-round.git@" + sha),
            "```",
            "",
        ]
    if quant_command:
        lines += [
            "```bash",
            _rewrite_quant_command(quant_command, base=base, repo_name=repo_name,
                                   model=run_cfg.model),
            "```",
        ]
    else:
        lines += [
            "```bash",
            "# quantization command was not recorded for this run",
            "```",
        ]

    signoff_line = (f"Signed-off-by: {signoff}" if signoff
                    else "Signed-off-by: {member name} <{member email}>")
    lines += ["", _FIXED_SECTIONS, "", signoff_line, ""]
    return "\n".join(lines)


def write_model_card(
    run_cfg: RunConfig,
    *,
    out_dir: str | Path,
    repo_name: str | None = None,
    result: EvalResult | None = None,
    baseline: EvalResult | None = None,
    quant_dir: str | Path | None = None,
    signoff: str | None = None,
) -> Path:
    """Write ``<out_dir>/model_card.md`` atomically; ``signoff=None`` = draft."""
    out_dir = Path(out_dir)
    name = repo_name or (
        run_cfg.provider.upload.repo_name
        if run_cfg.provider.upload is not None and run_cfg.provider.upload.repo_name
        else canonical_artifact_name(run_cfg))
    text = render_model_card(
        run_cfg, out_dir=out_dir, repo_name=name, result=result,
        baseline=baseline, quant_dir=quant_dir, signoff=signoff)
    path = out_dir / CARD_FILE
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path
