"""``aqa`` command-line interface.

Verbs mirror NeMo Evaluator's ``nel``: run / eval / quant / validate / list /
compare / gate / report / traces -- a clear, declarative surface over the pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import __version__
from .config import load_run_config
from .eval.registry import BenchmarkRegistry
from .logging_util import AqaError, get_logger
from .registry import EXECUTORS, all_registries

_log = get_logger("cli")


@click.group()
@click.version_option(__version__, prog_name="aqa")
def main() -> None:
    """autoquant-agent: agent-driven quantize / evaluate / self-heal."""


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--only", default=None,
              help="Comma-separated candidate names to run (default: all).")
@click.option("--dry-run", is_flag=True,
              help="Print the candidate plans + quant commands without executing.")
def sweep(config: str, only: str | None, dry_run: bool) -> None:
    """Run multiple quantization candidates (quant.recipe_sweep) and compare.

    Each candidate runs as a full derived pipeline in its own subdirectory
    (<output.dir>/<candidate>); the bf16 baseline is evaluated once (first
    candidate) and reused for the comparison table.  A failing candidate is
    recorded and does not abort the rest of the sweep.
    """
    from .agent.optimize import analyze_model
    from .quantize import build_autoround_command
    from .quantize.candidates import apply_candidate, resolve_candidates

    cfg = load_run_config(config)
    if not (cfg.quant and cfg.quant.recipe_sweep):
        raise click.ClickException(
            "quant.recipe_sweep is not enabled in this yaml; set `quant.recipe_sweep: true`")

    analysis = analyze_model(cfg.model)
    if not analysis.ok:
        raise click.ClickException(f"model structure analysis failed: {analysis.error}")
    candidates = resolve_candidates(analysis, cfg.quant)
    if only:
        wanted = {n.strip() for n in only.split(",") if n.strip()}
        unknown = wanted - {c.name for c in candidates}
        if unknown:
            raise click.ClickException(
                f"unknown candidate(s) {sorted(unknown)}; available: {[c.name for c in candidates]}")
        candidates = [c for c in candidates if c.name in wanted]

    base_dir = Path(cfg.output.dir)
    num_gpus = cfg.executor.num_gpus or 1
    plan_rows: list[dict] = []
    for cand in candidates:
        qcfg = apply_candidate(cfg.quant, cand)
        cmd = build_autoround_command(cfg.model, qcfg, base_dir / cand.name / "quantized",
                                      num_gpus=num_gpus)
        plan_rows.append({"name": cand.name, "rationale": cand.rationale,
                          "quant": qcfg.model_dump(exclude_none=True), "command": cmd})
    if dry_run:
        for row in plan_rows:
            click.echo(f"== {row['name']}: {row['rationale']}")
            click.echo(f"   {' '.join(row['command'])}")
        return

    click.echo(f"sweep: {len(candidates)} candidate(s) for {cfg.model} -> {base_dir}")
    executor = EXECUTORS.create(cfg.executor.type, **cfg.executor.options)
    report: dict = {"model": cfg.model, "base_dir": str(base_dir),
                    "candidates": []}
    baseline_scores: dict | None = None
    for i, (cand, row) in enumerate(zip(candidates, plan_rows)):
        qcfg = apply_candidate(cfg.quant, cand).model_copy(
            update={"recipe_sweep": False, "recipe_sweep_candidates": None})
        eval_cfg = cfg.eval if (i == 0 or not cfg.eval.baseline) \
            else cfg.eval.model_copy(update={"baseline": False})
        if cand.eval_model_args:
            lm_opts = eval_cfg.lm_eval.model_copy(update={
                "model_args": {**eval_cfg.lm_eval.model_args, **cand.eval_model_args}})
            eval_cfg = eval_cfg.model_copy(update={"lm_eval": lm_opts})
        out_cfg = cfg.output.model_copy(update={"dir": str(base_dir / cand.name)})
        rcfg = cfg.model_copy(update={"quant": qcfg, "eval": eval_cfg, "output": out_cfg})
        click.echo(f"\n===== [{i + 1}/{len(candidates)}] {cand.name}: {cand.rationale}")
        entry: dict = {"name": cand.name, "rationale": cand.rationale,
                       "command": row["command"], "output_dir": str(base_dir / cand.name)}
        try:
            result = executor.run(rcfg)
            entry["state"] = result.state.value
        except Exception as exc:  # noqa: BLE001 - one failure must not kill the sweep
            _log.warning("candidate %s failed: %s", cand.name, exc)
            entry["state"] = f"failed: {exc}"
        scores = _read_scores(base_dir / cand.name / "eval" / "accuracy.json")
        entry["scores"] = scores
        if i == 0 and cfg.eval.baseline:
            baseline_scores = _read_scores(base_dir / cand.name / "baseline" / "accuracy.json")
        report["candidates"].append(entry)
    report["baseline"] = baseline_scores
    base_dir.mkdir(parents=True, exist_ok=True)  # 全候选早夭时目录可能从未被建
    (base_dir / "sweep_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _print_sweep_table(report)
    # sweep 级 sink：各候选 run 只回传自己的子目录，对比报告在 base 目录——
    # 单独 dispatch 一次（best-effort，sink 失败不影响 sweep 结果）。
    try:
        from .pipeline import _artifact_name, _dispatch_sinks
        from .report.sinks import RunMeta

        states = [c.get("state", "") for c in report["candidates"]]
        done = sum(1 for s in states if s == "done")
        _dispatch_sinks(cfg, base_dir, RunMeta(
            model=cfg.output.report_model or cfg.model,
            artifact_name=_artifact_name(cfg, "sweep"),
            run_root=base_dir,
            status=f"Sweep {done}/{len(states)} done"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("sweep sink dispatch failed (non-fatal): %s", exc)
    click.echo(f"\nsweep report -> {base_dir / 'sweep_report.json'}")


def _read_scores(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return {k: v.get("score") for k, v in (data.get("results") or {}).items()}


def _print_sweep_table(report: dict) -> None:
    benches: list[str] = sorted(
        {b for c in report["candidates"] for b in (c.get("scores") or {})})
    if not benches:
        click.echo("\n(no benchmark scores collected)")
        return
    header = f"{'benchmark':<14}{'baseline':>10}" + "".join(
        f"{c['name']:>14}" for c in report["candidates"])
    click.echo("\n" + header)
    for b in benches:
        row = f"{b:<14}"
        base = (report.get("baseline") or {}).get(b)
        row += f"{base:>10.4f}" if isinstance(base, float) else f"{'-':>10}"
        for c in report["candidates"]:
            s = (c.get("scores") or {}).get(b)
            row += f"{s:>14.4f}" if isinstance(s, float) else f"{'-':>14}"
        click.echo(row)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def inspect(run_dir: str) -> None:
    """Validate a run output directory against the artifact layout contract
    (aqa/report/layout.yaml) and print what is present / missing."""
    from .report.layout import format_report, inspect_run_dir

    click.echo(format_report(inspect_run_dir(run_dir)))


@main.group()
def knowledge() -> None:
    """Shared knowledge store (fix-lessons + curated auto-round references)."""


@knowledge.command("show")
@click.option("--knowledge-root", default=None,
              help="Knowledge store root (default: shared host store).")
def knowledge_show(knowledge_root: str | None) -> None:
    """Show the references manifest (per-file freshness) and store stats."""
    import json as _json

    from .agent.knowledge import default_local_dir

    root = Path(knowledge_root) if knowledge_root else default_local_dir()
    manifest = root / "references" / "MANIFEST.json"
    if manifest.is_file():
        m = _json.loads(manifest.read_text())
        click.echo(f"references @ {root / 'references'}")
        click.echo(f"  updated_at: {m.get('updated_at')}  baseline auto_round_ref: "
                   f"{m.get('auto_round_ref') or '(unset)'}")
        files = m.get("files") or {}
        if isinstance(files, list):  # legacy manifest shape
            files = {f: {} for f in files}
        if m.get("note"):
            click.echo(f"  note: {m['note']}")
        for name, entry in files.items():
            ref = entry.get("auto_round_ref") if isinstance(entry, dict) else ""
            click.echo(f"  {name:<36} verified@ {ref or m.get('auto_round_ref') or '?'}")
    else:
        click.echo(f"no curated references at {root / 'references'} "
                   "(packaged copies in aqa/agent/recipes/ are the fallback)")
    lessons = root / "lessons"
    if lessons.is_dir():
        click.echo(f"lessons: {len(list(lessons.glob('*.jsonl')))} file(s)")


@knowledge.command("push-references")
@click.argument("from_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--repo", required=True,
              help="HF dataset repo hosting the knowledge store (org/name).")
@click.option("--path-in-repo", default="knowledge", show_default=True)
@click.option("--token-env", default=None,
              help="Env var with the write token (default: HF_TOKEN).")
@click.option("--auto-round-ref", default="",
              help="auto-round version/commit the docs were verified against.")
@click.option("--note", default="", help="Free-form change note.")
@click.option("--file", "single_file", default=None,
              help="Incremental update of one doc (merge); default: replace whole tree.")
@click.option("--knowledge-root", default=None,
              help="Local knowledge store root (default: shared host store).")
def knowledge_push_references(from_dir: str, repo: str, path_in_repo: str,
                              token_env: str | None, auto_round_ref: str,
                              note: str, single_file: str | None,
                              knowledge_root: str | None) -> None:
    """Publish a refreshed references tree (auto-round knowledge) to the store.

    Whole-tree replace by default; ``--file <name>`` updates a single doc
    incrementally (only its per-file freshness entry is bumped).  Example:

        aqa knowledge push-references aqa/agent/recipes \\
            --repo lvkaokao/lb_local --auto-round-ref v0.15.0
    """
    from .agent.knowledge import KnowledgeStore, default_local_dir

    store = KnowledgeStore(
        local_dir=Path(knowledge_root) if knowledge_root else default_local_dir(),
        repo=repo, path_in_repo=path_in_repo, token_env=token_env)
    ok = store.push_references(Path(from_dir), auto_round_ref=auto_round_ref,
                               note=note, file=single_file)
    click.echo("pushed references -> "
               f"{repo} ({path_in_repo}/references/)" if ok else
               "push skipped/failed (check token & repo)")


@knowledge.command("diff")
@click.option("--auto-round-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Local auto-round repo to diff against the recorded ref.")
@click.option("--knowledge-root", default=None,
              help="Knowledge store root (default: shared host store).")
@click.option("--max-commits", default=200, show_default=True)
def knowledge_diff(auto_round_dir: str, knowledge_root: str | None,
                   max_commits: int) -> None:
    """Scope what changed in auto-round since the references were verified.

    Reads ``references/MANIFEST.json``'s ``auto_round_ref`` and lists the
    upstream commits since then that touch knowledge-relevant areas (CLI,
    schemes, compressors, export, modeling, algorithms) — the work list for a
    knowledge refresh, so the owner re-reads the delta instead of everything.
    """
    import json as _json
    import subprocess

    from .agent.knowledge import default_local_dir

    root = Path(knowledge_root) if knowledge_root else default_local_dir()
    manifest_path = root / "references" / "MANIFEST.json"
    last_ref = ""
    if manifest_path.is_file():
        last_ref = _json.loads(manifest_path.read_text()).get("auto_round_ref", "")
    if not last_ref:
        click.echo("no auto_round_ref recorded; run a full review instead of a diff")
        return
    relevant = ("auto_round/cli/", "auto_round/schemes.py", "auto_round/compressors/",
                "auto_round/export/", "auto_round/modeling/", "auto_round/algorithms/",
                "auto_round/sign_sgd.py", "auto_round/utils/")
    rng = f"{last_ref}..HEAD"
    try:
        log = subprocess.run(
            ["git", "-C", auto_round_dir, "log", rng, "--oneline",
             f"-{max_commits}", "--", *relevant],
            capture_output=True, text=True, check=False)
    except OSError as exc:
        raise click.ClickException(f"git failed: {exc}")
    if log.returncode != 0:
        raise click.ClickException(
            f"git log failed for range {rng} (ref 不存在？先在 auto-round 仓库 git fetch):\n"
            f"{log.stderr.strip()}")
    lines = [l for l in log.stdout.splitlines() if l.strip()]
    click.echo(f"auto-round changes since {last_ref} (knowledge-relevant paths): "
               f"{len(lines)} commit(s)")
    for l in lines:
        click.echo(f"  {l}")
    if not lines:
        click.echo("references are up to date for these paths; check release notes "
                   "for changes outside them before skipping a refresh")


@main.command("log-action")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--kind", default="note",
              type=click.Choice(["env", "command", "patch", "note", "decision"]),
              show_default=True, help="Action category.")
@click.option("--text", required=True, help="What was done and why.")
@click.option("--command", default=None, help="The exact command run, if any.")
def log_action(run_dir: str, kind: str, text: str, command: str | None) -> None:
    """Append an owner/human action to <run_dir>/owner_actions.jsonl.

    Entries are embedded into run_manifest.json when the run finishes (and
    uploaded with the sinks), so owner-side env tweaks and extra commands
    become part of the auditable record instead of tribal knowledge.
    """
    import json as _json
    import time

    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, "text": text}
    if command:
        entry["command"] = command
    path = Path(run_dir) / "owner_actions.jsonl"
    with path.open("a") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    click.echo(f"logged -> {path}")


@main.command("capture-patch")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--repo", "repos", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Git worktree to capture (repeatable), e.g. the auto-round checkout.")
@click.option("--label", default=None,
              help="Patch file label (default: repo dir name).")
def capture_patch(run_dir: str, repos: tuple[str, ...], label: str | None) -> None:
    """Capture the owner's working-tree fixes as patches into the run dir.

    Writes ``git diff HEAD`` of each given repo to ``<run_dir>/patches/owner/
    <label>.patch`` (skipped when the tree is clean) and logs an
    ``owner_actions.jsonl`` entry, so owner-side fixes ride the same artifacts
    as aqa's own patch-capture output and get uploaded by the sinks.
    """
    import json as _json
    import subprocess
    import time

    run_path = Path(run_dir)
    dest_dir = run_path / "patches" / "owner"
    dest_dir.mkdir(parents=True, exist_ok=True)
    actions_path = run_path / "owner_actions.jsonl"
    captured: list[str] = []
    for repo in repos:
        repo_path = Path(repo).resolve()
        name = label or repo_path.name
        diff = subprocess.run(["git", "-C", str(repo_path), "diff", "HEAD"],
                              capture_output=True, text=True)
        if diff.returncode != 0:
            raise click.ClickException(f"git diff failed in {repo_path}: {diff.stderr.strip()}")
        if not diff.stdout.strip():
            click.echo(f"{repo_path}: clean tree, nothing to capture")
            continue
        patch_path = dest_dir / f"{name}.patch"
        patch_path.write_text(diff.stdout)
        captured.append(f"patches/owner/{name}.patch")
        click.echo(f"captured {repo_path} -> {patch_path} "
                   f"({len(diff.stdout.splitlines())} diff lines)")
    if captured:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": "patch",
                 "text": f"owner captured working-tree patches: {', '.join(captured)}"}
        with actions_path.open("a") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")


@main.command("publish-patch")
@click.argument("patch_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--repo-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Local clone of the fork (its worktree is NOT touched; a temp "
                   "worktree off upstream/base is used).")
@click.option("--upstream", default="https://github.com/intel/auto-round", show_default=True,
              help="Upstream repo URL to sync from.")
@click.option("--base", default="main", show_default=True, help="PR base branch.")
@click.option("--fork-repo", default=None,
              help="owner/name of the fork for `gh pr create` (default: parsed from origin).")
@click.option("--pr-repo", default=None,
              help="Repo the PR targets (default: the fork itself; set to the upstream "
                   "owner/name for a cross-repo upstream PR, head becomes <fork-owner>:<branch>).")
@click.option("--title", required=True, help="PR title (also the commit message head).")
@click.option("--body", default="", help="PR body (why/evidence/test results).")
@click.option("--branch", default=None,
              help="Branch name (default: aqa/<patch-stem>-<date>).")
@click.option("--dry-run", is_flag=True,
              help="Print the plan + `git apply --check` result; no fetch/push/PR.")
def publish_patch(patch_file: str, repo_dir: str, upstream: str, base: str,
                  fork_repo: str | None, pr_repo: str | None, title: str, body: str,
                  branch: str | None, dry_run: bool) -> None:
    """Turn a patch into a PR on the auto-round fork.

    Flow: sync fork (push upstream/<base> -> origin/<base>) -> temp worktree off
    upstream -> `git apply` the patch -> commit -> push branch -> `gh pr create`.
    Auth comes ONLY from the environment (GH_TOKEN/GITHUB_TOKEN for gh, or git
    credentials) — never embed tokens in URLs or files.
    """
    import re
    import shlex as _shlex
    import shutil
    import subprocess
    import tempfile
    import time

    def git(*args: str, cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run(["git", *args], cwd=cwd or repo_dir,
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise click.ClickException(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r

    repo_dir = str(Path(repo_dir).resolve())
    patch_path = Path(patch_file).resolve()
    patch_text = patch_path.read_text()
    touched = sorted({m.group(1) for m in
                      re.finditer(r"^(?:diff --git a/|--- a/)(\S+)", patch_text, re.M)})
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", patch_path.stem)
    branch = branch or f"aqa/{stem}-{time.strftime('%Y%m%d')}"
    if fork_repo is None:
        url = git("remote", "get-url", "origin").stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        if not m:
            if not dry_run:
                raise click.ClickException(
                    f"cannot parse fork owner/name from origin url: {url}")
            fork_repo = "(non-github origin; dry-run only)"
        else:
            fork_repo = m.group(1)

    click.echo(f"patch: {patch_path} ({len(touched)} file(s): {', '.join(touched)})")
    click.echo(f"fork: {fork_repo}  upstream: {upstream}  base: {base}  branch: {branch}")
    if dry_run:
        ref = f"refs/remotes/upstream/{base}"
        have_ref = git("rev-parse", "--verify", ref, check=False).returncode == 0
        with tempfile.TemporaryDirectory() as tmp:
            if have_ref:
                # real check against the last-fetched upstream tree (no network)
                git("worktree", "add", "--detach", tmp, f"upstream/{base}")
                chk = subprocess.run(["git", "apply", "--check", str(patch_path)],
                                     cwd=tmp, capture_output=True, text=True)
                git("worktree", "remove", "--force", tmp, check=False)
                ok = chk.returncode == 0
                click.echo(f"git apply --check (upstream/{base} tree): "
                           f"{'OK' if ok else 'FAILS'}")
                if not ok:
                    click.echo(chk.stderr.strip())
            else:
                subprocess.run(["git", "init", "-q", tmp], check=True)
                chk = subprocess.run(["git", "apply", "--check", str(patch_path)],
                                     cwd=tmp, capture_output=True, text=True)
                ok = chk.returncode == 0
                click.echo(f"git apply --check (bare tree): {'OK' if ok else 'FAILS'}")
                click.echo("(no local upstream ref fetched; run `git fetch upstream` "
                           "for a realistic check)")
        click.echo("[dry-run] would: sync fork base, worktree+apply, commit, push, gh pr create")
        return

    if not shutil.which("gh"):
        raise click.ClickException("publish-patch needs the `gh` CLI (PR creation)")
    # 1. sync the fork's base with upstream
    remotes = git("remote").stdout.split()
    if "upstream" not in remotes:
        git("remote", "add", "upstream", upstream)
    git("fetch", "upstream", base)
    git("push", "origin", f"upstream/{base}:{base}")
    click.echo(f"fork {fork_repo}:{base} synced with upstream/{base}")
    # 2. temp worktree off upstream — the user's dirty checkout is never touched
    with tempfile.TemporaryDirectory() as tmp:
        git("worktree", "add", "--detach", tmp, f"upstream/{base}")
        git("checkout", "-b", branch, cwd=tmp)
        chk = git("apply", "--check", str(patch_path), cwd=tmp, check=False)
        if chk.returncode != 0:
            raise click.ClickException(
                f"patch does not apply on upstream/{base}: {chk.stderr.strip()}")
        git("apply", str(patch_path), cwd=tmp)
        git("add", "-A", cwd=tmp)
        diffstat = git("diff", "--cached", "--stat", cwd=tmp).stdout.strip()
        git("commit", "-m", f"{title}\n\n{body}".strip(), cwd=tmp)
        click.echo(f"committed on {branch}:\n{diffstat}")
        # 3. push + PR (fork 内 PR 或跨仓上游 PR：--pr-repo 指定上游时
        #    head 必须是 <fork-owner>:<branch>)
        git("push", "-u", "origin", branch, cwd=tmp)
        target_repo = pr_repo or fork_repo
        head = f"{fork_repo.split('/')[0]}:{branch}" if pr_repo else branch
        pr = subprocess.run(
            ["gh", "pr", "create", "--repo", target_repo, "--base", base,
             "--head", head, "--title", title, "--body", body or title],
            cwd=tmp, capture_output=True, text=True)
        if pr.returncode != 0:
            raise click.ClickException(f"gh pr create failed: {pr.stderr.strip()}")
        click.echo(f"PR: {pr.stdout.strip()}")
    git("worktree", "prune")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Validate and print the plan without executing.")
@click.option("--resume", is_flag=True, help="Resume an interrupted run if detected.")
def run(config: str, dry_run: bool, resume: bool) -> None:
    """Run a full quantize -> evaluate -> gate pipeline from a run.yaml."""
    cfg = load_run_config(config)
    if cfg.quant and cfg.quant.recipe_sweep:
        raise click.ClickException(
            "this yaml enables quant.recipe_sweep; run it with `aqa sweep <yaml>` "
            "(multi-candidate orchestration), not `aqa run`")
    executor = EXECUTORS.create(cfg.executor.type, **cfg.executor.options)
    result = executor.run(cfg, dry_run=dry_run, resume=resume)
    click.echo(f"state={result.state.value} output={result.output_dir}")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def validate(config: str) -> None:
    """Type-check + sanity-check a run.yaml (no execution)."""
    cfg = load_run_config(config)
    reg = BenchmarkRegistry()
    known = set(reg.names())
    unknown = [b for b in cfg.eval.benchmarks if b not in known]
    click.echo(f"OK: model={cfg.model} executor={cfg.executor.type} backend={cfg.eval.backend}")
    click.echo(f"    benchmarks={cfg.eval.benchmarks}")
    if unknown:
        click.echo(f"    note: no recipe file for {unknown} (will fall back to bare lm_eval tasks)")
    # Write-back completeness nudges: env provides the vars but the yaml forgot
    # the fields — deterministic reminder so runs don't silently stay local-only.
    import os

    have_ds = bool(os.environ.get("LOCAL_RESULTS_DATASET"))
    have_tok = bool(os.environ.get("LOCAL_RESULTS_HF_TOKEN"))
    has_hf_sink = any(s.type == "hf_dataset" for s in cfg.output.sinks)
    if have_ds and have_tok and not has_hf_sink:
        click.echo("    note: LOCAL_RESULTS_DATASET+LOCAL_RESULTS_HF_TOKEN are set but "
                   "output.sinks has no hf_dataset — results stay local-only "
                   "(add the sink block; see docs/usage/yaml_reference.md)")
    if cfg.agent.fix_loop and have_ds and not cfg.agent.knowledge.repo:
        click.echo("    note: fix_loop is on but agent.knowledge.repo is unset — "
                   "fix lessons/sessions won't be shared to the dataset")
    if cfg.executor.type not in EXECUTORS.names():
        raise click.ClickException(f"unknown executor {cfg.executor.type!r}")
    if cfg.eval.backend not in _backend_names():
        raise click.ClickException(f"unknown backend {cfg.eval.backend!r}")


@main.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.option("--base-url", required=True, help="OpenAI-compatible endpoint of the served model.")
@click.option("--model", "served_model", required=True, help="Served model name (endpoint id).")
@click.option("--api-key-env", default=None, help="Env var holding the endpoint API key.")
@click.option("--out", type=click.Path(), default="./results/swebench", help="Output directory.")
@click.option("--limit", type=int, default=None, help="Cap the number of instances.")
@click.option("--instance-id", "instance_ids", multiple=True, help="Only these instance ids.")
@click.option("--difficulty", default=None, help="Only instances of this difficulty.")
@click.option("--concurrency", type=int, default=4, help="Instances graded in parallel.")
@click.option("--max-turns", type=int, default=40, help="ReAct turn budget per instance.")
@click.option("--scaffold", default="aqa-react",
              help="Agent scaffold / infra: aqa-react (default), codex, swe-agent.")
@click.option("--namespace", default="swebench", help="Docker namespace for eval images.")
def swebench(
    dataset: str, base_url: str, served_model: str, api_key_env: str | None, out: str,
    limit: int | None, instance_ids: tuple[str, ...], difficulty: str | None,
    concurrency: int, max_turns: int, scaffold: str, namespace: str,
) -> None:
    """Evaluate a served model on SWE-bench Verified (agentic, official scoring)."""
    import os

    from .agent_eval.client import ModelClient
    from .swebench import load_swebench, run_swebench
    from .swebench.run import default_sandbox_factory

    instances = load_swebench(
        dataset, instance_ids=list(instance_ids) or None,
        limit=limit, difficulty=difficulty)
    if not instances:
        raise click.ClickException("no SWE-bench instances selected")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    client = ModelClient(base_url=base_url, model=served_model, api_key=api_key)
    _, report = run_swebench(
        instances, client, model_name=served_model, out_dir=out,
        concurrency=concurrency, max_turns=max_turns, scaffold=scaffold,
        sandbox_factory=default_sandbox_factory(namespace=namespace),
    )
    resolved = sum(1 for r in report.results if r.resolved)
    click.echo(f"SWE-bench: resolved {resolved}/{len(report.results)} "
               f"({100 * report.resolved_rate:.1f}%) -> {out}")


@main.group()
def traces() -> None:
    """Trajectory datafication: export agent trajectories as SFT/RL datasets."""


@traces.command("export")
@click.option("--path", "paths", multiple=True, required=True,
              type=click.Path(exists=True),
              help="Run dir with trajectories.jsonl, a trajectories.jsonl file, "
                   "or a fix-loop knowledge root (repeatable).")
@click.option("--source", type=click.Choice(["agent_eval", "fixloop", "auto"]),
              default="auto", help="Trajectory source; auto-detects per path.")
@click.option("--filter", "success_filter",
              type=click.Choice(["success", "all"]), default="success",
              help="success keeps solved/reward>0 trajectories only.")
@click.option("--view", type=click.Choice(["sft", "rl"]), default="sft",
              help="Dataset view: sft (OpenAI messages, Nemotron-Agentic style) or "
                   "rl (NeMo-Gym style task rows: responses_create_params + verifier).")
@click.option("--sharegpt", is_flag=True, help="Add a ShareGPT `conversations` column.")
@click.option("--out", "out_jsonl", type=click.Path(), default=None,
              help="Write rows as JSONL (default: ./traces.jsonl when no other target).")
@click.option("--parquet", "parquet_path", type=click.Path(), default=None,
              help="Also write rows as parquet (needs `datasets`).")
@click.option("--push", is_flag=True, help="Push the dataset to the HF hub.")
@click.option("--repo", default=None, help="Target HF dataset repo for --push (org/name).")
def traces_export(paths, source, success_filter, sharegpt, view, out_jsonl, parquet_path,
                  push, repo) -> None:
    """Export trajectories (ATIF / fix-loop knowledge) as an SFT/RL dataset."""
    from .agent import traces_export as te

    success_only = success_filter == "success"
    rows: list[dict] = []
    for raw in paths:
        src = source
        if src == "auto":
            src = te.detect_source(raw)
            if src is None:
                _log.warning("skipping %s: no trajectories.jsonl / knowledge store found", raw)
                continue
        if src == "agent_eval":
            rows += te.export_agent_eval([raw], success_only=success_only, view=view)
        else:
            if view == "rl":
                _log.warning("fixloop knowledge has no RL view; exporting SFT rows for %s", raw)
            rows += te.export_fixloop(raw, success_only=success_only)
    if not rows:
        click.echo("no trajectory rows found")
        return
    if sharegpt:
        te.add_sharegpt(rows)
    targets: list[str] = []
    if out_jsonl or not (parquet_path or push):
        targets.append(str(te.write_jsonl(rows, out_jsonl or "traces.jsonl")))
    if parquet_path:
        targets.append(str(te.write_parquet(rows, parquet_path)))
    if push:
        if not repo:
            raise click.ClickException("--push needs --repo org/name")
        te.push_to_hub(rows, repo)
        targets.append(f"hf.co/datasets/{repo}")
    click.echo(f"exported {len(rows)} row(s) -> {', '.join(targets)}")


@traces.command("convert")
@click.option("--path", "paths", multiple=True, required=True,
              type=click.Path(exists=True),
              help="Raw agent-backend session log (kimi wire.jsonl / claude / codex "
                   "rollout / copilot / openclaw; repeatable).")
@click.option("--backend", default=None,
              type=click.Choice(["kimi", "claude", "codex", "copilot", "openclaw"]),
              help="Force the adapter (default: auto-detect from the artifact).")
@click.option("--model-name", default="", help="Model name when the log lacks one.")
@click.option("--instruction", default=None,
              help="Override the first user message (e.g. the task statement).")
@click.option("--out", "out_dir", type=click.Path(), default=None,
              help="Directory for trajectories.jsonl (default: ./traces_atif).")
def traces_convert(paths, backend, model_name, instruction, out_dir) -> None:
    """Convert raw agent session logs to ATIF trajectories.jsonl.

    The output directory feeds directly into `aqa traces export` /
    `distill` / `publish`.
    """
    from .agent.adapters import convert as convert_session

    dest = Path(out_dir or "traces_atif")
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / "trajectories.jsonl"
    n = 0
    with out_file.open("w") as f:
        for raw in paths:
            atif = convert_session(raw, backend=backend, model_name=model_name,
                                   instruction=instruction)
            f.write(json.dumps(atif, ensure_ascii=False) + "\n")
            n += 1
            click.echo(f"{raw}: {len(atif['steps'])} steps "
                       f"(session {atif['session_id']}, backend={atif['agent']['name']})")
    click.echo(f"converted {n} session(s) -> {out_file}")


@traces.command("distill")
@click.option("--path", "paths", multiple=True, required=True,
              type=click.Path(exists=True),
              help="Run dir with trajectories.jsonl or the file itself (repeatable).")
@click.option("--knowledge-root", default=None,
              help="Knowledge store root to write entries into "
                   "(default: the shared host store).")
@click.option("--phase", default="trajectory", show_default=True,
              help="Phase bucket the distilled entries are filed under.")
def traces_distill(paths, knowledge_root, phase) -> None:
    """Distill fix knowledge from trajectories into the knowledge store."""
    from .agent import knowledge_extract as ke
    from .agent.knowledge import default_local_dir

    root = Path(knowledge_root) if knowledge_root else default_local_dir()
    entries = []
    for raw in paths:
        entries.extend(ke.extract_from_file(raw))
    if not entries:
        click.echo("no knowledge entries found (no error→progress patterns)")
        return
    written = ke.write_back(entries, root, phase=phase)
    click.echo(f"distilled {written} knowledge entrie(s) -> {root} (phase={phase})")


@traces.command("publish")
@click.option("--path", "paths", multiple=True, required=True,
              type=click.Path(exists=True),
              help="Run dir with trajectories.jsonl or the file itself (repeatable).")
@click.option("--repo", default=None, help="Target HF dataset repo (org/name).")
@click.option("--token-env", default=None,
              help="Env var holding the HF write token (default: HF_TOKEN...).")
@click.option("--path-in-repo", default="", show_default=True,
              help="Subdirectory inside the dataset repo to upload under.")
@click.option("--stage-dir", type=click.Path(), default=None,
              help="Where to stage the archive (default: a temp dir).")
@click.option("--stage-only", is_flag=True,
              help="Only stage the archive locally; do not upload.")
def traces_publish(paths, repo, token_env, path_in_repo, stage_dir, stage_only) -> None:
    """Archive trajectories to a HF dataset repo in layered directory mode.

    Layout per session: trajectories/<session_id>/{trajectory.json,
    dataset_sft.jsonl, dataset_rl.jsonl}.
    """
    import tempfile

    from .agent import traces_publish as tp

    if not stage_only and not repo:
        raise click.ClickException("publishing needs --repo org/name (or --stage-only)")

    if stage_dir:
        dest = Path(stage_dir)
        manifest = tp.stage_trajectory_archive(paths, dest)
    else:
        dest = Path(tempfile.mkdtemp(prefix="aqa_traces_"))
        manifest = tp.stage_trajectory_archive(paths, dest)
    click.echo(f"staged {manifest['sessions']} session(s), "
               f"{len(manifest['files'])} file(s) -> {dest}")
    if stage_only:
        return
    url = tp.publish_archive(dest, repo, token_env=token_env,
                             path_in_repo=path_in_repo)
    click.echo(f"published -> {url}")


@main.command(name="list")
@click.option("--what", type=click.Choice(["benchmarks", "plugins", "all"]), default="all")
def list_cmd(what: str) -> None:
    """List available benchmarks and/or registered plugins."""
    if what in ("benchmarks", "all"):
        click.echo("benchmarks:")
        for name in BenchmarkRegistry().names():
            click.echo(f"  {name}")
    if what in ("plugins", "all"):
        click.echo("plugins:")
        for group, reg in all_registries().items():
            click.echo(f"  {group}: {', '.join(reg.names()) or '<none>'}")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(), default="./results", help="Output directory.")
def eval(config: str, out: str) -> None:
    """Evaluate a model (no quantization) from a run.yaml, locally in-process."""
    from .eval.runner import run_evaluation
    from .eval.thinking import apply_mode_to_cfg, resolve_thinking_modes

    cfg = load_run_config(config)
    out_dir = Path(out)
    for mode in resolve_thinking_modes(cfg.eval, cfg.model):
        cmode = apply_mode_to_cfg(cfg.eval, mode)
        result = run_evaluation(cfg.model, cmode, out_dir if mode == "on"
                                else out_dir.with_name(out_dir.name + "_thinking_off"))
        click.echo(json.dumps(result.to_dict(), indent=2))


@main.command()
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("candidate", type=click.Path(exists=True))
def compare(baseline: str, candidate: str) -> None:
    """Compare two accuracy.json result dirs/files and print score deltas."""
    b = _load_accuracy(baseline)
    c = _load_accuracy(candidate)
    click.echo(f"{'benchmark':<20}{'baseline':>10}{'candidate':>12}{'delta':>10}")
    for name in sorted(set(b) | set(c)):
        bs, cs = b.get(name), c.get(name)
        delta = (cs - bs) if (bs is not None and cs is not None) else float("nan")
        click.echo(f"{name:<20}{_fmt(bs):>10}{_fmt(cs):>12}{delta:>10.4f}")


def _fmt(v):
    return "-" if v is None else f"{v:.4f}"


def _load_accuracy(path: str) -> dict[str, float]:
    p = Path(path)
    if p.is_dir():
        p = p / "accuracy.json"
    data = json.loads(p.read_text())
    return {k: v.get("score") for k, v in data.get("results", {}).items()}


def _backend_names() -> list[str]:
    from .registry import BACKENDS

    return BACKENDS.names()


def main_entry() -> None:  # pragma: no cover
    try:
        main()
    except AqaError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main_entry()
