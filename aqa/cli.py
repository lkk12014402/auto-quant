"""``aqa`` command-line interface.

Verbs mirror NeMo Evaluator's ``nel``: run / eval / quant / validate / list /
compare / gate / report -- a clear, declarative surface over the pipeline.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import click

from . import __version__
from .config import load_run_config
from .eval.registry import BenchmarkRegistry
from .logging_util import AqaError, get_logger
from .registry import EXECUTORS, all_registries
from .report.owner_actions import OWNER_ACTION_KINDS, append_owner_action
from .session.journal import KINDS as _JOURNAL_KINDS

_log = get_logger("cli")


@click.group()
@click.version_option(__version__, prog_name="aqa")
def main() -> None:
    """autoquant-agent: agent-driven quantize / evaluate / self-heal."""


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Validate and print the plan without executing.")
@click.option("--resume", is_flag=True,
              help="Resume from <output.dir>/session: skip phases whose artifacts "
                   "already exist (e.g. don't re-quantize). Requires a prior session; "
                   "errors if none is found (local executor only).")
def run(config: str, dry_run: bool, resume: bool) -> None:
    """Run a full quantize -> evaluate -> gate pipeline from a run.yaml."""
    cfg = load_run_config(config)
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
    if cfg.executor.type not in EXECUTORS.names():
        raise click.ClickException(f"unknown executor {cfg.executor.type!r}")
    if cfg.eval.backend not in _backend_names():
        raise click.ClickException(f"unknown backend {cfg.eval.backend!r}")


@main.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.option("--variant", default=None,
              help="Variant id (auto-selected when the manifest has one).")
@click.option("--mode", type=click.Choice(["full", "eval-only"]), default="full",
              help="full = replay quantization + evaluation; eval-only = evaluate an existing checkpoint.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None,
              help="Reproduce run directory (default: ./reproduce/<manifest>-<variant>).")
@click.option("--checkpoint", default=None,
              help="eval-only checkpoint (default: variant hf_url / MODEL_PATH.default).")
@click.option("--abs-tol", type=float, default=0.01, help="Absolute score tolerance.")
@click.option("--rel-tol", type=float, default=0.02, help="Relative score tolerance.")
@click.option("--on-input-mismatch", type=click.Choice(["warn", "fail"]), default="warn",
              help="warn = record mismatch and continue; fail = verdict not_reproduced.")
@click.option("--dry-run", is_flag=True, help="Print the resolved plan; no execution.")
@click.option("--force", is_flag=True,
              help="If --out is non-empty, use a timestamped sibling instead of failing.")
@click.option("--json", "as_json", is_flag=True, help="Print the full JSON report.")
def reproduce(
    manifest: str, variant: str | None, mode: str, out_dir: str | None,
    checkpoint: str | None, abs_tol: float, rel_tol: float,
    on_input_mismatch: str, dry_run: bool, force: bool, as_json: bool,
) -> None:
    """Reproduce one INC recipe variant and verify input/structure/numbers.

    Exit codes: 0 reproduced, 1 not_reproduced, 2 inconclusive.  The JSON report
    is also written to <out>/reproduction_report.json (except for --dry-run).
    """
    from .reproduce import reproduce_manifest

    try:
        report = reproduce_manifest(
            manifest, variant=variant, mode=mode, out_dir=out_dir,
            checkpoint=checkpoint, abs_tol=abs_tol, rel_tol=rel_tol,
            on_input_mismatch=on_input_mismatch, dry_run=dry_run, force=force)
    except AqaError as exc:
        raise click.ClickException(str(exc)) from exc

    if dry_run or as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        click.echo(f"verdict={report.get('verdict')} mode={report.get('mode')} "
                   f"variant={report.get('variant')} out={report.get('output_dir')}")
        inp = report.get("input_equivalence") or {}
        quant = report.get("quantization_reproduction") or {}
        ev = report.get("evaluation_reproduction") or {}
        click.echo(f"  input:  {inp.get('status')}")
        for diff in inp.get("diffs") or []:
            click.echo(f"    - {diff}")
        click.echo(f"  quant:  {quant.get('status')}"
                   + (f"  ({len(quant.get('group_mismatches') or [])} mismatch(es))"
                      if quant.get("group_mismatches") else ""))
        click.echo(f"  eval:   {ev.get('status')}")
        for row in ev.get("per_benchmark") or []:
            mark = "ok" if row.get("within") else "MISS"
            click.echo(f"    [{mark}] {row.get('id')}: expected={row.get('expected')} "
                       f"got={row.get('got')}")
        if report.get("error"):
            click.echo(f"  error:  {report['error']}")
    if not dry_run:
        code = {"reproduced": 0, "not_reproduced": 1,
                "inconclusive": 2}.get(str(report.get("verdict")), 2)
        if code:
            raise SystemExit(code)


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

    cfg = load_run_config(config)
    result = run_evaluation(cfg.model, cfg.eval, Path(out))
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


@main.command(name="log-action")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--kind", type=click.Choice(OWNER_ACTION_KINDS), required=True)
@click.option("--text", required=True, help="Human-readable reason or action summary.")
@click.option("--command", default=None, help="Command associated with the action.")
def log_action(run_dir: Path, kind: str, text: str, command: str | None) -> None:
    """Append an owner action and refresh run_manifest.json when present."""
    try:
        append_owner_action(run_dir, kind=kind, text=text, command=command)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not log owner action: {exc}") from exc
    click.echo(f"logged {kind} action -> {run_dir / 'owner_actions.jsonl'}")


@main.command(name="capture-patch")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--repo",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def capture_patch(run_dir: Path, repo: Path) -> None:
    """Capture tracked working-tree changes into patches/owner without applying them."""
    repo = repo.resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
        if root.returncode:
            raise click.ClickException(root.stderr.strip() or f"{repo} is not a git repository")
        diff = subprocess.run(
            ["git", "-C", str(root.stdout.strip()), "diff", "--binary", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise click.ClickException(f"could not run git diff: {exc}") from exc
    if diff.returncode:
        raise click.ClickException(diff.stderr.strip() or "git diff failed")
    if not diff.stdout:
        click.echo("clean tree; no patch captured")
        return

    repo_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(root.stdout.strip()).name).strip("-")
    patch_dir = run_dir / "patches" / "owner"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"{repo_name or 'repo'}.patch"
    patch_path.write_text(diff.stdout, encoding="utf-8")
    try:
        append_owner_action(
            run_dir, kind="patch",
            text=f"Captured tracked working-tree diff from {root.stdout.strip()}",
            command=f"git -C {root.stdout.strip()} diff --binary HEAD",
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"patch saved at {patch_path}, but action log failed: {exc}"
        ) from exc
    click.echo(f"captured patch -> {patch_path}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def inspect(run_dir: str) -> None:
    """Validate a run dir against the artifact layout contract (aqa/report/layout.yaml).

    Reports which expected entries are present / missing for this run (the
    ``when`` conditions are detected from the dir itself). Exits non-zero when
    the run is INCOMPLETE, so owners (human or agent) can gate on it instead of
    eyeballing the layout.
    """
    from .report.layout import format_report, inspect_run_dir

    report = inspect_run_dir(Path(run_dir))
    click.echo(format_report(report))
    if not report.ok:
        raise SystemExit(1)


@main.group()
def session() -> None:
    """Operator tools for a run's session/ (state machine, events, resume status).

    Mirrors Hyperloom's ``session <state|report|events>`` operator surface: the
    observable session artifacts are the contract.
    """


def _session_dir(run_dir: str) -> Path:
    d = Path(run_dir)
    return d / "session" if (d / "session").exists() else d


@session.command(name="state")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def session_state(run_dir: str) -> None:
    """Print the raw session state.json (phase statuses + decisions + gate)."""
    from .session.state import state_path

    path = state_path(_session_dir(run_dir))
    if not path.exists():
        raise click.ClickException(f"no session state at {path}")
    click.echo(path.read_text().rstrip())


@session.command(name="report")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def session_report(run_dir: str) -> None:
    """Human-readable phase progress + current best + decisions."""
    from .session.state import SessionState, state_path

    path = state_path(_session_dir(run_dir))
    if not path.exists():
        raise click.ClickException(f"no session state at {path}")
    st = SessionState.from_dict(json.loads(path.read_text()))
    lines = [f"run_id: {st.run_id}", f"current phase: {st.phase}", "phases:"]
    for name, status in st.phases.items():
        lines.append(f"  [{status:<8}] {name}")
    if st.baseline:
        lines.append(f"baseline: mean={_fmt(st.baseline.get('mean_score'))}")
    if st.current_best:
        lines.append(f"current_best: {json.dumps(st.current_best, ensure_ascii=False)}")
    if st.gate:
        lines.append(f"gate: {st.gate.get('verdict')} "
                     f"({'; '.join(st.gate.get('reasons', []))})")
    for d in st.decisions:
        lines.append(f"decision[{d.get('phase')}]: {d.get('what')} — {d.get('why')}")
    click.echo("\n".join(lines))


@session.command(name="events")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def session_events(run_dir: str) -> None:
    """Print the append-only session events.jsonl (one event per line)."""
    from .session.events import events_path

    path = events_path(_session_dir(run_dir))
    if not path.exists():
        raise click.ClickException(f"no session events at {path}")
    click.echo(path.read_text().rstrip())


@main.group()
def artifacts() -> None:
    """Owner hard-flow artifact checks (AGENTS.md 硬流程)."""


@artifacts.command(name="check")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def artifacts_check(run_dir: str) -> None:
    """Verify hard-flow owner artifacts in a run dir (files, sections, gate match)."""
    from .report.artifacts import check_artifacts, format_report

    report = check_artifacts(Path(run_dir))
    click.echo(format_report(report))
    if not report.ok:
        raise SystemExit(1)


@main.group()
def recipe() -> None:
    """Read-only Recipe KB: consult validated quant recipes before a run.

    External-agent analog of Hyperloom's ``hyperloom-kb`` CLI. Exposes the pinned
    ``recipes/quant/*.json`` corpus and ``RecipeKB`` progressive-relaxation
    lookup, so an agent driving aqa with ``agent.optimize`` *off* can still
    hot-start from a validated recipe. GPU-free; no model load.
    """


def _recipe_kb():
    from .agent.recipe_kb import RecipeKB

    return RecipeKB.from_config(None)


@recipe.command(name="list")
def recipe_list() -> None:
    """List every known recipe (source, model, family, schemes)."""
    rows = _recipe_kb().entries_summary()
    if not rows:
        click.echo("no recipes found")
        return
    for r in rows:
        schemes = ",".join(r["schemes"]) if r["schemes"] else "-"
        click.echo(f"{r['source']:<7} {r['model']:<42} "
                   f"family={r['family'] or '-':<14} schemes={schemes}")


@recipe.command(name="show")
@click.argument("model")
def recipe_show(model: str) -> None:
    """Print the full pinned recipe JSON for an exact model id."""
    rec = _recipe_kb().show(model)
    if rec is None:
        raise click.ClickException(
            f"no exact-model recipe for {model!r} — try "
            f"`aqa recipe lookup {model}` for a relaxed (family) match, "
            f"or `aqa recipe list`.")
    click.echo(json.dumps(rec, ensure_ascii=False, indent=2))


@recipe.command(name="lookup")
@click.argument("model")
@click.option("--scheme", default=None, help="Requested scheme (e.g. mxfp8, W4A16).")
@click.option("--model-type", default=None,
              help="Architecture-family hint for relaxed matching (e.g. qwen3).")
@click.option("--json", "as_json", is_flag=True,
              help="Emit the machine-readable hit dict instead of the prose block.")
def recipe_lookup(model: str, scheme: str | None, model_type: str | None,
                  as_json: bool) -> None:
    """Closest validated recipe via progressive relaxation (hot-start block).

    Prints the ADVISORY hot-start context (validated command, ignore_layers,
    lessons, past failures) the agent should ADAPT to this run's paths and
    re-validate — never copy blindly. Exits non-zero on a cold start (no match).
    """
    from .agent.recipe_kb import RecipeKB

    hit = _recipe_kb().lookup(model, scheme=scheme, model_type=model_type)
    if hit is None:
        raise click.ClickException(
            f"no recipe matched {model!r}"
            + (f" (scheme={scheme})" if scheme else "")
            + " — cold start. See `aqa recipe list` for the corpus.")
    if as_json:
        click.echo(json.dumps(hit.to_dict(), ensure_ascii=False, indent=2))
    else:
        click.echo(RecipeKB.as_hot_start_context(hit))


@main.group()
def journal() -> None:
    """External-agent run journal: capture plan + reasoning + revisions as artifacts.

    The pipeline writes structured per-phase artifacts and an internal decisions
    log, but the driving agent's *narrative* — the plan, and the plan-then-revise
    history as pits are hit — otherwise lived only in the chat. ``journal add``
    persists it into the run (``journal.md`` + session ``events.jsonl``),
    append-only, so provenance travels with ``results/``. GPU-free.
    """


def _read_body(body: str | None) -> str:
    """Resolve a ``--body`` value: ``@file`` reads a file, ``-`` reads stdin."""
    if not body:
        return ""
    if body == "-":
        import sys
        return sys.stdin.read()
    if body.startswith("@"):
        return Path(body[1:]).read_text(encoding="utf-8")
    return body


@journal.command(name="add")
@click.option("--run", "-r", "run_dir", required=True,
              type=click.Path(file_okay=False),
              help="Run output dir (bootstrapped if absent; the pipeline appends here).")
@click.option("--title", "-t", required=True, help="One-line entry title.")
@click.option("--body", "-b", default="",
              help="Entry body. '@file' reads a file; '-' reads stdin.")
@click.option("--kind", "-k", type=click.Choice(list(_JOURNAL_KINDS)), default="note",
              help="plan | revise | decision | note | observation.")
@click.option("--phase", "-p", default="", help="Phase tag (e.g. ANALYZE, PLAN, QUANT).")
@click.option("--author", default="external-agent", help="Who wrote this entry.")
@click.option("--ref", default="", help="Optional reference (path, URL, prior run).")
@click.option("--supersedes", default="",
              help="For kind=revise: the entry id this one corrects (e.g. j002).")
@click.option("--decision", "as_decision", is_flag=True,
              help="Also mirror into state.json decisions[] (shows in session report).")
def journal_add(run_dir: str, title: str, body: str, kind: str, phase: str,
                author: str, ref: str, supersedes: str, as_decision: bool) -> None:
    """Append one journal entry to a run and rebuild journal.md."""
    from .session.journal import append_journal, mirror_decision

    entry = append_journal(run_dir, title=title, body=_read_body(body), kind=kind,
                           phase=phase, author=author, ref=ref, supersedes=supersedes)
    msg = f"journaled {entry['id']} ({kind}) -> {Path(run_dir) / 'journal.md'}"
    if as_decision:
        ok = mirror_decision(run_dir, phase=phase, what=title,
                             why=entry["body"] or title, ref=ref)
        msg += "  [+state.json]" if ok else "  [state.json absent — event only]"
    click.echo(msg)


@journal.command(name="show")
@click.option("--run", "-r", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False), help="Run output dir.")
def journal_show(run_dir: str) -> None:
    """Print the run journal (rebuilt from the session event stream)."""
    from .session.journal import render_markdown

    click.echo(render_markdown(run_dir).rstrip())


def _fmt(v):
    return "-" if v is None else f"{v:.4f}"


def _load_accuracy(path: str) -> dict[str, float]:
    p = Path(path)
    if p.is_dir():
        p = p / "accuracy.json"
    data = json.loads(p.read_text())
    return {k: v.get("score") for k, v in data.get("results", {}).items()}


@main.group()
def knowledge() -> None:
    """Inspect or retry local-first knowledge synchronization."""


@knowledge.command("status")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def knowledge_status(run_dir: str) -> None:
    """Print a run's knowledge_sync.json (or missing)."""
    path = Path(run_dir) / "knowledge_sync.json"
    if not path.is_file():
        click.echo(json.dumps({"status": "missing", "path": str(path)}))
        return
    click.echo(path.read_text(encoding="utf-8"))


@knowledge.command("sync")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--pull-only", is_flag=True, help="Only pull remote lessons/patches/references.")
@click.option("--push-only", is_flag=True, help="Only push local pending files.")
def knowledge_sync(run_dir: str, pull_only: bool, push_only: bool) -> None:
    """Retry this run's knowledge pull/push and refresh knowledge_sync.json."""
    run = Path(run_dir)
    config = run / "run.yaml"
    if not config.is_file():
        raise click.ClickException(f"run.yaml not found in {run}")
    cfg = load_run_config(config)
    from .agent.knowledge import KnowledgeStore
    from .agent.knowledge_sync import KnowledgeSync
    from .session._io import atomic_write_json

    store = KnowledgeStore.from_config(cfg.agent.knowledge)
    if store is None:
        raise click.ClickException("agent.knowledge is disabled in run.yaml")
    # Explicit sync must override a background/off run mode.
    sync = KnowledgeSync(store, mode="auto")
    if not push_only:
        sync.pull()
    if not pull_only:
        sync.push()
    status = sync.status()
    atomic_write_json(run / "knowledge_sync.json", status)
    manifest_path = run / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["knowledge"] = status
        atomic_write_json(manifest_path, manifest)
    click.echo(json.dumps(status, ensure_ascii=False, indent=2))


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
