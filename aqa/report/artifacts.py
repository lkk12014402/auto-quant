"""Owner-artifact contract check (AGENTS.md 硬流程 hard-flow section).

Deterministic validation that a run dir carries the five-step owner
artifacts: required files exist, required section headers are present,
``report.md``'s gate verdict agrees with ``gate.json``, and
``model_card.md`` is present whenever the run publishes weights
(``provider.upload.repo`` in the run yaml) and ends with a
``Signed-off-by: Name <email>`` line.  Templates live in ``templates/``;
this module encodes their *contract* so a run can be checked instead of
trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .model_card import signoff_problem


@dataclass(frozen=True)
class ArtifactSpec:
    path: str                       # run-dir-relative file
    desc: str                       # which hard-flow step produces it
    sections: tuple[str, ...] = ()  # keywords that must appear in markdown headers
    when: str = "always"            # "always" | "upload" (weights published)


_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("analysis.md", "①结构分析与量化方案",
                 ("查阅清单", "架构判定", "量化方案", "评估计划")),
    ArtifactSpec("setup_venv.sh", "②环境构建脚本"),
    ArtifactSpec("requirements.txt", "②依赖冻结快照"),
    ArtifactSpec("journal.md", "全程叙事日志"),
    ArtifactSpec("report.md", "⑤回溯报告",
                 ("结果总览", "经验沉淀路由表")),
    ArtifactSpec("model_card.md", "④模型卡（发布权重时必填）", when="upload"),
)

_YAML_SUFFIXES = {".yaml", ".yml"}
_VERDICTS = ("GO", "NO-GO", "INCONCLUSIVE")
_VERDICT_RE = re.compile(r"\b(?:NO-GO|INCONCLUSIVE|GO)\b")
_PLACEHOLDER_RE = re.compile(r"<!--\s*[A-Z0-9_①-⑳][A-Z0-9_①-⑳ /\-]*\s*-->")
_RUN_ID_FILES = (
    "session/state.json", "session/manifest.json", "env_lock.json",
    "quant_plan.json", "analysis.json", "recipe_kb.json",
    "eval_report.json", "run_manifest.json",
)


@dataclass
class Finding:
    level: str  # "error" | "warn"
    message: str


@dataclass
class ArtifactsReport:
    run_dir: Path
    present: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)


def _missing_sections(text: str, keywords: tuple[str, ...]) -> list[str]:
    headers = "\n".join(l for l in text.splitlines() if l.startswith("#"))
    return [k for k in keywords if k not in headers]


def _run_yamls(run_dir: Path) -> list[Path]:
    """Top-level YAMLs that look like run configs, not generated recipe manifests.

    ``recipe.yaml`` and the ``<model.id>.yaml`` strict sidecar are output
    artifacts; treating them as the run config would lose ``provider.upload``
    detection and pollute the artifact report.
    """
    out: list[Path] = []
    for path in sorted(p for p in run_dir.iterdir()
                       if p.is_file() and p.suffix in _YAML_SUFFIXES):
        if path.name == "recipe.yaml":
            continue
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            out.append(path)
            continue
        if (isinstance(data, dict) and data.get("schema_version") == "1.0"
                and isinstance(data.get("variants"), list)):
            continue  # INC / aqa recipe manifest
        out.append(path)
    return out


def _publishes_weights(yamls: list[Path]) -> bool | None:
    """True if any run yaml has ``provider.upload.repo``; None if undecidable."""
    saw_config = False
    for p in yamls:
        try:
            data = yaml.safe_load(p.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        saw_config = True
        prov = data.get("provider")
        if isinstance(prov, dict):
            upload = prov.get("upload")
            if isinstance(upload, dict) and upload.get("repo"):
                return True
    return None if not saw_config else False


def _gate_verdict(run_dir: Path) -> str | None:
    try:
        verdict = json.loads((run_dir / "gate.json").read_text()).get("verdict")
    except (OSError, json.JSONDecodeError):
        return None
    return str(verdict).upper() if verdict else None


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _run_id_map(run_dir: Path) -> dict[str, str]:
    ids: dict[str, str] = {}
    for rel in _RUN_ID_FILES:
        data = _load_json(run_dir / rel)
        rid = data.get("run_id")
        if isinstance(rid, str) and rid:
            ids[rel] = rid
    recipe = run_dir / "recipe.yaml"
    if recipe.is_file():
        try:
            data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
            rid = ((data or {}).get("provenance") or {}).get("run_id")
            if isinstance(rid, str) and rid:
                ids["recipe.yaml:provenance"] = rid
        except (OSError, yaml.YAMLError):
            pass
    return ids


def _eval_limit(yamls: list[Path]) -> int | None:
    for path in yamls:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        eval_cfg = data.get("eval")
        if not isinstance(eval_cfg, dict):
            continue
        limit = eval_cfg.get("limit")
        if isinstance(limit, int) and limit > 0:
            return limit
    return None


def _routing_table_rows(text: str) -> list[list[str]] | None:
    """Data rows of the 经验沉淀路由表 section, as cell lists; None if absent."""
    m = re.search(r"^#+ [^\n]*经验沉淀路由表[^\n]*\n(.*?)(?=^#|\Z)", text,
                  re.MULTILINE | re.DOTALL)
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|") or "经验类型" in line or set(line) <= set("|-: "):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def check_artifacts(run_dir: str | Path) -> ArtifactsReport:
    """Check ``run_dir`` against the hard-flow owner-artifact contract."""
    run_dir = Path(run_dir)
    report = ArtifactsReport(run_dir=run_dir)

    yamls = _run_yamls(run_dir)
    if yamls:
        report.present.append(yamls[0].name)
    else:
        report.findings.append(Finding(
            "error", "run.yaml 缺失（③的初始配置须拷进 run 目录顶层）"))
    upload = _publishes_weights(yamls)

    for spec in _SPECS:
        path = run_dir / spec.path
        if not path.exists():
            if spec.when == "upload" and upload is not True:
                continue
            note = "" if spec.when == "always" else "（provider.upload 已配置）"
            report.findings.append(Finding(
                "error", f"{spec.path} 缺失——{spec.desc}{note}"))
            continue
        report.present.append(spec.path)
        missing = _missing_sections(path.read_text(errors="replace"), spec.sections)
        for kw in missing:
            report.findings.append(Finding(
                "error", f"{spec.path} 缺必备章节「{kw}」"))

    card = run_dir / "model_card.md"
    if card.exists():
        problem = signoff_problem(card.read_text(errors="replace"))
        if problem:
            # A draft card (no user-confirmed sign-off) is expected on
            # non-publishing runs; only a publishing run is a hard error.
            report.findings.append(Finding("error" if upload is True else "warn",
                                           problem))

    report_md = run_dir / "report.md"
    if report_md.exists():
        text = report_md.read_text(errors="replace")
        placeholders = _PLACEHOLDER_RE.findall(text)
        if placeholders:
            report.findings.append(Finding(
                "error",
                "report.md 仍有模板占位符未填：" + ", ".join(placeholders[:4])))
        verdict = _gate_verdict(run_dir)
        tokens = set(_VERDICT_RE.findall(text))
        if verdict and verdict not in tokens:
            report.findings.append(Finding(
                "error",
                f"report.md 的 gate 结论与 gate.json 不一致：gate.json={verdict}，"
                f"report.md 中未出现该结论"))
        rows = _routing_table_rows(text)
        if rows is not None:
            empty = sum(1 for r in rows if not r[-1])
            if rows and empty == len(rows):
                report.findings.append(Finding(
                    "error", "经验沉淀路由表未填写（每行须标 已落地/明确不沉淀）"))
            elif empty:
                report.findings.append(Finding(
                    "warn", f"经验沉淀路由表有 {empty} 行未填状态"))

    ids = _run_id_map(run_dir)
    if len(set(ids.values())) > 1:
        detail = ", ".join(f"{name}={rid}" for name, rid in ids.items())
        report.findings.append(Finding(
            "warn", "run_id 在产物间不一致（resume 旧 bug；新 run 已复用 state.run_id）："
                    + detail))

    limit = _eval_limit(yamls)
    if limit and (run_dir / "model_card.md").is_file():
        report.findings.append(Finding(
            "warn", f"run.yaml eval.limit={limit}（run 自己的 eval 是 smoke）；"
                    "若 card/recipe 取自独立全量 run，请核对来源后再发布"))
    recipe_yaml = run_dir / "recipe.yaml"
    if limit and recipe_yaml.is_file():
        try:
            recipe_text = recipe_yaml.read_text(errors="replace")
        except OSError:
            recipe_text = ""
        if "--limit" in recipe_text and "Smoke evaluation" not in recipe_text:
            report.findings.append(Finding(
                "warn", f"recipe.yaml 的评估仍带 --limit（run.yaml eval.limit={limit}），"
                        "但 result_tables 未标记 smoke"))

    kb = _load_json(run_dir / "recipe_kb.json")
    record = kb.get("record") if isinstance(kb.get("record"), dict) else None
    if kb and record is None:
        report.findings.append(Finding(
            "warn", "recipe_kb.json 没有 CLOSE 回写结果（旧的静默失败；"
                    "新 run 会记录 record.status）"))
    elif record:
        status = str(record.get("status") or "?")
        if status == "failed":
            report.findings.append(Finding(
                "error", "Auto KB 回写失败：" + str(record.get("error") or "unknown")))
        elif status == "fallback":
            report.findings.append(Finding(
                "warn", "Auto KB 写入 run 目录 fallback（代码库只读）；"
                        "需把 " + str(record.get("path") or "recipe_kb/auto") + " 拷回代码库"))
        elif status == "skipped":
            report.findings.append(Finding(
                "warn", "Auto KB 回写被跳过：" + str(record.get("reason") or "")))
        auto = record.get("auto_sync")
        if isinstance(auto, dict) and auto.get("status") == "failed":
            report.findings.append(Finding(
                "warn", "Recipe auto_sync 失败：" + str(auto.get("error") or "")))

    ks = _load_json(run_dir / "knowledge_sync.json")
    if ks:
        status = str(ks.get("status") or "?")
        if status == "failed":
            report.findings.append(Finding(
                "error", "knowledge_sync 失败：" + "; ".join(
                    str(e) for e in (ks.get("errors") or [])[:3])))
        elif status == "pending":
            report.findings.append(Finding(
                "warn", "knowledge_sync 有 pending 文件，下次 run/显式 sync 重试"))
        elif status == "local_only":
            report.findings.append(Finding(
                "warn", "knowledge 仅本地（未配 repo/token）；未跨机器同步"))

    card = run_dir / "model_card.md"
    if card.exists():
        card_text = card.read_text(errors="replace")
        if "The model can produce factually incorrect output" not in card_text:
            report.findings.append(Finding(
                "warn", "model_card.md 的 Ethical Considerations 段落偏离模板固定文本"))
        if "The license on this model does not constitute legal advice" not in card_text:
            report.findings.append(Finding(
                "warn", "model_card.md 的 Disclaimer 段落偏离模板固定文本"))

    return report


def format_report(report: ArtifactsReport) -> str:
    lines = [f"run dir: {report.run_dir}",
             f"status: {'OK' if report.ok else 'INCOMPLETE'}"]
    for name in report.present:
        lines.append(f"  [ok   ] {name}")
    for f in report.findings:
        lines.append(f"  [{f.level:<5}] {f.message}")
    return "\n".join(lines)
