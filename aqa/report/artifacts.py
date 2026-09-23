"""Owner-artifact contract check (AGENTS.md 硬流程 hard-flow section).

Deterministic validation that a run dir carries the five-step owner
artifacts: required files exist, required section headers are present,
``report.md``'s gate verdict agrees with ``gate.json``, and
``model_card.md`` is present whenever the run publishes weights
(``provider.upload.repo`` in the run yaml).  Templates live in
``templates/``; this module encodes their *contract* so a run can be
checked instead of trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
    return sorted(p for p in run_dir.iterdir()
                  if p.is_file() and p.suffix in _YAML_SUFFIXES)


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

    report_md = run_dir / "report.md"
    if report_md.exists():
        text = report_md.read_text(errors="replace")
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

    return report


def format_report(report: ArtifactsReport) -> str:
    lines = [f"run dir: {report.run_dir}",
             f"status: {'OK' if report.ok else 'INCOMPLETE'}"]
    for name in report.present:
        lines.append(f"  [ok   ] {name}")
    for f in report.findings:
        lines.append(f"  [{f.level:<5}] {f.message}")
    return "\n".join(lines)
