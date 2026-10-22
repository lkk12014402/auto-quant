from pathlib import Path

from aqa.report.sinks import LocalFsSink, RunMeta


def test_local_fs_sink_does_not_copy_its_own_destination(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "analysis.md").write_text("analysis")
    meta = RunMeta(model="org/model", artifact_name="artifact")
    sink = LocalFsSink(path=str(run_dir))

    sink.upload(run_dir, meta)
    sink.upload(run_dir, meta)

    mirror = run_dir / "artifact"
    assert (mirror / "analysis.md").read_text() == "analysis"
    assert not (mirror / "artifact").exists()


def test_local_fs_sink_excludes_nested_destination_without_skipping_siblings(
        tmp_path: Path):
    run_dir = tmp_path / "run"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "keep.txt").write_text("keep")
    meta = RunMeta(model="org/model", artifact_name="artifact")
    sink = LocalFsSink(path=str(results_dir))

    sink.upload(run_dir, meta)

    mirror = results_dir / "artifact"
    assert (mirror / "results" / "keep.txt").read_text() == "keep"
    assert not (mirror / "results" / "artifact").exists()
