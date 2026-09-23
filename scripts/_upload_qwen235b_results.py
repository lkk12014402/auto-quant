"""One-off: upload consolidated Qwen3-235B-A22B eval results to the lb_local dataset
under the canonical results/Qwen/<artifact>/ path."""
import os
from pathlib import Path
from aqa.report.sinks import HfDatasetSink, RunMeta

RUN_ROOT = Path("results/qwen3_235b_a22b/_consolidated_upload")
RESULTS = {
    "gsm8k": 0.9219, "piqa": 0.8161, "mmlu": 0.8391, "hellaswag": 0.6692,
    "aime26": 0.80, "gpqa_diamond": 0.702,
}

sink = HfDatasetSink(
    repo=os.environ.get("LOCAL_RESULTS_DATASET", "lvkaokao/lb_local"),
    token_env="LOCAL_RESULTS_HF_TOKEN",
    disable_xet=True,
)
meta = RunMeta(
    model="Qwen/Qwen3-235B-A22B",
    scheme="MXFP8",
    method="model_free",
    artifact_name="Qwen3-235B-A22B-MXFP4-Mixed-CT-RTN-AutoRound",
    run_root=RUN_ROOT,
    run_id="Qwen3-235B-A22B_MXFP4-Mixed_20260909",
    status="Finished",
    extra={"results": RESULTS},
)
url = sink.upload(RUN_ROOT, meta)
print("DATASET_UPLOADED:", url)
