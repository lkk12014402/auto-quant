"""One-off: upload the Qwen3-235B-A22B MXFP4-mixed quantized weights to HF model repo."""
import os
from pathlib import Path
from aqa.report.sinks import HfModelSink, RunMeta

MODEL_DIR = "/models/auto-quant/results/qwen3_235b_a22b/mixed_mxfp4e_mxfp8/20260909-064235/20260909-064409/quantized"
RESULTS = {
    "gsm8k": 0.9219, "piqa": 0.8161, "mmlu": 0.8391, "hellaswag": 0.6692,
    "aime26": 0.80, "gpqa_diamond": 0.702,
}

sink = HfModelSink(
    repo="lvkaokao",
    repo_name="Qwen3-235B-A22B-MXFP4-Mixed-CT-RTN-AutoRound",
    token_env="LOCAL_RESULTS_HF_TOKEN",
    disable_xet=True,
)
meta = RunMeta(
    model="Qwen/Qwen3-235B-A22B",
    scheme="MXFP8",
    method="model_free",
    artifact_name="Qwen3-235B-A22B-MXFP4-Mixed-CT-RTN-AutoRound",
    extra={"results": RESULTS},
)
url = sink.upload(Path(MODEL_DIR), meta)
print("WEIGHTS_UPLOADED:", url)
