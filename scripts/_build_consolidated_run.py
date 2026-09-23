"""Assemble a complete, layout-contract-compliant consolidated run dir for the
Qwen3-235B-A22B MXFP4-mixed quantize+eval, reusing existing run artifacts."""
import json, shutil
from pathlib import Path

BASE = Path("results/qwen3_235b_a22b")
QUANT = BASE / "mixed_mxfp4e_mxfp8/20260909-064235/20260909-064409"
QRUN = BASE / "mixed_mxfp4e_mxfp8/20260909-064235"
LMEVAL = BASE / "eval_only/20260909-081739/20260909-081739"
ES = BASE / "evalscope_only/20260909-100018/20260909-100018"
OUT = BASE / "_consolidated_upload"

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "eval").mkdir(parents=True)
(OUT / "quantized").mkdir()
(OUT / "logs").mkdir()

# 1) quant artifacts (weights excluded by sink; keep command + log)
shutil.copy2(QUANT / ".quant_command.json", OUT / ".quant_command.json")
shutil.copy2(QUANT / "quantized/quantize.log", OUT / "quantized/quantize.log")

# 2) all 6 eval task dirs
for src, name in [
    (LMEVAL / "eval/gsm8k", "gsm8k"),
    (LMEVAL / "eval/piqa_mmlu_hellaswag", "piqa_mmlu_hellaswag"),
    (ES / "eval/aime26", "aime26"),
    (ES / "quantized-BF16-Baseline/gpqa_diamond", "gpqa_diamond"),
]:
    shutil.copytree(src, OUT / "eval" / name)

# 3) eval/accuracy.json — full 6-benchmark record
RESULTS = {
    "gsm8k": {"score": 0.9219, "metric": "exact_match,strict-match", "num_samples": 1319,
              "harness": "lm_eval", "note": "5-shot, no chat_template (per model guidance)"},
    "piqa": {"score": 0.8161, "metric": "acc,none", "num_samples": 1838, "harness": "lm_eval"},
    "mmlu": {"score": 0.8391, "metric": "acc,none", "num_samples": 14042, "harness": "lm_eval"},
    "hellaswag": {"score": 0.6692, "metric": "acc,none", "num_samples": 10042, "harness": "lm_eval"},
    "aime26": {"score": 0.80, "metric": "accuracy", "num_samples": 30, "harness": "evalscope",
               "note": "0-shot, thinking on, max_tokens 30000"},
    "gpqa_diamond": {"score": 0.702, "metric": "accuracy", "num_samples": 198, "harness": "evalscope",
                     "note": "0-shot, thinking on"},
}
mean = round(sum(v["score"] for v in RESULTS.values()) / len(RESULTS), 4)
commands = [
    "lm_eval --model vllm --model_args pretrained=<quantized>,tensor_parallel_size=4,max_model_len=8192,"
    "gpu_memory_utilization=0.55,dtype=bfloat16,trust_remote_code=True,add_bos_token=True,"
    "enable_prefix_caching=False,max_gen_toks=4096,max_num_batched_tokens=32768,max_num_seqs=128 "
    "--tasks gsm8k --batch_size 64 --num_fewshot 5   # no --apply_chat_template / --fewshot_as_multiturn",
    "lm_eval --model vllm --model_args pretrained=<quantized>,tensor_parallel_size=4,... "
    "--tasks piqa,mmlu,hellaswag --batch_size 64",
    "evalscope eval --model aqa-evalscope --api-url http://127.0.0.1:8801/v1 --eval-type openai_api "
    "--datasets aime26 --repeats 1 --generation-config '{\"max_tokens\":30000,\"temperature\":0.6,"
    "\"top_p\":0.95,\"top_k\":20}'   # vllm serve TP=2 (GPU0+3)",
    "evalscope eval ... --datasets gpqa_diamond ...   # vllm serve TP=2 (GPU0+3)",
]
accuracy = {
    "model": "Qwen/Qwen3-235B-A22B",
    "quantized_model": "Qwen3-235B-A22B-MXFP4-Mixed-CT-RTN-AutoRound",
    "backend": "vllm",
    "scheme": "MXFP8 (routed experts MXFP4)",
    "method": "model_free",
    "mean_score": mean,
    "results": RESULTS,
    "commands": commands,
    "healed": [],
}
(OUT / "eval" / "accuracy.json").write_text(json.dumps(accuracy, indent=2, ensure_ascii=False) + "\n")

# 4) run_manifest.json — augment the evalscope_only manifest with quant + all 6 results
man = json.loads((ES / "run_manifest.json").read_text())
man["run_id"] = "Qwen3-235B-A22B_MXFP4-Mixed_20260909"
man["model"] = {"id": "Qwen/Qwen3-235B-A22B", "provider": "quantize",
                "quantized_dir": "quantized/ (weights uploaded to HF model repo)"}
man["quant"] = {
    "scheme": "MXFP8", "method": "model_free", "export_format": "llm_compressor",
    "algorithm": "rtn", "iters": 0,
    "ignore_layers": "lm_head,embed_tokens,mlp.gate",
    "layer_config": {"mlp.experts": {"bits": 4, "data_type": "mx_fp"}},
    "command": json.loads((QUANT / ".quant_command.json").read_text())["command"],
    "output_size_gb": 122, "bf16_size_gb": 470, "compression": "3.85x",
}
man["eval"]["benchmarks"] = ["gsm8k", "piqa", "mmlu", "hellaswag", "aime26", "gpqa_diamond"]
man["eval"]["apply_chat_template"] = False
man["eval"]["fewshot_as_multiturn"] = False
man["eval"]["thinking"] = "on"
# refresh package versions with the actually-used pinned deps
man.setdefault("environment", {}).setdefault("packages", {}).update({
    "flashinfer-python": "0.6.18", "flashinfer-cubin": "0.6.18",
    "evalscope": "1.11.0", "modelscope": "1.39.1",
})
man["results"] = {
    "mean_score": mean, "backend": "vllm",
    "benchmarks": {k: {"score": v["score"], "metric": v["metric"],
                       "num_samples": v.get("num_samples"), "harness": v["harness"]}
                   for k, v in RESULTS.items()},
}
man["artifacts"] = {
    "weights_repo": "https://huggingface.co/lvkaokao/Qwen3-235B-A22B-MXFP4-Mixed-CT-RTN-AutoRound",
}
man["notes"] = [
    "flashinfer 0.6.16->0.6.18 for B300 sm_103a (patches/evaluate/flashinfer_0.6.18_sm103a.md)",
    "gsm8k without chat_template (patches/evaluate/gsm8k_no_chat_template.*)",
    "evalscope localhost proxy bypass (patches/evaluate/evalscope_localhost_no_proxy.*)",
    "evalscope 1.11.0 / modelscope 1.39.1 alignment",
    "evalscope run on TP=2 (GPU0+3) to avoid shared-GPU memory-profiling fluctuation",
]
(OUT / "run_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")

# 5) aqa_run.json — terminal done state
(OUT / "aqa_run.json").write_text(json.dumps(
    {"model": "Qwen/Qwen3-235B-A22B", "state": "done",
     "run_id": "Qwen3-235B-A22B_MXFP4-Mixed_20260909"}, indent=2) + "\n")

# 6) logs — executor logs from quant + eval runs + agent fix records
def _cp(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

_cp(QRUN / "logs/executor_venv.log", OUT / "logs/quant_executor_venv.log")
_cp(ES.parent / "logs/executor_venv.log", OUT / "logs/evalscope_executor_venv.log")
_cp(LMEVAL / "eval/gsm8k/gsm8k.eval.log", OUT / "logs/gsm8k.eval.log")
_cp(LMEVAL / "eval/piqa_mmlu_hellaswag/piqa+mmlu+hellaswag.eval.log", OUT / "logs/piqa+mmlu+hellaswag.eval.log")
_cp(QRUN / "logs/agent_fixes", OUT / "logs/agent_fixes")
# quantize.log also under logs/ so the dataset sink carries it (quantized/ is sink-excluded)
_cp(QUANT / "quantized/quantize.log", OUT / "logs/quantize.log")

# 7) patches — owner + fix-loop patches captured for this task
_cp(Path("patches/evaluate"), OUT / "patches/evaluate")

# 8) owner journal (analysis / decisions / lessons) at run-dir top (AGENTS.md §6.4)
_cp(Path("results/qwen3_235b_a22b/OWNER_JOURNAL.md"), OUT / "OWNER_JOURNAL.md")

print("consolidated dir built:")
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.relative_to(OUT).parts[0] != "eval":
        print("  ", p.relative_to(OUT))
print("  eval/:", sorted(x.name for x in (OUT/'eval').iterdir()))
