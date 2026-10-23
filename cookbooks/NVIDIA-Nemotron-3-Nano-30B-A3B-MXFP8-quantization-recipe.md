# NVIDIA Nemotron 3 Nano 30B A3B BF16 → MXFP8 + static FP8 KV

Validated on 2026-10-14 from the local BF16 checkpoint
`/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. The approved construction card
contains the investigation and run history:
[`NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-施工卡.md`](NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-施工卡.md).
Full artifacts are local at
`/home/xuehaosu/aqa_results/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-AutoRound_20261013`;
the quantization run itself did not upload weights or results and did not commit
repository changes. Afterward, the user explicitly authorized changing the
model repository to public. The quantized model directory and English
`README.md` were uploaded to
`intel-ai/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-FP8KV-CT-AutoRound` in
commit `68348e84fa9cb86dde6e20eb1c364e0aacff0317`. Remote inventory and sizes
match the 18 local files (33,647,253,181 bytes); all eight LFS objects match
local SHA-256 values, other files match Git blob IDs, and the public README is
byte-identical to the local copy. The original private upload attempt was
blocked by the organization's storage quota before the user authorized public
visibility.

## Follow-on variant: pure MXFP8 weights with BF16 KV

A separate local run on 2026-10-14 quantized the same BF16 checkpoint to MXFP8
without static FP8 KV or attention scales. The approved procedure and acceptance
criteria are in
[`NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-施工卡.md`](NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-施工卡.md);
full evidence is retained at
`/home/xuehaosu/aqa_results/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP8-AutoRound_20261014`.
This follow-on was not uploaded to Hugging Face; the original sections below
continue to describe the separate MXFP8 + static FP8 KV variant.

The new export passed the header audit: 5,968 `F8_E4M3` weights with 5,968
U8 group-32 scales, 206 retained BF16 tensors and 69 retained F32 tensors.
All 6,243 source parameter keys are represented, the seven indexed shards are
complete, and no KV/attention scales were exported. The tensor payload is
33,616,949,632 bytes. vLLM 0.31.0 loaded the native architecture with
`trust_remote_code=false`, Humming MXFP8 MoE and
`FlashInferCutlassMxfp8LinearKernel`; it used `kv_cache_dtype=auto` with a BF16
model dtype and `max_model_len=131072`.

| Task | BF16 baseline | Pure MXFP8 | Δ vs BF16 | MXFP8 + FP8 KV* | MXFP8 + FP8 KV + FP8 Attn* |
|---|---:|---:|---:|---:|---:|
| GSM8K (strict-match) | 0.948446 | 0.946171 | -0.2274 pp | 0.950720 | 0.952995 |
| PIQA (acc) | 0.805223 | 0.811752 | +0.6529 pp | 0.808487 | 0.810120 |
| MMLU (acc) | 0.712149 | 0.710013 | -0.2136 pp | 0.710084 | 0.709728 |
| HellaSwag (acc) | 0.598188 | 0.598287 | +0.0100 pp | 0.595897 | 0.597291 |
| **Unweighted mean** | **0.766001** | **0.766556** | **+0.0554 pp** | **0.766297** | **0.767533** |

The four-task candidate used batch size 32 and passed the `max_accuracy_drop=0.02`
gate; its largest BF16-relative drop was 0.002274 (0.2274 pp). The BF16
baseline used TP4×DP1+EP, while the candidate used TP2×DP2+EP; the prior FP8
KV variants also differ in KV/attention protocol and topology. These results
are protocol references, not causal ablations. The 128K model-length cap was
configured, but the four tasks are not long-context stress tests. This
checkout does not apply `eval.thinking` to lm_eval, so the model template
default applies.

The canonical `aqa run` quantization/export succeeded, but its embedded GSM8K
evaluation failed because no vLLM server was listening on `127.0.0.1:8000`.
The full four-task evaluation then completed with the staged external vLLM
runner and `aqa eval`; `aqa_run.json` correctly remains `failed`, and the
initial failure manifest is preserved in the local run directory.

## Model and precision plan

The 31.58B-parameter `nemotron_h` checkpoint has 52 hybrid layers: Mamba2,
routed/shared MoE, and full attention at layers 5, 12, 19, 26, 33, and 42.
Quantize routed experts, shared experts, and Mamba `in_proj`/`out_proj` outside
the attention-adjacent layers. Preserve attention, router, embeddings/head, and
the adjacent Mamba projections at layers 4, 11, 18, 25, 32, and 41.

The validated AutoRound command is recorded verbatim in the run's
`.quant_command.json`. Its core flags are:

```bash
AR_DISABLE_META_LOAD=1 auto-round \
  --model_name /models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --scheme MXFP8 --algorithm rtn --iters 0 \
  --static_kv_dtype fp8 \
  --ignore_layers 'lm_head,backbone.embeddings,mixer.q_proj,mixer.k_proj,mixer.v_proj,mixer.o_proj,mixer.gate,mixer.conv1d,model.layers.4.mixer.in_proj,model.layers.4.mixer.out_proj,model.layers.11.mixer.in_proj,model.layers.11.mixer.out_proj,model.layers.18.mixer.in_proj,model.layers.18.mixer.out_proj,model.layers.25.mixer.in_proj,model.layers.25.mixer.out_proj,model.layers.32.mixer.in_proj,model.layers.32.mixer.out_proj,model.layers.41.mixer.in_proj,model.layers.41.mixer.out_proj' \
  --device_map 0,1 --format llm_compressor --output_dir <output-dir> \
  --seqlen 2048 --batch_size 1 --nsamples 128 --disable_trust_remote_code
```

Use the repository AQA YAML flow, not a hand-built quantization run. The
canonical YAML and environment files are kept with the local run (`run.yaml`,
`setup_venv.sh`, `requirements.txt`). `AR_DISABLE_META_LOAD=1` was needed
because AutoRound 0.16.0's fused-MoE meta loader left the new K/V scale
parameters unmaterialized. The default Mamba2 reference fallback attempted a
64 GiB allocation; `seqlen=2048`, `batch_size=1` completed on the selected
quantization GPUs. Runtime module names use `model.layers.*` even though the
checkpoint index uses `backbone.layers.*`; verify exported dtypes rather than
trusting the ignore string.

The export audit passed: all 6,243 source parameter keys are present;
5,968 `F8_E4M3` target weights have matching U8 group-32 scales; 12 static
K/V scales are finite, nonzero F32; the six attention layers and 12 protected
adjacent Mamba projections retain the planned dtypes. The indexed tensor
payload is 33,616,949,680 bytes (about 31.32 GiB), in line with the 33.6 GB
estimate. The audit JSON is
`logs/quantized_header_audit_attempt05.json` in the local run.

## Inference and evaluation

Use native `NemotronHForCausalLM` with `trust_remote_code=false`. Verified stack:
Conda Python 3.12.15, uv 0.13.0, AutoRound 0.16.0, torch 2.13.0/CUDA 13.0,
Transformers 5.17.0, vLLM 0.31.0, and lm-eval 0.4.13.

The BF16 baseline used TP4 × DP1 + EP, PP=1, automatic/BF16 KV cache. The
quantized candidate used TP2 × DP2 + EP, PP=1, explicit Humming MoE, MXFP8
linear kernels, static FP8 KV cache, and `max_model_len=131072`. The automatic
Marlin selection failed on RTX 5090/SM120 with an invalid thread configuration;
explicit Humming passed both the generation smoke and full four-task run.

Both sides used batch size 32, the same tokenizer/template/seed, 128K maximum
context setting, and 2,048-token generation cap. Counts are GSM8K 1,319, PIQA
1,838, MMLU 14,042, and HellaSwag 10,042.

| Task (metric) | BF16 baseline | MXFP8 + FP8 KV | Δ | Candidate / baseline |
|---|---:|---:|---:|---:|
| GSM8K (strict-match) | 0.948446 | 0.950720 | +0.227 pp | 100.2398% |
| PIQA (acc) | 0.805223 | 0.808487 | +0.326 pp | 100.4054% |
| MMLU (acc) | 0.712149 | 0.710084 | −0.207 pp | 99.7100% |
| HellaSwag (acc) | 0.598188 | 0.595897 | −0.229 pp | 99.6171% |
| **Unweighted mean** | **0.766001** | **0.766297** | **+0.030 pp** | **100.0386%** |

The configured `max_accuracy_drop=0.02` gate returned **GO**. This is a
deployment-protocol comparison, not an isolated weight-quantization ablation:
the candidate uses FP8 KV while the BF16 baseline uses automatic/BF16 KV, and
the two validated server topologies differ. `max_model_len=131072` was
configured, but these four tasks were not 128K-context stress tests. No
throughput, thinking-off, multimodal, or broader benchmark evaluation was run.

GSM8K logs contain two rows per `doc_id` because `lm_eval` records both
`strict-match` and `flexible-extract`; the generations were identical and were
deduplicated for the sample audit. There were no empty responses. The literal
`Question:` marker appeared in 19 baseline and 16 candidate unique responses;
sampled examples show it in same-question reasoning followed by a final answer,
so it is not an API stop reason. Re-encoding decoded responses found 12 and 11
responses at or above 2,048 tokens, respectively; this is only a conservative
cap indicator, not the server's `finish_reason`.

## Reproduction and caveats

- Local run YAML, environment script, frozen dependencies, runner, raw stdout,
  per-task logs, baseline/candidate accuracy, sample audit, gate, and manifest
  are retained under the run directory above. The staged server/eval runner is
  `logs/run_external_eval.sh`.
- This checkout's `aqa run` cannot restart an external vLLM service between
  phases. The initial full-pipeline evaluation attempt is retained as failed in
  `aqa_run.json`; the quantized model's independent vLLM smoke and the staged
  `aqa eval` baseline/candidate evaluations completed successfully. Do not
  describe the original `aqa run` as end-to-end successful.
- The references workflow is not wired in this checkout. No HF dataset or
  knowledge/trajectory upload was configured. The model weights were later
  published to the public model repository after explicit user authorization.
  The runtime MoE smoke requirement is also recorded in
  `AGENTS.md` and `skills/evaluate/SKILL.md`; the static-KV meta-loading
  workaround is recorded in `skills/quantize/SKILL.md`.
