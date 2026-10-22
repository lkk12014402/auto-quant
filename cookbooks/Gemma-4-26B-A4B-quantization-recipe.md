# Gemma-4-26B-A4B MXFP8 + static FP8 KV recipe

Validated on 2026-09-28 with `google/gemma-4-26B-A4B` from
`/models/gemma-4-26B-A4B`. This is an AQA-run recipe; do not invoke AutoRound or
lm_eval as a separate hand-built run. The completed evidence is in
`/home/xuehaosu/aqa_results/Gemma-4-26B-A4B-MXFP8-FP8KV-AutoRound_20260927-180251`
(run id `gemma-4-26B-A4B_MXFP8_20260927-220842_a0c394`).

## Model facts and precision plan

The checkpoint has 30 text layers, 128 routed experts per layer (top-8), 25
sliding-attention layers and 5 full-attention layers. The two fused expert
matrices per layer are expanded to per-expert gate/up/down weights by
AutoRound. Preserve `router`, the shared `mlp` branch, vision tower, vision
embedding projection, and token embeddings; quantize the routed experts and
the self-attention projections present in the checkpoint.

Validated ignore list:

```text
router,vision_tower,embed_vision,embed_tokens,mlp
```

`multi_modal_projector` is not a Gemma-4-26B-A4B checkpoint key and must not be
used as an ignore pattern. The five full-attention layers (indices 5, 11, 17,
23, 29) do not contain `v_proj`; the exported MXFP8 tensors cover all 115
self-attention projection weights that actually exist in the checkpoint.
AutoRound warns that `embed_tokens` does not match a supported quantizable
module; this is benign because the embedding is not a `Linear` target. Its
checkpoint/export key exists and remains BF16.

## Verified precision allocation

Counts below are from the source safetensors index and exported tensor headers.
Weight-element counts exclude U8 block scales and F32 KV scales.

| Module / tensor family | Export precision | Tensor count | Weight elements | Decision / evidence |
|---|---|---:|---:|---|
| Routed expert gate/up/down weights | MXFP8 (`F8_E4M3`) + U8 block scales | 11,520 | 22,837,985,280 | 30 layers × 128 experts; fused checkpoint tensors expanded by AutoRound |
| Existing self-attention projections | MXFP8 (`F8_E4M3`) + U8 block scales | 115 | 1,110,179,840 | q/k/o in all layers; v in 25 sliding-attention layers; five full-attention layers have no source `v_proj` |
| Shared MLP branch | BF16 | 90 | 535,265,280 | Always-on branch; explicitly ignored |
| Router | BF16 | 90 | 10,901,760 | Explicitly ignored |
| Vision tower | BF16 | 355 | 569,550,384 | Explicitly ignored; text-only evaluation |
| Vision embedding projection | BF16 | 1 | 3,244,032 | Explicitly ignored |
| Token embedding | BF16 | 1 | 738,197,504 | Remains BF16; not an AutoRound `Linear` target |
| Other retained tensors (norms, patch/position tensors, etc.) | BF16 | 301 | 612,126 | Retained by model-free defaults |
| **Stored model weights total** | **F8_E4M3 + BF16** | **12,473** | **25,805,936,206** | **23,948,165,120 F8_E4M3 + 1,857,771,086 BF16 elements** |
| Block scales (metadata; not model weights) | U8 | 11,635 | — | One exported scale tensor per F8_E4M3 weight tensor |
| Static KV scales (metadata; not model weights) | F32 | 114 | — | 60 text K/V scales finite and nonzero; 54 vision scales zero after text-only calibration |

The source index contains 1,013 BF16 tensors and `total_size=51,611,872,412`
bytes, i.e. 25,805,936,206 unique serialized weight elements. The model config
sets `tie_word_embeddings: true`; keep this serialized-index count separate
from the recipe pin's nominal `total_params_B: 26.544` figure. For quantization
sanity checks, compare stored tensor elements/dtypes and file size, not the
nominal model count.

## Environment and reproducible AQA run

Use the verified `autoround-test` Conda environment without changing package
versions. Before the run, inspect GPU ownership with `nvidia-smi`; choose exactly
three idle physical GPUs, and wait if any selected card is busy. The YAML uses
logical device map `0,1,2` and vLLM TP=1/PP=3. The successful run used physical
GPU 1, 2 and 3; GPU 0 was deliberately excluded.

Verified run parameters:

| Setting | Value | Notes |
|---|---|---|
| Model | `/models/gemma-4-26B-A4B` | Local BF16 checkpoint |
| AutoRound method / scheme | `model_free` / `MXFP8` | RTN path; no tuning `iters` |
| MXFP8 format | 8-bit `mx_fp`, group size 32 | From the repository MXFP8 scheme |
| Export | `llm_compressor` | Compressed-tensors MXFP8 checkpoint |
| Static KV quantization | `static_kv_fp8: true` → `--static_kv_dtype fp8` | Static FP8 K/V scale calibration |
| Ignore layers | `router,vision_tower,embed_vision,embed_tokens,mlp` | Protect sensitive/shared modules |
| Quantization device map | `0,1,2` | Logical IDs map to three selected physical GPUs |
| Required environment | `AR_DISABLE_META_LOAD=1` | Required for this AutoRound/Transformers stack to materialize new K/V scales |
| Evaluation tasks | PIQA, MMLU, HellaSwag, GSM8K | Repository `lm_eval` task recipes |
| Evaluation KV / context | `kv_cache_dtype=fp8`, `max_model_len=131072` | Applied to both baseline and quantized evaluation |
| vLLM parallelism | TP=1 / PP=3; `language_model_only=true` | Three selected physical GPUs; text-only workload |

`model_free` avoids tuning iterations, but `static_kv_fp8` still loads the
checkpoint and runs calibration forwards to produce KV scales.

The run YAML must retain these settings:

```yaml
quant:
  scheme: MXFP8
  method: model_free
  export_format: llm_compressor
  static_kv_fp8: true
  ignore_layers: "router,vision_tower,embed_vision,embed_tokens,mlp"
  device_map: "0,1,2"

eval:
  backend: vllm
  harness: lm_eval
  benchmarks: [piqa, mmlu, hellaswag, gsm8k]
  baseline: true
  thinking: "off"
  lm_eval:
    apply_chat_template: false
    fewshot_as_multiturn: false
    model_args:
      kv_cache_dtype: fp8
      max_model_len: 131072
      tensor_parallel_size: 1
      pipeline_parallel_size: 3
      language_model_only: true
```

Create a fresh run directory/config rather than reusing the completed output
path. After confirming the physical GPU IDs are idle, activate the verified
environment and invoke the wrapper:

```bash
conda activate autoround-test
export CUDA_VISIBLE_DEVICES=1,2,3  # replace only with three verified idle physical IDs
bash cookbooks/quant_gemma-4-26B-A4B.sh /path/to/fresh/run.yaml
```

The wrapper validates and dry-runs the AQA YAML before starting the pipeline.
It rejects a missing/busy GPU selection and exports `AR_DISABLE_META_LOAD=1`.
This environment override is required by the verified AutoRound/Transformers
combination described below. The successful pipeline command was recorded in
the run's `.quant_command.json`, manifest and `owner_actions.jsonl`.

## Static KV and meta-loading pitfall

The source BF16 checkpoint does not contain `self_attn.k_scale` /
`self_attn.v_scale`. AutoRound 0.16.0.dev193 with Transformers 5.17.0 initially
used its fused-MoE meta skeleton and failed materializing those newly added
parameters. The AutoRound error recommends `AR_DISABLE_META_LOAD=1`; this
disables that meta path and lets Transformers initialize the added scales while
loading the model through CPU. On the validated host the peak RAM was 64.14 GiB
(188 GiB installed), and the retry completed without changing the requested
precision or KV mode.

The successful AutoRound invocation includes `--static_kv_dtype fp8`. The
exported `quantization_config.json` records an 8-bit, static (`dynamic: false`)
float `kv_cache_scheme`; all 60 language-layer K/V scales (30 layers × K/V) are
finite and nonzero. The 54 vision K/V scales are zero because calibration and
evaluation were text-only (`language_model_only: true`); the vision tower is
preserved in BF16 and is outside the evaluated workload.

## Verified output and evaluation

The exported checkpoint is
`quantized/gemma-4-26B-A4B-mxfp-w8g32/`. A fresh file-size audit found the six
`.safetensors` shards total 28,415,413,660 bytes (28.42 GB / 26.46 GiB), and
the full directory totals 28,450,171,113 bytes (28.45 GB / 26.50 GiB; about
1.82× smaller than the 51.61 GB BF16 source). A
metadata-only header audit found:

- 11,635 `F8_E4M3` weight tensors and 11,635 `U8` block-scale tensors, including
  all 11,520 routed-expert weight tensors and all 115 self-attention projection
  tensors.
- 838 BF16 tensors, including protected shared/vision modules, and 114 F32
  static K/V scales.
- 953 original non-fused source tensor keys preserved by name; the 60 fused
  expert keys are expanded to per-expert tensors. No other source keys were
  absent from the exported index.

The baseline and quantized checkpoint used the same vLLM FP8 KV cache, 131,072
maximum model length, TP=1/PP=3 and repository benchmark recipes. A 128K
`max_model_len` was configured, but these standard benchmark prompts are not a
128K long-context stress test.

| Benchmark (AQA primary metric) | BF16 baseline | MXFP8 | Δ (percentage points) |
|---|---:|---:|---:|
| PIQA (`acc`) | 0.8226 | 0.8286 | +0.60 |
| MMLU (`acc`) | 0.7434 | 0.7441 | +0.07 |
| HellaSwag (`acc`) | 0.6325 | 0.6347 | +0.22 |
| GSM8K (`exact_match,strict-match`) | 0.7180 | 0.7324 | +1.44 |
| Mean | 0.7291 | 0.7350 | +0.58 |

`gate.json` is **GO** (`max_accuracy_drop: 0.02`; all gate checks passed).
HellaSwag and PIQA `acc_norm` are also recorded in the lm_eval logs; the AQA
gate/report comparison uses the primary `acc` scores above.

The official/local Gemma-4 tokenizer configuration has no chat template, and
`chat_template.jinja` is absent. Therefore GSM8K uses the repository recipe's
bare-prompt path (`apply_chat_template: false`,
`fewshot_as_multiturn: false`) while retaining 5-shot, seed 42 and the 2,048
generation-token budget. A repeat BF16 baseline in a prior failed attempt had
GSM8K strict-match 0.7286 instead of 0.7180; the run-to-run difference is
preserved in `logs/retry-03/`. Gate comparisons use the baseline and quantized
scores from the same successful run, without attributing an unverified cause
to the repeat difference.

## RULER 131,072-token evaluation

Completed 2026-10-10 with the BF16 source and the existing MXFP8 export. Each
model was evaluated on the full 13-task lm_eval RULER group with `--limit 100`,
seed 42, and the native `131072,none` score metric. Every task has 100 effective
samples (lm_eval constructs 500 synthetic examples per task before applying
the limit). Both runs used `max_model_len=131072`, the local Gemma-4 tokenizer,
tokenized prompt IDs with an explicit BOS, TP=2/DP=2/EP enabled/PP=1 on
same-NUMA physical GPUs 0–3, and left GPUs 4–7 idle. The vLLM server used
`max_num_seqs=1` per DP worker; lm_eval used `batch_size=2` and
`num_concurrent=2`.

This is a deployment-protocol comparison, not a weights-only ablation:
BF16 used `kv_cache_dtype=auto`; MXFP8 used the export's calibrated static FP8
KV scales with `kv_cache_dtype=fp8`. The tokenized client caps the effective
API context at 131,072 and left-truncates any excess RULER `gen_prefix` tokens
at the boundary. Scores are task metrics, not a formal AQA gate; the RULER
evaluation-only runs have no `gate.json`.

| RULER task (`131072,none`) | BF16 | MXFP8 + FP8 KV | Δ (percentage points) |
|---|---:|---:|---:|
| niah_single_1 | 1.0000 | 1.0000 | +0.00 |
| niah_single_2 | 1.0000 | 1.0000 | +0.00 |
| niah_single_3 | 1.0000 | 1.0000 | +0.00 |
| niah_multikey_1 | 0.9700 | 0.9500 | -2.00 |
| niah_multikey_2 | 0.9900 | 0.9800 | -1.00 |
| niah_multikey_3 | 1.0000 | 0.9800 | -2.00 |
| niah_multiquery | 0.9775 | 0.9600 | -1.75 |
| niah_multivalue | 0.9875 | 0.9750 | -1.25 |
| ruler_vt | 0.9880 | 0.9920 | +0.40 |
| ruler_cwe | 0.8280 | 0.8010 | -2.70 |
| ruler_fwe | 0.9800 | 0.943333 | -3.67 |
| ruler_qa_squad | 0.518333 | 0.5150 | -0.33 |
| ruler_qa_hotpot | 0.5500 | 0.5400 | -1.00 |
| **Unweighted 13-task mean** | **0.906872** | **0.895103** | **-1.18** |

The largest observed drops are `ruler_fwe` (-3.67 pp) and `ruler_cwe`
(-2.70 pp); `ruler_vt` is +0.40 pp. The mean is the arithmetic mean of the
13 task scores and is not a sample-weighted aggregate. The RULER results do
not change the original four-task `gate.json` verdict or establish a
weights-only quantization effect.

## Scope and provenance

The 131,072-token RULER bucket is now measured, but this is not a generic
exact-length saturation test across arbitrary prompt layouts. Still not
measured: multimodal/vision quality, thinking-on mode, agentic tasks,
throughput, or latency. RULER run evidence is in
`/home/xuehaosu/aqa_results/Gemma-4-26B-A4B-BF16-RULER-128K_20261009-1600`
and
`/home/xuehaosu/aqa_results/Gemma-4-26B-A4B-MXFP8-FP8KV-RULER-128K_20261009-1425`;
the original quantization run remains at the path noted above. No weights were
uploaded and no Git commit was made.
