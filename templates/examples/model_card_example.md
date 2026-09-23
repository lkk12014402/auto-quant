---
license: mit
license_name: mit
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
tags:
- quantized
- mxfp8
- compressed-tensors
- auto-round
- vllm
- glm5_next
- moe
- multimodal
---

# GLM-5.3-Flash-MXFP8-Plus-AutoRound

MXFP8 (OCP Microscaling, 8-bit block FP with an E8M0 scale per 32 elements) quantized checkpoint of
**GLM-5.3-Flash** — a 321 B-parameter, ~18 B-active natively multimodal MoE model — produced with
[Intel AutoRound](https://github.com/intel/auto-round) in **model-free RTN** mode (no calibration
dataset, no model load), exported as **compressed-tensors** (`format: mxfp8-quantized`).

All **37 287** quantized layers are MXFP8 W8A8: every routed expert matrix (37 152), every shared
expert (129), the dense MLPs of layers 1-2 (6), and nothing else. All attention — both the 34 KDA
linear-attention layers and the 12 sparse-MLA layers — stays BF16, as do embeddings, `lm_head`, the
vision tower, hyper-connections, the MoE routers, the DSA indexer and the MTP structural pieces.

**This is the evaluated, vLLM-validated checkpoint of the pair.** Headline result against the BF16
reference on the same four tasks: **AVG 0.8411 vs 0.8399 — +0.12 pp, with every individual delta
positive and ≤ 0.35 σ.** On these benchmarks the quantization is indistinguishable from BF16.

> **Read this first: the "Plus" in the directory name is a misnomer, and a useful one.**
> The run was launched with the *plus* intent — quantize the 48 sparse-MLA projections on top of the
> experts — but AutoRound registers a hard-coded, non-removable predefined ignore list for
> `model_type=glm5_next` that includes a bare **`self_attn`** substring pattern and **`layers.0.mlp`**.
> Those two entries silently swallowed the 48 MLA projections and layer 0's dense MLP: **37 287
> layers instead of the audited 37 338**, with no error and no warning. §2.1 documents the
> measurement. What matters operationally is that the resulting scope is exactly the one vLLM can
> load: the sibling `…-AutoRound-aligned`, where the AutoRound side *was* fixed and all 37 338
> official-scope layers really are MXFP8, **fails at weight loading** (see §7). If you want an 8-bit
> GLM-5.3-Flash that serves, use this one; if you want the size win, use the MXFP4-Mixed recipe.

Sizes, layer counts and every score in §3 were measured on the machine in §5; the BF16 row of §3.3
is the reference table supplied for the base checkpoint, not a local run.

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Name | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (native block-FP8) / `zai-org/GLM-5.3-Flash-BF16` (quantization input) | GLM-5.3-Flash-MXFP8-Plus-AutoRound |
| Architecture | `Glm5NextForConditionalGeneration` / `model_type: glm5_next` (natively multimodal, hybrid 34× KDA linear attention + 11× sparse MLA (DSA), 288-expert MoE, mHC hyper-connections, 1-layer MTP) | identical |
| Parameters | **321.323 B** total by safetensors-header census (Z.AI bills the model as 320 B total / 18 B active; ≈96.99 % of the parameters are routed experts) | 321.323 B (unchanged; weights re-encoded) |
| Layers | 45 decoder (`first_k_dense_replace=3` → layers 0-2 dense MLP, 3-44 MoE) + layer 45 = MTP; 11 `deepseek_sparse_attention` at 3,7,…,43, the other 34 are `linear_attention` (KDA) | identical |
| MoE | 288 routed experts, 8 active (`topk_method=noaux_tc`, sigmoid scoring, `routed_scaling_factor=2.5`) + 1 shared, `moe_intermediate_size=2048`, `moe_router_dtype=float32` | identical (experts MXFP8, router BF16) |
| MLA / DSA | `kv_lora_rank=512`, `q_lora_rank=1536`, `v_head_dim=256`, `qk_nope_head_dim=256`, `qk_rope_head_dim=0` (NoPE), `index_topk=2048`, `index_kpool=4` | **all BF16 here** (see §2.1) |
| Weight tensors (index entries) | 38 770 (`BF16` 38 479 + `F32` 291) for the BF16 source | **76 057** = 37 287 × `F8_E4M3` + 37 287 × `U8` (E8M0 scales) + 1 192 × `BF16` + 291 × `F32` |
| Size on disk (safetensors bytes) | BF16 source 642.65 GB / 598.51 GiB | **339.39 GB / 316.08 GiB**, 120 shards |
| Compression ratio vs BF16 | 1.00× | **1.89×** |
| Effective bits/param | 16.0 | 8.45 on disk (8.25 on the quantized span: 8-bit weight + 8/32-bit E8M0 scale) |
| Quantization | — | W8A8 MXFP8, `group_size=32`, symmetric, dynamic activations |
| Format | safetensors | `compressed-tensors` / `mxfp8-quantized` |
| Context length | 1 048 576 native | unchanged |
| License | MIT (Z.AI Co., Ltd) | same (see §8) |

Serving was validated at **TP=4 on 4× B300** (§3.4, §6.4). TP must divide both
`num_attention_heads=64` and the KDA `linear_attn_config.num_heads=64`; PP is gated off for this
architecture (`vllm/models/glm5next/common/model.py:735-737`).

## 2. Precision plan

Of the **37 862** 2-D Linear layers in the model, **37 287 are MXFP8** and **575** stay BF16:

| Bit-width | Modules (layer count) | Params | Share of Linear |
|---|---|---|---|
| **8 bit (MXFP8)** | `mlp.experts.{gate,up,down}_proj` 43 L × 288 E × 3 = 37 152 · `mlp.shared_experts.{gate,up,down}_proj` 43×3 = 129 · dense `mlp.{gate,up,down}_proj` **layers 1-2 only** = 6 | **313.038 B** | **97.4 %** |
| 16 bit (BF16) | *all* attention: KDA `self_attn.{q,k,v,o}_proj`, `{q,k,v}_conv1d`, `{b,f_a,f_b,g_a,g_b}_proj` (34 layers) **and** sparse-MLA `self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b_proj,o}_proj` (12 layers) · DSA `indexer.*` · `mlp.gate` router · `hc_*` · `visual.*` · `embed_tokens` / `lm_head` · layer-0 dense MLP · MTP pieces | 8.221 B | 2.6 % |

The exported `quantization_config.json` carries **679** enumerated `ignore` names:

| Group | Entries | Reason |
|---|---|---|
| KDA layers `self_attn.{q,k,v,o}_proj`, `{q,k,v}_conv1d`, `{b,f_a,f_b,g_a,g_b}_proj` | 34 × 12 = 408 | official checkpoint ships them raw; vLLM builds KDA with `quant_config=None` (`common/kda.py:189-194`); `b_proj` is `[·,64]`, below the MX `N ≥ 128` floor |
| `model.visual.*` (24 blocks × 5, merger × 4, `patch_embed.proj`, `downsample`) | 126 | text evals do not exercise it; vLLM passes `quant_config=None` to the tower (`common/model.py:1096-1105`) |
| `mlp.gate` router | 43 | 288-way sigmoid routing in FP32; quantizing it changes the routing distribution directly |
| DSA `self_attn.indexer.{wk,wq_b,weights_proj}` | 12 × 3 = 36 | decides which KV pages are read; 0.09 B params, nothing to gain |
| **sparse-MLA `self_attn.{q_a,q_b,kv_a_proj_with_mqa,o_proj}`** | **12 × 4 = 48** | ⚠ **not a design decision — swallowed by AutoRound's predefined `self_attn`; see §2.1.** Only `kv_b_proj` (12, also here) is BF16 in the official checkpoint |
| layer-0 dense `mlp.{gate,up,down}_proj` | **3** | ⚠ same cause: predefined `layers.0.mlp`, produced by a `first_k_dense_replace` lookup that misses `text_config` and falls back to 1 |
| `self_attn.kv_b_proj` | 12 | the one sparse-MLA projection Z.AI genuinely left unquantized |
| `lm_head`, `embed_tokens`, MTP `eh_proj` | 3 | 0.634 B each / plain `nn.Linear` in vLLM (`common/mtp.py:49`) |

Hyper-connection parameters, RMSNorm weights, `A_log` / `dt_bias` / `e_score_correction_bias` and the
MTP `shared_head` / `enorm` / `hnorm` pieces are BF16 or F32 on disk too (1 192 + 291 non-quantized
tensors in total) but are not enumerated in `ignore`, because they are not Linear modules — the
`hc_`, `norm`, `A_log`, `dt_bias`, `enorm`, `hnorm`, `shared_head` patterns in §6.2 are explicit
insurance for that set.

### 2.1 ★ Deviation from the audited plan (measured, not predicted)

The pre-flight audit (`audit_GLM-5.3-Flash-MXFP8-Plus-AutoRound.log`) and the artifact disagree by
exactly 51 layers, and the audit log is *not* stale — it is byte-identical to the one for the fixed
sibling run:

| | Layers | Params | Bytes |
|---|---|---|---|
| Audited plan (`--scheme MXFP8`, plus profile) | 37 338 | 314.397 B | ≈338.07 GB |
| **This artifact (measured)** | **37 287** | 313.038 B | **339.39 GB** |
| Missing from the artifact | **51** (48 sparse-MLA + 3 layer-0 dense MLP) | 1.359 B | −1.32 GB |
| Anything quantized that should not be | **0** | 0 | — |

Cause, visible in the first seconds of the run log:

```
INFO model_free.py L794: Using predefined ignore_layers from config:
                         indexer, layers.0.mlp, self_attn, weights_proj
```

AutoRound unions this list into `--ignore_layers` with **no CLI switch to subtract from it**, and
matching is substring-based, so a bare `self_attn` covers the sparse-MLA projections no matter what
the command line asked for. (`layers.0.mlp` comes from `get_glm_flash_ignore_layers` reading
`config.first_k_dense_replace` at the top level, where this multimodal config does not have it — it
lives in `text_config` — so the value falls back to 1 dense layer instead of 3.) The audit script does
not merge the predefined list, which is why the audit predicted 37 338. Note the artifact is
**smaller in layer count but larger in bytes** than the plan: the 51 layers stay BF16 (2 bytes/param)
instead of MXFP8+scale (1.03 bytes/param).

Consequence, and the reason this card is the interesting one: `MODE=mxfp8` (the "aligned" profile)
and `MODE=mxfp8_plus` produce the **same** artifact on an unpatched AutoRound, and that artifact is
the one vLLM loads. The deviation is what makes this checkpoint deployable.

### 2.2 Size accounting

| | BF16 source | Official block-FP8 | **This checkpoint** | `…-aligned` sibling |
|---|---|---|---|---|
| File bytes (header sum) | 642.65 GB | 328.33 GB | **339.39 GB** | 338.07 GB |
| `du` on this NFS volume | 605 GiB | 306 GiB | 322 GiB | 321 GiB |
| Weight bytes | 642.65 GB | 314.40 GB | 313.04 GB | 314.40 GB |
| Scale bytes | — | 0.077 GB (F32 / 128×128) | **9.78 GB** (U8 / 1×32) | 9.83 GB |
| Ratio vs BF16 | 1.00× | 1.96× | **1.89×** | 1.90× |
| Loads in vLLM | *not run here* | *not run here* | ✅ **measured (§3.4)** | ❌ crashes at load (measured) |

**Compare sizes by header/file bytes, never by `du`** — `du` mixes GiB with block-allocation slack and
produced a bogus "MXFP8 loses 33 GB vs FP8" claim in an earlier draft of the recipe. The honest gap to
official FP8 is **+11.06 GB (+3.37 %)**: 9.75 GB of that is the MX format tax (one E8M0 byte per 32
weights = 3.1 % of weight bytes, versus one F32 per 128×128 block = 0.024 %), and the remaining
~1.3 GB is the 51 BF16 layers of §2.1. **8-bit MX is not a size play on this model** — MX only pays
off at 4 bits (§7).

## 3. Evaluation results

Harness `lm-eval` 0.4.13, `vllm` backend, **TP=4 on 4× B300**, `seed=42`, `batch_size=32`;
`--model_args` in §6.4. These numbers are for **this** checkpoint
(`GLM-5.3-Flash-MXFP8-Plus-AutoRound`), and were produced by two runs on 2026-09-21.

### 3.1 Scores

| Task | Setting | Metric | Score | stderr | Samples |
|---|---|---|---|---|---|
| gsm8k | 5-shot, chat template, `--fewshot_as_multiturn`, greedy (`do_sample=false`) | exact_match (**strict**) | **0.9742** | ±0.0044 | 1,319 |
| gsm8k | same | exact_match (flexible) | 0.9735 | ±0.0044 | 1,319 |
| piqa | 0-shot | acc | **0.8313** | ±0.0087 | 1,838 |
| piqa | 0-shot | acc_norm | 0.8542 | ±0.0082 | 1,838 |
| mmlu | 0-shot, 57 subjects | acc | **0.8668** | ±0.0028 | 14,042 |
| hellaswag | 0-shot | acc | **0.6919** | ±0.0046 | 10,042 |
| hellaswag | 0-shot | acc_norm | 0.8606 | ±0.0035 | 10,042 |
| **AVG** (gsm8k strict, mmlu, piqa acc, hellaswag acc) | | | **0.8411** | | |

MMLU category aggregates: STEM 0.8693 · Other 0.8861 · Social Sciences 0.9181 · Humanities 0.8189.
Weakest subjects: `mmlu_virology` 0.5361, `mmlu_high_school_mathematics` 0.6926,
`mmlu_college_chemistry` 0.6600, `mmlu_professional_law` 0.6995 — i.e. the usual MMLU tail, not a
quantization signature.

Wall time: gsm8k **3 396 s**; piqa + mmlu + hellaswag **2 082 s**. Raw files:
`lm_eval_results/__models__lkk__cri_models__GLM-5.3-Flash-MXFP8-Plus-AutoRound/results_2026-09-21T18-25-43*.json`
(gsm8k) and `…T19-00-34*.json` (0-shot suite).

### 3.2 What was held fixed

Recorded verbatim in both result files: `language_model_only=true` (text-only — the vision tower is
never exercised), `enable_thinking=false`, `reasoning_parser=glm45`, `dtype=bfloat16`,
`max_model_len=8192`, `max_num_seqs=256`, `block_size=128`, `enable_prefix_caching=false`,
`add_bos_token=true`, `max_gen_toks=2048`, `batch_size=32`, and `random_seed` / `numpy_seed` /
`torch_seed` / `fewshot_seed` all **42**. gsm8k is generative with the chat template, greedy
(`do_sample=false`, `temperature=0.0`, `until=["Question:", "</s>", "<|im_end|>"]`), 5-shot; the 0-shot
suite is log-likelihood with no chat template. Engine-level internals (`max_num_batched_tokens`,
cudagraph mode, MoE backend selection) are **not** recorded in the result files — see §3.4 for what is
actually attributable to these two runs.

### 3.3 vs. the BF16 base model

BF16 reference supplied for `zai-org/GLM-5.3-Flash` on the same four tasks and metric variants:

| | GSM8K (strict) | MMLU | PIQA | HellaSwag | **AVG** |
|---|---|---|---|---|---|
| BF16 `zai-org/GLM-5.3-Flash` | 0.9735 | 0.8666 | 0.8292 | 0.6903 | **0.8399** |
| **MXFP8 (this checkpoint)** | 0.9742 | 0.8668 | 0.8313 | 0.6919 | **0.8411** |
| Δ | **+0.07 pp** | **+0.02 pp** | **+0.21 pp** | **+0.16 pp** | **+0.12 pp** |
| Δ in units of MXFP8 stderr | +0.17σ (±0.44) | +0.08σ (±0.28) | +0.24σ (±0.87) | +0.35σ (±0.46) | — |

Reading it honestly:

- **All four deltas are positive and all are far inside one standard error.** The largest, PIQA
  +0.21 pp, is 0.24σ on a ±0.87 pp metric. Nothing here is separable from sampling/decoding noise.
- **On these four tasks the checkpoint is indistinguishable from BF16** — which is the expected
  outcome for W8A8 MXFP8 on a model whose quantized span is 97.4 % of Linear parameters at only 8.25
  effective bits, and it is the honest claim to make: *no measurable loss on this protocol*, not
  "lossless".
- **What these runs cannot tell you**: anything about generation-heavy, long-horizon or agentic
  quality (the tasks the base model is actually marketed on — coding, `terminal-bench`, `DeepSWE`,
  HLE-with-tools), and anything about throughput. §7 lists the follow-ups that would matter.
- **Caveat on the comparison**: the BF16 row is a supplied reference without its own stderr, so the σ
  column normalises by the MXFP8 error bars only. To close it properly, run §6.4 against
  `/models/lkk/cri_models/GLM-5.3-Flash-BF16` at TP=4.

### 3.4 Functional evidence (and its provenance)

**Direct evidence for this checkpoint**: both runs produced complete, full-sample scores — gsm8k 1,319
generated-and-parsed samples through the chat template with `--fewshot_as_multiturn`, plus 1,838 /
14,042 / 10,042 log-likelihood evaluations — so the full 316 GiB loaded across 4 ranks, the
compressed-tensors path resolved, and generation worked. The exact `--model_args` recorded in the two
result files are in §6.4.

**Not recorded for these two runs**: no engine stdout was kept, so there is no weight-load time,
no KV-cache token count, and no MoE-kernel name attributable to *this* checkpoint. The engine-level
defaults quoted below (`max_num_batched_tokens=16384` with chunked prefill,
`cudagraph_mode=FULL_AND_PIECEWISE`, `moe_backend='auto'`, and the *"All limits of multimodal
modalities … running in text-only mode"* line) come from a run of the **same build with the same args
against the `-aligned` sibling** (`test_mxfp8_aligned.log`); that run failed for a checkpoint-specific
reason (§7) but its engine configuration is identical. Treat those as environment facts, not as
measurements of this artifact.

The pre-flight checks that did pass for this checkpoint are the ones the eval wrapper prints before
launching: registry hit for `Glm5NextForConditionalGeneration`, `deep_gemm` importable,
`block_size=128 = index_kpool 4 × 32`, TP=4 dividing both head counts, and experts at 8 bit with 16
scale columns per rank — which is why no `moe_backend` override was needed (§7).

## 4. Usage

### 4.1 vLLM (the path validated for the scores above)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /models/lkk/cri_models/GLM-5.3-Flash-MXFP8-Plus-AutoRound \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --language-model-only \
  --reasoning-parser glm45
```

`--block-size 128` is a hard constraint, not a tuning knob: it must be a multiple of
`index_kpool × 32` (= 4 × 32) or the platform rejects it (`vllm/platforms/cuda.py:430-447`).
`--reasoning-parser` is **`glm45`**, not `qwen3`. `--trust-remote-code` is *not* required for serving —
`glm5_next` config and modelling are in-tree in the build documented in §5.3 (lm_eval passes it anyway).

Hard constraints for `glm5_next` (each verified at source level; violating one fails at start-up):

- **`deep_gemm` must be importable** — the DSA indexer hard-requires it
  (`models/glm5next/nvidia/sparse_indexer.py:662`) and it is *not* in the vLLM wheel's `requires_dist`.
- **TP must divide 64 and 64** (`num_attention_heads`, KDA heads). **PP is gated off**
  (`common/model.py:735-737`); the eval script refuses `PIPELINE_PARALLEL != 1`.
- **`max_model_len` must be capped explicitly** (8 192 here): indexer decode logits reserve
  ≈ `max_num_seqs × max_model_len × 4 B`, which reaches multiple GiB on a 1 M-context model (the fix
  #57701 is on main, not in v0.30.0).
- **`max_num_seqs` couples to the KV page geometry** — the KDA/mamba state page must fit inside the MLA
  page (`kv_cache_utils.py:1249-1254`, error text *"increase tensor parallelism or use a wider KV cache
  dtype"*). 256 is the conservative value used for every number in this card.
- **`dtype=bfloat16` is mandatory** (hyper-connection scratch is BF16-hardcoded,
  `layers/mhc.py:667,702`).
- **KV cache: `auto`/`fp8`/`fp8_e4m3` are allowed, `fp8_ds_mla` / `nvfp4_ds_mla` are not** — those
  packing layouts assert `pe_dim == 64` and GLM is NoPE (`qk_rope_head_dim=0`; open issues #57578/#55773).
- **No expert parallelism**: the CUTLASS MXFP4/MXFP8 expert kernels require `ep_size == 1`
  (`fused_moe/experts/cutlass_moe.py:1041-1044`).
- **No speculative decoding during evaluation**: MTP's `eh_proj` is a plain `nn.Linear` (BF16 here, as
  required), but quantized checkpoints + `shared_head.head` still hit an open load-time false positive
  (#57532).

### 4.2 Transformers (weights-only inspection)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "/models/lkk/cri_models/GLM-5.3-Flash-MXFP8-Plus-AutoRound"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype="auto", device_map="auto")
```

Requires a `transformers` build that registers `model_type=glm5_next` (written by `transformers
5.16.0`; importable on 5.16.1 / 5.17.0) plus `compressed-tensors 0.17.0`. This is a 316 GiB checkpoint:
§4.2 is an inspection path, not a serving path — nothing in this workspace loads it outside vLLM TP=4.

Default sampling (`generation_config.json`): `temperature=1.0, top_p=0.95`,
`eos_token_id=[154820, 154827, 154829]`. The upstream card documents `reasoning_effort`
(`low`/`high`/`max`, default `max`) and `clear_thinking` (defaults `false` in the chat template); pass
them explicitly if either matters. Note the scores in §3 are **thinking-off** — Z.AI's own convention
is thinking-on, so do not compare §3 to their published table.

## 5. Reproducibility

### 5.1 Hardware / OS

| | Value |
|---|---|
| GPU | 4× **NVIDIA B300 SXM6 AC**, 275,040 MiB each, compute capability **10.3** (`sm103`) |
| Driver | **580.159.04** (NVIDIA UNIX open kernel module, x86_64) |
| CUDA | runtime **13.3.33** (`/usr/local/cuda-13.3`); PyTorch built against **CUDA 13.0** |
| cuDNN | system 9.23.0 |
| CPU | 2× Intel Xeon 6776P, 64 cores/socket, 2 threads/core (256 logical), 4 NUMA nodes |
| RAM | 4,031 GB (quantization peak **5.54 GB**; vLLM reported 871 GB available at load time) |
| OS / kernel | Ubuntu 24.04.4 LTS (Noble), Linux 6.8.0-124-generic, glibc 2.39, GCC 13.3.0 |
| Storage | Ceph-backed NFS mount, 10 TB volume — 642.65 GB source + 339.39 GB output |
| System prereq | `libdw-dev` (provides `elfutils/libdwfl.h`; absent by default on 24.04, DeepGEMM fails to build without it), `nvcc`, `cmake`, `ninja`, `uv` |

Quantization is disk-I/O bound and used **no** calibration data and effectively no GPU compute (one
GPU visible via `CUDA_VISIBLE_DEVICES=2`). Evaluation used all 4 GPUs.

### 5.2 Two separate environments

**Quantization env** — conda env `muse_verify` (`/models/miniforge3/envs/muse_verify`), Python 3.12.14:

```
auto-round             0.16.0 (source __version__; editable install @ /models/lkk/auto-round,
                                git 51003909 = v0.14.0-155-g51003909; dist-info metadata reports
                                0.15.0.dev84+gd6ead479 — the editable metadata is stale, the
                                checked-out source is what runs)
                       state at run time: UNPATCHED for glm5_next — REQUIRED for this artifact, §5.4
torch 2.13.0+cu130     transformers 5.16.1     compressed-tensors 0.17.0
accelerate 1.14.0      datasets 5.0.1          tokenizers 0.23.1
safetensors 0.8.0      huggingface-hub 1.29.0  numpy 2.2.6
```

**Evaluation / serving env** — `/models/lkk/venv_glm53`, built by `./setup_vllm_env_glm53.sh` from
`/usr/bin/python3` (3.12.3), all values verified with `CHECK_ONLY=1`:

```
vllm                   0.29.1rc1.dev471+g82daf9f57   (index https://wheels.vllm.ai/nightly/cu130, --pre)
torch                  2.13.0+cu130                  (torch.version.cuda == 13.0)
deep_gemm              2.8.0+e1f418c                 (source build via vLLM's tools/install_deepgemm.sh)
flashinfer-python      0.6.18.post1     flashinfer-cubin 0.6.18.post1   (index https://flashinfer.ai/whl/)
transformers           5.17.0           compressed-tensors 0.17.0       lm-eval 0.4.13
accelerate 1.15.0      datasets 5.0.1   tokenizers 0.23.2   safetensors 0.9.0rc0
numpy 2.3.5            triton 3.7.1     tilelang 0.1.12     nvidia-cutlass-dsl 4.7.1     xgrammar 0.2.7
registry check         Glm5NextForConditionalGeneration / Glm5NextForCausalLM / Glm5NextMTPModel = True
device capability      torch.cuda.get_device_capability(0) == (10, 3)  → sm103
```

### 5.3 Why this exact vLLM, and not a release

| Check | Result |
|---|---|
| Support PR | `vllm-project/vllm#53906`, merged **2026-09-03** (merge commit `98ed0856`, 94 files, +17 926/−265); module path `vllm.models.glm5next` |
| Lowest tag containing it | **v0.29.1rc0**; first "stable" tag **v0.30.0** |
| PyPI | newest is **0.29.0** → `pip install vllm` cannot work today |
| `latest` Docker image | same digest as v0.29.0 → does **not** contain support |
| Why Python patching can't substitute | the PR also edits `csrc/libtorch_stable/cache_kernels.cu`: the dequant-cache `head_dim` whitelist goes `{320,576}` → `{320,512,576}`, and **512 is GLM's MLA latent head width**. A 0.29.0 binary still enforces the old list |
| Why not pin v0.30.0 | main carries 10 further `glm5next` fixes (incl. #57701 indexer decode workspace, #57477 kpool tail addressing) that v0.30.0 lacks → follow nightly |
| Zero-cost alternative | `docker run --gpus all --network host -v /models:/models -e HF_HOME=/models/huggingface -it vllm/vllm-openai:nightly bash`, then add only `auto-round`/`lm-eval` (do **not** pip-install vllm inside). The purpose-built `vllm/vllm-openai:glm53-flash-x86_64-cu130` tag is stale (2026-09-09) |

Rebuild with `./setup_vllm_env_glm53.sh` (idempotent: if the registry check and `deep_gemm` import
already pass it only runs the health check and exits). `PIN=0` follows the moving nightly;
`CHECK_ONLY=1` is a ~14-second audit that flags `DRIFT` against the versions above.

### 5.4 ★ This artifact depends on an UNPATCHED AutoRound

The opposite caveat from the sibling card, and just as important for reproduction. At run time
(2026-09-21 16:59) `/models/lkk/auto-round` was at stock `glm5_next` behaviour:

```python
register_ignore_layers(
    matchers=[ModelTypeMatcher(r"glm5_next", mode="full")],
    ignore_layers=[get_glm_flash_ignore_layers, "weights_proj", "indexer", "self_attn"],
)
```

Later the same evening that block was patched locally to a config-driven generator (34 layer-anchored
`layers.{i}.self_attn.o_proj` names + the KDA-specific projections + `kv_b_proj`), which is what
produced the `-aligned` sibling. Therefore:

- **To reproduce this checkpoint byte-for-byte in scope**, use an unpatched AutoRound, *or* run the
  `aligned` profile (`MODE=mxfp8`) — both yield 37 287 layers, because the predefined bare
  `self_attn` dominates either way.
- On a **patched** AutoRound, `MODE=mxfp8` yields **37 290** layers (layer 0's dense MLP comes back;
  3 layers, ~0.05 GB) and `MODE=mxfp8_plus` yields 37 338 (the `-aligned` artifact, which does not
  load). Neither reproduces this artifact exactly, so record which tree you ran against.
- The scores in §3 belong to the 37 287-layer scope. The 3-layer and 51-layer differences are small
  enough that they will not move these benchmarks materially, but do not mix tables across scopes.

### 5.5 Environment variables

```bash
# quantization
source /models/miniforge3/etc/profile.d/conda.sh && conda activate muse_verify   # or ENVBIN=/models/miniforge3/envs/muse_verify/bin
export HF_HOME=/models/huggingface
export AR_MODEL_FREE_SHARD_PARALLELISM=1     # caps host RAM during the 642 GB streaming rewrite
export CUDA_VISIBLE_DEVICES=2                # single GPU id for the model-free path (I/O bound)

# evaluation
export PATH=/models/lkk/venv_glm53/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 6. Reproduce the artifact

### 6.1 Quantize

```bash
cd /models/lkk/cri_models
df -h /models                                       # needs ~340 GB free for the output
nvidia-smi --query-gpu=index,memory.used --format=csv     # never share a GPU
MODE=mxfp8_plus DRY_RUN=1 ./quant_GLM-5.3-Flash-mxfp.sh   # audit + print the command, no writes
MODE=mxfp8_plus      ./quant_GLM-5.3-Flash-mxfp.sh        # the run (~25 min)
```

The wrapper (`quant_GLM-5.3-Flash-mxfp.sh`; `MODE` ∈ `mxfp4_mixed|mxfp4_mixed_plus|mxfp8|mxfp8_plus`,
`SOURCE` ∈ `bf16|fp8`) resolves the ignore list with `gen_glm_ignore.py <src> --profile plus` (computed
from the config, never hard-coded), refuses to quietly secondary-quantize the local block-FP8 copy
(`SOURCE=fp8` warns and waits 5 s), runs the tensor-level pre-flight audit `audit_precision_plan.py`
and **exits 2** if the plan violates a vLLM MX kernel constraint, runs AutoRound, then greps the log for
`falling back to the regular AutoRound flow` (a silent model-free bail-out that would mean really
loading 642 GB) and runs `check_quant_export.py`. For this artifact the audit "passed" and the
predefined ignore still overrode the plan (§2.1) — the audit does not merge AutoRound's predefined
list, which is the known blind spot being reported upstream-style in the recipe.

### 6.2 The exact AutoRound invocation

```bash
auto-round \
  --model_name /models/lkk/cri_models/GLM-5.3-Flash-BF16 \
  --scheme MXFP8 \
  --ignore_layers lm_head,embed_tokens,visual,hc_,indexer,mlp.gate.,eh_proj,enorm,hnorm,shared_head,norm,A_log,dt_bias,self_attn.b_proj,f_a_proj,f_b_proj,g_a_proj,g_b_proj,conv1d,self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.kv_b_proj,'(language_model.)?layers.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30|32|33|34|36|37|38|40|41|42|44).self_attn.o_proj' \
  --format llm_compressor \
  --output_dir /models/lkk/cri_models/GLM-5.3-Flash-MXFP8-Plus-AutoRound \
  --model_free \
  --device_map auto
```

No `--layer_config` in this mode (uniform 8-bit; the `mxfp4_mixed*` modes add
`'{"mlp.experts":{"bits":4,"data_type":"mx_fp"}}'`). `mlp.gate.` keeps its **trailing dot** — a bare
`mlp.gate` substring-matches the dense layers' `mlp.gate_proj`; and `self_attn.b_proj` is written in
full because a bare `b_proj` would also hit `q_b_proj` / `kv_b_proj` (measured: silently lost 12
layers). Patterns must contain **no commas** — the separator is a comma. Expect the run to land on
37 287 layers only with the §5.4 unpatched tree.

Measured run (2026-09-21 16:59:59 → 17:24 local): **1 473.44 s**, **5.54 GB** peak RAM, 120/120
shards, shard parallelism 1, **37 287 quantized / 679 ignored**, device `cuda`, no `falling back`
line. Log lines to expect and not panic about:

- `Using predefined ignore_layers from config: indexer, layers.0.mlp, self_attn, weights_proj` —
  **this is the line that rewrites your plan.** Read it before the shard loop starts.
- `MXFP optimized RTN is enabled: evaluating the baseline E8M0 scale, 2x scale, and 0.5x scale
  independently for each group` — pass `--disable_opt_rtn` for plain RTN.
- `Detected N layer(s) incompatible with model-free RTN: language_model.embed_tokens,
  ….self_attn.conv1d, visual.rotary_pos_emb` → force-ignored automatically. Expected.
- `Found 3-D tensor '…self_attn.{q,k,v}_conv1d.weight' while splitting expert tensors; it will be
  kept unchanged` — the KDA depthwise convs are 3-D and BF16 by design anyway.

Logs: `quant_GLM-5.3-Flash-MXFP8-Plus-AutoRound.log` (AutoRound),
`audit_GLM-5.3-Flash-MXFP8-Plus-AutoRound.log` (pre-flight plan).

### 6.3 Validate

```bash
ENVBIN=/models/miniforge3/envs/muse_verify/bin \
  python check_quant_export.py GLM-5.3-Flash-MXFP8-Plus-AutoRound
# format=mxfp8-quantized  method=compressed-tensors  ignore=679   group_0: W8A8 g32 targets=1
# 2D Linear 37862 = quantized 37287 + plaintext 575  ->  OK

ENVBIN=/models/lkk/venv_glm53/bin SMOKE_ONLY=1 TP=4 \
  ./eval_GLM-5.3-Flash-quant.sh GLM-5.3-Flash-MXFP8-Plus-AutoRound
```

The smoke script (`smoke_vllm_ckpt.py`) must be invoked **as a file** — vLLM's spawn children re-exec
`__main__`, so feeding it through `python - <<PY` fails on `<stdin>` for reasons unrelated to the
checkpoint. The eval wrapper also refuses, before touching a GPU, if the registry lacks
`Glm5NextForConditionalGeneration`, if `deep_gemm` is missing, if TP does not divide the head counts,
or if `block_size` is not a multiple of `index_kpool × 32`.

### 6.4 Evaluate

```bash
cd /models/lkk/cri_models
TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
ENVBIN=/models/lkk/venv_glm53/bin \
./eval_GLM-5.3-Flash-quant.sh /models/lkk/cri_models/GLM-5.3-Flash-MXFP8-Plus-AutoRound
```

Other useful invocations: `RUN_SMOKE=1` (smoke then bench), `SMOKE_ONLY=1` (load check only),
`RUN_GSM8K=0 TASKS=piqa LIMIT=16` (quick slice), `DRY_RUN=1` (print the pre-flight verdict and the arg
JSON without touching a GPU). The script reads `moe_intermediate_size`, `index_kpool`, head counts and
the expert bit-width **from the artifact** rather than from memory.

The `--model_args` it generates — reproduced **verbatim** from the two runs behind §3, and it must be
one JSON object on this lm_eval build (it `json.loads` the first token and dies with `Invalid JSON` on
a comma-separated `k=v` list containing a nested dict):

```json
{"pretrained": "/models/lkk/cri_models/GLM-5.3-Flash-MXFP8-Plus-AutoRound",
 "tensor_parallel_size": 4, "max_model_len": 8192, "max_num_seqs": 256, "block_size": 128,
 "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": true,
 "add_bos_token": true, "enable_prefix_caching": false, "max_gen_toks": 2048,
 "enable_thinking": false, "language_model_only": true, "reasoning_parser": "glm45"}
```

gsm8k adds `--apply_chat_template --fewshot_as_multiturn`; both invocations use
`--batch_size 32 --seed 42 --output_path /models/lkk/cri_models/lm_eval_results`. Results land in
`lm_eval_results/<sanitized-ckpt-path>/results_<timestamp>.json`. For the BF16 reference, point
`pretrained` at `/models/lkk/cri_models/GLM-5.3-Flash-BF16` and keep `tensor_parallel_size: 4`.

## 7. Known issues and caveats

- **"Plus" does not mean the sparse MLA is quantized** (§2.1): 48 MLA projections and layer 0's dense
  MLP are BF16, contradicting both the directory name and the pre-flight audit. Cross-check any
  claim about this artifact's scope against its `quantization_config.ignore` (679) rather than its
  name.
- **Do not "fix" this by shipping the `-aligned` sibling.** `GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned`
  really does quantize all 37 338 official-scope layers and then **fails to load in vLLM** (measured
  2026-09-21 23:19, `test_mxfp8_aligned.log`, at ~5 % of shard loading):

  ```
  vllm/models/glm5next/common/model.py:831   if _try_load_fp8_attn_proj(
  vllm/models/glm5next/common/model.py:1295     weight_bf16 = _dequant_fp8_block(weight_fp8, scale_inv, block_size)
  vllm/models/glm5next/common/model.py:1231     out = (w.to(torch.float32) * s_full).to(torch.bfloat16)
  RuntimeError: The size of tensor a (512) must match the size of tensor b (16384) at non-singleton dimension 0
  ```

  Mechanism: vLLM dequantizes the sparse-MLA projections to BF16 at load time (`_FP8_ATTN_PROJS`,
  `common/model.py:1238-1246`) and accepts the `weight_scale` name, but assumes block-FP8 scale
  geometry `[⌈N/bs⌉, K/bs]` and `repeat_interleave`s **both** dims. MXFP8 stores `[N, K/32]` — the
  batch dim is already full — so `N` inflates 32× and the first MLA projection hit
  (`kv_a_proj_with_mqa [512,4096]`) dies with exactly `512 vs 16384`. There is no memory win even if
  it worked, since those modules end up BF16 on the device anyway. This checkpoint avoids the whole
  issue by leaving those layers BF16.
- **8-bit MX is the wrong format if you are chasing size.** 1.89× vs the official FP8's 1.96×, and
  339 GB vs 328 GB. The shippable MX recipe is `MODE=mxfp4_mixed`: routed experts (97 % of parameters)
  at MXFP4 W4A4 for **≈183 GB / 3.5×**, reproducing
  [`INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound`](https://huggingface.co/INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound)
  layer-for-layer, whose reported AVG on this same four-task protocol is **0.8366 vs 0.8399 BF16
  (99.61 %)**. So the 4-bit variant is half the size of this checkpoint and costs ≈0.3 pp of AVG;
  this 8-bit one costs ≈0 pp and is not smaller. Choose deliberately.
- **Only four academic benchmarks.** gsm8k / mmlu / piqa / hellaswag are log-likelihood-or-short-generation
  probes; they do not cover long-context (everything here ran at `max_model_len=8192` on a 1 M-native
  model), agentic/tool use, code, or reasoning-on decoding (`enable_thinking=false`). Absence of
  regression here is not evidence of parity on those.
- **Multimodal path unvalidated.** The vision tower is untouched BF16 and would load, but every number
  here is `language_model_only=true`. Dropping that flag is untested.
- **No throughput data.** Load time, KV-cache sizing, and tokens/s were not recorded for the two runs
  behind §3, and this card makes no performance claim.
- **The MXFP8-MoE scale-swizzle bug that bit Qwen3.8-Flash-Next at TP≥2 does not apply here.** The
  trigger is per-rank scale columns `moe_intermediate_size/TP/32`, which for GLM is `2048/TP/32 =
  64/TP` → 64/32/16/8 at TP=1/2/4/8, all multiples of 4. No `moe_backend` override is needed (and none
  was used: the runs above resolved `moe_backend='auto'`).
- **Requires a nightly vLLM** (§5.3): `vllm==0.29.0` (PyPI, `latest` image) cannot load this
  architecture at all, and the limitation is compiled into the C++/CUDA kernels.
- **Long start-up is normal, not a hang.** `No available shared memory broadcast block found in 60
  seconds` at 0 % GPU utilisation is vLLM's keepalive while a rank loads weights or captures CUDA
  graphs — with 43×288 experts that is minutes. Do not "fix" it with `enforce_eager=True`. First engine
  start ≫ subsequent starts (inductor AOT caches + FlashInfer autotune cache in `~/.cache/vllm/`).
- **`triton_kernels.matmul_ogs` import errors** can appear in logs; non-fatal — that module is only
  needed if the MoE resolves to the Triton backend, which it does not here.
- **Ignore lists are not portable between the models in this workspace.** GLM needs `mlp.gate.` with a
  trailing dot (dense layers exist), `self_attn.b_proj` spelled in full, and — for the plus profile —
  layer-anchored `o_proj` written with **bare dots** (`layers.(0|1)…`, never `layers\.(0|1)`:
  `to_standard_regex` re-escapes the backslash and the pattern silently matches nothing).
  Qwen3.8-Flash-Next needs `in_proj_a`/`in_proj_b` and `ple`; dense Qwen3.8-27B must *not* list
  `mlp.gate` at all.

## 8. License and attribution

Base model [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) is **MIT**
(Z.AI Co., Ltd); the `LICENSE` file in this repository applies to this derivative unchanged.

Quantization performed with [Intel AutoRound](https://github.com/intel/auto-round) (Apache-2.0) in
model-free RTN mode — **no calibration corpus was used**, so no dataset attribution applies.
Serving via [vLLM](https://github.com/vllm-project/vllm) (nightly; `glm5_next` support from PR #53906)
with DeepGEMM / FlashInfer / CUTLASS-DSL kernels; evaluation via
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.13. MXFP8 follows the
[OCP Microscaling Formats (MX) specification](https://opencompute.org): E8M0 shared scale per
32-element block.

Full design rationale — the tensor-level parameter census, the safetensors-header diff against the
official block-FP8 checkpoint, the predefined-ignore post-mortem behind §2.1, the vLLM source-level
constraint map with `file:line` references, the `-aligned` load-failure analysis, and the
MXFP4-Mixed (G0) / MXFP4-Mixed-Plus (G0+) variants of the same recipe — is in
`GLM-5.3-Flash-quantization-recipe.md` in the workspace root. Scripts:
`quant_GLM-5.3-Flash-mxfp.sh`, `gen_glm_ignore.py`, `audit_precision_plan.py`,
`check_quant_export.py`, `compare_quant_scopes.py`, `verify_glm_scope.py`,
`setup_vllm_env_glm53.sh`, `eval_GLM-5.3-Flash-quant.sh`, `smoke_vllm_ckpt.py`. The
[`…-AutoRound-aligned`](../GLM-5.3-Flash-MXFP8-Plus-AutoRound-aligned/README.md) sibling has its own
card: it documents the patched-AutoRound path to the full official scope (37 338 layers) and the
load-time failure that scope currently causes.
