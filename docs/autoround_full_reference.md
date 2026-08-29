# AutoRound 量化算法与配置全参考（源码核实版）

> 来源：`/home/kaokaolv/auto_agent/auto-round` @ `bd9a13ce`（2026-08-25，main 分支）。
> 所有结论均带 `文件:行号` 引用，路径相对仓库根。**写量化命令前以本文档和源码为准，不要凭记忆编造参数。**
> 注意：这是重构后的新版 auto-round（`scheme` + `alg_configs` 驱动），与旧版有显著差异，见第 11 节"常见误解"。

## 1. 预定义 Scheme 全表（`auto_round/schemes.py:804-829`）

`QuantizationScheme` 字段（`schemes.py:197-210`）：`bits=4, group_size=128, sym=True, data_type="int", act_bits, act_group_size, act_sym, act_data_type, act_dynamic, super_bits, super_group_size, rotation_config`。表中只列显式设置的字段。

| Scheme | 权重 | 激活 | 定义位置 | 备注 |
|---|---|---|---|---|
| `W4A16` | 4bit, g128, int, sym | 16bit | `schemes.py:538` | 默认 scheme |
| `W8A16` | 8bit, g128, int, sym | 16bit | `schemes.py:608` | |
| `W3A16` | 3bit, g128, int, sym | 16bit | `schemes.py:598` | |
| `W2A16` | 2bit, g128, int, sym | 16bit | `schemes.py:568` | 推荐配 `--enable_alg_ext` |
| `W2A16G64` / `W2A16G32` | 2bit, g64/g32 | 16bit | `schemes.py:578/588` | |
| `W5A16` / `W6A16` | 5/6bit, g128 | 16bit | `schemes.py:548/558` | |
| `MXFP4` | 4bit, g32, `mx_fp` | 4bit `mx_fp`, g32, sym, dynamic | `schemes.py:618` | OCP MX，E8M0 scale |
| `MXFP4_RCEIL` | 同上 | `mx_fp_rceil`（仅此项不同） | `schemes.py:644` | MoE 常用，利于 vLLM fused-moe |
| `MXFP6` | 6bit, g32, `mx_fp` | 6bit `mx_fp` | `schemes.py:631` | |
| `MXFP8` / `MXFP8_RCEIL` | 8bit, g32, `mx_fp` | 8bit `mx_fp`(/`_rceil`) | `schemes.py:657/670` | |
| `MXINT4` | 4bit, g32, `mx_int` | 4bit `mx_int`, dynamic | `schemes.py:683` | |
| `NVFP4` | 4bit, g16, `nv_fp` | `nv_fp4_with_static_gs`, g16, dynamic | `schemes.py:696` | E4M3 block scale + FP32 global scale |
| `NVFP4_E5M3` | 4bit, g16, `nvfp4_v2` | `nvfp4_v2`, g16 | `schemes.py:709` | UE5M3 scale，gs 允许 16/32；与 NVFP4 是两套实现 |
| `FPW8A16` | 8bit, per-tensor, `fp` | 16bit | `schemes.py:721` | |
| `FP8_BLOCK` | 8bit, g(128,128), `fp` | 8bit `fp`, g128, dynamic | `schemes.py:731` | 唯一 tuple group_size 预设（DeepSeek 风格） |
| `FP8_STATIC` | 8bit, per-channel, `fp` | 8bit `fp`, per-tensor, **static** | `schemes.py:752` | 需校准 act max |
| `INT8` | 8bit, per-channel, int | 8bit int, dynamic | `schemes.py:765` | |
| `INT4` | 4bit, per-channel, int | 4bit int, dynamic | `schemes.py:779` | |
| `BF16` | 16bit | 16bit | `schemes.py:794` | 主要作 AutoScheme 候选/layer_config 用 |
| `W4A16_MIXED` | = W4A16 | | `schemes.py:867-898` | 触发特殊配方：experts 4bit、其余 8bit、lm_head 8bit |

GGUF 预设（大小写不敏感，需 `GGUF:` 前缀；事实表 `schemes.py:22-141`）：
`Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q6_K, Q2_K_S, Q2_K_MIXED, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M`。
K-quant 带双重量化参数（super_bits/super_group_size）；`Q2_K_MIXED` 是特殊配方（embedding/lm_head→Q8_0、MoE 3D expert→Q2_K_S、其余 Linear→Q4_K_S，`schemes.py:853-866`）。

scheme 解析规则（`parse_scheme`，`schemes.py:496-535`）：
- 字符串必须命中预设名，否则 `KeyError`；dict 只接受上述 12 个字段，**未知 key 被静默丢弃**。
- 缺省填充：act_dynamic→True；act_group_size→group_size；act_bits→16；act_sym→sym；act_data_type→权重 dtype（act_bits<16 时）。
- `--group_size` 传 `"128,128"` 元组仅允许 fp8 权重 + 动态 fp8 激活（`algorithms/quantization/config.py:105-151`）。

## 2. data_type 清单（`auto_round/data_type/`）

| 族 | 值 | 说明 |
|---|---|---|
| int | `int_sym`, `int_asym`, `int_sym_gptq`（zp 固定中点） | 普通整数量化，bits 2/3/4/8（`int.py`） |
| MX（E8M0 共享指数） | `mx_fp`/`mx_fp4(e2m1)`, `mx_fp6(e2m3/e3m2)`, `mx_fp8(e4m3/e5m2)`, `mx_fp16`, `mx_bf16`, `mx_int2/4/8`；每个自动派生 `_rceil` 和 `opt_rtn_` 变体（`mxfp.py:409-412`） | rceil=共享指数用 ceil 取整（利于 vLLM kernel）；`mx_fp4_rceil_v2`（`mxfp.py:414`）为 MXAttention 论文方法。**部署要求 group_size=32**（仅 warning，`algorithms/quantization/config.py:173-174`） |
| NVFP | `nv_fp4`（g16，E4M3 scale+FP32 global）、`nv_fp4_with_static_gs`、`nvfp4_v2`（UE5M3 scale，g16/32） | `nvfp.py`；global scale 强制 g16（`nvfp.py:58`） |
| FP8 | `fp8_sym`/`fp8`/`fp8_e4m3`, `fp8_e5m2`, `block_fp8_sym`（2D block）, `fp8_unit_scale`, `fp8_gaudi3_sym` | `fp8.py`；`fp8_static` 是分类标记非注册函数 |
| GGUF 双重量化 | `int_sym_dq`, `int_asym_dq`, `int_asym_float_zp` | `gguf.py`；`_dq` 需配 super_bits/super_group_size |
| 混合 | `fp8_to_int_sym` | 两阶段渐进量化（`w4fp8.py:21`） |

运行时量化函数解析顺序：`opt_rtn_{dtype}`（iters=0 且未 disable）→ `rtn_{dtype}` → `block_{dtype}`（tuple gs）→ `{dtype}`（`data_type/utils.py:105-176`）。

## 3. 算法清单与组合规则

组合架构：若干 **preprocessor**（可堆叠，按序执行）+ **恰好一个 block-quantizer**（`algorithms/composer.py:159-165`；缺省时自动补 RTN，`:127-131`）。

| 算法 | 类型 | 作用 | 启用方式 | 关键参数 | 源码 |
|---|---|---|---|---|---|
| `auto_round`（SignRound，别名 autoround/sign_round/signround） | quantizer | 符号梯度下降逐 block 调 rounding + min/max | 默认（iters>0）；`--algorithm auto_round` | `--iters --lr --minmax_lr --momentum --nblocks --enable_minmax_tuning --gradient_accumulate_steps --enable_quanted_input --not_use_best_mse --enable_lfq` | `algorithms/quantization/sign_round/quantizer.py:46` |
| `rtn` | quantizer | round-to-nearest | `--iters 0` 或 `--algorithm rtn` | `--disable_opt_rtn / --enable_opt_rtn` | `rtn/quantizer.py:27` |
| opt_rtn | rtn 变体 | imatrix 加权 scale/zp 搜索 | iters=0 且未 `--disable_opt_rtn` 时默认开启；W8A16/W8A8 int 自动关（`autoround.py:269-276`）；MoE expert 默认关（`quantization/base.py:212-224`） | 同上 | `rtn/quantizer.py:71-140` |
| alg_ext（SignRoundV2） | auto_round 变体 | imatrix 感知 init scale + 低比特 outlier-suppressed loss + GGUF dq 逐迭代处理 | **`--enable_alg_ext`（必须显式传，recipe 不会自动开）** | — | `algorithms/registry.py:204-205` → `sign_roundv2/quantizer.py:321` |
| AdamRound | auto_round 变体 | AdamW 替代 SignSGD | `--enable_adam` | — | `adam_round/adam.py:24` |
| `awq` | preprocessor | 激活感知 smooth scaling + 可选 clip | `--algorithm awq,auto_round` | `--awq_duo_scaling(默认True) --awq_n_grid(20) --awq_seqlen(512) --awq_apply_clip --awq_clip_as_init` | `transforms/awq/base.py:214` |
| `hadamard` | preprocessor（rotation） | 逐 Linear 块对角 Hadamard 旋转 | `--algorithm hadamard --rotation_type {hadamard,random_hadamard} --rotation_backend {auto,inplace,transform}` | `--rotation_block_size`（mx_fp 默认32/nv_fp 默认16） | `transforms/hadamard/dispatcher.py:54-152` |
| QuaRot / SpinQuant | 架构级旋转 | R1–R4 位置 Hadamard 旋转 | **仅 Python API**：`rotation_config="quarot"`（固定，推荐）/ `"spinquant"`（可训练，实验性） | — | `transforms/__init__.py:142-153`、`spinquant/preprocessor.py:64` |
| `svdquant` | preprocessor | SVD 低秩分解吸收 outlier | `--algorithm svdquant,auto_round` | `--svdquant-rank(32)` 等 | `transforms/svdquant/apply.py:111` |
| LFQ | auto_round 子开关 | 最后一个 block 用 LM 交叉熵 loss | `--enable_lfq` | — | `sign_round/quantizer.py:486-492` |

组合约束（代码强制）：
- 多于一个 quantizer → `ValueError`；同一 preprocessor 重复 → 报错。
- CLI 规则（`cli/algorithms.py:103-146`）：`iters=0` 时 auto_round 一律替换为 rtn；只写 `awq` 无 quantizer 时自动补 rtn；AWQ 的 `disable_opt_rtn`（默认 True）传染给同行 RTN。
- AWQ / SVDQuant **要求 nblocks=1**（`awq/base.py:275-281`、`svdquant/apply.py:128-130`）；AWQ 默认 `skip_moe=True` 不 smooth routed experts。
- Hadamard `inplace` 后端只能导出 `fake` 格式（`dispatcher.py:125`）；`transform` 后端（fused Triton kernel）仅支持 MXFP4/NVFP4 且仅 `hadamard`/`random_hadamard`。
- SVDQuant 强制 `format="svdquant_nunchaku"`（`autoround.py:676-678`），目前主要面向 FLUX 扩散模型。
- GGUF 导出且 `iters!=0` 时建议 `iters=0` 或开 `enable_alg_ext`（`export/formats/backends/gguf.py:128-134`）。

## 4. CLI 内置 recipe 的真实覆盖（`cli/main.py:117-141`）

只覆盖用户未显式设置的字段。**本版不再覆盖 `enable_alg_ext`**（旧 README 残留说法，需用时显式传）。

| 入口命令 | batch_size | iters | seqlen | nsamples | lr | 其他 |
|---|---|---|---|---|---|---|
| `auto-round`（默认） | 8 | 200 | 2048 | 128 | 自动 | — |
| `auto-round-best` | 8 | 1000 | 2048 | 512 | 自动 | — |
| `auto-round-light` | 8 | 50 | 2048 | 128 | 5e-3 | — |
| `auto-round-rtn` | 8 | 0 | 2048 | 1 | — | `disable_opt_rtn=True` |
| `auto-round-opt-rtn` | 8 | 0 | 2048 | 128 | — | `disable_opt_rtn=False` |

lr 自动启发式（`sign_round/config.py:110-121`）：`iters>=1000 且 bits<=3` → `2.0/iters`，否则 `1.0/iters`。

## 5. CLI 参数全表（`auto-round [quantize]`）

### 5.1 常用量化参数（`cli/parser.py:32-91`）
`--scheme`（默认 W4A16，必须是预设名）、`--bits`、`--group_size`（-1 per-channel / 0 per-tensor / `"128,128"` block-fp8）、`--asym`、`--data_type`（int/mx_fp/nv_fp/mx_int/fp）、`--act_bits`、`--act_group_size`、`--act_asym`、`--act_data_type`、`--disable_act_dynamic`、`--super_bits`、`--super_group_size`、`--scale_dtype`、`--ignore_layers/--fp_layers`（同义词，逗号分隔模糊匹配）、`--quant_lm_head`（仅 auto_round/fake/mlx 格式允许）、`--to_quant_block_names`。

### 5.2 运行时参数（`cli/parser.py:99-209`）
`model`（位置参数）/`--model_name`、`--model_dtype`、`--platform`(hf/model_scope)、`--batch_size`、`--seqlen`、`--nsamples`、`--device_map`（"0"/"0,1"/"auto"/"cpu"）、`--dataset`（默认 `NeelNanda/pile-10k`）、`--seed`(42)、`--format`（默认 auto_round，逗号分隔多格式）、`--algorithm`（逗号分隔：rtn/auto_round/awq/svdquant/hadamard）、`--output_dir`（./tmp_autoround）、`--low_gpu_mem_usage`、`--disable_low_cpu_mem_usage`（低 CPU 内存默认开）、`--enable_torch_compile/--disable_torch_compile`（None 时按设备自动）、`--layer_config`（JSON 字符串）、`--static_kv_dtype fp8`、`--static_kv_granularity tensor/head`、`--static_attention_dtype fp8`、`--static_attention_granularity`（均实验性）。

### 5.3 AutoScheme（混合位宽自动搜索）
`--avg_bits 3.5 --options "W4A16,W8A16" [--shared_layers ...] [--ignore_scale_zp_bits]`（`cli/main.py:370-385`）。逐层 delta-loss 评分为每层选候选 scheme，使平均位宽≈目标。**强制需要校准数据**。Python：`AutoRound(model, scheme=AutoScheme(avg_bits=..., options=...))`（`auto_scheme/gen_auto_scheme.py:29-42`）。

### 5.4 算法参数组
- auto_round 组：`--iters --lr --minmax_lr --momentum --nblocks --enable_minmax_tuning/--disable_minmax_tuning(默认开) --enable_norm_bias_tuning(实验性) --gradient_accumulate_steps --enable_alg_ext --not_use_best_mse --enable_quanted_input/--disable_quanted_input(默认开) --enable_adam --enable_lfq`（`cli/algorithms.py:399-469`）
- rtn 组：`--disable_opt_rtn / --enable_opt_rtn`（`cli/algorithms.py:366-390`）
- awq / hadamard / svdquant 组：见第 3 节表格。

### 5.5 评测组
`--tasks`（默认 12 任务：lambada_openai,hellaswag,winogrande,piqa,mmlu,wikitext,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge）、`--eval_backend hf/vllm`、`--eval_bs`、`--limit`、`--num_fewshot`、`--eval_task_by_task`。也可用独立入口 `auto-round-eval`。

### 5.6 其他
`--model_free` / `--disable_model_free`（见第 10 节）、`--mllm`（废弃，多模态已自动检测）、`--quant_nontext_module`（量化 vision/audio tower）、`--extra_data_dir`、`--template`、diffusion 组（`--prompt_file --metrics --guidance_scale` 等）。

## 6. Python API（`auto_round/autoround.py:789-845`）

```python
from auto_round import AutoRound
ar = AutoRound(
    model,                    # nn.Module 或 str（model_free 必须 str）
    scheme="MXFP4",           # str / dict / QuantizationScheme / AutoScheme
    layer_config=None,        # dict[str, str|dict|QuantizationScheme]
    iters=200,                # 0 → 自动切 RTN
    seqlen=2048, nsamples=128, batch_size=8,
    dataset="NeelNanda/pile-10k",   # 也可传 DataLoader/list
    device_map=0,             # "0,1" / "auto" / dict
    low_gpu_mem_usage=False,
    alg_configs=None,         # 算法链；旧名 algorithm= 已废弃
    rotation_config="quarot", # 架构级旋转（仅 API）
    enable_alg_ext=True,      # 直传 SignRound kwarg（仍支持，告警推荐 config 对象）
    ignore_layers="",         # 无 fp_layers 参数（那是 CLI 别名）
    quant_lm_head=False,
)
ar.quantize_and_save(output_dir, format="llm_compressor")  # inplace=True 默认；多格式自动 inplace=False
```

`quantize_and_save(output_dir="tmp_autoround", format=None, inplace=True, **kwargs)`（`compressors/base.py:1925-2022`）；实际保存目录会自动派生后缀（如 `模型名-w4g128`，`base.py:1870-1923`）。

## 7. 导出格式矩阵（`auto_round/export/formats/`，`SUPPORTED_FORMATS` 见 `utils/common.py:643-670`）

| format | 支持的 scheme（代码 `support_schemes`） | 加载方 | 关键限制 |
|---|---|---|---|
| `auto_round`（含 `:auto_gptq`/`:gptqmodel`/`:auto_awq`/`:llm_compressor`/`:fp8`/`:mlx` 后缀） | W2/3/4/8A16 系、MXFP4/MXFP8/MXINT4、NVFP4(_E5M3)、FPW8A16、FP8_STATIC、FP8_BLOCK、BF16、W4A16_MIXED（`backends/autoround.py:35-52`） | AutoRound extension（CPU/XPU/CUDA/HPU）、Transformers、vLLM | act_bits<16 且不匹配已知分支 → ValueError 建议 fake（`autoround.py:81-86`） |
| `llm_compressor` | MXFP/NVFP 系、FP8 系、INT8、W2/3/4/8A16（`backends/llm_compressor.py:28-44`） | vLLM（compressed-tensors） | 需 `compressed_tensors` 库；MXFP4/NVFP4 **官方推荐格式** |
| `auto_gptq` / `gptqmodel` | W2/3/4/8A16 int 系 | vLLM/TGI/AutoGPTQ | **拒绝 mx_fp/nv_fp/fp**（`auto_gptq.py:42-58`） |
| `auto_awq` | 仅 W4A16 int | AutoAWQ/vLLM | 不支持 Conv1D，in_features 须整除 group_size |
| `fake` | 全部 scheme | 研究/精度验证 | QDQ 假量化，不 pack |
| `fp8` | 仅 FP8_BLOCK | DeepSeek 风格 block-fp8 | — |
| `gguf` / `gguf:*` | GGUF 预设全系列 | llama.cpp | 与其他非 fake 格式**互斥**（`resolver.py:43-49`）；需 gguf-py+sentencepiece；单一 GGUF 导出 scale_dtype 强制 fp32 |
| `mlx` | W2~W8A16、BF16 | Apple Silicon（mlx-lm） | — |
| `svdquant_nunchaku` | 仅 MXFP4 | Nunchaku 扩散模型 | — |

全局校验规则（`export/formats/base.py`）：
- scheme 不在目标格式 `support_schemes` 内 → `FormatCompatibilityError`。
- 权重/激活为非 block 的 fp8 时只允许 `auto_round`/`llm_compressor`（`base.py:189-203`）。
- `act_bits<=8` 且非标准静态 fp → 任何格式被重置为 `fake` 并 warning（`base.py:205-214`）。
- **"MXFP4 只能导 llm_compressor"是误解**：`auto_round` 格式同样支持 MXFP4（落盘权重与 llm_compressor 一致，`compressors/model_free.py:1729-1737`），但 vLLM 部署推荐 `llm_compressor`。

## 8. layer_config / ignore_layers / fp_layers

解析核心：`compressors/layer_config/resolver.py:369-434`。

**匹配规则**（`resolver.py:289-323`）：key 按 ①完整层名精确 → ②存在但不支持的模块类型（丢弃+warning）→ ③正则/子串模糊匹配（无正则元字符时自动两侧包 `.*`）展开。无匹配会 warning。

**value 三种形式**：预设名字符串（`"W4A16"`、`"BF16"`、`"GGUF:Q6_K"`）/ `QuantizationScheme` 对象 / dict（可含 `"scheme"` 键 + 字段覆盖；非法 key 直接 ValueError）。

```python
layer_config = {
    "lm_head": {"bits": 16},             # ≥16bit 即保持原精度
    "model.layers.*.self_attn": "BF16",  # 正则匹配整组
    "lm_head": "GGUF:Q6_K",              # GGUF 场景
}
```

**优先级与关联**：
- `ignore_layers`（CLI `--fp_layers` 是其别名）先于 layer_config 生效，强制 `bits=16, data_type=float, fixed_by_user=True`（`resolver.py:204-212`）。匹配方式：逗号分隔、精确名 + 子串模糊（`resolver.py:327-366`）。
- 自动保护：shape 不整除的层强制 fp16（`resolver.py:68-90`）；`quant_lm_head` 在 tie_word_embeddings 且非 GGUF 时强制 False（`resolver.py:58-62`）。
- 正则配置会写入导出 `extra_config` 供 vLLM 按 regex 匹配（`export_to_nvfp_mx.py:238-247`）。

## 9. 特殊模型自动处理（`auto_round/special_model_handler.py`）

- **兜底规则**：无特定 rule 命中且判定为 MoE 时，自动忽略所有**模块名以 `.gate` 结尾**的层（router/gate，`special_model_handler.py:1135-1155`）。
- 预定义 ignore 表：Longcat（classifier）、GLM-Flash/GLM5（前 k 层 dense mlp、weights_proj）、Step3.5（g_proj/moe.gate/MTP 层）、Kimi-K2.5（vision_tower/mm_projector）、BAGEL（moe_gen+self_attn+顶部整层）。详见 `special_model_handler.py:1023-1132`。
- 应用点：`compressors/base.py:943-967`（GGUF 导出时跳过预定义 ignore）。
- **VLM/多模态**：默认只量化文本塔；`--quant_nontext_module` 才把 vision/audio block 纳入（强制 batch_size=1，`compressors/mllm_mixin.py:54-69`）。
- 同一 block 内 MoE expert 的量化方案由 block 级机制统一处理，无需逐 expert 配置。

## 10. model_free 模式与校准数据集

**model_free**（免校准直出，`utils/model.py:2574-2631`）：
- 显式：`--model_free` / `model_free=True`（model 必须是 str 路径）。
- 自动路由（可 `--disable_model_free` 关）：model 为 str + `iters==0` + `disable_opt_rtn=True` + scheme 受支持 + format 为 auto_round（int 系）或 llm_compressor（mx_fp 系）+ 未开 static kv/attn。
- 支持的 scheme：W2A16(G32/G64)/W4A16(_MIXED)/W8A16/MXFP4/MXFP8/NVFP4_E5M3/BF16（`utils/model_free_utils.py:52-63`）。
- 与 AWQ/SVDQuant smooth 等需校准的预处理器不兼容（ValueError）。

**校准数据集**（`auto_round/calib_dataset.py`）：默认 `NeelNanda/pile-10k`；内置 `pile-val-backup`、`CCI3-HQ`、`github-code-clean`、`opencode-instruct`、`ultrachat_200k`、`Ultra-FineWeb`、`new-title-chinese`（中文）、`mbpp`、`audiocaps`、本地 json/jsonl。语法：`name:split=train:num=100:concat=false:apply_chat_template=true`，逗号分隔混合多数据集。

## 11. 与旧文档/常见误解的差异（防幻觉要点）

1. **`enable_quarot_opt` 不存在**。QuaRot 正确用法：Python `rotation_config="quarot"`；CLI 只有 per-linear `--rotation_type hadamard|random_hadamard`。
2. **CLI `--rotation_type quarot_hadamard` 疑似 bug**：`RotationConfig.hadamard_type` 校验器只接受 `hadamard/random_hadamard/inplace_quarot_hadamard/inplace_hadamard/inplace_random`（`transforms/hadamard/config.py:53-55,120-125`），传 `quarot_hadamard` 会被 pydantic 拒绝。正确写法 `inplace_quarot_hadamard`（静态分析结论，未实际运行验证）。
3. **recipe 不再自动开 `enable_alg_ext`**：W2A16/MXFP4 追求精度时需显式 `--enable_alg_ext`。
4. **`fp_layers` 只是 CLI 上 `--ignore_layers` 的别名**（`cli/parser.py:80-85`）；Python API 只有 `ignore_layers`。
5. **MXFP4 不是只能导 llm_compressor**：`auto_round` 格式同样支持；但 vLLM 部署推荐 llm_compressor。
6. **AutoGPTQ/AutoAWQ 只是导出格式**，算法仍是 SignRound 调优；且它们只接受 int 系 scheme。
7. `algorithm=` 旧 API 名已废弃，用 `alg_configs=`；`AutoRoundLLM/AutoRoundAdam/AutoRoundMLLM` 均为兼容包装。
8. `low_cpu_mem_usage` 默认开启；CLI 旧 flag 废弃，用 `--disable_low_cpu_mem_usage` 关闭。

## 12. 常用命令模板（以代码为准的最小集）

```bash
# Dense W4A16（默认调优）
auto-round --model_name <model> --scheme W4A16 --format auto_round --output_dir <out>

# MXFP4（dense，调优 + alg_ext 提精度；vLLM 部署用 llm_compressor）
auto-round --model_name <model> --scheme MXFP4 --iters 200 --enable_alg_ext \
  --format llm_compressor --output_dir <out>

# MoE MXFP4_RCEIL（router/gate 自动忽略；大模型可 iters=0 走 opt_rtn）
auto-round --model_name <model> --scheme MXFP4_RCEIL --iters 0 \
  --format llm_compressor --output_dir <out>

# 免校准 model-free（等价自动路由条件：iters=0 + disable_opt_rtn + llm_compressor）
auto-round --model_name <model> --scheme MXFP4 --iters 0 --disable_opt_rtn \
  --format llm_compressor --model_free --output_dir <out>

# AWQ + AutoRound 组合
auto-round --model_name <model> --scheme W4A16 --algorithm awq,auto_round --iters 200

# GGUF
auto-round --model_name <model> --scheme "GGUF:Q4_K_M" --iters 0 --format gguf --output_dir <out>

# 混合位宽自动搜索
auto-round --model_name <model> --avg_bits 3.5 --options "W4A16,W8A16" --format auto_round

# 量化后直接评测
auto-round --model_name <model> --scheme MXFP4 --format llm_compressor \
  --tasks mmlu,gsm8k --eval_backend hf --output_dir <out>
```

> 提醒：MoE 模型的 router/gate（`.gate` 结尾）会被自动忽略，无需手动配置；lm_head 默认不量化（除非 `--quant_lm_head`）；VLM 默认只量化文本塔。额外的 ignore（如 self_attn、shared_experts）属**模型级 recipe 决策**，参考同目录各模型分析文档，不要默认照抄。

## 13. 已有 per-model recipe 索引（同目录）

以下文档均为**实跑验证过**的模型级 recipe + 分析过程，写法范式：结论速览 → 模型结构分析（config.json + safetensors index 实测，不加载权重）→ AutoRound 自动处理 → 全张量 pattern 命中模拟 → 推荐命令 → 变体 → 注意事项。**新模型的 recipe 应复用这套范式。**

> ⚠️ 版本基线：这些文档基于 **auto-round 0.15.0**（commit `f17d9cd4`，位于 `/home/kaokaolv/auto-round`）编写；本文档第 1–12 节基于新版重构仓库（`auto_agent/auto-round` @ `bd9a13ce`）。涉及 model_free 内部实现（如 `_BLOCK_NAME_TO_IGNORE`、`split_fused_expert_tensors`）的结论以 0.15.0 为准，复用前对新版重核。

| 文档 | 模型/主题 | 核心结论 |
|---|---|---|
| `mxfp4_recipes.md` | MXFP4 通用调研（25+ HF 模型） | MXFP4 定义、dense vs MoE 配置差异、官方 recipe（fp_layers/iters）、导出与硬件要求 |
| `GLM-5.1-AutoRound量化配置.md` | GLM-5.1（MLA+DSA MoE，78 层+MTP） | experts MXFP4 + 其余 MXFP8；常规流程需手动 ignore `mlp.gate`；实跑踩坑（空 block 除零、cuDNN SDPA） |
| `LongCat-Flash-Lite-Sparse_mxfp4_quantization_analysis.md` | LongCat 69B（MLA+LSA MoE） | model_free 下 `--scheme MXFP8 + {mlp.experts: bits4}` 的三档语义；`MXFP4_RCEIL` 不在 model_free 白名单 |
| `DeepSeek-V4-Flash-0731_W4A16_quantization_analysis.md` | DeepSeek-V4（原生 MXFP 混合精度源） | model_free 支持 deepseek_v4 专用反量化→INT4 重量化；`wo_a` 需 layer_config bits=16；`AR_MODEL_FREE_SHARD_PARALLELISM` 控制内存 |
| `Gemma-4-31B_quantization_analysis.md` | Gemma-4-31B（dense 多模态） | 31B 应走校准 iters=200 而非 model_free；量化主体是文本 MLP；视觉塔自动保 BF16 |
| `Qwen3.5-397B-A17B_AMD_variants_comparison.md` | AMD Quark 三变体对比 + 复刻 | Quark vs auto-round 数值格式等价性；`mm_mxfp8 requires N>=128`（in_proj_a/b 必须 BF16）；model_free 无法量化 router |
| `Qwen3.8-27B_MXFP4_量化问题分析与解决.md` | vLLM `KeyError` 排查 | 导出 ignore 列表必须覆盖所有实际未量化的 Linear；`mlp.gate` 子串误伤 `gate_proj`（dense 模型）；MTP 建议 ignore |
| `Qwen3.8-2.4T-A95B_mxfp4_quantization_analysis.md` | 2.4T MoE（fused 3-D experts） | fused expert 自动拆分；只量化 routed experts 覆盖 96.9% 参数；与 AMD Quark 规则对账 |
| `Qwen3.8_vLLM_排查记录.md` | vLLM 部署环境 | config 类型 bug 需 nightly、flashinfer 版本检查旁路、flash_attn ABI 问题卸载法 |

**跨文档共性经验**（已被实跑验证，直接继承）：

1. ignore_layers 是**子串/正则匹配**——写每个 pattern 前先在完整 index 上模拟命中（如 `mlp.gate` 会误伤 `gate_proj`，router 精确匹配用 `mlp.gate$`）。
2. 导出配置的 `ignore` 列表必须与实际未量化权重**逐项一致**，否则 vLLM 按量化层构建明文权重 → `KeyError`。
3. MXFP8 的 `mm_mxfp8` kernel 要求 `out_features >= 128`，小投影（如 N=64 的 `in_proj_a/b`）必须保 BF16；且 MXFP8 是 MX-block FP8，无法字节复刻 per-channel FP8 recipe。
4. model_free 的预定义 skip（`.gate.`、`embed`、`conv`、`shared_expert_gate.`）**优先于 layer_config**，router 在 model_free 下恒为 BF16。
5. llm_compressor 导出按有效 weight bits 重建完整 preset：`{mlp.experts:{bits:4,data_type:mx_fp}}` 在默认 MXFP8 下得到的是完整 W4A4，不是 W4A8。
6. 大 shard 模型先设 `AR_MODEL_FREE_SHARD_PARALLELISM=1` 控制 host RAM 峰值。
