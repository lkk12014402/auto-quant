# Patch: 让 vLLM 的 DeepSeek-V4 `wo_a` 走 compressed-tensors MXFP8 BMM

## 背景

DeepSeek-V4 的 attention 输出投影 `wo_a` 不是普通 Linear，而是**按 group 的 batched GEMM**：
vLLM 把它建模为 `ColumnParallelLinear(is_bmm=True, bmm_batch_size=n_local_groups)`，
输入是 `[tokens, n_groups, in_features]`，输出 `[tokens, n_groups, o_lora_rank]`。

原生 DeepSeek checkpoint 的 `wo_a` 是 128×128 block FP8 + UE8M0 scale，
`vllm/models/deepseek_v4/nvidia/ops/o_proj.py` 用 `fp8_einsum(recipe=(1,1,128))` 处理，
只接受这种 block scale 布局。**compressed-tensors MXFP8** 的 scale 是 per-32（uint8 E8M0），
`weight_scale.dim()==2` 但列数 = in/32 ≠ in/128，进 `fp8_einsum` 会 shape/断言失败
（实测错误形如 `t.dim() == N`）。

## 原状（vLLM nightly `46d2b23ac`）

- `CompressedTensorsW8A8Mxfp8` 在 `__init__` 里 `init_mxfp8_linear_kernel()`（无 bmm），
  即使 layer 带 `is_bmm=True` 也不会选到 `DeepGemmMxfp8BmmLinearKernel`。
- `o_proj.py` 只有两个分支：原生 FP8（读 `wo_a.weight` + `fp8_einsum`）和 bf16
  （`weight.view(n_groups, ...)` + `torch.bmm`）；CT 量化权重会落进 FP8 分支并崩。
- 参考：vLLM 自带的 modelopt `kMxfp8Static` 分支已经按同样方式重建 kernel
  （`modelopt.py:2546`），本 patch 只是把同样的接线补到 compressed-tensors 上。

## 改动（两处，最小）

1. `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py`
   `process_weights_after_loading`：若 `layer.is_bmm`，用
   `init_mxfp8_linear_kernel(bmm_batch_size=layer.bmm_batch_size)` 重建 kernel，
   再把权重后处理成 3D `[groups, N, K]`（`deepgemm_post_process_fp8_weight_block(is_bmm=True)`）。
2. `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`：识别 CT MXFP8 wo_a 后跳过 FP8
   einsum，改为 `wo_b(wo_a(bf16_rotated_input).flatten(1))`；输入由
   `fused_inv_rope_fp8_quant(..., quantize=False)` 产出（bf16 `[T,G,K]`）。

   判别条件（两次踩坑后的最终版）：`is_bmm and hasattr(wo_a, "weight_scale") and
   not hasattr(wo_a, "weight_scale_inv")`。
   - 不能用 `type(layer.quant_method).__name__`：`layer.quant_method` 是
     `CompressedTensorsLinearMethod` 包装，不是 scheme。
   - 不能用 `wo_a.weight.ndim == 3`：**原生 DeepSeek FP8** 的 wo_a 在
     `process_weights_after_loading` 里也会 reshape 成 3D（供 fp8_einsum 的 bmm 布局），
     会误判成 CT。
   - 可靠标记：CT 是 uint8 per-32 `weight_scale`；原生是 float32 128×128 `weight_scale_inv`。

## 用法 / 验证

无需改 yaml；compressed-tensors mixed-precision checkpoint 自动命中该分支。
本次 `DeepSeek-V4-Flash-0731` MXFP4(experts)+MXFP8(其余) 的 4 任务全量评估即在该
patch 下完成。

补丁文件：`patches/evaluate/vllm_dsv4_ct_mxfp8_bmm_wo_a.patch`
