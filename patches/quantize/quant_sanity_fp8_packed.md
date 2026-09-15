# Patch: quant_sanity 识别 FP8/MXFP8/MXFP4/NVFP4 打包权重

## 症状
diffusion MXFP8 量化：auto-round 成功导出（全部层量化，transformer 存为 F8_E4M3 weight +
weight_scale，pipeline 完整），但 aqa quantize phase 被判失败，fix-loop 反复重量化
（并被 drift.py 的 "tokenizer" 正则误分类成 tokenizer_error）。

## 根因
after_quantize 校验器 `quant_sanity`（aqa/hooks/sanity.py）的 `_has_packed_weights`
只认 int 打包的 `qweight` 张量。MXFP8/FP8 导出用 `weight`(F8_E4M3) + `weight_scale`，
无 qweight → 校验器误判 "0 packed layers ... fake-quantized" → 失败。
（实测导出键：`...to_k.weight`=F8_E4M3 + `...to_k.weight_scale`，has qweight=False。）

## 修复
`_has_packed_weights` 除 `qweight` 外，也认 `weight_scale`/`weight_packed`/`weight_global_scale`
（覆盖 FP8/MXFP8/MXFP4/NVFP4 + compressed-tensors 打包）。已单测：对本次 MXFP8 导出返回 True。
