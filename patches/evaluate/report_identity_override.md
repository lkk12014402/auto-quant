# Patch: dataset sink 身份可配置（避免 results/unknown 与误判 BF16-Baseline）

## 症状
`model:` 指向本地 checkpoint 路径（无 Org/Name）时，dataset sink 落到
`results/unknown/...`；eval-only 跑已量化目录时无 quant 块，artifact 误命名为
`<dir>-BF16-Baseline`。

## 根因
`RunMeta.org` 从 model 字符串切 org 段（本地路径→空→unknown）；
`canonical_artifact_name`/`_artifact_name` 从 model basename + quant 块推导
（eval-only 无 quant→BF16-Baseline）。

## 修复
新增 `output.report_model`（canonical Org/Model，定 org 与 artifact short 名）与
`output.artifact_name`（显式 artifact 覆盖）。pipeline/cli 的 RunMeta 用
`_report_model(run_cfg)` 作 model（org 正确），`_artifact_name` 优先返回
`output.artifact_name`。默认(未设)行为不变。

## 用法
    output:
      report_model: Qwen/Qwen3-235B-A22B
      artifact_name: Qwen3-235B-A22B-MXFP4-Mixed-CT-RTN-AutoRound   # eval-only 才需显式
