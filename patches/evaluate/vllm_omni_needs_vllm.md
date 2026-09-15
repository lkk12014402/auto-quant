# Patch: vllm_omni backend 需显式装 vllm

## 症状
diffusion 评估（backend vllm_omni）报 missing_tool："vllm-omni is not installed;
install it to use the diffusion harness"——但 vllm-omni 已装（0.28.0）。

## 根因
diffusion harness `available()` 实为 `from vllm_omni.entrypoints.omni import Omni`，
而 vllm_omni 内部 `from vllm.config.lora import LoRAConfig` 需要 **vllm** 包；
vllm-omni 的 Requires-Dist 未声明 vllm。aqa `container_packages` 的 vllm_omni 分支
只装 vllm-omni，不装 vllm → 导入失败被误报成 "not installed"。

## 修复
aqa/executors/_remote.py：vllm_omni 分支的包清单加入 "vllm"。
（本机 .venv-zimage 已手动 `pip install vllm==0.28.0` 补齐；vllm 0.28.0 对齐 torch 2.13.0，
vllm-omni 0.28.0 仍可导入 Omni。）
