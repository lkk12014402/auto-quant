# Patch: 让生成类 lm_eval 任务可关闭 chat template（Qwen3 gsm8k）

## 背景
用户实证：Qwen3-235B-A22B 的 gsm8k 用 `--apply_chat_template` + `--fewshot_as_multiturn`
反而掉点；经典 5-shot 补全格式更好。

## 原状
aqa/eval/harness/lm_eval.py build_command：生成类任务无条件追加 `--apply_chat_template`，
`lm_eval.apply_chat_template` 选项只能把任务*强制*成 generative，无法关闭该 flag。

## 改动（最小、无回归）
`--apply_chat_template` 仅在 `self.apply_chat_template is not False` 时追加；
显式 `lm_eval.apply_chat_template: false` 才会关闭。`--fewshot_as_multiturn`（需依赖
chat template）改为嵌套在 apply_chat_template 之下，避免 no-template+multiturn 的非法组合。
默认(None)/True 行为完全不变（已用单测校验：default 仍带两个 flag）。

## 用法（run yaml）
    eval:
      lm_eval:
        apply_chat_template: false
        fewshot_as_multiturn: false
只影响生成类 lm_eval 任务（如 gsm8k）；loglikelihood 任务(piqa/mmlu/hellaswag)本就不套模板。
补丁文件：patches/evaluate/gsm8k_no_chat_template.patch
