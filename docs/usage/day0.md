# Day-0 量化用法（新架构 / 官方未支持）

> 完整 playbook + Qwen3.8-Flash-Next 实证：`docs/day0_quantization_recipe.md`。
> 这页是速查。

## 什么时候走这条路

新发布的模型架构，transformers / 官方 vllm release / auto-round 有任一不认。
正常路径（`examples/ssh_docker/run_5000_qwen3.8_27B.yaml`）假设三件套都认；day-0 先花小
成本确认"谁不认、怎么补"。

## 判定（分钟级，不下载权重）

1. 读 `config.json` 的 `architectures` + `model_type`——唯一锚点，名字会骗人；
2. 三件套探针：
   - transformers：`AutoConfig.for_model('<model_type>')` 认不认；
   - vllm：`vllm/model_executor/models/registry.py` 有无该类名；
   - auto-round：新架构大多走通用 Linear 路径，特殊层型过坑清单。

## 量化路径选型

- **先跑 model_free RTN 建下限**（不用真加载，transformers 不认也能跑，
  INT 系只认 `--format auto_round`）；
- 调优（真加载）要 transformers 认架构——不认就升版本或打 patch；
- RTN 已近乎无损时（实证 -0.0007），调优可能零收益甚至倒退。

## 推理路径

vllm 官方不认时按顺序查：HF model card 部署段 → recipes.vllm.ai/<org>/<model> →
vllm PR 搜索 → 官方定制镜像（有就别自己编译）。PR 含新 CUDA kernel 就别 patch，
走源码构建或定制镜像。

## 验证顺序（成本递增）

config 解析 → bf16 单请求冒烟 → **量化产物加载冒烟**（最危险放最前，
`scripts/smoke_vllm.py`）→ 小 limit 评估 → 全量 + gate。

## 参考案例

`examples/local_docker/run_qwen38_flash_next*.yaml`（RTN / tuning / 纯评估三个）+
`recipes/quant/Qwen__Qwen3.8-Flash-Next.json`（pinned recipe）+
`patches/autoround_qwen4exp_fixes.patch`（运行前提的 auto-round 补丁，
应用方式见 `patches/README.md`）。
