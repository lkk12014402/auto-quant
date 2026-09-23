# model_card.md —— <Model>-<Scheme>-AutoRound 模型卡（第④步产物）

> 模板使用：发布量化权重（`provider.upload`）时必填，作为模型 repo 的 README.md。
> 实例参照（逐章填出来的样子）：`templates/examples/model_card_example.md`。
> 章节不许删（不需要的写「本次无，原因」）。命名规范见 `aqa/report/names.py`。
>
> 四条写作纪律（违反任何一条 = 卡不合格）：
> 1. **measured, not predicted**——每个数字注明测量方式与日期；预测值不许混进表格。
> 2. **诚实措辞**——结论限定在已测协议/benchmark 集内（"本协议内不可区分"，
>    不写"无损"）；明确列出未覆盖面。
> 3. **偏差必须成章**——量化计划与产物不一致时，偏差表+根因+后果是硬章节；
>    一致也要写一句"已核对一致"。
> 4. **约束可验证**——Usage 的每条硬约束标注依据（源码 file:line 或实测报错），
>    不靠记忆。

---
license: <与 base model 一致>
license_name: <...>
license_link: LICENSE
library_name: transformers
pipeline_tag: <text-generation | image-text-to-text>
base_model: <org>/<BaseModel>
tags:
- quantized
- <scheme 小写>
- compressed-tensors
- auto-round
- vllm
- <model_type>
---

# <Model>-<Scheme>-AutoRound

<三段开头：
① 产物身份——什么 scheme（格式规格一句话）+ 什么方法（RTN/tuning、有无校准数据）
  + 导出格式；量化范围精确到层数（多少层量化、哪些族保高精度）。
② 头条结论——对 baseline 的 Δ（带口径与 σ 换算）。
③ "Read this first"——命名/范围与事实不符之处、选型警告，必须置顶。没有就写无。>

## 1. Model summary

| | Base model | This checkpoint |
|---|---|---|
| Name / Architecture / Parameters | | |
| 层结构 / MoE / 注意力细节 | | |
| Weight tensors（index 条目数与 dtype 分布） | | |
| Size on disk（safetensors header bytes） | | |
| Compression ratio / Effective bits/param | | |
| Quantization / Format / Context / License | | |

<serving 验证形态一句话（TP×几卡什么 GPU）；TP/PP 的整除约束。>

## 2. Precision plan

<逐模块位宽表：bit-width | 模块（层数） | 参数量 | 占比。然后 ignore 分组表：
每组给理由（证据：实测或 file:line）。非 Linear 的保精度件单独一段说明。>

### 2.1 量化计划与产物的偏差（实测，非预测）

<审计层数 vs 产物层数 vs 差值表；根因；后果。无偏差写「审计=产物，已核对（命令）」。>

### 2.2 Size accounting

<vs BF16 源 / 官方量化版（如有）：header bytes 对比。**禁止用 du**。>

## 3. Evaluation results

<harness/版本/backend/TP/seed/batch 一行声明；分数表带 stderr 与样本数与口径。>

### 3.2 What was held fixed
<逐字记录固定项：model_args 全量、seed、生成参数；注明哪些引擎内部值**没有**记录。>

### 3.3 vs. baseline
<Δ 表 + σ 换算 + 诚实解读：哪些不可区分、哪些不能下结论、参照值的口径风险。>

### 3.4 Functional evidence (and its provenance)
<直接证据（本 checkpoint 实测了什么）与间接事实（环境/兄弟产物的数字）分开写。>

## 4. Usage

### 4.1 vLLM（已验证路径）
<完整可拷命令 + 硬约束清单：每条注明依据与违反时的死法（启动即崩/静默错分）。>

### 4.2 Transformers（inspection 路径）
<加载代码 + 版本要求 + "这是检查路径不是 serving 路径"的体积警示。>

## 5. Reproducibility

### 5.1 Hardware / OS
<GPU 型号×数量/cc、driver、CUDA、CPU、RAM、OS、存储、系统级前置包。>

### 5.2 环境（量化 / 评估分开列）
<精确版本块（来自 requirements.txt + setup_venv.sh 自检输出），含索引来源。>

### 5.3 版本选择的理由
<为什么不能 pip install 默认版：PR/tag/索引实测结论，含查证日期。>

### 5.4 产物级依赖警示
<如"本产物依赖未打补丁的 auto-round @ commit"——复现字节级一致的前提。无则写无。>

### 5.5 Environment variables
<分量化/评估两段，逐字可拷。>

## 6. Reproduce the artifact

### 6.1 Quantize / 6.2 精确量化命令 / 6.3 Validate / 6.4 Evaluate
<四段命令逐字可拷；关键参数的坑注释（子串陷阱、尾点、正则写法）；
评估的 --model_args 逐字 JSON；预期日志行与"不要恐慌"提示。>

## 7. Known issues and caveats

<含"不要做的事"（如不要强上某兄弟版本，附实测死法）+ 未覆盖面声明
（长上下文/agentic/吞吐/多模态/thinking-on）+ 已知良性噪音。>

## 8. License and attribution

<base model 许可 + 衍生适用；量化工具/服务栈/评估栈归因；
"未使用校准语料"之类的归因豁免（如适用）；指向完整 recipe 文档与脚本清单。>
