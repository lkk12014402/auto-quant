# run.yaml 字段参考（schema 真源：`aqa/config.py`，extra=forbid，写错字段直接报错）

顶层：`model`（必填，HF id 或本地路径）+ `provider / quant / eval / gate / agent / executor / output / hooks`。
支持 `_base_: 相对路径.yaml`（深合并继承，子键覆盖，`null` 删除继承键）与
`${VAR}` / `${VAR:-default}` 环境变量展开。

## provider（模型从哪来）

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `type` | `pretrained` | `pretrained`（原样评估）/ `quantize`（先量化）/ `endpoint`（已部署的 OpenAI 兼容端点） |
| `upload.repo` / `upload.token_env` | — | 配了才把量化权重推 HF model repo（自动模型卡） |

## quant（量化，等价于 provider.quantize；顶层写法自动折叠进 provider）

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `scheme` | str, `MXFP4` | MXFP4/MXFP8/W4A16/W8A16/NVFP4_E5M3/FP8_STATIC/GGUF 系等 |
| `method` | `model_free` | `model_free`（免校准）/ `autoround`（校准路径） |
| `export_format` | `llm_compressor` | 导出格式（`llm_compressor` / `auto_round` / gguf 系） |
| `recipe` | `auto` | agent-optimize 或 recipe store 的引用 |
| `iters` / `nsamples` | None | 0=RTN；200 量级=tuning；nsamples 校准样本数 |
| `ignore_layers` | None | 逗号分隔 ignore 清单（子串匹配层名） |
| `layer_config` | None | 按层混合精度，如 `{mlp.experts: {bits: 4, data_type: mx_fp}}` |
| `quant_lm_head` | false | `--quant_lm_head` |
| `algorithm` | None | 算法链：rtn/auto_round/awq/hadamard/random_hadamard/quarot_hadamard/svdquant；model_free 仅允许 rtn；block-quantizer 只能一个 |
| `enable_alg_ext` | false | `--enable_alg_ext`（仅 auto_round） |
| `hadamard_type` | None | hadamard/random_hadamard/quarot_hadamard |
| `opt_rtn` | None | true→`--enable_opt_rtn`，false→`--disable_opt_rtn` |
| `static_kv_fp8` / `static_attention_fp8` | false | FP8 KV cache / FP8 attention。当前 schema 允许 `static_kv_fp8` 搭配 `model_free` 或 `autoround`；`model_free + static_kv_fp8` 仍会加载模型并执行 KV scale 校准 forward。`static_attention_fp8` 仍需 `method: autoround` |
| `device_map` | None | 默认多卡 auto；可显式 `"0,1"` 等 |
| `extra_args` | [] | 原样透传 CLI 的逃生口 |
| `recipe_sweep` / `recipe_sweep_candidates` | false / None | 多候选对比（`aqa sweep`），见 AGENTS.md §3.2 |

## eval（评估）

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `benchmarks` | `[piqa, mmlu, hellaswag]` | 任务名列表；逐任务参数在 `recipes/benchmarks/<task>.yaml` |
| `backend` | `hf` | `hf` / `vllm` / `local_completions` / `vllm_xpu` / `vllm_omni`（lm_eval 用；`local_completions` 通过本机已有的 OpenAI-compatible completions 服务评估，默认 `http://127.0.0.1:8000/v1/completions`；evalscope 自起 vllm serve） |
| `harness` | `lm_eval` | 裸任务的默认 harness：`lm_eval` / `evalscope` / `native` |
| `batch_size` | `auto` | 全局默认；recipe 可逐任务覆盖 |
| `baseline` | false | true 先测 bf16 原模型（gate 对比前提） |
| `limit` | None | 每任务样本上限（冒烟用） |
| `generative_tasks` | `[gsm8k]` | 强制按 generative 处理的任务 |
| `gen_max_toks` | 2048 | 生成预算 |
| `thinking` | `auto` | thinking 双模式：`auto`（检出即双测）/ `both` / `on` / `off` |
| `thinking_off_model_args` | `{enable_thinking: false}` | "关"模式的传参（特殊模型覆盖） |
| `lm_eval.model_args` | {} | backend 额外参数（`reasoning_parser`/`max_num_seqs`/`kv_cache_dtype` 等） |
| `lm_eval.gen_kwargs` / `apply_chat_template` / `fewshot_as_multiturn` / `seed` / `group` | — | 生成参数 / recipe 模板设置覆盖（`null` 沿用 recipe；显式 `false` 可关闭 generative task 的模板默认值）/ 种子(42) / 合并调用(true)。vLLM 的 `max_gen_toks` 由 `model_args` 传入，额外参数（如 `until`）通过 `--gen_kwargs` 覆盖 task recipe；fewshot-as-multiturn 仅在 chat template 启用时传入 |
| `diffusion.*` | — | diffusion harness 参数（步数/guidance/分辨率/scorer 等） |
| `evalscope` | {} | evalscope harness 透传：`repeats` / `eval_batch_size` / `generation_config` / `dataset_args` / `server_args` / `port` / `startup_timeout` |

## gate（可选，不配不判）

| 字段 | 说明 |
|---|---|
| `max_accuracy_drop` | 量化 vs baseline 允许的最大掉点（超了 NO-GO） |
| `min_accuracy` | 量化模型均分下限 |
| `significance` / `alpha` | 掉点须统计显著才判 NO-GO / 显著性水平（0.05） |

thinking 双模式时 gate 按同模式配对判定，取最差 verdict。

## agent（aqa 内部 agent）

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `fix_loop` | true | 相位失败进 agent 修复循环 |
| `optimize` | false | 量化前由 agent 分析结构并推荐命令（用户显式字段是硬约束） |
| `tiers` | `[openclaw, minimax, opus]` | 修复 tier 阶梯（需对应密钥：MINIMAX_API_KEY / COPILOT_GITHUB_TOKEN） |
| `max_fix_attempts` / `tier_attempt_budget` / `drift_threshold` | 5 / 3 / 2 | 总尝试上限 / 单 tier 预算 / 同类错误连击升级阈值 |
| `agent_timeout` | 1200 | 单次尝试墙钟（秒） |
| `context` | None | 附加上下文（内联文本或 `@文件`） |
| `patch.capture` / `patch.review` / `patch.track_vllm` | true / false / true | 工作区补丁快照 / 只读评审 / vllm 跟踪 |
| `knowledge.enabled` | true | 共享修复知识库 |
| `knowledge.repo` / `path_in_repo` / `token_env` | None / `knowledge` / None | HF dataset 后端（不配 repo 则纯本地 `~/.cache/aqa/knowledge`） |
| `knowledge.replay` / `push` | true / true | 先回放已录补丁 / 修复成功后回传 |
| `knowledge.use_references` | true | optimize 是否读增量 references 层（false 锁回打包层） |

## executor（执行环境，详见 environments.md）

| 字段 | 说明 |
|---|---|
| `type` | `local`（默认）/ `ssh` / 插件 |
| `num_gpus` | 显式卡数（TP 参数）；local 不选卡号（用 CUDA_VISIBLE_DEVICES） |
| `ssh.host/gpus/accelerator/auth/jump/mounts` | ssh：主机/卡串/ cuda 或 xpu / 认证（密码只走 env）/ 跳板 / 额外挂载 |
| `image.gpu/xpu/dockerfile/build_args/pip_install` | 基础镜像 / Dockerfile / 构建参数 / 追加 pip 包 |
| `proxy.http_proxy/https_proxy/no_proxy` | 远端代理 |
| `env.type` | `current`（默认）/ `venv` / `docker` |
| `env.path` / `env.create` / `env.packages` | venv 路径（必填）/ 缺失时 provision(true) / `auto` 推导或显式列表 |
| `env.mounts` / `env.network` | local docker 挂载 / 网络（默认 host） |
| `required_env` | 前置必填环境变量（缺失即 fast-fail） |

## output（落盘）

| 字段 | 说明 |
|---|---|
| `dir` | 输出目录（默认 `./results`） |
| `unique_dir` | true 加时间戳子目录 + `latest` 软链（默认 false，保留 resume 语义） |
| `sinks` | 默认 `[local_fs]`；`hf_dataset`（`repo/path_in_repo/token_env/full_run/sanitize/status`）/ `hf_model` |

## hooks

`None`=内置默认（quant_sanity + eval_sanity）；显式列表=只用这些；`[]`=全关。
