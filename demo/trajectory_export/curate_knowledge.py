#!/usr/bin/env python3
"""Curated knowledge sediment — 把 session 的关键经验人工提炼进共享 knowledge store。

用法（repo 根目录下）：

    source env.sh          # 提供 LOCAL_RESULTS_DATASET / LOCAL_RESULTS_HF_TOKEN
    python demo/trajectory_export/curate_knowledge.py            # 全流程（含 push）
    python demo/trajectory_export/curate_knowledge.py --no-push  # 只写本地 store

流程（幂等，可重复执行）：
  1. sync_down：拉远端共享 store（HF dataset 的 knowledge/ 子树），合并而非覆盖；
  2. 合并本目录各 session 的机械 distill（knowledge/trajectory.jsonl，按 signature 去重）
     进 store 的 trajectories/trajectory.jsonl；
  3. 写入 CURATED 里的人工提炼 lessons（LessonStore，按 signature 去重）——
     这是脚本的核心资产：error signature → 修复方案，fix-loop 命中同签名错误时
     可不调 agent 直接重放；
  4. push_up：把整个 store 推回 HF dataset（--no-push 跳过）。

新增 session 的经验时：在 CURATED 里追加条目（phase/error_context/status/solution_note），
error_context 尽量用真实报错原文（signature 从它提取，决定日后能否被命中）。
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aqa.agent.knowledge import KnowledgeStore  # noqa: E402

DEMO = Path(__file__).resolve().parent

CURATED = [
    # ---- quantize ----
    dict(phase="quantize", status="fixed", model_id="Qwen/Qwen3.5-9B", scheme="MXFP8",
         error_context="ValueError: Found different quantization schemes for the shards of "
                       "language_model.model.layers.0.mlp.gate_up_proj. "
                       "vLLM requires all to use the same scheme.",
         solution_note="ignore_layers 里有裸 `gate`（正则 re.search 命中 mlp.gate_proj）→ gate_proj 留 bf16 而 "
                       "up_proj 被量化，vLLM 融合 gate_up_proj 时 shard scheme 不一致直接拒载。纪律：① ignore 禁止裸 gate，"
                       "要保路由器写 `mlp.gate`（精确命中 41 个）；② 写 ignore 前查 vLLM 融合边界（stacked_params_mapping），"
                       "同组融合权重（gate_up_proj / in_proj_qkvz / in_proj_ba）必须同进同出；③ 主层 MoE 专家是单个融合模块 "
                       "mlp.experts，只能整体量化或整体保留。9B 实测硬故障、35B 实测命中 379 模块静默漏量。"),
    dict(phase="quantize", status="fixed", model_id="Qwen/Qwen3.5-9B", scheme="MXFP4",
         error_context="ValueError: There is no module or parameter named 'visual' in Qwen3_5Model "
                       "(plus checkpoint 体积 ≈2x 预期、键名出现 model.language_model.language_model.* 双前缀)",
         solution_note="method: autoround（model-load）嵌套 config 的 VLM 必须加 --disable_low_cpu_mem_usage。默认 "
                       "low_cpu_mem_usage=True 用 from_config 造 meta 骨架，保存时权重改名表 _weight_conversions 缺失，"
                       "PrefixChange 被丢弃 → 键名双前缀且量化张量全挂伪名（正确名下是 bf16，= 量化静默失效）。"
                       "修复实测：887 键 / 双前缀 0 / 192 量化张量全在正确前缀 / 44GB→12GB。model_free 路径免疫。"),
    dict(phase="quantize", status="fixed", model_id="Qwen/Qwen3.5-35B-A3B", scheme="MXFP8",
         error_context="ValueError: There is no module or parameter named "
                       "'layers.0.mlp.shared_expert_gate.weight_scale' in Qwen3_5MoeForCausalLM",
         solution_note="model-load 的 predefined ignore 用 name.endswith('.gate') 兜底，保得住 mlp.gate 但保不住 "
                       "mlp.shared_expert_gate（_gate 结尾）→ 被量化写出 weight_scale，而 vLLM 侧它是透传属性无处映射。"
                       "V2/V3（autoround 路径）ignore 必须手写 shared_expert_gate（model_free 的子串清单能自动保住）。"),
    dict(phase="quantize", status="note", model_id="Qwen/Qwen3.5-9B", scheme="MXFP4",
         error_context="ignore_layers containing backslashes (e.g. model\\.visual) silently fails to match on "
                       "the model-load path; protected modules get quantized",
         solution_note="method: autoround 把 ignore 序列化成 `re:.*<pat>.*` 并对反斜杠二次转义 → `model\\.visual` 永不匹配 "
                       "→ vision 被静默量化（无报错，只能靠体积/census 抓）。model-load 的 ignore 一律写无反斜杠裸词；"
                       "model_free 展开成字面量清单不受影响。量化后必做 census：体积、量化模块数、双前缀、保护清单抽验。"),
    dict(phase="quantize", status="fixed", model_id="Qwen/Qwen3.5-9B", scheme="MXFP8",
         error_context="ValueError: Cannot determine num_attention_heads from config. Config must define "
                       "either `num_attention_heads` or both `hidden_size` and `head_dim`. Qwen3_5Config",
         solution_note="compressed-tensors 的 get_num_attn_heads/_kv_heads/_head_dim 只读 root config，composite config "
                       "（Qwen3.5/Qwen3-VL 这类，字段只在 text_config）不落 root → 静态 KV/attn scale 初始化崩。"
                       "补丁：aqa patches/quantize/compressed_tensors_nested_config_attn_dims.patch（_attn_config_scope 下沉一层，"
                       "扁平 config 零行为变化），基线 compressed-tensors 0.17.0，值得提上游 PR。仅 static_kv/attention（v2/v3）触发。"),
    dict(phase="quantize", status="note", model_id="Qwen/Qwen3.5-35B-A3B", scheme="MXFP8",
         error_context="V2 checkpoint is 0.76 GiB LARGER than V1 despite same ignore list; "
                       "quantized module census differs by exactly 776 (all mtp.*)",
         solution_note="model-load（autoround iters=0）按 named_modules() 选层，Qwen3.5 系列模块树里没有 mtp.* → V2/V3 不量化 MTP，"
                       "model_free V1 量（776 个）。“V2 = V1 + FP8 KV”不严格：后端也从 MXFP optimized RTN 退成 plain RTN。"
                       "vLLM 服务图不加载 MTP 权重，差异只花磁盘不花显存；开投机解码时注意 V2/V3 的 MTP 是 bf16。"),
    dict(phase="quantize", status="note", model_id="Qwen/Qwen3.5-35B-A3B", scheme="MXFP4",
         error_context="ignore 清单选型：照搬同模型 MXFP8 清单到 MXFP4 导致精度显著下降",
         solution_note="MXFP4 清单必须分档自探（结构分析免下载 + 实测归因）。9B(dense)：attention 每族进 4bit 约 −2.7pp gsm8k "
                       "⇒ 只量 dense MLP（97.79%）。35B(MoE 256专家 top-8)：只量路由专家即 99.53%（误差被专家数摊薄）；"
                       "照抄 MXFP8 清单（连 shared_expert/attn 也 4bit）掉到 97.48%，配对检验四项 p<0.01；"
                       "非专家 Linear 被量化时 piqa 最先掉（降 8bit 也挡不住）。9B 的“piqa 对 attention 量化不敏感”不可外推到 35B。"),
    # ---- evaluate ----
    dict(phase="evaluate", status="fixed", model_id="Qwen/Qwen3.5-9B", scheme="MXFP4",
         error_context="gsm8k score collapses to ~0.40 (expected ~0.93) on a thinking-default model; "
                       "strict-match extraction finds no final answer",
         solution_note="Qwen3.5 默认 thinking 且不支持 /think /nothink 软开关：不关 thinking 时思考链在 max_gen_toks=2048 "
                       "截断，strict 抽取失败（实测 0.398→修后 0.9075，与量化无关）。lm_eval 生成任务必须带 "
                       "model_args: enable_thinking=False, reasoning_parser=qwen3；aqa 侧用 eval.thinking 双模式管理。"
                       "evalscope 侧 thinking 默认开属另一口径（max_tokens 30000），两套数字不可跨表比。"),
    dict(phase="evaluate", status="fixed", model_id="muse/Muse-Spark", scheme="MXFP4",
         error_context="RuntimeError: CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY undefined; "
                       "Ninja build failed (trtllm-gen FMHA JIT, sm_103a / B300)",
         solution_note="flashinfer 0.6.17 在 B300(sm_103a) 上 JIT 编 trtllm-gen FMHA 失败 → 装 flashinfer-python + "
                       "flashinfer-cubin 双 0.6.18（预编译核免 JIT）。aqa patches/evaluate/flashinfer_0.6.18_sm103a.md。"
                       "另：日志里 `Failed to import Triton kernels ... triton_kernels.matmul_ogs` 是无害告警，"
                       "FlashInfer MXFP4 路径不需要 triton_kernels。"),
    dict(phase="evaluate", status="fixed", model_id="Qwen/Qwen3.5-9B", scheme="MXFP4",
         error_context="lm_eval process does not exit after results are fully written "
                       "(EngineCore SyncMPClient stuck in futex_wait); aqa waits forever",
         solution_note="vLLM V1 跨进程拆机挂死：results json 已落盘但 lm_eval 不退出。TP=1 时启动前 "
                       "export VLLM_ENABLE_V1_MULTIPROCESSING=0 把 EngineCore 收进本进程。另注意每个 task group 起一次新引擎，"
                       "profiling 窗口每 group 各出现一次；孤儿 vllm worker 按 PGID 清理（proc_kill_orphaned patch）。"),
    dict(phase="evaluate", status="fixed", model_id="Qwen/Qwen3.5-9B", scheme="MXFP4",
         error_context="AssertionError: Error in memory profiling. Initial free memory X GiB, "
                       "current free memory Y GiB (Y > X)",
         solution_note="共享卡显存竞态：co-tenant 在 vLLM profiling 窗口内释放显存触发断言，gpu_memory_utilization 防不住。"
                       "当可重试错误处理：重试循环要包住整个 run（暴露次数 = task group 数），不是单次调用；"
                       "可配 enforce_eager 跳过 cudagraph 估算、调低 max_num_batched_tokens 缩短窗口；挑卡看时段稳定性不看瞬时 free。"),
    dict(phase="evaluate", status="note", model_id="Qwen/Qwen3.5-35B-A3B", scheme="MXFP8",
         error_context="checkpoint on CEPH/network FS: vLLM weight load takes ~2227s",
         solution_note="网络盘必须 model_args 加 safetensors_load_strategy=prefetch（vLLM 默认只对 NFS/Lustre 预取），"
                       "实测 2227s → 45~86s（~26x）。另建议 TP1：TP2 的 WorkerProc 退出残留信号量/共享内存，"
                       "同 run 第二次 engine init 报 WorkerProc initialization failed。"),
    dict(phase="evaluate", status="note", model_id="Qwen/Qwen3.5-35B-A3B", scheme="MXFP4",
         error_context="interpreting small-sample benchmark deltas (aime25 ±5pp between variants; "
                       "quantized model scores above BF16 baseline)",
         solution_note="evalscope avg@5 只压同题解码噪声（aime25 ~4pp），压不掉题目集抽样误差（30 题 SE ~9pp）。"
                       "判差异用配对检验：lm_eval 用 McNemar（同批题），evalscope 按题聚合 avg@5 后符号检验 + 对题 bootstrap。"
                       ">100% 的 Ratio 一律读作噪声不是提升；35B 六变体 12 组配对全不显著 ⇒ “未检测到退化”。"
                       "gpqa 198 题 SE≈2.8pp 相对可用；等值读数（如恰等于 BF16）多为离散步长巧合。"),
    dict(phase="evaluate", status="note", model_id="Z-Image/FLUX.2 (diffusion)", scheme="MXFP8",
         error_context="diffusion MXFP8 生成内容崩坏但 checkpoint 的 text encoder 是 bf16（与 bf16 模型一致）",
         solution_note="vllm-omni OmniINCConfig.get_quant_method 的 MXFP8 特判不查 block_name_to_quantize → "
                       "text encoder 在加载时被 IncMxfp8OfflineLinearMethod 误量化（checkpoint 无辜）。"
                       "修复已合上游 main + aqa patches/fix_mxfp8_text_encoder_quantization.patch（fork PR #1）。"
                       "排查法：/tmp/mxfp8_forward.py 逐层对比反量化参考输出找发散点；先核对 checkpoint 里各组件实际 dtype 再怀疑加载路径。"),
]


def main() -> None:
    push = "--no-push" not in sys.argv[1:]
    store = KnowledgeStore(
        local_dir=Path(os.environ.get("AQA_KNOWLEDGE_DIR", "~/.cache/aqa/knowledge")).expanduser(),
        repo=os.environ.get("LOCAL_RESULTS_DATASET"),
        path_in_repo="knowledge",
        token_env="LOCAL_RESULTS_HF_TOKEN",
        push=push,
    )
    print("sync_down:", store.sync_down())

    # 2) 机械 distill 归集（signature 去重）
    by_sig: dict[str, dict] = {}
    for row in store.trajectories.load("trajectory"):
        by_sig[row.get("signature", "")] = row
    before = len(by_sig)
    for f in sorted(DEMO.glob("kimi_*/knowledge/trajectory.jsonl")) + \
             sorted(DEMO.glob("copilot_*/knowledge/trajectory.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_sig[obj.get("signature", line)] = obj
    path = store.trajectories._path("trajectory")
    path.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in by_sig.values()) + "\n")
    print(f"trajectory lessons: {before} -> {len(by_sig)}")

    # 3) 人工提炼 lessons
    for lesson in CURATED:
        ctx = lesson.pop("error_context")
        store.lessons.save(lesson.pop("phase"), ctx, lesson.pop("status"),
                           lesson.pop("solution_note"), model_id=lesson.pop("model_id"),
                           scheme=lesson.pop("scheme", ""), **lesson)
    print(f"curated lessons saved: {len(CURATED)}")
    for ph in ("quantize", "evaluate", "baseline", "trajectory"):
        print(f"  lessons/{ph}.jsonl: {len(store.lessons.load(ph))}")

    # 4) push
    print("push_up:", store.push_up() if push else "skipped (--no-push)")


if __name__ == "__main__":
    main()
