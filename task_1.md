你是 owner agent，本仓库的 AGENTS.md 已生效，请先通读并遵守其中的铁律
（先 yaml 后运行、全程留痕、不与他人抢 GPU、把关 fix-loop、偏离要可复现）。

任务：对 Qwen/Qwen3-30B-A3B-Instruct-2507 做三档量化+评估对比，等价于
/home/kaokaolv/auto_agent/my_commit/quant_30B.md 的手工流程。仓库里已有现成
yaml（我已审查过内容），不要新写：
  examples/local_venv/run_qwen3_30b_mixed.yaml        （档1：MXFP8主体+专家MXFP4）
  examples/local_venv/run_qwen3_30b_fp8kv.yaml        （档2：档1+静态FP8 KV）
  examples/local_venv/run_qwen3_30b_fp8kv_fp8attn.yaml（档3：档2+静态FP8 attention）

执行要求：
1. 模型路径：yaml 里写的是 /models/Qwen3-30B-A3B-Instruct-2507，本机若没有
   就改用 HF id Qwen/Qwen3-30B-A3B-Instruct-2507 下载（改 yaml 里 model 字段）。
2. 先用 nvidia-smi 确认空闲卡。三个档串行跑即可，CUDA_VISIBLE_DEVICES 用
   第 X 号卡（你在启动前告诉我哪张）。不要动别人在用的卡。
3. 首次运行会先 provision ~/.venv-qwen30b（conda/uv 优先，auto-round+lm-eval+vllm），
   让它跑完；后续档幂等复用。provision/量化/评估日志都在 output.dir 下。
4. 每档命令：aqa run examples/local_venv/<档>.yaml。逐档跑，一档完成再跑下一档。
5. 每档完成后 aqa inspect <run_dir> 验证产物完整性，并记录 accuracy.json 分数。
6. 三档完成后用 aqa compare 两两对比，汇总一张对比表（含模型体积 du -sh）。
7. 全程记工作日志到 ./results/qwen3_30b_a3b/OWNER_JOURNAL.md：每步命令、
   日志路径、分数、遇到的问题与解决、你的分析。
8. 如果量化/评估报错且 fix-loop 介入，你要审查 logs/agent_fixes/<phase>/ 的
   session 记录确认它改得对，不要盲目放行；修不通就停下来报告，不要无限尝试。
9. 不要 commit 任何文件（尤其 env.sh）；不要手改 venv 里的包——需要改依赖
   先问我。

完成后汇报：三档的 gsm8k/piqa/mmlu/hellaswag 分数对比、模型体积、
与 bf16 相比的判断、产物目录路径。

