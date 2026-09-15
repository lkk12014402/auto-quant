# Patch: flashinfer 0.6.18 for B300 (sm_103a) eval

## 症状
eval 阶段 vllm engine core 初始化失败：flashinfer JIT 编译 trtllm-gen FMHA
(`fmha_gen/csrc_trtllm_fmha_kernel_launcher.cu`) 报
`CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE` /
`CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY` /
`CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK` undefined → Ninja build failed。

## 根因
venv 由 `packages: auto` provision，vllm 0.28.0 拉入 **flashinfer-python 0.6.16.post3**
且**无 flashinfer-cubin**，只能在 B300(compute_103a) 上 JIT 编译 trtllm-gen FMHA，
其头文件引用的 CUDA 符号在当前 toolkit 缺失 → 编译失败。

## 修复（owner 应用，用户指定参考 muse_new 环境）
参考 `source /models/miniforge3/etc/profile.d/conda.sh && conda activate muse_new`
（flashinfer 0.6.18 + flashinfer-cubin 0.6.18，torch 2.13.0+cu130，可在本机正常 serve）。
flashinfer 0.6.18 自带 **flashinfer-cubin**（预编译 sm_103a 核，免 JIT）。两包均为
`py3-none-any` purelib，同为 py3.12，直接镜像进 venv：

    VENV_SP=/root/.venv-qwen235b/lib/python3.12/site-packages
    MN_SP=/models/miniforge3/envs/muse_new/lib/python3.12/site-packages
    rm -rf $VENV_SP/flashinfer $VENV_SP/flashinfer_python-*.dist-info $VENV_SP/flashinfer_cubin*
    cp -a $MN_SP/flashinfer $MN_SP/flashinfer_cubin \
          $MN_SP/flashinfer_python-0.6.18.dist-info \
          $MN_SP/flashinfer_cubin-0.6.18.dist-info $VENV_SP/

（等价：`pip install flashinfer-python==0.6.18 flashinfer-cubin==0.6.18`，若配置了
flashinfer cu130 wheel 索引 https://flashinfer.ai/whl/cu130/；代理 PyPI 无 cubin 0.6.18。）

## 复现建议
从零 provision 会重装 0.6.16.post3 → 需在 provision 后重复上面镜像步骤；
或后续 run 用 `env.create: false` 复用已修好的 ~/.venv-qwen235b。
