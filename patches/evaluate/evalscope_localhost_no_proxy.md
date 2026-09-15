# Patch: evalscope harness 对 localhost 绕过 HTTP 代理

## 症状
evalscope 阶段：vllm serve 已 "Application startup complete"，但 harness 的
`_wait_ready` 一直探测不到 ready（挂起，直到 startup_timeout=3600s 才失败）。
手测：`curl http://127.0.0.1:8801/v1/models` 经代理返回 403，`--noproxy '*'` 返回 200。

## 根因
env.sh 导出 LOCAL_HTTP_PROXY/HTTPS_PROXY（公司代理）。harness 在 127.0.0.1 起
vllm serve，但 `_wait_ready`(urllib) 与 evalscope 的 OpenAI 客户端请求本地 server
时都会走代理 → 代理对 localhost 返回 403。健康探测永不 200 → 挂起。

## 修复（reproducible，patches/evaluate/evalscope_localhost_no_proxy.patch）
aqa/eval/harness/evalscope.py：
1. 新增 `_localhost_no_proxy_env()`：在现有 no_proxy/NO_PROXY 追加
   127.0.0.1,localhost,0.0.0.0,::1（保留其余代理，数据集下载仍走代理）。
2. `_wait_ready`：用 `build_opener(ProxyHandler({}))` 对本地探测绕过代理。
3. `run_group`：evalscope eval 子进程用 `stream_subprocess(..., env=_localhost_no_proxy_env())`，
   使 evalscope→本地 server 的请求也绕过代理。

只对 localhost 绕过，不影响需要代理的外部访问；无代理环境行为不变。
