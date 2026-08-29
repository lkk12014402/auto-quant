调研完成。以下是结构化报告（中文，技术术语保留英文），所有结论均附源码/文档链接。本地 aqa 侧已读：`aqa/sandbox/{base,local}.py`、`aqa/agent_eval/{scaffold,stats,atif,proxy}.py`、`aqa/eval/code_exec.py`、`aqa/hooks/__init__.py`。

---

# OpenAI Codex (codex-rs) harness 能力调研 → aqa 借鉴清单

## 0. 总览：codex-rs 现在的形态

codex-rs 已演进为 ~70 个 crate 的 workspace（[codex-rs 目录树](https://github.com/openai/codex/tree/main/codex-rs)）。对 harness 视角最重要的分层：

- `protocol/` — 全部 wire types（Submission/EventMsg、approvals、permissions、models），与 UI 解耦（[protocol/src](https://github.com/openai/codex/tree/main/codex-rs/protocol/src)）
- `core/` — agent loop + session/turn 状态机 + sandbox/exec 编排（[core/src](https://github.com/openai/codex/tree/main/codex-rs/core/src)）
- `execpolicy/` — 声明式命令审批规则引擎 + CLI（[execpolicy](https://github.com/openai/codex/tree/main/codex-rs/execpolicy)）
- `sandboxing/` + `bwrap/` + `linux-sandbox/` + `network-proxy/` + `process-hardening/` — OS 级沙箱（[目录树](https://github.com/openai/codex/tree/main/codex-rs)）
- `rollout/` + `thread-store/` — session rollout 持久化/检索/resume（[rollout/src/lib.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/lib.rs)）
- `exec/` — 非交互 `codex exec`（[exec/src](https://github.com/openai/codex/tree/main/codex-rs/exec/src)）
- `hooks/` — 生命周期 hook 引擎（schema + runtime）（[hooks crate](https://github.com/openai/codex/tree/main/codex-rs/hooks)）

为什么说它是好 *harness* 而不仅是 agent：每一层都是独立 crate + 独立 CLI 入口（`codex execpolicy check`、`codex sandbox linux`、`codex exec --json`），policy/事件/持久化全部 schema 化（serde + JSON Schema + ts-rs 导出），headless 路径（`codex exec`）与 TUI 共用同一个 core，事件流本身就是机器可读的评测数据源。

---

## 1. Agent loop 架构

- **Loop 形状**：`SessionTask` trait（`kind()` + `run()`），具体实现为 `RegularTask` / `ReviewTask` / `CompactTask` / `UserShellTask`（[core/src/tasks/](https://github.com/openai/codex/tree/main/codex-rs/core/src/tasks)、[tasks/regular.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/tasks/regular.rs)）。`RegularTask.run` 发 `TurnStarted` 事件 → 循环调 `run_turn(sess, ctx, input, ..., CancellationToken)`（[core/src/session/turn.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs)），turn 间检查 `input_queue.has_pending_input`（支持 steering/interrupt），`terminal_error` 置位即停。
- **TurnContext**：每次 turn 的不可变上下文（model client、cwd、sandbox policy、approval policy、turn 计时等），`per-turn` 重建而非全局可变（[core/src/session/turn_context.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn_context.rs)）。
- **Tool 协议**：shell / unified_exec（PTY）、`apply_patch`（虚拟 CLI：主二进制检测到 `arg1 == --codex-run-as-apply-patch` 时变身 patch 应用器，见 [core/README.md](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/README.md) 与 `arg0/` crate）、web_search（`disabled|cached|indexed|live`，默认 cached 走 OpenAI 索引以降 prompt-injection 面）、MCP tools、plan/todo、多 agent collab 工具（[config reference](https://developers.openai.com/codex/config-reference)）。
- **Model client 抽象**：`model_providers.<id>` 声明式 provider（`base_url` / `env_key` / `wire_api="responses"` / `request_max_retries` / `stream_idle_timeout_ms` / `stream_max_retries`），内置 openai/ollama/lmstudio（[config reference](https://developers.openai.com/codex/config-reference)）；client 本体在 [core/src/client.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/client.rs)（102KB，含 SSE 重试 `responses_retry.rs`）。
- **Hooks 已内建进 turn loop**：`run_hooks_and_record_inputs` 在 turn 输入落盘前执行（[tasks/regular.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/tasks/regular.rs)），runtime 在 [core/src/hook_runtime.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/hook_runtime.rs)，EventMsg 有 `HookStarted/HookCompleted`（[rollout/src/policy.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/policy.rs)）。

---

## 2. Sandbox 模型（首要借鉴对象）

### 2.1 两层控制：sandbox mode × approval policy

- `sandbox_mode = read-only | workspace-write | danger-full-access`；`approval_policy = untrusted | on-request | never | { granular = {...} }`（`on-failure` **已 deprecated**）。granular 可按类别单独开关：`sandbox_approval / rules / mcp_elicitations / request_permissions / skill_approval`（[config reference](https://developers.openai.com/codex/config-reference)）。
- 新版 `[permissions.<name>]` split 文件系统策略：路径/glob → `read|write|deny`，支持 `extends`、`:workspace_roots` 作用域、嵌套 carveout（write 下再 deny、deny 下再 reopen write），不可安全执行的策略 **fail closed**（[config reference](https://developers.openai.com/codex/config-reference)、[core/README.md](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/README.md)）。
- workspace-write 的 protected paths：`<root>/.git`（含 `gitdir:` 指针解析目标）、`.agents`、`.codex` 递归只读（[sandbox 文档](https://developers.openai.com/codex/sandbox)）。

### 2.2 Linux 执行机制（现在是 bwrap，不再是 landlock 默认）

来自 [linux-sandbox/README.md](https://raw.githubusercontent.com/openai/codex/main/codex-rs/linux-sandbox/README.md)：

- 默认 **bubblewrap**：`--ro-bind / /`（全盘只读起步）→ `--bind` 叠加 writable roots → 对 protected subpaths（`.git`/`.codex`）再 `--ro-bind`；重叠策略按 path-specificity 排序应用。
- `--unshare-user` + `--unshare-pid`；网络受限且无 proxy 时 `--unshare-net`；进程内 `PR_SET_NO_NEW_PRIVS` + **seccomp 网络过滤器**。
- deny glob（如 `**/*.env = "none"`）在启动前用 `rg --files` 展开并 mask；writable root 内的 symlink/缺失 protected 路径用 `/dev/null` 挂载封堵。
- Landlock 仅作 legacy fallback（`features.use_legacy_landlock=true`）；CLI 仍叫 `codex debug landlock` / `codex sandbox linux`。
- 托管 proxy 模式下 `--unshare-net` + TCP→UDS→TCP 桥，桥起后 seccomp 再禁新 AF_UNIX/socketpair。
- macOS = Seatbelt（`sandbox-exec -p`），Windows = restricted token/elevated backend（[sandbox 文档](https://developers.openai.com/codex/sandbox)、[core/src/sandboxing/mod.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/sandboxing/mod.rs)）。
- **网络 domain 策略**（`network-proxy` crate）：allowlist-first，`*.example.com` 只管子域、`**.` 含 apex、`deny` 恒胜；默认 block loopback/private（`allow_local_binding=false`）；allow 前做 DNS/IP 分类检查防 DNS rebinding（[sandbox 文档](https://developers.openai.com/codex/sandbox)）。

### 2.3 execpolicy 规则引擎（读了实际源码）

- **文件格式**：`.rules` 文件用 **Starlark**（无副作用方言），builtins：`prefix_rule(pattern, decision?, justification?, match?, not_match?)`、`network_rule(host, protocol, decision, ...)`、`host_executable(name, paths)`（[parser.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/parser.rs)、[execpolicy/README.md](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/README.md)、[rules 文档](https://developers.openai.com/codex/rules)）。
- **匹配语义**：命令按 argv token 列表匹配；`pattern` 元素可为字面量或字面量列表（alternatives）；`Policy` 内部 `MultiMap<first_token, RuleRef>` 按首 token 索引；可选 basename fallback（`/usr/bin/git` → `git` 规则，受 `host_executable` 白名单约束）（[policy.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/policy.rs)、[rule.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/rule.rs)）。
- **Decision**：`Allow < Prompt < Forbidden`，多规则命中取 **max（最严格者胜）**（[decision.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/decision.rs)、[policy.rs `Evaluation::from_matches`](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/policy.rs)）。无命中时走 `heuristics_fallback`（core 侧的可疑命令启发式，`HeuristicsRuleMatch`）。
- **内联单测**：`match` / `not_match` 示例在 **加载时校验**，不通过则 policy 加载失败（[rule.rs `validate_match_examples`](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/rule.rs)）。
- **复合命令防走私**：`bash -lc "a && b"` 若为纯线性链（无重定向/替换/变量/glob/控制流），用 **tree-sitter** 拆成单命令逐条过规则，最严格者胜；否则整条按 `["bash","-lc",script]` 保守匹配（[rules 文档](https://developers.openai.com/codex/rules)）。
- **闭环**：审批弹窗可携带 `proposed_execpolicy_amendment`，用户接受 `ApprovedExecpolicyAmendment` 后写入 `~/.codex/rules/default.rules`，下次免问（[protocol/src/approvals.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/protocol/src/approvals.rs)、[rules 文档](https://developers.openai.com/codex/rules)）；admins 可从 `requirements.toml` 强制 forbidden 规则。
- **调试 CLI**：`codex execpolicy check --rules ... [--resolve-host-executables] -- cmd args`，输出 JSON `{matchedRules, decision}`（[execpolicycheck.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/execpolicycheck.rs)）。

---

## 3. Rollout / session 日志（对照 ATIF）

- **文件布局**：`$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>[_<rollout_id>].jsonl`（reverted thread 追加第二个 id）；可选 zstd 压缩 + 后台 compression worker；另有 SQLite state DB、session index、reverse JSONL scanner（[rollout_file_name.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/rollout_file_name.rs)、[rollout/src/lib.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/lib.rs)）。
- **行格式**：`RolloutLine { timestamp, ordinal?, item: RolloutItem }`；`RolloutItem = SessionMeta | ResponseItem | Compacted | TurnContext | EventMsg | WorldState | SecurityRiskScore | InterAgentCommunication | RealtimeItem`（[rollout/src/lib.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/lib.rs)、[rollout/src/policy.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/policy.rs)）。
- **持久化白名单**（`is_persisted_rollout_item`）：ResponseItem 里 message/reasoning/function_call/output/web_search/compaction 全存；EventMsg 里只存 durable 的（`TokenCount`、`TurnStarted`、`TurnComplete`、`ItemCompleted`(Paginated 模式)、`ThreadRolledBack`…），**transient 事件（delta、approval request、StreamError 等）一律不落盘** —— 这是"日志即恢复状态"设计的关键：rollout 既是审计日志又是 resume 的状态源（[rollout/src/policy.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/rollout/src/policy.rs)）。
- **Resume**：`InitialHistory / ResumedHistory` 从 rollout 重建模型上下文（[core/src/session/rollout_reconstruction.rs](https://github.com/openai/codex/blob/main/codex-rs/core/src/session/rollout_reconstruction.rs)）；`codex exec resume --last | <SESSION_ID>`（[noninteractive 文档](https://developers.openai.com/codex/noninteractive)）；`--ephemeral` 可不写盘。
- **Token 核算**：`EventMsg::TokenCount` 持久化，含 total/last（`last_token_usage` 即 aqa `parse_usage_from_text` 现在抓的字段）；另有实验性 `features.rollout_budget`（token 预算 + reminder 间隔）（[config reference](https://developers.openai.com/codex/config-reference)）。
- **对 ATIF 的差距**：ATIF/aqa 目前是"评测后导出"（trajectory-cumulative，无 per-step 指标，scaffold 轨迹只有 1 个 agent step + 4000 字符尾巴，`atif.py` docstring 自述）；codex rollout 是"运行中 append-only 持久化 + 每行 timestamp/ordinal + per-turn token + 可 resume"。

---

## 4. `codex exec` 非交互模式与 JSON 事件

- **事件 schema**（`--json` → stdout JSONL；progress 在 stderr，final message 可 `-o` 落盘）：`ThreadEvent = thread.started | turn.started | turn.completed{usage} | turn.failed | item.started/updated/completed | error`。`Usage { input_tokens, cached_input_tokens, cache_write_input_tokens, output_tokens, reasoning_output_tokens }`。`ThreadItem` details：`agent_message / reasoning / command_execution{command, aggregated_output, exit_code, status} / file_change{changes[{path,kind}],status} / mcp_tool_call / web_search / todo_list / collab_tool_call / error`（完整定义：[exec/src/exec_events.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/exec/src/exec_events.rs)；渲染器：[event_processor_with_jsonl_output.rs](https://github.com/openai/codex/blob/main/codex-rs/exec/src/event_processor_with_jsonl_output.rs)；官方示例：[noninteractive 文档](https://developers.openai.com/codex/noninteractive)）。
- ⚠️ 注意：当前 main 的 `turn.completed.usage` 是新 schema；aqa scaffold 抓的 `last_token_usage` 是 legacy `EventMsg::token_count`（`{"msg":{"type":"token_count","info":{"last_token_usage":...}}}`）形状。`parse_usage_from_text` 的 DFS 找 key 写法两种都能兜住，但要意识到 schema 漂移是常态。
- **评测向特性**：`--output-schema <json-schema>` 让 final message 变成 schema 约束的结构化输出（配合 `-o` 写文件）；`--ephemeral`；`--sandbox read-only` 是 exec 默认值；`--skip-git-repo-check`；`--ignore-user-config` / `--ignore-rules`（干净自动化环境）；`mcp_servers.<id>.required=true` 时 MCP 初始化失败即非零退出（fail-fast 依赖检查）；`codex exec resume --last` 两段式 pipeline（[noninteractive 文档](https://developers.openai.com/codex/noninteractive)）。
- **官方没有独立 evals crate**（repo 里没有 eval 目录；评测逻辑靠 tests + `codex exec --json` 的可消费性，第三方 harness 如 [supabase/evals](https://www.marktechpost.com/2026/08/01/supabase-releases-evals-an-open-source-benchmark-that-scores-claude-code-codex-and-opencode-on-real-supabase-tasks/) 就是这么驱动 Codex 的）。另有 opt-in **OTel**：`codex.conversation_starts / api_request / sse_event(含 token) / tool_decision(approved|denied, source) / tool_result`（[sandbox 文档 §telemetry](https://developers.openai.com/codex/sandbox)）。

---

## 5. Verdict：aqa 该借什么、跳过什么

### 借（按优先级）

1. **execpolicy 规则引擎 → 新模块 `aqa/sandbox/policy.py`，替换/补强 `base.py:is_unsafe_command` 的正则启发式。**
   - 移植语义而非实现：argv token 化（`shlex.split`，aqa 现在跑 `sh -c` 整串，需要先拆）；prefix pattern + alternatives；`allow/prompt/forbidden` 三态 + strictest-wins；`match`/`not_match` 内联单测（加载时校验，直接可做成 pytest）；rules 文件用 TOML/Python DSL 即可，**不必引 Starlark**。
   - 照搬"复合命令拆分"规则：`sh -c "a && b"` 只对纯线性链拆分逐条判定，否则保守整体匹配 —— 正好补上 aqa 现在正则对 `curl ... | sh` 这类拼接的脆弱性（源码依据见 §2.3）。
   - `approval_policy` 映射到 aqa 没有交互的场景：`prompt` 在 headless 下即 `forbidden`（codex 在 `never` 下也是这么处理的，[decision.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/execpolicy/src/decision.rs) 注释）。
   - 附赠 CLI：`aqa sandbox check <cmd>`（对标 `codex execpolicy check`），让规则可单测可调试。

2. **bwrap 后端 → `aqa/sandbox/` 新增 `BwrapSandbox`（或给 `LocalSandbox` 加 `isolation="bwrap"`）。** aqa 主机是 Linux，bubblewrap 零守护、无 root 依赖（user namespace）。照搬参数模板：`--ro-bind / /` + `--bind <workdir>` + `--ro-bind` 保护 `.git`/`.codex` + `--unshare-net`（无网时）+ `--unshare-pid`；网络开关成为一等配置（对应 codex `network_access`）。这让"LocalSandbox + 正则"从 *best-effort guard* 升级为 *实际隔离*，且比 DockerSandbox 轻（依据：[linux-sandbox/README.md](https://raw.githubusercontent.com/openai/codex/main/codex-rs/linux-sandbox/README.md)）。deny glob（`**/*.env = "none"` → 启动前 rg 展开 + mask）对 aqa 防 API key 泄漏进沙箱特别值钱。

3. **CodexScaffold 升级 → `aqa/agent_eval/scaffold.py`。**
   - `parse_usage` 增加新 schema 分支：`turn.completed.usage`（当前只认 `last_token_usage`，依据 [exec_events.rs](https://raw.githubusercontent.com/openai/codex/main/codex-rs/exec/src/exec_events.rs)）。
   - 加 `--output-schema` 透传（`extra_args` 已支持，建议在 ScaffoldContext 加显式字段）：让 codex 直接吐结构化 verdict/summary，替代现在"output 尾巴塞 ReActStep"。
   - 多阶段 eval 用 `codex exec resume --last`（两段式 quant→fix pipeline 可直接复用）。
   - 关键改进：**别丢 rollout**。scaffold 跑完后从沙箱内 `$CODEX_HOME/sessions/**/rollout-*.jsonl` 捞 rollout 文件喂给 `atif.py` —— 每行 ResponseItem + TurnContext + TokenCount 就是完整逐步轨迹，直接修掉 atif.py "scaffold 只有单步摘要"的缺口（§3）。

4. **Rollout 式持久化 → `aqa/agent_eval/atif.py` / `aqa/agent/trajectories.py`。** 借三点：append-only JSONL + 每行 `timestamp/ordinal`；durable/transient 持久化白名单（只落 token_count、turn 边界、item 完成，不落 delta）；`InitialHistory` 式 resume（aqa 的 fix_loop 重跑目前是无状态的）。per-turn usage（`turn.completed`）还能补上 ATIF per-step metrics 的空缺。

5. **Hook 事件模型 → `aqa/hooks/`（phase-hook 层）。** codex `[hooks]` 的 taxonomy：matcher group + handler，`PreToolUse/PostToolUse/PermissionRequest/PreCompact/PostCompact/SessionStart/SessionEnd`，handler 支持 `async=true`（后台不阻塞）和 `additionalContextLimit`（hook 输出 token 上限，超出落盘只给预览）（[config reference](https://developers.openai.com/codex/config-reference)、[hooks crate](https://github.com/openai/codex/tree/main/codex-rs/hooks)）。aqa 现在只有 phase-boundary hook（after_quantize/after_benchmark）；值得借：(a) `async` 标志（`run_hook_chain` 目前是同步串行）；(b) 输出体积上限（防 hook 污染 fix-loop 签名）；(c) 若 aqa-react loop 要加 PreToolUse 级拦截，直接对 `react.py` 的 Command 执行点加 hook 相位，与 execpolicy 的 `prompt/forbidden` 判定天然衔接。

6. **Token 预算 → `aqa/agent_eval/proxy.py`。** codex `features.rollout_budget`（limit_tokens + reminder_interval_tokens + prefill/sampling 权重）可映射为 ServingProxy 的 token 预算（现在只有 max_turns）；fail-closed 语义照 codex：`max_turns` 用尽抛 AqaError 已一致。

### 跳过

- **Rust 特有实现**：Starlark 求值器、tree-sitter 拆分器（借语义即可，Python 用 `shlex`+简单 tokenize 够 aqa 的场景）、serde/ts-rs 导出。
- **TUI / app-server / IDE / ChatGPT desktop**：整套 approval 人机交互、editor 集成、notifications。
- **MCP / connectors / multi-agent collab / memories / realtime / cloud-tasks**：aqa 无对应面。
- **Seatbelt/Windows sandbox**：aqa 只跑 Linux。
- **OTel exporter**：aqa 已有 `report/sinks.py`；只需借 `tool_decision`/`tool_result` 事件*命名*做日志规范，不必引 OTel 依赖。
- **`danger-full-access` 之外无中间态的老 `SandboxPolicy`**：codex 自己也在向 `[permissions]` split 策略迁移，aqa 新写就直接上 split 模型（read/write/deny per path），别抄 legacy 三档。

### 一句话总结

对 aqa 最值钱的不是 agent loop，而是 codex 把 harness 三件套做成了**可独立测试的 schema 化组件**：execpolicy（声明式命令规则 + 内联单测 + strictest-wins）、bwrap 沙箱参数模板（ro-bind 全盘起步 + 白名单可写 + 网络开关）、rollout（append-only JSONL 同时充当审计日志与 resume 状态）。三者分别落在 `aqa/sandbox/policy.py`（新建）、`aqa/sandbox/` bwrap 后端、`aqa/agent_eval/{scaffold,atif}.py` 的 rollout 消费链上。