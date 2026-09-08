#!/bin/bash
# 沉淀：本会话（kimi 7f851d8a，含子代理）+ milan_copilot 全部 copilot session
set -u
AQA=~/.venv-aqa-gpu/bin/aqa
DEMO=/home/kaokaolv/auto_agent/my_commit/auto-quant/demo/trajectory_export
KSESS=/home/kaokaolv/.kimi-code/sessions/wd_kaokaolv_25b9c78aa31b/session_7f851d8a-b474-4658-9969-f1ec7246362e
CSESS=/home/kaokaolv/milan_copilot/session-state
LOG=$DEMO/batch_export_$(date +%Y%m%d_%H%M%S).log

process_dir() {  # $1=目标目录（已含 trajectories.jsonl）
    local dest=$1
    ( cd "$dest" && \
      $AQA traces export --path trajectories.jsonl --view sft --out sft.jsonl && \
      $AQA traces export --path trajectories.jsonl --view rl --out rl.jsonl && \
      $AQA traces export --path trajectories.jsonl --view sft --sharegpt --out sharegpt.jsonl && \
      $AQA traces distill --path trajectories.jsonl --knowledge-root "$dest/knowledge" ) >> "$LOG" 2>&1 \
      && echo "OK   $dest" | tee -a "$LOG" || echo "FAIL $dest" | tee -a "$LOG"
}

echo "=== kimi main session ===" | tee -a "$LOG"
dest=$DEMO/kimi_7f851d8a
mkdir -p "$dest" && rm -f "$dest"/*.jsonl && rm -rf "$dest/knowledge"
$AQA traces convert --path "$KSESS/agents/main/wire.jsonl" --backend kimi --out "$dest" >> "$LOG" 2>&1 \
  && process_dir "$dest" || echo "FAIL convert kimi main" | tee -a "$LOG"

echo "=== kimi subagents ===" | tee -a "$LOG"
for w in "$KSESS"/agents/agent-*/wire.jsonl; do
    [ -f "$w" ] || continue
    name=$(basename "$(dirname "$w")")           # agent-N
    dest=$DEMO/kimi_7f851d8a_$name
    mkdir -p "$dest" && rm -f "$dest"/*.jsonl && rm -rf "$dest/knowledge"
    $AQA traces convert --path "$w" --backend kimi --out "$dest" >> "$LOG" 2>&1 \
      && process_dir "$dest" || echo "FAIL convert $name" | tee -a "$LOG"
done

echo "=== milan_copilot sessions ===" | tee -a "$LOG"
for d in "$CSESS"/*/; do
    ev="$d/events.jsonl"
    [ -f "$ev" ] || continue
    sid=$(basename "$d")
    dest=$DEMO/copilot_${sid:0:8}
    mkdir -p "$dest" && rm -f "$dest"/*.jsonl && rm -rf "$dest/knowledge"
    $AQA traces convert --path "$ev" --backend copilot --out "$dest" >> "$LOG" 2>&1 \
      && process_dir "$dest" || echo "FAIL convert $sid" | tee -a "$LOG"
done
echo "ALL DONE" | tee -a "$LOG"
