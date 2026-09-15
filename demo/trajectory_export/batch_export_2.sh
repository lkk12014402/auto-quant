#!/bin/bash
# 沉淀：kimi 7f851d8a（本 session，含子代理）+ kimi 4f880e4d（前序 session，含子代理）
set -u
AQA=~/.venv-aqa-gpu/bin/aqa
DEMO=/home/kaokaolv/auto_agent/my_commit/auto-quant/demo/trajectory_export
BASE=/home/kaokaolv/.kimi-code/sessions/wd_kaokaolv_25b9c78aa31b
LOG=$DEMO/batch_export_$(date +%Y%m%d_%H%M%S).log

process_dir() {  # $1=目标目录（已含 trajectories.jsonl）
    local dest=$1
    ( cd "$dest" && \
      $AQA traces export --path trajectories.jsonl --view sft --out sft.jsonl && \
      $AQA traces export --path trajectories.jsonl --view rl --out rl.jsonl && \
      $AQA traces export --path trajectories.jsonl --view sft --sharegpt --out sharegpt.jsonl && \
      rm -rf knowledge && \
      $AQA traces distill --path trajectories.jsonl --knowledge-root "$dest/knowledge" ) >> "$LOG" 2>&1 \
      && echo "OK   $dest" | tee -a "$LOG" || echo "FAIL $dest" | tee -a "$LOG"
}

do_session() {  # $1=session 目录名（session_xxx） $2=导出前缀（kimi_xxx）
    local sess=$1 prefix=$2
    echo "=== $prefix (main) ===" | tee -a "$LOG"
    local dest=$DEMO/$prefix
    mkdir -p "$dest" && rm -f "$dest"/*.jsonl
    $AQA traces convert --path "$BASE/$sess/agents/main/wire.jsonl" --backend kimi --out "$dest" >> "$LOG" 2>&1 \
      && process_dir "$dest" || echo "FAIL convert $prefix main" | tee -a "$LOG"
    for w in "$BASE/$sess"/agents/agent-*/wire.jsonl; do
        [ -f "$w" ] || continue
        local name=$(basename "$(dirname "$w")")
        dest=$DEMO/${prefix}_${name}
        mkdir -p "$dest" && rm -f "$dest"/*.jsonl
        $AQA traces convert --path "$w" --backend kimi --out "$dest" >> "$LOG" 2>&1 \
          && process_dir "$dest" || echo "FAIL convert $prefix $name" | tee -a "$LOG"
    done
}

do_session session_7f851d8a-b474-4658-9969-f1ec7246362e kimi_7f851d8a
do_session session_4f880e4d-7721-4c6f-868e-2327b922825a kimi_4f880e4d
echo "ALL DONE" | tee -a "$LOG"
