#!/usr/bin/env python3
"""Regenerate prompts.jsonl from the vendored dpg_bench.csv (see README.md).

Usage: python prepare_prompts.py  (run from this directory)
"""
from pathlib import Path

from aqa.eval.dpgbench import convert_dpg_csv

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    records = convert_dpg_csv(HERE / "dpg_bench.csv", HERE / "prompts.jsonl")
    n_q = sum(len(r["questions"]) for r in records)
    print(f"wrote {len(records)} prompts ({n_q} questions) -> {HERE / 'prompts.jsonl'}")
