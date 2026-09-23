#!/usr/bin/env python3
"""Regenerate prompts.jsonl from the WISE_Verified release (see README.md).

Fetches ``data_verified/merge.json`` from the GitHub repo ``PKU-YuanGroup/WISE``
(unless ``--source`` points at a local copy) and converts it via
``aqa.eval.wise.convert_wise_prompts``.

Usage:
  python prepare_prompts.py
  python prepare_prompts.py --source /path/to/merge.json
"""
import argparse
import urllib.request
from pathlib import Path

from aqa.eval.wise import convert_wise_prompts

HERE = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/PKU-YuanGroup/WISE/main/data_verified/merge.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="local merge.json (default: fetch from GitHub)")
    args = parser.parse_args()
    if args.source:
        source = Path(args.source)
    else:
        source = HERE / "merge.json"
        with urllib.request.urlopen(URL) as resp:
            source.write_bytes(resp.read())
    records = convert_wise_prompts(source, HERE / "prompts.jsonl")
    print(f"wrote {len(records)} prompts -> {HERE / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
