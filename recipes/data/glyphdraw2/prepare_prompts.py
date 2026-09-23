#!/usr/bin/env python3
"""Regenerate prompts.jsonl from the GlyphDraw2 prompt files (see README.md).

Reads ``prompts/{complex,poster}_{en,zh}.txt`` either from a local checkout
(``--source-dir``) or by fetching them from the GitHub repo
``OPPO-Mente-Lab/GlyphDraw2``, then converts via
``aqa.eval.glyphdraw2.convert_glyphdraw_prompts``.

Usage:
  python prepare_prompts.py
  python prepare_prompts.py --source-dir /path/to/GlyphDraw2/prompts
"""
import argparse
import urllib.request
from pathlib import Path

from aqa.eval.glyphdraw2 import PROMPT_FILES, convert_glyphdraw_prompts

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/OPPO-Mente-Lab/GlyphDraw2/main/prompts"


def _fetch_prompts(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for stem, _lang, _kind in PROMPT_FILES:
        with urllib.request.urlopen(f"{RAW}/{stem}.txt") as resp:
            (dest / f"{stem}.txt").write_bytes(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=None,
                        help="directory holding the four prompt .txt files "
                             "(default: fetch from GitHub)")
    args = parser.parse_args()
    if args.source_dir:
        source = Path(args.source_dir)
    else:
        source = HERE / "source"
        _fetch_prompts(source)
    records = convert_glyphdraw_prompts(source, HERE / "prompts.jsonl")
    print(f"wrote {len(records)} prompts -> {HERE / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
