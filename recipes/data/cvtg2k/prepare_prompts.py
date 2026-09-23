#!/usr/bin/env python3
"""Regenerate prompts.jsonl from the official CVTG-2K release (see README.md).

Downloads ``CVTG-2K.zip`` from the HuggingFace dataset ``dnkdnk/CVTG-2K``
(unless ``--source-dir`` points at an already-extracted tree) and converts it via
``aqa.eval.cvtg2k.convert_cvtg_dir``.

Usage:
  python prepare_prompts.py                 # download + convert
  python prepare_prompts.py --source-dir /path/to/CVTG-2K
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from aqa.eval.cvtg2k import convert_cvtg_dir

HERE = Path(__file__).resolve().parent
URL = "https://huggingface.co/datasets/dnkdnk/CVTG-2K/resolve/main/CVTG-2K.zip"


def _download_zip() -> Path:
    with urllib.request.urlopen(URL) as resp:
        data = resp.read()
    dest = HERE / "CVTG-2K.zip"
    dest.write_bytes(data)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=None,
                        help="extracted CVTG-2K/ tree (default: download the zip)")
    args = parser.parse_args()
    if args.source_dir:
        source = Path(args.source_dir)
    else:
        source = HERE / "CVTG-2K"
        if not source.is_dir():
            with zipfile.ZipFile(_download_zip()) as zf:
                zf.extractall(HERE)
    records = convert_cvtg_dir(source, HERE / "prompts.jsonl")
    print(f"wrote {len(records)} prompts -> {HERE / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
