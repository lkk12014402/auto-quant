# CVTG-2K data (vendored)

Provenance: `CVTG-2K.zip` from the HuggingFace dataset
[dnkdnk/CVTG-2K](https://huggingface.co/datasets/dnkdnk/CVTG-2K) @
`7823ea5a45f8acc53d281da83ef1a1df23812608` (Apache License 2.0 — see the
repo-root NOTICE).  Released alongside [TextCrafter](https://github.com/NJU-PCALab/TextCrafter)
(arXiv:2503.23461), whose `TextCrafter_Eval/unified_metrics_eval.py` defines the
scoring this repo ports.

The dataset ships **prompts only** (no generated images): 2,000 English prompts
in a plain `CVTG` subset (1,002) and a `CVTG-Style` subset (998) that adds
size/color/font attributes, each split by number of text regions (2-5):

| subset | area 2 | area 3 | area 4 | area 5 | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| CVTG | 200 | 301 | 300 | 201 | 1002 |
| CVTG-Style | 200 | 299 | 302 | 197 | 998 |

("C" is for **Complex**, not Chinese — the prompts are English.)

- `prompts.jsonl` — generated from the extracted `CVTG-2K/` tree by
  `prepare_prompts.py` (`aqa.eval.cvtg2k.convert_cvtg_dir`): one record per
  prompt `{idx, subset, area, index, prompt, gt_words}` in the reference's
  evaluation order (subset, then area 2..5, then the file's `data_list` order).
  `idx` is the flat generation index used by the diffusion harness.
  `gt_words` are the reference's ground-truth words (single-quoted strings,
  lowercased and split).

Regenerate: `python prepare_prompts.py` (downloads the zip; use
`--source-dir /path/to/CVTG-2K` for an extracted tree), with `aqa` installed.
