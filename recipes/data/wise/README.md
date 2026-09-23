# WISE (Verified protocol) data (vendored)

Provenance: `data_verified/merge.json` from
[PKU-YuanGroup/WISE](https://github.com/PKU-YuanGroup/WISE) @ main (Apache
License 2.0 — see the repo-root NOTICE).  Paper: arXiv:2503.07265 (ICML 2026).

1,000 verified prompts keyed by `prompt_id` 1..1000, across six domains.  The
repo's `data_verified/` splits them into three files
(`cultural_common_sense_verified.json` 1-400,
`spatio-temporal_reasoning_verified.json` 401-640,
`natural_science_verified.json` 641-1000); `merge.json` is the repo's combined
copy, which we vendor:

| judge category | prompt_id range | count |
| --- | --- | ---: |
| CULTURE | 1-400 | 400 |
| TIME | 401-520 | 120 |
| SPACE | 521-640 | 120 |
| BIOLOGY | 641-760 | 120 |
| PHYSICS | 761-880 | 120 |
| CHEMISTRY | 881-1000 | 120 |

- `prompts.jsonl` — generated from `merge.json` by `prepare_prompts.py`
  (`aqa.eval.wise.convert_wise_prompts`): one record per prompt
  `{idx, prompt_id, prompt, explanation, category, subcategory}` in ascending
  `prompt_id` order.  `idx = prompt_id - 1` is the flat generation index used by
  the diffusion harness.

Official aggregation (from `calculate_verified.py`, recorded in the recipe):
`overall = 0.40*CULTURE + 0.12*TIME + 0.12*SPACE + 0.12*BIOLOGY +
0.12*PHYSICS + 0.12*CHEMISTRY`, each category the mean of its binary 0/1 scores.

**Only the `WISE_Verified` protocol is implemented here.**  The legacy GPT-4o
protocol (`WISE_legacy/`) needs a paid API and its scores are *not* comparable
to the official WISE_Verified leaderboard; see `aqa/eval/wise.py`.

Regenerate: `python prepare_prompts.py` (fetches `merge.json` from GitHub; use
`--source /path/to/merge.json` for a local copy), with `aqa` installed.
