# DPG-Bench data (vendored)

Provenance: `dpg_bench/dpg_bench.csv` from
[TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) @master
(Apache License 2.0 — see the repo-root NOTICE).

- `dpg_bench.csv` — official benchmark metadata (14,392 proposition rows over
  1,065 dense prompts; the `text` column is the dense prompt itself, so the
  upstream `prompts/*.txt` files are not vendored).
- `prompts.jsonl` — generated from the CSV by `prepare_prompts.py`
  (`aqa.eval.dpgbench.convert_dpg_csv`): one record per prompt
  `{idx, item_id, prompt, questions: [{qid, question, answer, dependency,
  category}], category}`. The official `prepare_dpg_data` quirk of skipping the
  first data row is kept for score parity with upstream.

Regenerate: `python prepare_prompts.py` (from this directory, with `aqa`
installed).
