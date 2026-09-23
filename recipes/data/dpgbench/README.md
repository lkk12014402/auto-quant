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
  category}], category}`. The first data row is **kept**: on this CSV it is a
  genuine proposition (all 1,065 items start at `proposition_id` 1), so dropping
  it would leave dependencies dangling on the first item — see the
  `aqa/eval/dpgbench.py` module docstring.

Regenerate: `python prepare_prompts.py` (from this directory, with `aqa`
installed).
