# GlyphDraw2 data (vendored)

Provenance: `prompts/*.txt` from
[OPPO-Mente-Lab/GlyphDraw2](https://github.com/OPPO-Mente-Lab/GlyphDraw2)
@ master (MIT License — see the repo-root NOTICE).

Four prompt files, 440 prompts total (≈200 complex + ≈240 poster, across the
English and Chinese variants):

| file | language | kind | prompts |
| --- | --- | --- | ---: |
| `complex_en.txt` | en | complex | 100 |
| `complex_zh.txt` | zh | complex | 100 |
| `poster_en.txt` | en | poster | 120 |
| `poster_zh.txt` | zh | poster | 120 |

- `prompts.jsonl` — generated from the four `.txt` files by
  `prepare_prompts.py` (`aqa.eval.glyphdraw2.convert_glyphdraw_prompts`): one
  record per prompt `{idx, kind, lang, prompt, gt_texts}` in
  complex_en, complex_zh, poster_en, poster_zh order. `idx` is the flat
  generation index used by the diffusion harness. `gt_texts` are the quoted
  substrings of the prompt (the target strings to render).

Note: GlyphDraw2 ships OCR recognizer weights
(`third_party/ocr_weights/ppv3_rec*.pth`) but no runnable scorer; the Acc/NED
metric used here is defined by the AnyText evaluator. See the
`aqa/eval/glyphdraw2.py` module docstring for the exact substitutions.

Regenerate: `python prepare_prompts.py` (fetches the four files from GitHub; use
`--source-dir /path/to/GlyphDraw2/prompts` for a local checkout), with `aqa`
installed.
