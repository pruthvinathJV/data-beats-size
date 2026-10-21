# Artifacts

Everything needed to check the results without rerunning the pipeline. Images are not included; each pair links to its source through the CC12M `key` and `url`.

| Path | Contents | Rows |
|---|---|---:|
| `arms/{A,B,C,D,Dx,E}.parquet` | training set of each arm: `key`, `caption` (E adds the `critic` score). `Dx` is the exploratory arm D\*; `D` equals `C` | 25,000 each |
| `arm_e_loop/crit1.parquet` | first generated caption, critic score, near-duplicate flag | 25,000 |
| `arm_e_loop/crit2.parquet` | regenerated captions and their critic scores | 1,495 |
| `pool/pool.parquet` | every pool pair with its features and `removed_by` (empty = kept) | 100,820 |
| `pool/judge_scores.parquet` | selection-judge score for every pool pair | 100,820 |
| `eval_captions/{run}.parquet` | captions for the 5,000 Karpathy test images (`0` = base model) | 5,000 each |
| `handrate/ratings.csv` | the author's 200 blind ratings: score, failure tags, notes, audit notes | 200 |
| `handrate/items.parquet` | the rated pairs: `key`, `caption`, `url` | 200 |

`handrate/ratings.csv` matches the SHA-256 committed before any model scoring (`results/handrate_manifest.json`).
