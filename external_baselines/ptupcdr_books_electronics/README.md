# PTUPCDR Books -> Electronics External Baseline

This folder keeps PTUPCDR separate from the SG-GATv2-R experiment code.

## Data Adapter

`prepare_ptupcdr_data.py` reads:

- `../../source_books_filtered.csv`
- `../../target_electronics_filtered.csv`

It applies the same external-baseline protocol used for the SG-GATv2-R 5-core runs:

1. Keep users appearing in both Books and Electronics.
2. Apply iterative bipartite 5-core filtering separately to source and target.
3. Keep users still shared across both domains.
4. Split target Electronics interactions with fixed seed `42` into 80/10/10 train/validation/test.

The adapter writes PTUPCDR-style integer ids:

- `user_id_map.json`: raw Amazon user id -> integer `uid`
- `source_item_id_map.json`: raw Books item id -> integer source `iid`
- `target_item_id_map.json`: raw Electronics item id -> integer target `iid`

Source item ids start at `0`. Target item ids are offset after the source item range, matching the official PTUPCDR convention.

Generated data files:

- `data/train_src.csv`: `uid,iid,rating`
- `data/train_tgt.csv`: `uid,iid,rating`
- `data/train_meta.csv`: `uid,iid,rating,pos_seq`
- `data/val.csv`: `uid,iid,rating,pos_seq`
- `data/test.csv`: `uid,iid,rating,pos_seq`

`pos_seq` is the user source-domain positive history from Books interactions with `rating >= 4.0`, truncated to 20 ids by the runner.

## Evaluation Scope

This baseline is evaluated only for rating prediction:

- validation RMSE
- validation MAE
- test RMSE
- test MAE

NDCG@10 is not reported here because the official PTUPCDR implementation does not provide the same sampled-ranking evaluator used by the SG-GATv2-R experiments.

## Run

Prepare data:

```powershell
python prepare_ptupcdr_data.py --project-root ..\.. --output-dir data
```

Train/evaluate PTUPCDR for RMSE/MAE:

```powershell
python run_ptupcdr_rating.py --data-dir data --epochs 30 --patience 5
```

The runner saves:

```text
results_PTUPCDR_books_electronics_5core.csv
```
