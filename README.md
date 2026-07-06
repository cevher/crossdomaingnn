# SG-GATv2-R Books to Electronics Experiments

This repository contains the experiment code and generated outputs for a
Books -> Electronics cross-domain recommendation manuscript revision. The main
focus is SG-GATv2-R and related baselines evaluated on rating prediction and
sampled ranking metrics.

## Repository Layout

```text
.
|-- revision_ablation_books_electronics.py
|-- run_edge_dropout_disco.py
|-- run_edge_dropout_all_prepared.py
|-- prepare_item_texts_books_electronics.py
|-- results_*.csv
|-- mainlatest.ipynb
|-- main.ipynb
|-- Code.ipynb
`-- external_baselines/
    `-- ptupcdr_books_electronics/
        |-- prepare_ptupcdr_data.py
        |-- run_ptupcdr_rating.py
        |-- README.md
        `-- data/
            |-- metadata.json
            |-- train_src.csv
            |-- train_tgt.csv
            |-- val.csv
            |-- test.csv
            |-- user_id_map.json
            |-- source_item_id_map.json
            `-- target_item_id_map.json
```

## Main Scripts

`revision_ablation_books_electronics.py` contains the reusable experiment
components for the manuscript revision, including deterministic split
construction, SG-GATv2-R model code, ablation variants, RMSE/MAE evaluation, and
the sampled NDCG@10 evaluator.

`run_edge_dropout_disco.py` runs the DisCo-adapted baseline under the
edge-dropout protocol. It supports `--input-mode prepared_split`, which reads
the already generated iterative 5-core split CSVs.

`run_edge_dropout_all_prepared.py` is the unified prepared-split edge-dropout
runner for LightGCN, GAT-Base, DisCo-adapted, and SG-GATv2-R. It is intended to
produce a fully consistent replacement for the old Table 6, not rows to append
to old results.

The PTUPCDR scripts under `external_baselines/ptupcdr_books_electronics/` are
kept separate. They are for the external PTUPCDR rating baseline only and do
not provide the SG-GATv2-R sampled NDCG@10 evaluator.

## Data Modes

The original filtered raw interaction files are not included in this working
copy:

```text
source_books_filtered.csv
target_electronics_filtered.csv
```

For the current edge-dropout revision workflow, use the prepared split files:

```text
external_baselines/ptupcdr_books_electronics/data/
```

These files contain the prepared iterative 5-core Books -> Electronics split:

```text
train_src.csv   source Books training interactions
train_tgt.csv   target Electronics training interactions
val.csv         target validation interactions
test.csv        target test interactions
metadata.json   uid/item counts and split metadata
```

The prepared CSVs use integer ids and should not be remapped. Source item ids
start at `0`; target item ids start at `metadata["source_items"]`.

## Environment

Install the core Python dependencies:

```powershell
python -m pip install numpy pandas torch scikit-learn matplotlib
```

If generating semantic item embeddings from text, also install:

```powershell
python -m pip install sentence-transformers
```

CUDA is used automatically when available.
