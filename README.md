# SG-GATv2-R Cross-Domain Recommendation Experiments

This repository contains the experiment code, prepared diagnostic data files, and generated outputs associated with the manuscript:

**Bridging the Semantic Gap: LLM-Enriched Graph Attention Networks for Robust Cross-Domain Recommendation**

The manuscript evaluates SG-GATv2-R on Amazon cross-domain recommendation settings, including Books→Electronics, Movies→CDs, Home→Kitchen, and Clothing→Sports. This repository is intended to support reproducibility by providing the main implementation scripts, revision experiment code, generated result files, and the prepared Books→Electronics diagnostic split used in the additional ablation and robustness analyses.

## Repository Scope

The repository includes:

- SG-GATv2-R experiment and ablation code.
- Edge-dropout diagnostic runners.
- Generated result CSV files.
- A prepared iterative 5-core Books→Electronics split used for revision diagnostics.
- External PTUPCDR baseline preparation and rating-prediction scripts.

The original raw Amazon review files are not redistributed in this repository. Users should obtain the original Amazon review data from the public data source and follow the preprocessing scripts/protocols described in the manuscript.

The prepared Books→Electronics split is included for diagnostic reproducibility. Some full semantic embedding cache files may need to be regenerated from item text metadata if they are not present locally.

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
