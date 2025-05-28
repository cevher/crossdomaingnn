
# Cross-Domain Recommender System with GNNs

This repository contains the code for a **Graph Neural Network (GNN)-based cross-domain recommender system**, designed to transfer user preference information from one domain (e.g., Books) to another (e.g., Electronics) with high accuracy, even in cold-start scenarios. The approach integrates **contrastive learning**, **domain-adversarial training**, and **multi-relational graph modeling**, inspired by cutting-edge research in recommendation systems and large language models.

## Table of Contents
- [Background](#background)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Datasets](#datasets)
- [Results](#results)
- [License](#license)
- [References](#references)

## Background
The project is motivated by recent advances in **transformer models** (e.g., GPT-4) and **graph-based recommendation systems**. Inspired by the cross-domain adaptability of large language models, this system aims to transfer knowledge between different item domains (Books and Electronics) to overcome challenges such as data sparsity and cold-start users.

## Features
- Construction of separate user–item graphs for source and target domains.
- Integration of **contrastive learning** to align user embeddings across domains.
- **Domain-adversarial training** to enforce domain-invariant representation learning.
- Support for cold-start recommendation in sparse target domains.
- Modular and extensible architecture for experimenting with new domains or embedding strategies.

## Requirements
- Python 3.7+
- Required libraries (install via `pip install -r requirements.txt`):
  - `networkx`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `torch` (PyTorch)
  - `gzip` and `json` (for dataset preprocessing)
  
## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare datasets:
   - Place `Books.json.gz` and `Electronics.json.gz` files in the project directory.
4. Run preprocessing and model training:
   - Launch the Jupyter notebook `Code.ipynb` and follow the step-by-step cells to:
     - Extract user IDs from datasets.
     - Build user–item graphs for each domain.
     - Train the GNN model with cross-domain alignment and adversarial learning.

## Code Overview
- `Code.ipynb`: The main Jupyter notebook with code for preprocessing, graph construction, model training, and evaluation.
- `get_reviewer_ids()`: Function to extract user IDs from gzip-compressed JSON files (Books and Electronics datasets).
- Additional functions and modules will process data, implement contrastive learning, and integrate domain-adversarial components.

## Datasets
- Amazon Product Reviews datasets (`Books.json.gz`, `Electronics.json.gz`) available at: https://nijianmo.github.io/amazon/index.html  
- Use `get_reviewer_ids()` to extract user information for graph construction.

## Results
Extensive experiments demonstrate superior performance of our model compared to traditional CDR baselines, particularly in **cold-start scenarios** where the target domain lacks sufficient data. Performance metrics include precision, recall, and NDCG.

## License
This project is licensed under the [MIT License](LICENSE).

## References
- OpenAI’s GPT series and transformer architectures: https://openai.com/research
- Knowledge graph-based recommender systems: [Wang et al., 2019](https://arxiv.org/abs/1905.08108)
- Cross-domain contrastive learning for recommendations: [Li et al., 2024](https://doi.org/10.1016/j.knosys.2025.113109)
