
# Cross-Domain Recommender System with GNNs

## Description
This project implements a cross-domain recommendation framework using Graph Neural Networks (GNNs). It focuses on transferring user preferences from a source domain (Books) to a target domain (Electronics), addressing challenges such as data sparsity and cold-start users. The model integrates:
- Multi-relational user–item graphs
- Contrastive learning
- Domain-adversarial training

## Dataset Information
We use publicly available datasets:
- **Amazon Product Reviews** – curated and published by Ni, Li, and McAuley (2019), available at: [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html)
- Datasets used: `Books.json.gz` and `Electronics.json.gz`

## Code Information
The project contains:
- `Code.ipynb`: Jupyter notebook including all steps from preprocessing to evaluation
- `utils.py`: (optional, if modularized) for reusable preprocessing and modeling functions
- `get_reviewer_ids()`: function to extract and align users across domains

## Requirements
Install with:
```bash
pip install -r requirements.txt
```
Dependencies:
- Python ≥ 3.7
- networkx
- numpy
- pandas
- scikit-learn
- torch (PyTorch)
- gzip
- json

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/cevher/crossdomaingnn.git
   cd crossdomaingnn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add datasets (`Books.json.gz`, `Electronics.json.gz`) to the project root.
4. Run `Code.ipynb` to:
   - Preprocess raw data
   - Construct domain-specific graphs
   - Train and evaluate the GNN-based model

## Methodology
- **Graph Construction**: Build user–item bipartite graphs for both domains.
- **User Alignment**: Identify common users across datasets using reviewer IDs.
- **Contrastive Learning**: Encourage embedding similarity for the same user across domains.
- **Domain-Adversarial Training**: Enforce domain-invariant user representation through a gradient reversal layer.
- **Evaluation Metrics**: Precision@K, Recall@K, and NDCG.

## Evaluation
- **Evaluation Setting**: Cold-start scenario where the target domain lacks sufficient interaction data.
- **Baseline Comparison**: Compared against standard collaborative filtering and CDR baselines.
- **Ablation Study**: Included in the Results section of the manuscript to assess the contribution of each module (contrastive loss, adversarial layer).

## License
This project is released under the [MIT License](LICENSE).

## Citation
Please cite the dataset and model as:
> Ni J, Li J, McAuley J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. *EMNLP 2019*. [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html)

## Contribution Guidelines
We welcome contributions! To contribute:
1. Fork the repo
2. Create a feature branch
3. Submit a pull request with detailed explanation
