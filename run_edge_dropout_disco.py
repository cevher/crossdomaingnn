"""
DisCo edge-dropout stress test for Books -> Electronics.

This runner reuses the split construction and sampled NDCG@10 evaluator from
revision_ablation_books_electronics.py so that the DisCo rows are comparable to
the SG-GATv2-R robustness experiment. The prepared_split mode may read prepared
interaction CSVs generated for the external PTUPCDR baseline, but it does not
use the PTUPCDR model, runner, or evaluator.

No reusable DisCo baseline implementation exists in this repository. The model
below is therefore labeled "DisCo-adapted": it uses separate source/target graph
views, shared users, target rating prediction, and bidirectional contrastive
alignment between source-domain and target-domain user representations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from revision_ablation_books_electronics import (
    ExperimentConfig,
    SOURCE_CSV,
    SplitData,
    TARGET_CSV,
    build_edge_index_from_df,
    build_iterative_5core_split,
    build_split,
    df_to_tensors,
    rmse_mae,
    sampled_ndcg_at_k,
    set_all_seeds,
)


DROPOUT_RATIOS = (0.0, 0.1, 0.2, 0.3)
OUTPUT_CSV = "results_edge_dropout_disco_books_electronics.csv"
ORIGINAL_PROTOCOL_NOTE = (
    "DisCo-adapted; same Books->Electronics split/evaluator as SG-GATv2-R; "
    "edge dropout applied only within source_train and target_train before graph/training construction; "
    "validation/test unchanged; sampled NDCG@10 uses 99 negatives and rating>=4 relevance."
)
PREPARED_PROTOCOL_NOTE = (
    "protocol-compatible adapted DisCo-style baseline; prepared iterative 5-core split; "
    "same validation/test sampled NDCG@10 evaluator; not PTUPCDR; edge dropout applied only to train_src/train_tgt."
)
PREPARED_FILES = (
    "metadata.json",
    "train_src.csv",
    "train_tgt.csv",
    "val.csv",
    "test.csv",
)


class DisCoAdapted(nn.Module):
    def __init__(self, num_users: int, num_items: int, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.cfg = cfg
        self.user_emb = nn.Embedding(num_users, cfg.embedding_dim)
        self.item_emb = nn.Embedding(num_items, cfg.embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.source_proj = nn.Linear(cfg.embedding_dim, cfg.hidden_dim, bias=False)
        self.target_proj = nn.Linear(cfg.embedding_dim, cfg.hidden_dim, bias=False)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.0))
        self.rating_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def initial_nodes(self) -> torch.Tensor:
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

    @staticmethod
    def lightgcn_aggregate(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=num_nodes).float().clamp_min(1.0)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] / deg[dst].unsqueeze(-1))
        return 0.5 * (x + out)

    def encode_domain(self, edge_index: torch.Tensor, proj: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        x = proj(self.initial_nodes())
        z = self.lightgcn_aggregate(x, edge_index, self.num_users + self.num_items)
        return z[: self.num_users], z[self.num_users :]

    def encode(
        self,
        source_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_user_z, _ = self.encode_domain(source_edge_index, self.source_proj)
        target_user_z, target_item_z = self.encode_domain(target_edge_index, self.target_proj)
        return source_user_z, target_user_z, target_item_z

    def predict(self, users: torch.Tensor, items: torch.Tensor, user_z: torch.Tensor, item_z: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([user_z[users], item_z[items]], dim=-1)
        pred = self.rating_head(pair).squeeze(-1)
        pred = pred + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1) + self.global_bias
        return torch.clamp(pred, self.cfg.rating_min, self.cfg.rating_max)

    def alignment_loss(self, source_user_z: torch.Tensor, target_user_z: torch.Tensor) -> torch.Tensor:
        num_users = source_user_z.shape[0]
        max_pairs = min(num_users, 2048)
        users = torch.randperm(num_users, device=source_user_z.device)[:max_pairs]
        z_src = F.normalize(source_user_z[users], dim=-1)
        z_tgt = F.normalize(target_user_z[users], dim=-1)
        logits = z_src @ z_tgt.t() / self.cfg.temperature
        labels = torch.arange(max_pairs, device=source_user_z.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def copy_split_with_dropped_training(
    split: SplitData,
    dropout_ratio: float,
    seed: int,
) -> SplitData:
    source_train = edge_dropout_df(split.source_train, dropout_ratio, seed + 17)
    target_train = edge_dropout_df(split.target_train, dropout_ratio, seed + 31)
    return SplitData(
        source_train=source_train,
        target_train=target_train,
        target_val=split.target_val.copy(),
        target_test=split.target_test.copy(),
        user_to_idx=split.user_to_idx,
        item_to_idx=split.item_to_idx,
        target_item_indices=split.target_item_indices.copy(),
        num_users=split.num_users,
        num_items=split.num_items,
    )


def edge_dropout_df(df: pd.DataFrame, dropout_ratio: float, seed: int) -> pd.DataFrame:
    if dropout_ratio <= 0.0:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = rng.random(len(df)) >= dropout_ratio
    if not keep.any():
        raise RuntimeError(f"Edge dropout ratio {dropout_ratio} removed all training interactions.")
    return df.loc[keep].copy().reset_index(drop=True)


def resolve_prepared_split_dir(base_dir: Path) -> Path:
    if all((base_dir / name).exists() for name in PREPARED_FILES):
        return base_dir
    fallback = base_dir / "external_baselines" / "ptupcdr_books_electronics" / "data"
    if all((fallback / name).exists() for name in PREPARED_FILES):
        print(f"Prepared split files not found directly under {base_dir}.")
        print(f"Using prepared interaction files from: {fallback}")
        return fallback
    missing = [name for name in PREPARED_FILES if not (base_dir / name).exists()]
    raise SystemExit(
        "Missing prepared_split input file(s): "
        + ", ".join(missing)
        + f"\nExpected under: {base_dir}"
        + f"\nAlso checked: {fallback}"
        + "\nThis mode reads prepared interaction CSVs only; it does not run PTUPCDR."
    )


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return obj


def read_prepared_interactions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 3:
        raise SystemExit(f"{path.name} must contain at least uid,iid,rating columns.")
    df = raw.iloc[:, :3].copy()
    df.columns = ["user_idx", "item_idx", "rating"]
    df["user_idx"] = df["user_idx"].astype(np.int64)
    df["item_idx"] = df["item_idx"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    return df


def build_prepared_split(base_dir: Path) -> SplitData:
    split_dir = resolve_prepared_split_dir(base_dir)
    metadata = load_json(split_dir / "metadata.json")
    required = ("uid_all", "iid_all", "source_items", "target_items", "seed")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise SystemExit(f"metadata.json is missing required key(s): {', '.join(missing)}")

    num_users = int(metadata["uid_all"])
    num_items = int(metadata["iid_all"])
    source_items = int(metadata["source_items"])
    target_items = int(metadata["target_items"])
    target_start = source_items
    target_stop = source_items + target_items
    if target_stop > num_items:
        raise SystemExit(
            f"Invalid metadata: source_items + target_items = {target_stop}, "
            f"but iid_all = {num_items}."
        )

    source_train = read_prepared_interactions(split_dir / "train_src.csv")
    target_train = read_prepared_interactions(split_dir / "train_tgt.csv")
    target_val = read_prepared_interactions(split_dir / "val.csv")
    target_test = read_prepared_interactions(split_dir / "test.csv")
    for name, df in (
        ("train_src.csv", source_train),
        ("train_tgt.csv", target_train),
        ("val.csv", target_val),
        ("test.csv", target_test),
    ):
        if df.user_idx.min() < 0 or df.user_idx.max() >= num_users:
            raise SystemExit(f"{name} contains user ids outside [0, {num_users}).")
        if df.item_idx.min() < 0 or df.item_idx.max() >= num_items:
            raise SystemExit(f"{name} contains item ids outside [0, {num_items}).")
    for name, df in (("train_tgt.csv", target_train), ("val.csv", target_val), ("test.csv", target_test)):
        if df.item_idx.min() < target_start or df.item_idx.max() >= target_stop:
            raise SystemExit(
                f"{name} contains item ids outside target range [{target_start}, {target_stop})."
            )

    print("WARNING: prepared_split mode uses the available iterative 5-core split. "
          "Do not append these values to an old Table 6 unless LightGCN, GAT-Base, "
          "and Ours were produced under the same split/protocol.")
    print("Prepared split metadata")
    print(f"  protocol: {metadata.get('protocol', 'NA')}")
    print(f"  seed: {metadata['seed']}")
    print(f"  users: {num_users:,}")
    print(f"  items total: {num_items:,}")
    print(f"  source item ids: [0, {source_items})")
    print(f"  target item ids: [{target_start}, {target_stop})")
    print(
        "  interactions: "
        f"train_src={len(source_train):,}, train_tgt={len(target_train):,}, "
        f"val={len(target_val):,}, test={len(target_test):,}"
    )

    return SplitData(
        source_train=source_train,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        user_to_idx={str(i): i for i in range(num_users)},
        item_to_idx={str(i): i for i in range(num_items)},
        target_item_indices=np.arange(target_start, target_stop, dtype=np.int64),
        num_users=num_users,
        num_items=num_items,
    )


def build_domain_edges(split: SplitData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    source_edge_index, _, _ = build_edge_index_from_df(split.source_train, split, device)
    target_edge_index, _, _ = build_edge_index_from_df(split.target_train, split, device)
    return source_edge_index, target_edge_index


def evaluate_disco(
    model: DisCoAdapted,
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    eval_df: pd.DataFrame,
    train_df_for_ranking: pd.DataFrame,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    users, items, ratings = df_to_tensors(eval_df, device)
    model.eval()
    with torch.no_grad():
        _, target_user_z, target_item_z = model.encode(source_edge_index, target_edge_index)
        pred = model.predict(users, items, target_user_z, target_item_z)
    pred_np = pred.detach().cpu().numpy()
    true_np = ratings.detach().cpu().numpy()
    rmse, mae = rmse_mae(pred_np, true_np)
    ndcg = sampled_ndcg_at_k(
        lambda u, i: model.predict(u, i, target_user_z, target_item_z),
        eval_df,
        train_df_for_ranking,
        split.target_item_indices,
        cfg,
        device,
    )
    return {"rmse": rmse, "mae": mae, "ndcg@10": ndcg}


def evaluate_disco_rating(
    model: DisCoAdapted,
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    eval_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, float]:
    users, items, ratings = df_to_tensors(eval_df, device)
    model.eval()
    with torch.no_grad():
        _, target_user_z, target_item_z = model.encode(source_edge_index, target_edge_index)
        pred = model.predict(users, items, target_user_z, target_item_z)
    pred_np = pred.detach().cpu().numpy()
    true_np = ratings.detach().cpu().numpy()
    rmse, mae = rmse_mae(pred_np, true_np)
    return {"rmse": rmse, "mae": mae}


def train_one_dropout(
    base_split: SplitData,
    dropout_ratio: float,
    cfg: ExperimentConfig,
    device: torch.device,
    input_mode: str,
    protocol_note: str,
) -> Dict[str, object]:
    set_all_seeds(cfg.seed)
    split = copy_split_with_dropped_training(base_split, dropout_ratio, cfg.seed)
    source_edge_index, target_edge_index = build_domain_edges(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)

    model = DisCoAdapted(split.num_users, split.num_items, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_epoch = 0
    best_val_rmse = float("inf")
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses: List[float] = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            source_user_z, target_user_z, target_item_z = model.encode(source_edge_index, target_edge_index)
            pred = model.predict(train_users[idx], train_items[idx], target_user_z, target_item_z)
            rating_loss = F.mse_loss(pred, train_ratings[idx])
            align_loss = model.alignment_loss(source_user_z, target_user_z)
            loss = rating_loss + 0.1 * align_loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_disco_rating(
            model,
            source_edge_index,
            target_edge_index,
            split.target_val,
            device,
        )
        print(
            f"dropout={dropout_ratio:.1f} epoch={epoch:03d} "
            f"train_loss={float(np.mean(losses)):.6f} "
            f"val_RMSE={val_metrics['rmse']:.6f}",
            flush=True,
        )
        if val_metrics["rmse"] < best_val_rmse - 1e-5:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_disco(
        model,
        source_edge_index,
        target_edge_index,
        split.target_test,
        split.target_train,
        split,
        cfg,
        device,
    )
    return {
        "model": "DisCo-adapted",
        "dropout_ratio": dropout_ratio,
        "best_epoch": best_epoch,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_ndcg10": test_metrics["ndcg@10"],
        "seed": cfg.seed,
        "input_mode": input_mode,
        "protocol_note": protocol_note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=(
            "Example: python run_edge_dropout_disco.py --data-dir . --input-mode prepared_split "
            "--epochs 30 --patience 5 --output results_edge_dropout_disco_books_electronics.csv"
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_CSV))
    parser.add_argument(
        "--input-mode",
        choices=["original_filtered", "prepared_split"],
        default="original_filtered",
        help="original_filtered uses source_books_filtered.csv/target_electronics_filtered.csv; "
        "prepared_split uses metadata.json, train_src.csv, train_tgt.csv, val.csv, and test.csv.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--future-rerun-all-models",
        action="store_true",
        help="Reserved placeholder for a future unified prepared-split rerun of LightGCN, GAT-Base, Ours, and DisCo-adapted.",
    )
    parser.add_argument("--standard-split", action="store_true", help="Use the non-5-core split; default is iterative 5-core.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        temperature=0.2,
        ranking_mode="sampled",
        relevance_threshold=4.0,
        sampled_negatives=99,
    )
    set_all_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = args.data_dir.resolve()
    if args.future_rerun_all_models:
        raise SystemExit(
            "--future-rerun-all-models is a placeholder only. "
            "This script currently runs DisCo-adapted edge dropout."
        )
    print(f"Using device: {device}")
    print(f"Config: {asdict(cfg)}")
    protocol_note = PREPARED_PROTOCOL_NOTE if args.input_mode == "prepared_split" else ORIGINAL_PROTOCOL_NOTE
    print(f"Protocol: {protocol_note}")

    if args.input_mode == "prepared_split":
        base_split = build_prepared_split(base_dir)
    else:
        missing_inputs = [name for name in (SOURCE_CSV, TARGET_CSV) if not (base_dir / name).exists()]
        if missing_inputs:
            raise SystemExit(
                "Missing original SG-GATv2-R filtered interaction file(s): "
                + ", ".join(missing_inputs)
                + f"\nExpected under: {base_dir}"
                + "\nUse --input-mode prepared_split to read the generated split CSVs instead."
            )
        base_split = build_split(base_dir, cfg) if args.standard_split else build_iterative_5core_split(base_dir, cfg)

    results = [
        train_one_dropout(base_split, ratio, cfg, device, args.input_mode, protocol_note)
        for ratio in DROPOUT_RATIOS
    ]

    output_path = args.output if args.output.is_absolute() else base_dir / args.output
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved DisCo-adapted edge-dropout CSV: {output_path}")
    print(
        "Command example: python run_edge_dropout_disco.py --data-dir . --input-mode prepared_split "
        "--epochs 30 --patience 5 --output results_edge_dropout_disco_books_electronics.csv"
    )

    print("\nEdge Dropout | DisCo-adapted RMSE | DisCo-adapted NDCG@10")
    for row in results:
        pct = int(round(float(row["dropout_ratio"]) * 100))
        print(f"{pct:>3}%         | {row['test_rmse']:.6f}   | {row['test_ndcg10']:.6f}")


if __name__ == "__main__":
    main()
