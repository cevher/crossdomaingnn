"""
Minimal PTUPCDR rating baseline for Books -> Electronics iterative 5-core.

This runner is intentionally external to SG-GATv2-R. It follows the official
PTUPCDR MF-style stages: source pretraining, target pretraining, source-target
user mapping, and personalized meta-network transfer. It reports RMSE/MAE only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LookupEmbedding(nn.Module):
    def __init__(self, uid_all: int, iid_all: int, emb_dim: int):
        super().__init__()
        self.uid_embedding = nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        return torch.cat([uid_emb, iid_emb], dim=1)


class MetaNet(nn.Module):
    def __init__(self, emb_dim: int, meta_dim: int):
        super().__init__()
        self.event_k = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1, bias=False))
        self.event_softmax = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(nn.Linear(emb_dim, meta_dim), nn.ReLU(), nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea: torch.Tensor, seq_index: torch.Tensor) -> torch.Tensor:
        mask = (seq_index == 0).float()
        event_k = self.event_k(emb_fea)
        att = self.event_softmax(event_k - torch.unsqueeze(mask, 2) * 1e8)
        his_fea = torch.sum(att * emb_fea, 1)
        return self.decoder(his_fea).squeeze(1)


class PTUPCDRModel(nn.Module):
    def __init__(self, uid_all: int, iid_all: int, emb_dim: int, meta_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor, stage: str):
        if stage == "train_src":
            emb = self.src_model(x)
            return torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        if stage in {"train_tgt", "eval_tgt"}:
            emb = self.tgt_model(x)
            return torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        if stage in {"train_meta", "eval_meta"}:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            source_history = self.src_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net(source_history, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            return torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
        if stage == "train_map":
            src_emb = self.mapping(self.src_model.uid_embedding(x.unsqueeze(1)).squeeze())
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        raise ValueError(stage)


def parse_history(text: str, max_len: int = 20) -> List[int]:
    try:
        values = ast.literal_eval(text)
    except Exception:
        values = []
    values = [int(v) for v in values[:max_len]]
    return values + [0] * max(0, max_len - len(values))


def load_triplet_loader(path: Path, batch_size: int, device: torch.device, shuffle: bool = True) -> DataLoader:
    rows = []
    labels = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            rows.append([int(row[0]), int(row[1])])
            labels.append(float(row[2]))
    x = torch.tensor(rows, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def load_history_loader(path: Path, batch_size: int, device: torch.device, shuffle: bool = True) -> DataLoader:
    rows = []
    labels = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            rows.append([int(row[0]), int(row[1])] + parse_history(row[3]))
            labels.append(float(row[2]))
    x = torch.tensor(rows, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def load_map_loader(path: Path, batch_size: int, device: torch.device) -> DataLoader:
    users = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            users.add(int(row[0]))
    x = torch.tensor(sorted(users), dtype=torch.long, device=device)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)


def train_rating(model: PTUPCDRModel, loader: DataLoader, optimizer: torch.optim.Optimizer, stage: str) -> float:
    model.train()
    losses = []
    criterion = nn.MSELoss()
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, stage)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_mapping(model: PTUPCDRModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    losses = []
    criterion = nn.MSELoss()
    for (x,) in loader:
        optimizer.zero_grad(set_to_none=True)
        src_emb, tgt_emb = model(x, "train_map")
        loss = criterion(src_emb, tgt_emb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(model: PTUPCDRModel, loader: DataLoader, stage: str) -> Tuple[float, float]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x, stage)
            preds.append(pred.detach().cpu())
            trues.append(y.detach().cpu())
    pred_t = torch.cat(preds).float()
    true_t = torch.cat(trues).float()
    mae = torch.mean(torch.abs(pred_t - true_t)).item()
    rmse = math.sqrt(torch.mean((pred_t - true_t) ** 2).item())
    return rmse, mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=10)
    parser.add_argument("--meta-dim", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir.resolve()
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    train_src = load_triplet_loader(data_dir / "train_src.csv", args.batch_size, device)
    train_tgt = load_triplet_loader(data_dir / "train_tgt.csv", args.batch_size, device)
    train_meta = load_history_loader(data_dir / "train_meta.csv", args.batch_size, device)
    val = load_history_loader(data_dir / "val.csv", args.batch_size, device, shuffle=False)
    test = load_history_loader(data_dir / "test.csv", args.batch_size, device, shuffle=False)
    map_loader = load_map_loader(data_dir / "train_meta.csv", min(args.batch_size, 128), device)

    model = PTUPCDRModel(metadata["uid_all"], metadata["iid_all"], args.emb_dim, args.meta_dim).to(device)
    opt_src = torch.optim.Adam(model.src_model.parameters(), lr=args.lr)
    opt_tgt = torch.optim.Adam(model.tgt_model.parameters(), lr=args.lr)
    opt_map = torch.optim.Adam(model.mapping.parameters(), lr=args.lr)
    opt_meta = torch.optim.Adam(model.meta_net.parameters(), lr=args.lr)

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        src_loss = train_rating(model, train_src, opt_src, "train_src")
        tgt_loss = train_rating(model, train_tgt, opt_tgt, "train_tgt")
        map_loss = train_mapping(model, map_loader, opt_map)
        meta_loss = train_rating(model, train_meta, opt_meta, "train_meta")
        val_rmse, val_mae = evaluate(model, val, "eval_meta")
        print(
            f"epoch={epoch:03d} src_loss={src_loss:.6f} tgt_loss={tgt_loss:.6f} "
            f"map_loss={map_loss:.6f} meta_loss={meta_loss:.6f} "
            f"val_rmse={val_rmse:.6f} val_mae={val_mae:.6f}"
        )
        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_rmse, val_mae = evaluate(model, val, "eval_meta")
    test_rmse, test_mae = evaluate(model, test, "eval_meta")
    output = data_dir.parent / "results_PTUPCDR_books_electronics_5core.csv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["baseline", "best_epoch", "val_rmse", "val_mae", "test_rmse", "test_mae"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "baseline": "PTUPCDR",
                "best_epoch": best_epoch,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
            }
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
