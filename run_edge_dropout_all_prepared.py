"""
Unified prepared-split edge-dropout diagnostic for Books -> Electronics.

This script reruns LightGCN, GAT-Base, DisCo-adapted, and SG-GATv2-R under one
prepared iterative 5-core split and one edge-dropout protocol. It does not read
source_books_filtered.csv or target_electronics_filtered.csv, and it does not
import or run PTUPCDR model/evaluator code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from revision_ablation_books_electronics import (
    BOOKS_EMBEDDINGS_PT,
    BOOKS_ITEM_ID_TO_INDEX_JSON,
    ELECTRONICS_EMBEDDINGS_PT,
    ELECTRONICS_ITEM_ID_TO_INDEX_JSON,
    ExperimentConfig,
    SGGATv2,
    SplitData,
    VariantConfig,
    build_edge_index_from_df,
    build_user_semantic_profiles,
    df_to_tensors,
    evaluate_graph_model,
    rmse_mae,
    sampled_ndcg_at_k,
    set_all_seeds,
)


DROPOUT_RATIOS = (0.0, 0.1, 0.2, 0.3)
OUTPUT_CSV = "results_edge_dropout_all_prepared_books_electronics.csv"
PROTOCOL_NOTE = (
    "prepared iterative 5-core Books->Electronics split; same retained train interactions per dropout "
    "ratio across all models; validation/test unchanged; sampled NDCG@10 with fixed negatives; "
    "not old Table 6; not PTUPCDR evaluator"
)
PREPARED_FILES = (
    "metadata.json",
    "train_src.csv",
    "train_tgt.csv",
    "val.csv",
    "test.csv",
    "user_id_map.json",
    "source_item_id_map.json",
    "target_item_id_map.json",
)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_usage_error(message: str) -> None:
    raise SystemExit(message)


def resolve_prepared_split_dir(data_dir: Path) -> Path:
    if all((data_dir / name).exists() for name in PREPARED_FILES):
        return data_dir
    fallback = data_dir / "external_baselines" / "ptupcdr_books_electronics" / "data"
    if all((fallback / name).exists() for name in PREPARED_FILES):
        print(f"Prepared split files not found directly under {data_dir}.")
        print(f"Using prepared interaction files from: {fallback}")
        return fallback
    missing = [name for name in PREPARED_FILES if not (data_dir / name).exists()]
    save_usage_error(
        "Missing prepared split input file(s): "
        + ", ".join(missing)
        + f"\nExpected under: {data_dir}"
        + f"\nAlso checked: {fallback}"
    )


def read_prepared_interactions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 3:
        save_usage_error(f"{path.name} must contain at least uid,iid,rating columns.")
    df = raw.iloc[:, :3].copy()
    df.columns = ["user_idx", "item_idx", "rating"]
    df["user_idx"] = df["user_idx"].astype(np.int64)
    df["item_idx"] = df["item_idx"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    return df


def build_prepared_split(split_dir: Path) -> Tuple[SplitData, Dict[str, object]]:
    metadata_obj = load_json(split_dir / "metadata.json")
    if not isinstance(metadata_obj, dict):
        save_usage_error("metadata.json must contain a JSON object.")
    metadata = metadata_obj
    for key in ("uid_all", "iid_all", "source_items", "target_items"):
        if key not in metadata:
            save_usage_error(f"metadata.json is missing required key: {key}")
    num_users = int(metadata["uid_all"])
    num_items = int(metadata["iid_all"])
    source_items = int(metadata["source_items"])
    target_items = int(metadata["target_items"])
    target_start = source_items
    target_stop = source_items + target_items
    if target_stop > num_items:
        save_usage_error(
            f"Invalid metadata: source_items + target_items = {target_stop}, iid_all = {num_items}."
        )

    split = SplitData(
        source_train=read_prepared_interactions(split_dir / "train_src.csv"),
        target_train=read_prepared_interactions(split_dir / "train_tgt.csv"),
        target_val=read_prepared_interactions(split_dir / "val.csv"),
        target_test=read_prepared_interactions(split_dir / "test.csv"),
        user_to_idx={str(i): i for i in range(num_users)},
        item_to_idx={str(i): i for i in range(num_items)},
        target_item_indices=np.arange(target_start, target_stop, dtype=np.int64),
        num_users=num_users,
        num_items=num_items,
    )

    for name, df in (
        ("train_src.csv", split.source_train),
        ("train_tgt.csv", split.target_train),
        ("val.csv", split.target_val),
        ("test.csv", split.target_test),
    ):
        if df.user_idx.min() < 0 or df.user_idx.max() >= num_users:
            save_usage_error(f"{name} contains user ids outside [0, {num_users}).")
        if df.item_idx.min() < 0 or df.item_idx.max() >= num_items:
            save_usage_error(f"{name} contains item ids outside [0, {num_items}).")
    for name, df in (("train_tgt.csv", split.target_train), ("val.csv", split.target_val), ("test.csv", split.target_test)):
        if df.item_idx.min() < target_start or df.item_idx.max() >= target_stop:
            save_usage_error(f"{name} contains item ids outside target range [{target_start}, {target_stop}).")

    print("Prepared split metadata")
    print(f"  protocol: {metadata.get('protocol', 'NA')}")
    print(f"  seed: {metadata.get('seed', 42)}")
    print(f"  users: {num_users:,}")
    print(f"  items total: {num_items:,}")
    print(f"  source item ids: [0, {source_items})")
    print(f"  target item ids: [{target_start}, {target_stop})")
    print(
        "  interactions: "
        f"train_src={len(split.source_train):,}, train_tgt={len(split.target_train):,}, "
        f"val={len(split.target_val):,}, test={len(split.target_test):,}"
    )
    return split, metadata


def edge_dropout_mask(n: int, dropout_ratio: float, seed: int) -> np.ndarray:
    if dropout_ratio <= 0:
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    mask = rng.random(n) >= dropout_ratio
    if not mask.any():
        save_usage_error(f"Edge dropout ratio {dropout_ratio} removed all interactions.")
    return mask


def dropped_split_from_masks(base: SplitData, source_mask: np.ndarray, target_mask: np.ndarray) -> SplitData:
    return SplitData(
        source_train=base.source_train.loc[source_mask].copy().reset_index(drop=True),
        target_train=base.target_train.loc[target_mask].copy().reset_index(drop=True),
        target_val=base.target_val.copy(),
        target_test=base.target_test.copy(),
        user_to_idx=base.user_to_idx,
        item_to_idx=base.item_to_idx,
        target_item_indices=base.target_item_indices.copy(),
        num_users=base.num_users,
        num_items=base.num_items,
    )


def build_domain_edges(split: SplitData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    source_edge_index, _, _ = build_edge_index_from_df(split.source_train, split, device)
    target_edge_index, _, _ = build_edge_index_from_df(split.target_train, split, device)
    return source_edge_index, target_edge_index


class LightGCNPrepared(nn.Module):
    def __init__(self, num_users: int, num_items: int, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.cfg = cfg
        self.user_emb = nn.Embedding(num_users, cfg.embedding_dim)
        self.item_emb = nn.Embedding(num_items, cfg.embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.0))

    def initial_nodes(self) -> torch.Tensor:
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

    @staticmethod
    def aggregate(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        deg = torch.bincount(dst, minlength=num_nodes).float().clamp_min(1.0)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] / deg[dst].unsqueeze(-1))
        return out

    def encode(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.initial_nodes()
        x1 = self.aggregate(x0, edge_index, self.num_users + self.num_items)
        z = 0.5 * (x0 + x1)
        return z[: self.num_users], z[self.num_users :]

    def predict(self, users: torch.Tensor, items: torch.Tensor, user_z: torch.Tensor, item_z: torch.Tensor) -> torch.Tensor:
        pred = (user_z[users] * item_z[items]).sum(dim=-1)
        pred = pred + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1) + self.global_bias
        return torch.clamp(pred, self.cfg.rating_min, self.cfg.rating_max)


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

    def encode_domain(self, edge_index: torch.Tensor, proj: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        x = proj(self.initial_nodes())
        z = LightGCNPrepared.aggregate(x, edge_index, self.num_users + self.num_items)
        z = 0.5 * (x + z)
        return z[: self.num_users], z[self.num_users :]

    def encode(self, source_edge_index: torch.Tensor, target_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_user_z, _ = self.encode_domain(source_edge_index, self.source_proj)
        target_user_z, target_item_z = self.encode_domain(target_edge_index, self.target_proj)
        return source_user_z, target_user_z, target_item_z

    def predict(self, users: torch.Tensor, items: torch.Tensor, user_z: torch.Tensor, item_z: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([user_z[users], item_z[items]], dim=-1)
        pred = self.rating_head(pair).squeeze(-1)
        pred = pred + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1) + self.global_bias
        return torch.clamp(pred, self.cfg.rating_min, self.cfg.rating_max)

    def alignment_loss(self, source_user_z: torch.Tensor, target_user_z: torch.Tensor) -> torch.Tensor:
        max_pairs = min(source_user_z.shape[0], 2048)
        users = torch.randperm(source_user_z.shape[0], device=source_user_z.device)[:max_pairs]
        z_src = F.normalize(source_user_z[users], dim=-1)
        z_tgt = F.normalize(target_user_z[users], dim=-1)
        logits = z_src @ z_tgt.t() / self.cfg.temperature
        labels = torch.arange(max_pairs, device=source_user_z.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def eval_lightgcn(
    model: LightGCNPrepared,
    edge_index: torch.Tensor,
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
    include_ndcg: bool,
) -> Dict[str, float]:
    users, items, ratings = df_to_tensors(eval_df, device)
    model.eval()
    with torch.no_grad():
        user_z, item_z = model.encode(edge_index)
        pred = model.predict(users, items, user_z, item_z)
    rmse, mae = rmse_mae(pred.detach().cpu().numpy(), ratings.detach().cpu().numpy())
    metrics = {"rmse": rmse, "mae": mae}
    if include_ndcg:
        metrics["ndcg@10"] = sampled_ndcg_at_k(
            lambda u, i: model.predict(u, i, user_z, item_z),
            eval_df,
            train_df,
            split.target_item_indices,
            cfg,
            device,
        )
    return metrics


def eval_disco(
    model: DisCoAdapted,
    source_edge_index: torch.Tensor,
    target_edge_index: torch.Tensor,
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
    include_ndcg: bool,
) -> Dict[str, float]:
    users, items, ratings = df_to_tensors(eval_df, device)
    model.eval()
    with torch.no_grad():
        _, target_user_z, target_item_z = model.encode(source_edge_index, target_edge_index)
        pred = model.predict(users, items, target_user_z, target_item_z)
    rmse, mae = rmse_mae(pred.detach().cpu().numpy(), ratings.detach().cpu().numpy())
    metrics = {"rmse": rmse, "mae": mae}
    if include_ndcg:
        metrics["ndcg@10"] = sampled_ndcg_at_k(
            lambda u, i: model.predict(u, i, target_user_z, target_item_z),
            eval_df,
            train_df,
            split.target_item_indices,
            cfg,
            device,
        )
    return metrics


def train_lightgcn(split: SplitData, cfg: ExperimentConfig, device: torch.device) -> Dict[str, object]:
    model = LightGCNPrepared(split.num_users, split.num_items, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    full_graph_df = pd.concat([split.source_train, split.target_train], ignore_index=True)
    edge_index, _, _ = build_edge_index_from_df(full_graph_df, split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            user_z, item_z = model.encode(edge_index)
            pred = model.predict(train_users[idx], train_items[idx], user_z, item_z)
            loss = F.mse_loss(pred, train_ratings[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = eval_lightgcn(model, edge_index, split.target_val, split.target_train, split, cfg, device, include_ndcg=False)
        print(f"  LightGCN epoch={epoch:03d} train_loss={np.mean(losses):.6f} val_RMSE={val['rmse']:.6f}", flush=True)
        if val["rmse"] < best_val - 1e-5:
            best_val = val["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = eval_lightgcn(model, edge_index, split.target_test, split.target_train, split, cfg, device, include_ndcg=True)
    return {"model": "LightGCN", "best_epoch": best_epoch, "test_rmse": test["rmse"], "test_mae": test["mae"], "test_ndcg10": test["ndcg@10"]}


def train_disco(split: SplitData, cfg: ExperimentConfig, device: torch.device) -> Dict[str, object]:
    model = DisCoAdapted(split.num_users, split.num_items, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    source_edge_index, target_edge_index = build_domain_edges(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            source_user_z, target_user_z, target_item_z = model.encode(source_edge_index, target_edge_index)
            pred = model.predict(train_users[idx], train_items[idx], target_user_z, target_item_z)
            rating_loss = F.mse_loss(pred, train_ratings[idx])
            loss = rating_loss + 0.1 * model.alignment_loss(source_user_z, target_user_z)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = eval_disco(model, source_edge_index, target_edge_index, split.target_val, split.target_train, split, cfg, device, include_ndcg=False)
        print(f"  DisCo-adapted epoch={epoch:03d} train_loss={np.mean(losses):.6f} val_RMSE={val['rmse']:.6f}", flush=True)
        if val["rmse"] < best_val - 1e-5:
            best_val = val["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = eval_disco(model, source_edge_index, target_edge_index, split.target_test, split.target_train, split, cfg, device, include_ndcg=True)
    return {"model": "DisCo-adapted", "best_epoch": best_epoch, "test_rmse": test["rmse"], "test_mae": test["mae"], "test_ndcg10": test["ndcg@10"]}


def eval_sggat_rating(
    model: SGGATv2,
    edge_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eval_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, float]:
    edge_index, edge_user_node_idx, edge_item_node_idx = edge_tuple
    users, items, ratings = df_to_tensors(eval_df, device)
    model.eval()
    with torch.no_grad():
        user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        pred = model.predict(users, items, user_z, item_z)
    rmse, mae = rmse_mae(pred.detach().cpu().numpy(), ratings.detach().cpu().numpy())
    return {"rmse": rmse, "mae": mae}


def train_sggat_variant(
    model_name: str,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
    variant: VariantConfig,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
) -> Dict[str, object]:
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    full_graph_df = pd.concat([split.source_train, split.target_train], ignore_index=True)
    edge_tuple = build_edge_index_from_df(full_graph_df, split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            user_z, item_z = model.encode(*edge_tuple)
            pred = model.predict(train_users[idx], train_items[idx], user_z, item_z)
            rating_loss = F.mse_loss(pred, train_ratings[idx])
            cl_loss = (
                model.info_nce_loss(user_z, item_z, train_users[idx], train_items[idx])
                if variant.use_infonce and variant.lambda_cl > 0
                else torch.zeros((), device=device)
            )
            loss = rating_loss + variant.lambda_cl * cl_loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = eval_sggat_rating(model, edge_tuple, split.target_val, device)
        print(f"  {model_name} epoch={epoch:03d} train_loss={np.mean(losses):.6f} val_RMSE={val['rmse']:.6f}", flush=True)
        if val["rmse"] < best_val - 1e-5:
            best_val = val["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_graph_model(model, *edge_tuple, split.target_test, split.target_train, split.target_val, split, cfg, device)
    return {"model": model_name, "best_epoch": best_epoch, "test_rmse": test["rmse"], "test_mae": test["mae"], "test_ndcg10": test["ndcg@10"]}


def check_semantic_cache(project_root: Path) -> None:
    required = [
        project_root / BOOKS_EMBEDDINGS_PT,
        project_root / BOOKS_ITEM_ID_TO_INDEX_JSON,
        project_root / ELECTRONICS_EMBEDDINGS_PT,
        project_root / ELECTRONICS_ITEM_ID_TO_INDEX_JSON,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        save_usage_error(
            "Cannot run SG-GATv2-R exactly: required semantic embedding cache file(s) are missing:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\nDo not replace these with random embeddings; generate/load the manuscript embedding caches first."
        )


def load_prepared_item_semantics(project_root: Path, split_dir: Path, split: SplitData) -> np.ndarray:
    check_semantic_cache(project_root)
    source_raw_to_iid = load_json(split_dir / "source_item_id_map.json")
    target_raw_to_iid = load_json(split_dir / "target_item_id_map.json")
    if not isinstance(source_raw_to_iid, dict) or not isinstance(target_raw_to_iid, dict):
        save_usage_error("source_item_id_map.json and target_item_id_map.json must be JSON objects.")

    cache_specs = [
        (project_root / BOOKS_EMBEDDINGS_PT, project_root / BOOKS_ITEM_ID_TO_INDEX_JSON, source_raw_to_iid),
        (project_root / ELECTRONICS_EMBEDDINGS_PT, project_root / ELECTRONICS_ITEM_ID_TO_INDEX_JSON, target_raw_to_iid),
    ]
    item_semantic: Optional[np.ndarray] = None
    missing_cache_items = 0
    for emb_path, index_path, raw_to_iid in cache_specs:
        embeddings = torch.load(emb_path, map_location="cpu")
        if not torch.is_tensor(embeddings):
            save_usage_error(f"{emb_path.name} must contain a torch.Tensor.")
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        raw_to_cache = load_json(index_path)
        if not isinstance(raw_to_cache, dict):
            save_usage_error(f"{index_path.name} must contain a JSON object.")
        if item_semantic is None:
            item_semantic = np.zeros((split.num_items, embeddings.shape[1]), dtype=np.float32)
        elif item_semantic.shape[1] != embeddings.shape[1]:
            save_usage_error("Books and Electronics semantic embedding dimensions do not match.")
        for raw_id, iid in raw_to_iid.items():
            cache_idx = raw_to_cache.get(str(raw_id))
            if cache_idx is None:
                missing_cache_items += 1
                continue
            item_semantic[int(iid)] = embeddings[int(cache_idx)].cpu().numpy().astype(np.float32)
    if item_semantic is None:
        save_usage_error("No semantic embeddings were loaded.")
    if missing_cache_items:
        print(f"Warning: {missing_cache_items:,} prepared split items were missing from semantic caches.")
    return item_semantic


def zero_semantics(split: SplitData, dim: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((split.num_items, dim), dtype=np.float32),
        np.zeros((split.num_users, dim), dtype=np.float32),
    )


def add_degradation(rows: List[Dict[str, object]]) -> None:
    by_model: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if float(row["dropout_ratio"]) == 0.0:
            by_model[str(row["model"])] = row
    for row in rows:
        ref = by_model[str(row["model"])]
        ref_rmse = float(ref["test_rmse"])
        ref_ndcg = float(ref["test_ndcg10"])
        row["rmse_degradation_pct"] = ((float(row["test_rmse"]) - ref_rmse) / ref_rmse) * 100.0
        row["ndcg_degradation_pct"] = ((float(row["test_ndcg10"]) - ref_ndcg) / ref_ndcg) * 100.0 if ref_ndcg != 0 else float("nan")


def print_summary_tables(rows: List[Dict[str, object]]) -> None:
    models = ["LightGCN", "GAT-Base", "DisCo-adapted", "SG-GATv2-R"]
    by_key = {(str(r["model"]), float(r["dropout_ratio"])): r for r in rows}
    print("\nEdge Dropout | LightGCN RMSE | LightGCN NDCG@10 | GAT-Base RMSE | GAT-Base NDCG@10 | DisCo-adapted RMSE | DisCo-adapted NDCG@10 | SG-GATv2-R RMSE | SG-GATv2-R NDCG@10")
    for ratio in DROPOUT_RATIOS:
        vals = []
        for model in models:
            r = by_key[(model, ratio)]
            vals.extend([f"{float(r['test_rmse']):.6f}", f"{float(r['test_ndcg10']):.6f}"])
        print(f"{int(ratio * 100):>3}%         | " + " | ".join(vals))

    print("\nEdge Dropout | LightGCN RMSE degradation | LightGCN NDCG degradation | GAT-Base RMSE degradation | GAT-Base NDCG degradation | DisCo-adapted RMSE degradation | DisCo-adapted NDCG degradation | SG-GATv2-R RMSE degradation | SG-GATv2-R NDCG degradation")
    for ratio in DROPOUT_RATIOS:
        vals = []
        for model in models:
            r = by_key[(model, ratio)]
            vals.extend([f"{float(r['rmse_degradation_pct']):.2f}%", f"{float(r['ndcg_degradation_pct']):.2f}%"])
        print(f"{int(ratio * 100):>3}%         | " + " | ".join(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_CSV))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Directory containing SG-GATv2-R semantic embedding caches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    split_dir = resolve_prepared_split_dir(data_dir)
    base_split, metadata = build_prepared_split(split_dir)
    seed = int(args.seed if args.seed is not None else metadata.get("seed", 42))
    cfg = ExperimentConfig(
        seed=seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        temperature=0.2,
        gamma=0.1,
        ranking_mode="sampled",
        relevance_threshold=4.0,
        sampled_negatives=99,
    )
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = args.project_root.resolve()
    print(f"Using device: {device}")
    print(f"Config: {asdict(cfg)}")
    print(f"Protocol: {PROTOCOL_NOTE}")
    print("Sanity checks")
    print("  Does not read source_books_filtered.csv or target_electronics_filtered.csv.")
    print("  Does not import or use run_ptupcdr_rating.py.")
    print("  Validation/test interactions are copied unchanged per dropout ratio.")
    print("  Graphs are constructed only from retained train_src/train_tgt rows.")
    print("  Sampled NDCG@10 evaluator is imported from revision_ablation_books_electronics.py.")

    item_semantic = load_prepared_item_semantics(project_root, split_dir, base_split)

    rows: List[Dict[str, object]] = []
    for ratio in DROPOUT_RATIOS:
        print(f"\n=== Edge dropout {int(ratio * 100)}% ===", flush=True)
        source_mask = edge_dropout_mask(len(base_split.source_train), ratio, seed + 1000 + int(ratio * 1000))
        target_mask = edge_dropout_mask(len(base_split.target_train), ratio, seed + 2000 + int(ratio * 1000))
        split = dropped_split_from_masks(base_split, source_mask, target_mask)
        print(f"  retained train_src={len(split.source_train):,}/{len(base_split.source_train):,}")
        print(f"  retained train_tgt={len(split.target_train):,}/{len(base_split.target_train):,}")

        model_runs = [
            train_lightgcn(split, cfg, device),
        ]
        gat_item_sem, gat_user_sem = zero_semantics(split, dim=64)
        model_runs.append(
            train_sggat_variant(
                "GAT-Base",
                split,
                cfg,
                device,
                VariantConfig("GAT-Base", use_llm_init=False, use_semantic_gate=False, use_infonce=False, lambda_cl=0.0),
                gat_item_sem,
                gat_user_sem,
            )
        )
        model_runs.append(train_disco(split, cfg, device))
        user_semantic = build_user_semantic_profiles(
            pd.concat([split.source_train, split.target_train], ignore_index=True),
            item_semantic,
            split.num_users,
        )
        model_runs.append(
            train_sggat_variant(
                "SG-GATv2-R",
                split,
                cfg,
                device,
                VariantConfig(
                    "Full SG-GATv2-R",
                    use_llm_init=True,
                    use_semantic_gate=True,
                    use_infonce=True,
                    lambda_cl=0.1,
                    residual_fusion=True,
                ),
                item_semantic,
                user_semantic,
            )
        )

        for row in model_runs:
            row.update(
                {
                    "dropout_ratio": ratio,
                    "seed": seed,
                    "input_mode": "prepared_split",
                    "protocol_note": PROTOCOL_NOTE,
                }
            )
            rows.append(row)

    add_degradation(rows)
    columns = [
        "model",
        "dropout_ratio",
        "best_epoch",
        "test_rmse",
        "test_mae",
        "test_ndcg10",
        "rmse_degradation_pct",
        "ndcg_degradation_pct",
        "seed",
        "input_mode",
        "protocol_note",
    ]
    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    pd.DataFrame(rows)[columns].to_csv(output_path, index=False)
    print(f"\nSaved combined prepared-split edge-dropout CSV: {output_path}")
    print_summary_tables(rows)
    print("\nWARNING: These results are a fully rerun prepared-split diagnostic. They should replace, not be mixed with, the old Table 6 values.")


if __name__ == "__main__":
    main()
