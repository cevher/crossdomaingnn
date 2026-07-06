"""
Component-level ablations for Books -> Electronics cross-domain rating prediction.

This script intentionally does not depend on the original notebooks. It builds a
single deterministic split, reuses that split for every variant, trains only on
observed target-domain ratings, and never converts missing user-item pairs into
zero-valued ratings.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}


def require_dependencies(include_sentence_transformers: bool = True) -> None:
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    if include_sentence_transformers:
        try:
            __import__("sentence_transformers")
        except ImportError:
            missing.append("sentence-transformers")
    if missing:
        joined = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing required Python packages. Install them and rerun:\n"
            f"  python -m pip install {joined}\n"
            "If no precomputed item embeddings are present, also install sentence-transformers:\n"
            "  python -m pip install sentence-transformers\n"
            "The experiment needs these libraries for reproducible training, "
            "semantic embedding generation, metrics, and plotting."
        )


require_dependencies(include_sentence_transformers=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error


BOOKS = "books"
ELECTRONICS = "electronics"
SOURCE_CSV = "source_books_filtered.csv"
TARGET_CSV = "target_electronics_filtered.csv"
BOOKS_META = "meta_Books.jsonl.gz"
ELECTRONICS_META = "meta_Electronics.jsonl.gz"
BOOKS_ITEM_TEXTS_CSV = "books_item_texts_filtered.csv"
ELECTRONICS_ITEM_TEXTS_CSV = "electronics_item_texts_filtered.csv"
BOOKS_EMBEDDINGS_PT = "books_item_embeddings.pt"
ELECTRONICS_EMBEDDINGS_PT = "electronics_item_embeddings.pt"
BOOKS_ITEM_ID_TO_INDEX_JSON = "books_item_id_to_index.json"
ELECTRONICS_ITEM_ID_TO_INDEX_JSON = "electronics_item_id_to_index.json"
SEED = 42


@dataclass(frozen=True)
class VariantConfig:
    name: str
    use_llm_init: bool
    use_semantic_gate: bool
    use_infonce: bool
    lambda_cl: float
    semantic_only: bool = False
    no_graph: bool = False
    residual_fusion: bool = False


@dataclass
class ExperimentConfig:
    seed: int = SEED
    embedding_dim: int = 64
    hidden_dim: int = 64
    semantic_proj_dim: int = 64
    epochs: int = 120
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8192
    rating_min: float = 1.0
    rating_max: float = 5.0
    temperature: float = 0.2
    gamma: float = 0.1
    train_frac: float = 0.8
    val_frac: float = 0.1
    topk: int = 10
    ndcg_chunk_users: int = 128
    encode_batch_size: int = 128
    ranking_mode: str = "full"
    relevance_threshold: float = 0.0
    sampled_negatives: int = 99


@dataclass
class SplitData:
    source_train: pd.DataFrame
    target_train: pd.DataFrame
    target_val: pd.DataFrame
    target_test: pd.DataFrame
    user_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]
    target_item_indices: np.ndarray
    num_users: int
    num_items: int


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    lower_to_col = {c.lower(): c for c in df.columns}
    user_col = lower_to_col.get("user_id") or lower_to_col.get("userid") or lower_to_col.get("reviewerid")
    item_col = lower_to_col.get("item_id") or lower_to_col.get("itemid") or lower_to_col.get("asin")
    rating_col = lower_to_col.get("rating") or lower_to_col.get("overall")
    if not user_col or not item_col or not rating_col:
        raise ValueError(
            f"Could not infer user/item/rating columns from: {list(df.columns)}"
        )
    return user_col, item_col, rating_col


def load_interactions(path: Path, domain: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    user_col, item_col, rating_col = infer_columns(df)
    keep = df[[user_col, item_col, rating_col]].copy()
    keep.columns = ["raw_user_id", "raw_item_id", "rating"]
    keep["raw_user_id"] = keep["raw_user_id"].astype(str)
    keep["raw_item_id"] = keep["raw_item_id"].astype(str)
    keep["rating"] = keep["rating"].astype(float)
    keep["domain"] = domain
    keep["item_key"] = keep["domain"] + ":" + keep["raw_item_id"]
    return keep.dropna(subset=["raw_user_id", "raw_item_id", "rating"])


def infer_text_column(df: pd.DataFrame) -> str:
    preferred = ("text", "item_text", "metadata_text", "title_features", "description")
    lower_to_col = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_to_col:
            return lower_to_col[name]
    candidates = [
        c
        for c in df.columns
        if c.lower() not in {"item_id", "itemid", "asin", "parent_asin", "raw_item_id"}
    ]
    if not candidates:
        raise ValueError(f"Could not infer text column from: {list(df.columns)}")
    return candidates[0]


def infer_item_id_column(df: pd.DataFrame) -> str:
    lower_to_col = {c.lower(): c for c in df.columns}
    for name in ("item_id", "itemid", "asin", "parent_asin", "raw_item_id"):
        if name in lower_to_col:
            return lower_to_col[name]
    raise ValueError(f"Could not infer item_id column from: {list(df.columns)}")


def item_text_coverage_stats(
    item_text_path: Path,
    unique_items: Sequence[str],
    domain_label: str,
) -> Dict[str, object]:
    unique_item_set = set(str(x) for x in unique_items)
    if not item_text_path.exists():
        return {
            "domain": domain_label,
            "path": str(item_text_path),
            "exists": False,
            "rows": 0,
            "unique_items": len(unique_item_set),
            "matched_items": 0,
            "coverage": 0.0,
            "empty_or_short_text_count": len(unique_item_set),
            "empty_or_short_text_ratio": 1.0,
        }

    text_df = pd.read_csv(item_text_path)
    item_col = infer_item_id_column(text_df)
    text_col = infer_text_column(text_df)
    text_df[item_col] = text_df[item_col].astype(str)
    texts = text_df[text_col].fillna("").astype(str)
    item_ids = set(text_df[item_col])
    matched_items = unique_item_set.intersection(item_ids)

    text_by_item = (
        pd.DataFrame({"item_id": text_df[item_col], "text": texts})
        .drop_duplicates("item_id", keep="first")
        .set_index("item_id")["text"]
    )
    empty_or_short = 0
    for item_id in unique_item_set:
        text = text_by_item.get(item_id, "")
        if not isinstance(text, str) or len(text.strip()) < 10:
            empty_or_short += 1

    return {
        "domain": domain_label,
        "path": str(item_text_path),
        "exists": True,
        "rows": int(len(text_df)),
        "unique_items": len(unique_item_set),
        "matched_items": len(matched_items),
        "coverage": len(matched_items) / max(len(unique_item_set), 1),
        "empty_or_short_text_count": empty_or_short,
        "empty_or_short_text_ratio": empty_or_short / max(len(unique_item_set), 1),
    }


def print_random_item_text_examples(
    item_text_path: Path,
    domain_label: str,
    seed: int,
    n: int = 5,
) -> None:
    print(f"\n  Random {domain_label} item-text examples:")
    if not item_text_path.exists():
        print(f"    {item_text_path.name} not found.")
        return

    text_df = pd.read_csv(item_text_path)
    if text_df.empty:
        print(f"    {item_text_path.name} is empty.")
        return

    item_col = infer_item_id_column(text_df)
    text_col = infer_text_column(text_df)
    sample = text_df.sample(n=min(n, len(text_df)), random_state=seed)
    for _, row in sample.iterrows():
        item_id = str(row[item_col])
        raw_text = "" if pd.isna(row[text_col]) else row[text_col]
        text = str(raw_text).replace("\n", " ").replace("\r", " ")
        preview = text[:300]
        print(f"    item_id: {item_id}")
        print(f"    text: {preview}")


def validate_metadata_coverage(base_dir: Path) -> Dict[str, Dict[str, object]]:
    source = load_interactions(base_dir / SOURCE_CSV, BOOKS)
    target = load_interactions(base_dir / TARGET_CSV, ELECTRONICS)
    source_items = sorted(source.raw_item_id.unique())
    target_items = sorted(target.raw_item_id.unique())

    books_stats = item_text_coverage_stats(
        base_dir / BOOKS_ITEM_TEXTS_CSV,
        source_items,
        "Books source",
    )
    electronics_stats = item_text_coverage_stats(
        base_dir / ELECTRONICS_ITEM_TEXTS_CSV,
        target_items,
        "Electronics target",
    )

    print("\nMetadata coverage validation")
    for stats in (books_stats, electronics_stats):
        print(f"  {stats['domain']}:")
        print(f"    item text file: {Path(str(stats['path'])).name}")
        print(f"    file exists: {stats['exists']}")
        print(f"    item text rows: {stats['rows']:,}")
        print(f"    unique interaction items: {stats['unique_items']:,}")
        print(f"    exact matched item texts: {stats['matched_items']:,}")
        print(f"    exact item_id coverage: {stats['coverage']:.2%}")
        print(
            "    missing/empty/too-short texts (<10 chars): "
            f"{stats['empty_or_short_text_count']:,} ({stats['empty_or_short_text_ratio']:.2%})"
        )

    print_random_item_text_examples(base_dir / BOOKS_ITEM_TEXTS_CSV, "Books", SEED)
    print_random_item_text_examples(base_dir / ELECTRONICS_ITEM_TEXTS_CSV, "Electronics", SEED)

    if float(electronics_stats["coverage"]) < 0.90:
        raise SystemExit(
            "Stopping before model training: target Electronics metadata exact item_id "
            f"coverage is {float(electronics_stats['coverage']):.2%}, below the required 90%. "
            f"Expected sufficient matches in {ELECTRONICS_ITEM_TEXTS_CSV}."
        )

    return {"books": books_stats, "electronics": electronics_stats}


def build_split_from_filtered_interactions(
    source: pd.DataFrame,
    target: pd.DataFrame,
    cfg: ExperimentConfig,
    label: str,
    enforce_test_seen_threshold: bool = False,
) -> SplitData:
    shared_users = sorted(set(source.raw_user_id).intersection(set(target.raw_user_id)))
    source = source[source.raw_user_id.isin(shared_users)].reset_index(drop=True)
    target = target[target.raw_user_id.isin(shared_users)].reset_index(drop=True)
    user_to_idx = {u: i for i, u in enumerate(shared_users)}
    item_keys = sorted(set(source.item_key).union(set(target.item_key)))
    item_to_idx = {it: i for i, it in enumerate(item_keys)}

    for df in (source, target):
        df["user_idx"] = df.raw_user_id.map(user_to_idx).astype(np.int64)
        df["item_idx"] = df.item_key.map(item_to_idx).astype(np.int64)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(target))
    n_train = int(len(target) * cfg.train_frac)
    n_val = int(len(target) * cfg.val_frac)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    target_train = target.iloc[train_idx].reset_index(drop=True)
    target_val = target.iloc[val_idx].reset_index(drop=True)
    target_test = target.iloc[test_idx].reset_index(drop=True)

    target_items = np.array(
        [idx for key, idx in item_to_idx.items() if key.startswith(ELECTRONICS + ":")],
        dtype=np.int64,
    )

    print(f"Dataset statistics ({label})")
    print(f"  Shared users: {len(shared_users):,}")
    print(f"  Source Books interactions after shared-user filter: {len(source):,}")
    print(f"  Target Electronics interactions after shared-user filter: {len(target):,}")
    print(f"  Users: {len(user_to_idx):,}")
    print(f"  Items total: {len(item_to_idx):,}")
    print(f"  Target Electronics items: {len(target_items):,}")
    print(
        "  Target split: "
        f"train={len(target_train):,}, val={len(target_val):,}, test={len(target_test):,}"
    )
    train_items = set(target_train.item_idx.astype(int))
    test_items = set(target_test.item_idx.astype(int))
    unseen_test_items = test_items - train_items
    unseen_pct = len(unseen_test_items) / max(len(test_items), 1)
    print(f"  Test items unseen in target train: {len(unseen_test_items):,} ({unseen_pct:.2%})")
    if enforce_test_seen_threshold and unseen_pct > 0.05:
        raise SystemExit(
            "Stopping: iterative 5-core target split has test unseen item percentage "
            f"{unseen_pct:.2%}, exceeding the allowed 5%."
        )

    return SplitData(
        source_train=source,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        target_item_indices=target_items,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
    )


def build_split(base_dir: Path, cfg: ExperimentConfig) -> SplitData:
    source = load_interactions(base_dir / SOURCE_CSV, BOOKS)
    target = load_interactions(base_dir / TARGET_CSV, ELECTRONICS)
    shared_users = sorted(set(source.raw_user_id).intersection(set(target.raw_user_id)))
    source = source[source.raw_user_id.isin(shared_users)].reset_index(drop=True)
    target = target[target.raw_user_id.isin(shared_users)].reset_index(drop=True)
    return build_split_from_filtered_interactions(source, target, cfg, "standard shared-user split")


def build_iterative_5core_split(base_dir: Path, cfg: ExperimentConfig) -> SplitData:
    source = load_interactions(base_dir / SOURCE_CSV, BOOKS)
    target = load_interactions(base_dir / TARGET_CSV, ELECTRONICS)
    shared_users = sorted(set(source.raw_user_id).intersection(set(target.raw_user_id)))
    source = source[source.raw_user_id.isin(shared_users)].reset_index(drop=True)
    target = target[target.raw_user_id.isin(shared_users)].reset_index(drop=True)
    source = iterative_bipartite_kcore(source, 5)
    target = iterative_bipartite_kcore(target, 5)
    shared_after = sorted(set(source.raw_user_id).intersection(set(target.raw_user_id)))
    source = source[source.raw_user_id.isin(shared_after)].reset_index(drop=True)
    target = target[target.raw_user_id.isin(shared_after)].reset_index(drop=True)
    return build_split_from_filtered_interactions(
        source,
        target,
        cfg,
        "iterative 5-core shared-user split",
        enforce_test_seen_threshold=True,
    )


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def stable_hash(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def embedding_cache_paths(base_dir: Path, domain: str, item_ids: Sequence[str]) -> Tuple[Path, Path]:
    signature = stable_hash(sorted(item_ids))
    stem = f"{domain}_minilm_item_embeddings_{signature}"
    return base_dir / f"{stem}.npy", base_dir / f"{stem}_ids.json"


def try_load_precomputed(base_dir: Path, domain: str, item_ids: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
    cache_npy, cache_ids = embedding_cache_paths(base_dir, domain, item_ids)
    candidates = [
        (cache_npy, cache_ids),
        (base_dir / f"{domain}_item_embeddings.npy", base_dir / f"{domain}_item_ids.json"),
        (base_dir / f"item_embeddings_{domain}.npy", base_dir / f"item_ids_{domain}.json"),
    ]
    for emb_path, ids_path in candidates:
        if emb_path.exists() and ids_path.exists():
            ids = [str(x) for x in load_json(ids_path)]
            arr = np.load(emb_path)
            if len(ids) != len(arr):
                raise ValueError(f"Embedding/id count mismatch for {emb_path} and {ids_path}")
            print(f"Loaded precomputed {domain} embeddings from {emb_path.name}")
            return {item_id: arr[i].astype(np.float32) for i, item_id in enumerate(ids)}

    for pt_path in [base_dir / f"{domain}_item_embeddings.pt", base_dir / f"item_embeddings_{domain}.pt"]:
        if pt_path.exists():
            obj = torch.load(pt_path, map_location="cpu")
            if isinstance(obj, dict) and "item_ids" in obj and "embeddings" in obj:
                ids = [str(x) for x in obj["item_ids"]]
                arr = obj["embeddings"].detach().cpu().numpy() if torch.is_tensor(obj["embeddings"]) else np.asarray(obj["embeddings"])
                print(f"Loaded precomputed {domain} embeddings from {pt_path.name}")
                return {item_id: arr[i].astype(np.float32) for i, item_id in enumerate(ids)}
            if isinstance(obj, dict):
                print(f"Loaded precomputed {domain} embedding dictionary from {pt_path.name}")
                return {
                    str(k): (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)).astype(np.float32)
                    for k, v in obj.items()
                }
    return None


def item_text_from_meta(record: dict) -> str:
    title = record.get("title") or record.get("name") or ""
    features = record.get("features") or record.get("feature") or []
    if isinstance(features, str):
        feature_text = features
    elif isinstance(features, list):
        feature_text = " ".join(str(x) for x in features if x)
    else:
        feature_text = ""
    return f"{title} {feature_text}".strip()


def extract_metadata_texts(meta_path: Path, needed_item_ids: Sequence[str]) -> Dict[str, str]:
    needed = set(str(x) for x in needed_item_ids)
    found: Dict[str, str] = {}
    id_fields = ("parent_asin", "asin", "item_id")
    with gzip.open(meta_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_id = None
            for field in id_fields:
                value = obj.get(field)
                if value is not None and str(value) in needed:
                    raw_id = str(value)
                    break
            if raw_id is None:
                continue
            text = item_text_from_meta(obj)
            if text:
                found[raw_id] = text
            if len(found) == len(needed):
                break
    return found


def generate_minilm_embeddings(
    base_dir: Path,
    domain: str,
    meta_file: str,
    item_ids: Sequence[str],
    cfg: ExperimentConfig,
) -> Dict[str, np.ndarray]:
    meta_path = base_dir / meta_file
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    require_dependencies(include_sentence_transformers=True)
    from sentence_transformers import SentenceTransformer

    print(f"Extracting {domain} item text from {meta_path.name}")
    texts_by_id = extract_metadata_texts(meta_path, item_ids)
    if not texts_by_id:
        raise RuntimeError(
            f"No usable title/features text was found in {meta_path.name} for {domain} items. "
            "LLM semantic ablation cannot be run without item text or item embeddings."
        )

    ordered_ids = [item_id for item_id in item_ids if item_id in texts_by_id]
    texts = [texts_by_id[item_id] for item_id in ordered_ids]
    print(f"Encoding {len(ordered_ids):,}/{len(item_ids):,} {domain} items with all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    arr = model.encode(
        texts,
        batch_size=cfg.encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    emb_path, ids_path = embedding_cache_paths(base_dir, domain, item_ids)
    np.save(emb_path, arr)
    save_json(ids_path, ordered_ids)
    print(f"Cached {domain} embeddings to {emb_path.name}")
    return {item_id: arr[i] for i, item_id in enumerate(ordered_ids)}


def exact_embedding_cache_files(base_dir: Path, domain: str) -> Tuple[Path, Path]:
    if domain == BOOKS:
        return base_dir / BOOKS_EMBEDDINGS_PT, base_dir / BOOKS_ITEM_ID_TO_INDEX_JSON
    if domain == ELECTRONICS:
        return base_dir / ELECTRONICS_EMBEDDINGS_PT, base_dir / ELECTRONICS_ITEM_ID_TO_INDEX_JSON
    raise ValueError(f"Unknown domain: {domain}")


def item_text_csv_for_domain(base_dir: Path, domain: str) -> Path:
    if domain == BOOKS:
        return base_dir / BOOKS_ITEM_TEXTS_CSV
    if domain == ELECTRONICS:
        return base_dir / ELECTRONICS_ITEM_TEXTS_CSV
    raise ValueError(f"Unknown domain: {domain}")


def choose_embedding_batch_size() -> int:
    if not torch.cuda.is_available():
        return 256
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 512 if total_mem_gb >= 8 else 256


def load_exact_embedding_cache(base_dir: Path, domain: str) -> Optional[Dict[str, np.ndarray]]:
    emb_path, index_path = exact_embedding_cache_files(base_dir, domain)
    if not emb_path.exists() or not index_path.exists():
        return None

    embeddings = torch.load(emb_path, map_location="cpu")
    if not torch.is_tensor(embeddings):
        raise ValueError(f"{emb_path.name} must contain a torch.Tensor.")
    item_id_to_index = load_json(index_path)
    if not isinstance(item_id_to_index, dict):
        raise ValueError(f"{index_path.name} must contain a JSON object mapping item_id to index.")

    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
    arr = embeddings.cpu().numpy().astype(np.float32)
    max_index = max((int(v) for v in item_id_to_index.values()), default=-1)
    if max_index >= arr.shape[0]:
        raise ValueError(
            f"{index_path.name} references index {max_index}, but {emb_path.name} has "
            f"only {arr.shape[0]} rows."
        )

    print(f"Loaded cached {domain} embeddings from {emb_path.name} and {index_path.name}")
    return {str(item_id): arr[int(idx)] for item_id, idx in item_id_to_index.items()}


def read_item_text_csv(item_text_path: Path) -> Tuple[List[str], List[str]]:
    if not item_text_path.exists():
        raise SystemExit(
            f"{item_text_path.name} not found. Cannot create MiniLM item embedding cache "
            "without filtered item text CSVs."
        )
    text_df = pd.read_csv(item_text_path)
    if text_df.empty:
        raise SystemExit(f"{item_text_path.name} is empty; cannot create item embeddings.")
    item_col = infer_item_id_column(text_df)
    text_col = infer_text_column(text_df)
    text_df[item_col] = text_df[item_col].astype(str)
    text_df[text_col] = text_df[text_col].fillna("").astype(str)
    text_df = text_df.drop_duplicates(item_col, keep="first").reset_index(drop=True)
    item_ids = text_df[item_col].tolist()
    texts = text_df[text_col].tolist()
    return item_ids, texts


def generate_exact_embedding_cache(base_dir: Path, domain: str) -> Dict[str, np.ndarray]:
    require_dependencies(include_sentence_transformers=True)
    from sentence_transformers import SentenceTransformer

    item_text_path = item_text_csv_for_domain(base_dir, domain)
    item_ids, texts = read_item_text_csv(item_text_path)
    batch_size = choose_embedding_batch_size()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Creating {domain} embedding cache from {item_text_path.name}: "
        f"{len(item_ids):,} items, batch_size={batch_size}, device={device_name}"
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device_name)
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    tensor = F.normalize(torch.tensor(arr, dtype=torch.float32), p=2, dim=1)
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    emb_path, index_path = exact_embedding_cache_files(base_dir, domain)
    torch.save(tensor.cpu(), emb_path)
    save_json(index_path, item_id_to_index)
    print(f"Saved {domain} embeddings to {emb_path.name}")
    print(f"Saved {domain} item_id_to_index mapping to {index_path.name}")
    return {item_id: tensor[idx].cpu().numpy().astype(np.float32) for item_id, idx in item_id_to_index.items()}


def load_or_create_domain_embedding_cache(base_dir: Path, domain: str) -> Dict[str, np.ndarray]:
    loaded = load_exact_embedding_cache(base_dir, domain)
    if loaded is not None:
        return loaded
    emb_path, index_path = exact_embedding_cache_files(base_dir, domain)
    print(
        f"Embedding cache missing for {domain}: expected {emb_path.name} and "
        f"{index_path.name}. Generating from filtered item texts."
    )
    return generate_exact_embedding_cache(base_dir, domain)


def validate_embedding_cache(base_dir: Path) -> None:
    print("\nEmbedding cache validation")
    for domain in (BOOKS, ELECTRONICS):
        emb_path, index_path = exact_embedding_cache_files(base_dir, domain)
        print(f"  {domain}:")
        print(f"    embeddings: {emb_path.name} exists={emb_path.exists()}")
        print(f"    index map: {index_path.name} exists={index_path.exists()}")
        if not emb_path.exists() or not index_path.exists():
            raise SystemExit(
                "Embedding cache is incomplete. Run the item-text preparation and embedding "
                "creation steps before smoke testing or full ablation."
            )
        item_id_to_index = load_json(index_path)
        if not isinstance(item_id_to_index, dict):
            raise SystemExit(f"{index_path.name} must contain a JSON object.")
        embeddings = torch.load(emb_path, map_location="cpu")
        if not torch.is_tensor(embeddings):
            raise SystemExit(f"{emb_path.name} must contain a torch.Tensor.")
        print(f"    embedding shape: {tuple(embeddings.shape)}")
        print(f"    mapped item ids: {len(item_id_to_index):,}")


def load_semantic_embeddings_from_existing_cache(base_dir: Path, split: SplitData) -> np.ndarray:
    domain_items: Dict[str, List[Tuple[str, int]]] = {BOOKS: [], ELECTRONICS: []}
    for item_key, idx in split.item_to_idx.items():
        domain, raw_id = item_key.split(":", 1)
        domain_items[domain].append((raw_id, idx))

    item_sem: Optional[np.ndarray] = None
    missing = 0
    for domain in (BOOKS, ELECTRONICS):
        emb_path, index_path = exact_embedding_cache_files(base_dir, domain)
        if not emb_path.exists() or not index_path.exists():
            raise SystemExit(
                f"Existing embedding cache files are required in this mode. Missing {domain} cache."
            )
        item_id_to_index = load_json(index_path)
        embeddings = torch.load(emb_path, map_location="cpu")
        if not torch.is_tensor(embeddings):
            raise SystemExit(f"{emb_path.name} must contain a torch.Tensor.")
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        if item_sem is None:
            item_sem = np.zeros((split.num_items, embeddings.shape[1]), dtype=np.float32)
        elif item_sem.shape[1] != embeddings.shape[1]:
            raise SystemExit("Books and Electronics embedding dimensions do not match.")

        pairs = domain_items[domain]
        for raw_id, idx in pairs:
            cache_idx = item_id_to_index.get(raw_id)
            if cache_idx is None:
                missing += 1
                continue
            item_sem[idx] = embeddings[int(cache_idx)].cpu().numpy().astype(np.float32)
        del embeddings

    if item_sem is None:
        raise SystemExit("No embedding cache data was loaded.")
    if missing:
        print(f"Warning: {missing:,} split items were missing from the existing embedding cache.")
    return item_sem


def load_or_create_semantic_embeddings(base_dir: Path, split: SplitData, cfg: ExperimentConfig) -> np.ndarray:
    domain_items: Dict[str, List[Tuple[str, int]]] = {BOOKS: [], ELECTRONICS: []}
    for item_key, idx in split.item_to_idx.items():
        domain, raw_id = item_key.split(":", 1)
        domain_items[domain].append((raw_id, idx))

    domain_embedding_maps: Dict[str, Dict[str, np.ndarray]] = {}
    for domain in (BOOKS, ELECTRONICS):
        domain_embedding_maps[domain] = load_or_create_domain_embedding_cache(base_dir, domain)

    dims = [
        len(v)
        for emb_map in domain_embedding_maps.values()
        for v in emb_map.values()
    ]
    if not dims:
        raise SystemExit(
            "No item embeddings were loaded or generated. LLM semantic ablation cannot be run "
            "without item text or item embeddings."
        )
    sem_dim = dims[0]
    item_sem = np.zeros((split.num_items, sem_dim), dtype=np.float32)
    missing = 0
    for domain, pairs in domain_items.items():
        emb_map = domain_embedding_maps[domain]
        for raw_id, idx in pairs:
            emb = emb_map.get(raw_id)
            if emb is None:
                missing += 1
                continue
            item_sem[idx] = emb
    if missing:
        print(
            f"Warning: {missing:,} mapped items had no metadata embedding; "
            "their semantic vectors are zeros."
        )
    return item_sem


def build_user_semantic_profiles(
    training_interactions: pd.DataFrame,
    item_semantic: np.ndarray,
    num_users: int,
) -> np.ndarray:
    profiles = np.zeros((num_users, item_semantic.shape[1]), dtype=np.float32)
    weight_sums = np.zeros(num_users, dtype=np.float32)
    for row in training_interactions.itertuples(index=False):
        u = int(row.user_idx)
        i = int(row.item_idx)
        rating = float(row.rating)
        profiles[u] += rating * item_semantic[i]
        weight_sums[u] += abs(rating)
    nonzero = weight_sums > 0
    profiles[nonzero] /= weight_sums[nonzero, None]
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = profiles / np.maximum(norms, 1e-8)
    return profiles.astype(np.float32)


def build_edge_index(split: SplitData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_df = pd.concat([split.source_train, split.target_train], ignore_index=True)
    return build_edge_index_from_df(graph_df, split, device)


def build_edge_index_from_df(
    graph_df: pd.DataFrame,
    split: SplitData,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users = graph_df.user_idx.to_numpy(np.int64)
    items = graph_df.item_idx.to_numpy(np.int64) + split.num_users
    src = np.concatenate([users, items])
    dst = np.concatenate([items, users])
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    edge_item_node_idx = np.concatenate([items, items])
    edge_user_node_idx = np.concatenate([users, users])
    edge_item_node_idx_t = torch.tensor(edge_item_node_idx, dtype=torch.long, device=device)
    edge_user_node_idx_t = torch.tensor(edge_user_node_idx, dtype=torch.long, device=device)
    return edge_index, edge_user_node_idx_t, edge_item_node_idx_t


def build_domain_edge_indices(
    split: SplitData,
    device: torch.device,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    source_edges = build_edge_index_from_df(split.source_train, split, device)
    target_edges = build_edge_index_from_df(split.target_train, split, device)
    return source_edges, target_edges


def df_to_tensors(df: pd.DataFrame, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users = torch.tensor(df.user_idx.to_numpy(np.int64), dtype=torch.long, device=device)
    items = torch.tensor(df.item_idx.to_numpy(np.int64), dtype=torch.long, device=device)
    ratings = torch.tensor(df.rating.to_numpy(np.float32), dtype=torch.float32, device=device)
    return users, items, ratings


def segment_softmax(logits: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    max_per_dst = torch.full((num_nodes,), -torch.inf, device=logits.device, dtype=logits.dtype)
    max_per_dst.scatter_reduce_(0, dst, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - max_per_dst[dst])
    denom = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
    denom.scatter_add_(0, dst, exp)
    return exp / (denom[dst] + 1e-12)


class SemanticGATv2Layer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        semantic_dim: int,
        semantic_proj_dim: int,
        use_semantic_gate: bool,
        gamma: float,
    ) -> None:
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(out_dim))
        self.msg = nn.Linear(in_dim, out_dim, bias=False)
        self.use_semantic_gate = use_semantic_gate
        self.semantic_proj = nn.Linear(semantic_dim, semantic_proj_dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.msg.weight)
        nn.init.xavier_uniform_(self.semantic_proj.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_user_node_idx: torch.Tensor,
        edge_item_node_idx: torch.Tensor,
        node_semantic: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        h = torch.tanh(self.lin_l(x[src]) + self.lin_r(x[dst]))
        structural_logits = (h * self.att).sum(dim=-1)
        logits = structural_logits
        if self.use_semantic_gate:
            user_sem = self.semantic_proj(node_semantic[edge_user_node_idx])
            item_sem = self.semantic_proj(node_semantic[edge_item_node_idx])
            sem_prior = F.cosine_similarity(user_sem, item_sem, dim=-1, eps=1e-8)
            logits = logits + self.gamma * sem_prior
        alpha = segment_softmax(logits, dst, x.shape[0])
        messages = self.msg(x[src]) * alpha.unsqueeze(-1)
        out = torch.zeros(x.shape[0], messages.shape[1], device=x.device, dtype=messages.dtype)
        out.index_add_(0, dst, messages)
        return out

    def attention_debug(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_user_node_idx: torch.Tensor,
        edge_item_node_idx: torch.Tensor,
        node_semantic: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        src, dst = edge_index
        h = torch.tanh(self.lin_l(x[src]) + self.lin_r(x[dst]))
        structural_logits = (h * self.att).sum(dim=-1)
        user_sem = self.semantic_proj(node_semantic[edge_user_node_idx])
        item_sem = self.semantic_proj(node_semantic[edge_item_node_idx])
        semantic_prior = F.cosine_similarity(user_sem, item_sem, dim=-1, eps=1e-8)
        gated_logits = structural_logits + self.gamma * semantic_prior
        return {
            "semantic_prior": semantic_prior,
            "structural_logits": structural_logits,
            "gated_logits": gated_logits,
            "gamma": self.gamma.detach(),
        }


class SGGATv2(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        semantic_item: torch.Tensor,
        semantic_user: torch.Tensor,
        cfg: ExperimentConfig,
        variant: VariantConfig,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.cfg = cfg
        self.variant = variant
        sem_dim = semantic_item.shape[1]
        self.register_buffer("semantic_item", semantic_item)
        self.register_buffer("semantic_user", semantic_user)
        self.user_emb = nn.Embedding(num_users, cfg.embedding_dim)
        self.item_emb = nn.Embedding(num_items, cfg.embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        if variant.use_llm_init:
            self.item_init = nn.Linear(sem_dim, cfg.embedding_dim, bias=False)
            self.user_init = nn.Linear(sem_dim, cfg.embedding_dim, bias=False)
        else:
            self.item_init = None
            self.user_init = None

        self.layer = SemanticGATv2Layer(
            cfg.embedding_dim,
            cfg.hidden_dim,
            sem_dim,
            cfg.semantic_proj_dim,
            variant.use_semantic_gate,
            cfg.gamma,
        )
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(3.0))
        self.rating_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.residual_proj = (
            nn.Identity()
            if cfg.embedding_dim == cfg.hidden_dim
            else nn.Linear(cfg.embedding_dim, cfg.hidden_dim, bias=False)
        )
        self.beta_user = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.beta_item = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.user_fusion_norm = nn.LayerNorm(cfg.hidden_dim)
        self.item_fusion_norm = nn.LayerNorm(cfg.hidden_dim)

    def initial_nodes(self) -> torch.Tensor:
        user_x = self.user_emb.weight
        item_x = self.item_emb.weight
        if self.variant.use_llm_init:
            user_x = user_x + self.user_init(self.semantic_user)
            item_x = item_x + self.item_init(self.semantic_item)
        return torch.cat([user_x, item_x], dim=0)

    def node_semantics(self) -> torch.Tensor:
        return torch.cat([self.semantic_user, self.semantic_item], dim=0)

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_user_node_idx: torch.Tensor,
        edge_item_node_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.initial_nodes()
        h0 = self.residual_proj(x0)
        if self.variant.no_graph:
            return h0[: self.num_users], h0[self.num_users :]
        node_sem = self.node_semantics()
        h_gnn = F.elu(self.layer(x0, edge_index, edge_user_node_idx, edge_item_node_idx, node_sem))
        if self.variant.residual_fusion:
            user_final = self.user_fusion_norm(h0[: self.num_users] + self.beta_user * h_gnn[: self.num_users])
            item_final = self.item_fusion_norm(h0[self.num_users :] + self.beta_item * h_gnn[self.num_users :])
            return user_final, item_final
        h = h_gnn + h0
        return h[: self.num_users], h[self.num_users :]

    def attention_debug(
        self,
        edge_index: torch.Tensor,
        edge_user_node_idx: torch.Tensor,
        edge_item_node_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.layer.attention_debug(
            self.initial_nodes(),
            edge_index,
            edge_user_node_idx,
            edge_item_node_idx,
            self.node_semantics(),
        )

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        user_z: torch.Tensor,
        item_z: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.cat([user_z[users], item_z[items]], dim=-1)
        pred = self.rating_head(pair).squeeze(-1)
        pred = pred + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1) + self.global_bias
        return torch.clamp(pred, self.cfg.rating_min, self.cfg.rating_max)

    def info_nce_loss(self, user_z: torch.Tensor, item_z: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if users.numel() == 0:
            return torch.zeros((), device=user_z.device)
        max_pairs = min(users.numel(), 2048)
        perm = torch.randperm(users.numel(), device=user_z.device)[:max_pairs]
        u = users[perm]
        i = items[perm]
        z_u = F.normalize(user_z[u], dim=-1)
        z_i = F.normalize(item_z[i], dim=-1)
        logits = z_u @ z_i.t() / self.cfg.temperature
        labels = torch.arange(max_pairs, device=user_z.device)
        return F.cross_entropy(logits, labels)

    def shared_user_alignment_loss(
        self,
        source_user_z: torch.Tensor,
        target_user_z: torch.Tensor,
        shared_users: torch.Tensor,
    ) -> torch.Tensor:
        if shared_users.numel() == 0:
            return torch.zeros((), device=source_user_z.device)
        max_pairs = min(shared_users.numel(), 2048)
        perm = torch.randperm(shared_users.numel(), device=shared_users.device)[:max_pairs]
        users = shared_users[perm]
        z_src = F.normalize(source_user_z[users], dim=-1)
        z_tgt = F.normalize(target_user_z[users], dim=-1)
        logits = z_src @ z_tgt.t() / self.cfg.temperature
        labels = torch.arange(max_pairs, device=source_user_z.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class SemanticOnly(nn.Module):
    def __init__(self, user_semantic: torch.Tensor, item_semantic: torch.Tensor, cfg: ExperimentConfig):
        super().__init__()
        self.register_buffer("user_semantic", F.normalize(user_semantic, dim=-1))
        self.register_buffer("item_semantic", F.normalize(item_semantic, dim=-1))
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(3.0))
        self.cfg = cfg

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_semantic, self.item_semantic

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(self.user_semantic[users], self.item_semantic[items], dim=-1)
        pred = self.bias + self.scale * sim
        return torch.clamp(pred, self.cfg.rating_min, self.cfg.rating_max)


def rmse_mae(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    rmse = math.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return float(rmse), float(mae)


def ndcg_at_k(
    score_fn,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_item_indices: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    train_seen = set(zip(train_df.user_idx.astype(int), train_df.item_idx.astype(int)))
    val_seen = set(zip(val_df.user_idx.astype(int), val_df.item_idx.astype(int)))
    heldout_by_user: Dict[int, List[Tuple[int, float]]] = {}
    for row in test_df.itertuples(index=False):
        heldout_by_user.setdefault(int(row.user_idx), []).append((int(row.item_idx), float(row.rating)))
    if not heldout_by_user:
        return float("nan")

    target_items_t = torch.tensor(target_item_indices, dtype=torch.long, device=device)
    ndcgs = []
    for user, positives in heldout_by_user.items():
        candidates = target_item_indices
        mask = np.array(
            [
                (user, int(item)) not in train_seen and (user, int(item)) not in val_seen
                for item in candidates
            ],
            dtype=bool,
        )
        candidates = candidates[mask]
        if len(candidates) == 0:
            continue
        users_t = torch.full((len(candidates),), user, dtype=torch.long, device=device)
        items_t = torch.tensor(candidates, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = score_fn(users_t, items_t).detach().cpu().numpy()
        top_idx = np.argsort(-scores)[: cfg.topk]
        top_items = candidates[top_idx]
        rel_map = {item: rating for item, rating in positives}
        gains = np.array([rel_map.get(int(item), 0.0) for item in top_items], dtype=np.float32)
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum((np.power(2.0, gains) - 1.0) * discounts))
        ideal = sorted([rating for _, rating in positives], reverse=True)[: cfg.topk]
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum((np.power(2.0, np.array(ideal)) - 1.0) * ideal_discounts))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def sampled_ndcg_at_k(
    score_fn,
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_item_indices: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    rng = np.random.default_rng(cfg.seed)
    train_seen = set(zip(train_df.user_idx.astype(int), train_df.item_idx.astype(int)))
    eval_seen = set(zip(eval_df.user_idx.astype(int), eval_df.item_idx.astype(int)))
    positives_by_user: Dict[int, Dict[int, float]] = {}
    for row in eval_df.itertuples(index=False):
        rating = float(row.rating)
        if rating >= cfg.relevance_threshold:
            positives_by_user.setdefault(int(row.user_idx), {})[int(row.item_idx)] = 1.0
    if not positives_by_user:
        return float("nan")

    all_target = np.array(target_item_indices, dtype=np.int64)
    ndcgs = []
    for user, positives in positives_by_user.items():
        candidate_items = set(positives.keys())
        negative_pool = np.array(
            [
                int(item)
                for item in all_target
                if (user, int(item)) not in train_seen and (user, int(item)) not in eval_seen
            ],
            dtype=np.int64,
        )
        for _ in positives:
            if len(negative_pool) == 0:
                break
            sample_size = min(cfg.sampled_negatives, len(negative_pool))
            candidate_items.update(rng.choice(negative_pool, size=sample_size, replace=False).tolist())
        candidates = np.array(sorted(candidate_items), dtype=np.int64)
        if len(candidates) == 0:
            continue
        users_t = torch.full((len(candidates),), user, dtype=torch.long, device=device)
        items_t = torch.tensor(candidates, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = score_fn(users_t, items_t).detach().cpu().numpy()
        top_items = candidates[np.argsort(-scores)[: cfg.topk]]
        rels = [positives.get(int(item), 0.0) for item in top_items]
        ideal = [1.0] * min(len(positives), cfg.topk)
        idcg = dcg_from_relevances(ideal)
        if idcg > 0:
            ndcgs.append(dcg_from_relevances(rels) / idcg)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def print_rating_stats(name: str, df: pd.DataFrame) -> None:
    ratings = df.rating.astype(float)
    print(
        f"  {name}: min={ratings.min():.4f} max={ratings.max():.4f} "
        f"mean={ratings.mean():.4f} std={ratings.std(ddof=0):.4f}"
    )


def print_prediction_stats(
    name: str,
    model: SGGATv2,
    user_z: torch.Tensor,
    item_z: torch.Tensor,
    df: pd.DataFrame,
    device: torch.device,
) -> None:
    users, items, ratings = df_to_tensors(df, device)
    with torch.no_grad():
        pred = model.predict(users, items, user_z, item_z)
    pred_np = pred.detach().cpu().numpy()
    true_np = ratings.detach().cpu().numpy()
    print(
        f"  {name}: min={pred_np.min():.4f} max={pred_np.max():.4f} "
        f"mean={pred_np.mean():.4f} std={pred_np.std():.4f}"
    )
    print(f"  first 20 {name} observed pairs:")
    for true_rating, pred_rating in list(zip(true_np, pred_np))[:20]:
        print(f"    true={true_rating:.4f}, pred={pred_rating:.4f}")


def dcg_from_relevances(relevances: Sequence[float]) -> float:
    if not relevances:
        return 0.0
    gains = np.power(2.0, np.asarray(relevances, dtype=np.float32)) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains * discounts))


def score_items_for_user(
    model: SGGATv2,
    user_z: torch.Tensor,
    item_z: torch.Tensor,
    user: int,
    candidates: np.ndarray,
    device: torch.device,
    chunk_size: int = 8192,
) -> np.ndarray:
    scores = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        users_t = torch.full((len(chunk),), user, dtype=torch.long, device=device)
        items_t = torch.tensor(chunk, dtype=torch.long, device=device)
        with torch.no_grad():
            scores.append(model.predict(users_t, items_t, user_z, item_z).detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def ranking_debug_full_catalog(
    model: SGGATv2,
    user_z: torch.Tensor,
    item_z: torch.Tensor,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    train_seen = set(zip(split.target_train.user_idx.astype(int), split.target_train.item_idx.astype(int)))
    positives_by_user: Dict[int, Dict[int, float]] = {}
    for row in split.target_test.itertuples(index=False):
        rating = float(row.rating)
        if rating > 0:
            positives_by_user.setdefault(int(row.user_idx), {})[int(row.item_idx)] = rating

    ndcgs = []
    hits = []
    recalls = []
    candidate_sizes = []
    positives_per_user = []
    for user, positives in positives_by_user.items():
        candidates = np.array(
            [int(item) for item in split.target_item_indices if (user, int(item)) not in train_seen],
            dtype=np.int64,
        )
        if len(candidates) == 0:
            continue
        candidate_set = set(candidates.tolist())
        if not any(item in candidate_set for item in positives):
            continue
        scores = score_items_for_user(model, user_z, item_z, user, candidates, device)
        top_items = candidates[np.argsort(-scores)[: cfg.topk]]
        top_rels = [positives.get(int(item), 0.0) for item in top_items]
        ideal = sorted(positives.values(), reverse=True)[: cfg.topk]
        idcg = dcg_from_relevances(ideal)
        ndcgs.append(dcg_from_relevances(top_rels) / idcg if idcg > 0 else 0.0)
        hit_count = sum(1 for item in top_items if int(item) in positives)
        hits.append(1.0 if hit_count > 0 else 0.0)
        recalls.append(hit_count / max(len(positives), 1))
        candidate_sizes.append(len(candidates))
        positives_per_user.append(len(positives))

    return {
        "name": "Full-catalog ranking over all target Electronics items, excluding train items",
        "num_evaluable_users": len(ndcgs),
        "mean_ndcg10": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "median_ndcg10": float(np.median(ndcgs)) if ndcgs else float("nan"),
        "hit_rate10": float(np.mean(hits)) if hits else float("nan"),
        "recall10": float(np.mean(recalls)) if recalls else float("nan"),
        "candidate_sizes": candidate_sizes,
        "positives_per_user": positives_per_user,
        "users_with_positive_test_item": len(positives_by_user),
    }


def ranking_debug_sampled(
    model: SGGATv2,
    user_z: torch.Tensor,
    item_z: torch.Tensor,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    train_seen = set(zip(split.target_train.user_idx.astype(int), split.target_train.item_idx.astype(int)))
    test_seen = set(zip(split.target_test.user_idx.astype(int), split.target_test.item_idx.astype(int)))
    positives_by_user: Dict[int, Dict[int, float]] = {}
    for row in split.target_test.itertuples(index=False):
        rating = float(row.rating)
        if rating > 0:
            positives_by_user.setdefault(int(row.user_idx), {})[int(row.item_idx)] = rating

    ndcgs = []
    hits = []
    recalls = []
    candidate_sizes = []
    positives_per_user = []
    all_target = np.array(split.target_item_indices, dtype=np.int64)
    for user, positives in positives_by_user.items():
        candidate_items = set(positives.keys())
        negative_pool = np.array(
            [
                int(item)
                for item in all_target
                if (user, int(item)) not in train_seen and (user, int(item)) not in test_seen
            ],
            dtype=np.int64,
        )
        for _ in positives:
            if len(negative_pool) == 0:
                break
            sample_size = min(99, len(negative_pool))
            candidate_items.update(rng.choice(negative_pool, size=sample_size, replace=False).tolist())
        candidates = np.array(sorted(candidate_items), dtype=np.int64)
        if len(candidates) == 0:
            continue
        scores = score_items_for_user(model, user_z, item_z, user, candidates, device)
        top_items = candidates[np.argsort(-scores)[: cfg.topk]]
        top_rels = [positives.get(int(item), 0.0) for item in top_items]
        ideal = sorted(positives.values(), reverse=True)[: cfg.topk]
        idcg = dcg_from_relevances(ideal)
        ndcgs.append(dcg_from_relevances(top_rels) / idcg if idcg > 0 else 0.0)
        hit_count = sum(1 for item in top_items if int(item) in positives)
        hits.append(1.0 if hit_count > 0 else 0.0)
        recalls.append(hit_count / max(len(positives), 1))
        candidate_sizes.append(len(candidates))
        positives_per_user.append(len(positives))

    return {
        "name": "Sampled ranking with 99 random negative items per positive test item",
        "num_evaluable_users": len(ndcgs),
        "mean_ndcg10": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "median_ndcg10": float(np.median(ndcgs)) if ndcgs else float("nan"),
        "hit_rate10": float(np.mean(hits)) if hits else float("nan"),
        "recall10": float(np.mean(recalls)) if recalls else float("nan"),
        "candidate_sizes": candidate_sizes,
        "positives_per_user": positives_per_user,
        "users_with_positive_test_item": len(positives_by_user),
    }


def print_ranking_debug_result(result: Dict[str, object]) -> None:
    candidate_sizes = result["candidate_sizes"]
    positives_per_user = result["positives_per_user"]
    avg_candidates = float(np.mean(candidate_sizes)) if candidate_sizes else float("nan")
    min_candidates = int(np.min(candidate_sizes)) if candidate_sizes else 0
    max_candidates = int(np.max(candidate_sizes)) if candidate_sizes else 0
    avg_positives = float(np.mean(positives_per_user)) if positives_per_user else float("nan")
    print(f"\n{result['name']}")
    print(f"  users with at least one positive test item: {result['users_with_positive_test_item']:,}")
    print(f"  number of evaluable users: {result['num_evaluable_users']:,}")
    print(f"  average positive test items per evaluable user: {avg_positives:.4f}")
    print(f"  candidate set size per user: mean={avg_candidates:.2f}, min={min_candidates}, max={max_candidates}")
    print(f"  mean NDCG@10: {result['mean_ndcg10']:.6f}")
    print(f"  median NDCG@10: {result['median_ndcg10']:.6f}")
    print(f"  hit rate@10: {result['hit_rate10']:.6f}")
    print(f"  recall@10: {result['recall10']:.6f}")


def evaluate_graph_model(
    model: SGGATv2,
    edge_index: torch.Tensor,
    edge_user_node_idx: torch.Tensor,
    edge_item_node_idx: torch.Tensor,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    exclude_df: pd.DataFrame,
    split: SplitData,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    users, items, ratings = df_to_tensors(df, device)
    model.eval()
    with torch.no_grad():
        user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        pred = model.predict(users, items, user_z, item_z)
    pred_np = pred.detach().cpu().numpy()
    true_np = ratings.detach().cpu().numpy()
    rmse, mae = rmse_mae(pred_np, true_np)
    if cfg.ranking_mode == "sampled":
        ndcg = sampled_ndcg_at_k(
            lambda u, i: model.predict(u, i, user_z, item_z),
            df,
            train_df,
            split.target_item_indices,
            cfg,
            device,
        )
    else:
        ndcg = ndcg_at_k(
            lambda u, i: model.predict(u, i, user_z, item_z),
            df,
            train_df,
            exclude_df,
            split.target_item_indices,
            cfg,
            device,
        )
    return {"rmse": rmse, "mae": mae, "ndcg@10": ndcg}


def train_graph_variant(
    variant: VariantConfig,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    val_users, val_items, val_ratings = df_to_tensors(split.target_val, device)

    best_state = None
    best_val = float("inf")
    stale = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        batch_losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
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
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
            val_pred = model.predict(val_users, val_items, user_z, item_z)
            val_rmse = math.sqrt(F.mse_loss(val_pred, val_ratings).item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_rmse": val_rmse})
        if val_rmse < best_val - 1e-5:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_graph_model(
        model,
        edge_index,
        edge_user_node_idx,
        edge_item_node_idx,
        split.target_test,
        split.target_train,
        split.target_val,
        split,
        cfg,
        device,
    )
    metrics["best_val_rmse"] = best_val
    metrics["epochs"] = len(history)
    return metrics, {"variant": asdict(variant), "history": history}


def train_semantic_only(
    variant: VariantConfig,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    model = SemanticOnly(
        torch.tensor(user_semantic, dtype=torch.float32, device=device),
        torch.tensor(item_semantic, dtype=torch.float32, device=device),
        cfg,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    val_users, val_items, val_ratings = df_to_tensors(split.target_val, device)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model.predict(train_users[idx], train_items[idx])
            loss = F.mse_loss(pred, train_ratings[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            val_pred = model.predict(val_users, val_items)
            val_rmse = math.sqrt(F.mse_loss(val_pred, val_ratings).item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_rmse": val_rmse})
        if val_rmse < best_val - 1e-5:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_users, test_items, test_ratings = df_to_tensors(split.target_test, device)
    model.eval()
    with torch.no_grad():
        pred = model.predict(test_users, test_items)
    rmse, mae = rmse_mae(pred.detach().cpu().numpy(), test_ratings.detach().cpu().numpy())
    ndcg = ndcg_at_k(
        lambda u, i: model.predict(u, i),
        split.target_test,
        split.target_train,
        split.target_val,
        split.target_item_indices,
        cfg,
        device,
    )
    return (
        {"rmse": rmse, "mae": mae, "ndcg@10": ndcg, "best_val_rmse": best_val, "epochs": len(history)},
        {"variant": asdict(variant), "history": history},
    )


def run_variant(
    variant: VariantConfig,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    print(f"\nRunning {variant.name}")
    set_all_seeds(cfg.seed)
    if variant.semantic_only:
        return train_semantic_only(variant, split, item_semantic, user_semantic, cfg, device)
    return train_graph_variant(variant, split, item_semantic, user_semantic, cfg, device)


def limited_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def make_smoke_split(split: SplitData, cfg: ExperimentConfig) -> SplitData:
    rng = np.random.default_rng(cfg.seed)
    candidate_users = np.array(sorted(split.target_train.user_idx.unique()), dtype=np.int64)
    if len(candidate_users) > 1000:
        candidate_users = np.sort(rng.choice(candidate_users, size=1000, replace=False))
    selected_users = set(int(u) for u in candidate_users)

    source = split.source_train[split.source_train.user_idx.isin(selected_users)].copy()
    target_train = split.target_train[split.target_train.user_idx.isin(selected_users)].copy()
    target_val = split.target_val[split.target_val.user_idx.isin(selected_users)].copy()

    source = limited_df(source, 5000, cfg.seed)
    target_train = limited_df(target_train, 5000, cfg.seed)
    target_val = limited_df(target_val, 1000, cfg.seed)

    used_users = sorted(
        set(source.user_idx.astype(int))
        .union(set(target_train.user_idx.astype(int)))
        .union(set(target_val.user_idx.astype(int)))
    )
    used_items = sorted(
        set(source.item_key.astype(str))
        .union(set(target_train.item_key.astype(str)))
        .union(set(target_val.item_key.astype(str)))
    )
    old_to_new_user = {old: new for new, old in enumerate(used_users)}
    item_to_idx = {item_key: idx for idx, item_key in enumerate(used_items)}
    idx_to_raw_user = {idx: raw for raw, idx in split.user_to_idx.items()}
    user_to_idx = {idx_to_raw_user[old]: new for old, new in old_to_new_user.items()}

    def remap(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["user_idx"] = out.user_idx.astype(int).map(old_to_new_user).astype(np.int64)
        out["item_idx"] = out.item_key.astype(str).map(item_to_idx).astype(np.int64)
        return out.reset_index(drop=True)

    smoke_source = remap(source)
    smoke_train = remap(target_train)
    smoke_val = remap(target_val)
    smoke_test = smoke_val.iloc[0:0].copy()
    target_items = np.array(
        [idx for key, idx in item_to_idx.items() if key.startswith(ELECTRONICS + ":")],
        dtype=np.int64,
    )

    print("\nSmoke-test subset")
    print(f"  users: {len(user_to_idx):,}")
    print(f"  items: {len(item_to_idx):,}")
    print(f"  source interactions: {len(smoke_source):,}")
    print(f"  target train interactions: {len(smoke_train):,}")
    print(f"  validation interactions: {len(smoke_val):,}")

    return SplitData(
        source_train=smoke_source,
        target_train=smoke_train,
        target_val=smoke_val,
        target_test=smoke_test,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        target_item_indices=target_items,
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
    )


def run_smoke_test(
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    if split.target_train.empty or split.target_val.empty:
        raise SystemExit("Smoke-test subset needs non-empty target train and validation interactions.")

    variant = VariantConfig("A0 Full SG-GATv2 smoke", True, True, True, 0.1)
    set_all_seeds(cfg.seed)
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    val_users, val_items, val_ratings = df_to_tensors(split.target_val, device)

    model.train()
    perm = torch.randperm(train_users.numel(), device=device)
    losses = []
    for start in range(0, train_users.numel(), cfg.batch_size):
        idx = perm[start : start + cfg.batch_size]
        opt.zero_grad(set_to_none=True)
        user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        pred = model.predict(train_users[idx], train_items[idx], user_z, item_z)
        rating_loss = F.mse_loss(pred, train_ratings[idx])
        cl_loss = model.info_nce_loss(user_z, item_z, train_users[idx], train_items[idx])
        loss = rating_loss + variant.lambda_cl * cl_loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        val_pred = model.predict(val_users, val_items, user_z, item_z)
    val_pred_np = val_pred.detach().cpu().numpy()
    val_true_np = val_ratings.detach().cpu().numpy()
    val_rmse, val_mae = rmse_mae(val_pred_np, val_true_np)
    empty_exclusion = split.target_val.iloc[0:0].copy()
    val_ndcg = ndcg_at_k(
        lambda u, i: model.predict(u, i, user_z, item_z),
        split.target_val,
        split.target_train,
        empty_exclusion,
        split.target_item_indices,
        cfg,
        device,
    )

    print("\nSmoke-test metrics")
    print(f"  train loss: {float(np.mean(losses)):.6f}")
    print(f"  validation RMSE: {val_rmse:.6f}")
    print(f"  validation MAE: {val_mae:.6f}")
    print(f"  validation NDCG@10: {val_ndcg:.6f}")


def tensor_stats(prefix: str, values: torch.Tensor) -> None:
    detached = values.detach().float()
    print(f"  {prefix} mean: {detached.mean().item():.6f}")
    print(f"  {prefix} std: {detached.std(unbiased=False).item():.6f}")
    print(f"  {prefix} min: {detached.min().item():.6f}")
    print(f"  {prefix} max: {detached.max().item():.6f}")


def run_semantic_gate_debug(
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    if split.target_train.empty:
        raise SystemExit("Semantic-gate debug needs non-empty target train interactions.")

    set_all_seeds(cfg.seed)
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    variant_with_gate = VariantConfig("A0 Full SG-GATv2 debug", True, True, True, 0.1)
    variant_without_gate = VariantConfig("A0 gate disabled debug", True, False, True, 0.1)
    model_with_gate = SGGATv2(
        split.num_users,
        split.num_items,
        semantic_item_t,
        semantic_user_t,
        cfg,
        variant_with_gate,
    ).to(device)
    set_all_seeds(cfg.seed)
    model_without_gate = SGGATv2(
        split.num_users,
        split.num_items,
        semantic_item_t,
        semantic_user_t,
        cfg,
        variant_without_gate,
    ).to(device)
    model_without_gate.load_state_dict(model_with_gate.state_dict(), strict=False)

    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    train_users, train_items, _ = df_to_tensors(split.target_train, device)
    batch_idx = torch.arange(min(cfg.batch_size, train_users.numel()), device=device)
    batch_users = train_users[batch_idx]
    batch_items = train_items[batch_idx]

    model_with_gate.eval()
    model_without_gate.eval()
    with torch.no_grad():
        debug = model_with_gate.attention_debug(edge_index, edge_user_node_idx, edge_item_node_idx)
        semantic_prior = debug["semantic_prior"]
        structural_logits = debug["structural_logits"]
        gated_logits = debug["gated_logits"]
        gamma = debug["gamma"]
        logit_diff = (gated_logits - structural_logits).abs().mean()

        user_z_gate, item_z_gate = model_with_gate.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        pred_gate = model_with_gate.predict(batch_users, batch_items, user_z_gate, item_z_gate)
        user_z_no_gate, item_z_no_gate = model_without_gate.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        pred_no_gate = model_without_gate.predict(batch_users, batch_items, user_z_no_gate, item_z_no_gate)
        pred_diff = (pred_gate - pred_no_gate).abs().mean()

    print("\nSemantic-gate debug diagnostics")
    print(f"  initial gamma value: {gamma.item():.6f}")
    print(f"  structural_attention_logit mean: {structural_logits.mean().item():.6f}")
    print(f"  structural_attention_logit std: {structural_logits.std(unbiased=False).item():.6f}")
    print(f"  semantic_prior mean: {semantic_prior.mean().item():.6f}")
    print(f"  semantic_prior std: {semantic_prior.std(unbiased=False).item():.6f}")
    print(f"  gated_attention_logit mean: {gated_logits.mean().item():.6f}")
    print(f"  gated_attention_logit std: {gated_logits.std(unbiased=False).item():.6f}")
    print(
        "  mean absolute difference between gated logits and structural-only logits: "
        f"{logit_diff.item():.8f}"
    )
    print(f"  prediction mean with gate: {pred_gate.mean().item():.6f}")
    print(f"  prediction mean without gate: {pred_no_gate.mean().item():.6f}")
    print(f"  mean absolute prediction difference: {pred_diff.item():.8f}")


def diagnostic_encode_with_gate_override(
    model: SGGATv2,
    edge_index: torch.Tensor,
    edge_user_node_idx: torch.Tensor,
    edge_item_node_idx: torch.Tensor,
    gamma_value: Optional[float] = None,
    random_prior: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = model.initial_nodes()
    node_sem = model.node_semantics()
    src, dst = edge_index
    layer = model.layer
    h_att = torch.tanh(layer.lin_l(x0[src]) + layer.lin_r(x0[dst]))
    structural_logits = (h_att * layer.att).sum(dim=-1)
    user_sem = layer.semantic_proj(node_sem[edge_user_node_idx])
    item_sem = layer.semantic_proj(node_sem[edge_item_node_idx])
    semantic_prior = F.cosine_similarity(user_sem, item_sem, dim=-1, eps=1e-8)
    if random_prior:
        semantic_prior = torch.randn_like(semantic_prior)
    gamma = layer.gamma if gamma_value is None else torch.tensor(float(gamma_value), device=x0.device)
    logits = structural_logits + gamma * semantic_prior
    alpha = segment_softmax(logits, dst, x0.shape[0])
    messages = layer.msg(x0[src]) * alpha.unsqueeze(-1)
    out = torch.zeros(x0.shape[0], messages.shape[1], device=x0.device, dtype=messages.dtype)
    out.index_add_(0, dst, messages)
    encoded = F.elu(out) + model.residual_proj(x0)
    return encoded[: model.num_users], encoded[model.num_users :], logits


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def diagnostic_grad_groups(model: SGGATv2) -> Dict[str, Iterable[nn.Parameter]]:
    graph_params = [model.layer.lin_l.weight, model.layer.lin_r.weight, model.layer.att, model.layer.msg.weight]
    decoder_params: List[nn.Parameter] = list(model.rating_head.parameters())
    decoder_params.extend(list(model.user_bias.parameters()))
    decoder_params.extend(list(model.item_bias.parameters()))
    decoder_params.append(model.global_bias)
    groups: Dict[str, Iterable[nn.Parameter]] = {
        "user embedding parameters": model.user_emb.parameters(),
        "item embedding parameters": model.item_emb.parameters(),
        "graph encoder parameters": graph_params,
        "decoder parameters": decoder_params,
        "semantic projection parameters": model.layer.semantic_proj.parameters(),
        "gamma parameter": [model.layer.gamma],
    }
    return groups


def instantiate_sggat(
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    variant: VariantConfig,
    device: torch.device,
) -> SGGATv2:
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    return SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, cfg, variant).to(device)


def print_gradient_diagnostics(
    loss_name: str,
    model: SGGATv2,
    loss: torch.Tensor,
) -> None:
    model.zero_grad(set_to_none=True)
    loss.backward()
    print(f"  {loss_name} loss value: {loss.detach().item():.6f}")
    for group_name, params in diagnostic_grad_groups(model).items():
        print(f"    {group_name} grad norm: {grad_norm(params):.8f}")


def run_hard_diagnose_model_path(
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    if split.target_train.empty:
        raise SystemExit("Hard diagnosis needs a non-empty target train split.")
    variant = VariantConfig("A0 Full SG-GATv2", True, True, True, 0.1)
    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)
    batch_idx = torch.arange(min(cfg.batch_size, train_users.numel()), device=device)
    batch_users = train_users[batch_idx]
    batch_items = train_items[batch_idx]
    batch_ratings = train_ratings[batch_idx]

    set_all_seeds(cfg.seed)
    model = instantiate_sggat(split, item_semantic, user_semantic, cfg, variant, device)
    model.eval()
    with torch.no_grad():
        initial_nodes = model.initial_nodes()
        initial_user = initial_nodes[: model.num_users]
        initial_item = initial_nodes[model.num_users :]
        encoded_user, encoded_item = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        user_diff = (encoded_user - initial_user).abs().mean()
        item_diff = (encoded_item - initial_item).abs().mean()

    print("\nHard model-path diagnostics")
    print("  decoder input uses graph-encoded user embeddings: yes")
    print("  decoder input uses graph-encoded item embeddings: yes")
    print(f"  current gamma: {model.layer.gamma.detach().item():.6f}")
    print(f"  current beta_user: {model.beta_user.detach().item():.6f}")
    print(f"  current beta_item: {model.beta_item.detach().item():.6f}")
    print(f"  user embeddings before encoder shape: {tuple(initial_user.shape)}")
    print(f"  user embeddings after encoder shape: {tuple(encoded_user.shape)}")
    print(f"  item embeddings before encoder shape: {tuple(initial_item.shape)}")
    print(f"  item embeddings after encoder shape: {tuple(encoded_item.shape)}")
    print(f"  mean absolute difference initial vs encoded user embeddings: {user_diff.item():.8f}")
    print(f"  mean absolute difference initial vs encoded item embeddings: {item_diff.item():.8f}")

    print("\nSemantic gate kill-switch tests")
    cases = [
        ("normal gamma", None, False),
        ("force gamma = 0", 0.0, False),
        ("force gamma = 10", 10.0, False),
        ("random semantic_prior", None, True),
    ]
    normal_pred = None
    for name, gamma_override, random_prior in cases:
        with torch.no_grad():
            user_z, item_z, logits = diagnostic_encode_with_gate_override(
                model,
                edge_index,
                edge_user_node_idx,
                edge_item_node_idx,
                gamma_value=gamma_override,
                random_prior=random_prior,
            )
            pred = model.predict(batch_users, batch_items, user_z, item_z)
            if normal_pred is None:
                normal_pred = pred.detach().clone()
            pred_diff = (pred - normal_pred).abs().mean()
        print(f"  {name}:")
        print(f"    prediction mean/std: {pred.mean().item():.6f}/{pred.std(unbiased=False).item():.6f}")
        print(f"    mean absolute prediction difference from normal gamma: {pred_diff.item():.8f}")
        print(f"    attention logit mean/std: {logits.mean().item():.6f}/{logits.std(unbiased=False).item():.6f}")

    print("\nInfoNCE gradient test")
    for loss_name in ["RMSE-only", "InfoNCE-only", "total"]:
        set_all_seeds(cfg.seed)
        fresh = instantiate_sggat(split, item_semantic, user_semantic, cfg, variant, device)
        user_z, item_z = fresh.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
        pred = fresh.predict(batch_users, batch_items, user_z, item_z)
        rmse_loss = torch.sqrt(F.mse_loss(pred, batch_ratings) + 1e-8)
        infonce_loss = fresh.info_nce_loss(user_z, item_z, batch_users, batch_items)
        if loss_name == "RMSE-only":
            loss = rmse_loss
        elif loss_name == "InfoNCE-only":
            loss = infonce_loss
        else:
            loss = rmse_loss + variant.lambda_cl * infonce_loss
        print_gradient_diagnostics(loss_name, fresh, loss)

    print("\nFlag verification")
    for check_variant in [
        VariantConfig("A0", True, True, True, 0.1),
        VariantConfig("A1", True, True, False, 0.0),
        VariantConfig("A2", True, False, True, 0.1),
    ]:
        check_model = instantiate_sggat(split, item_semantic, user_semantic, cfg, check_variant, device)
        trainable = sum(p.numel() for p in check_model.parameters() if p.requires_grad)
        print(f"  {check_variant.name}:")
        print(f"    use_llm_init: {check_model.variant.use_llm_init}")
        print(f"    use_semantic_gate: {check_model.variant.use_semantic_gate}")
        print(f"    use_infonce: {check_model.variant.use_infonce}")
        print(f"    lambda_cl: {check_model.variant.lambda_cl}")
        print(f"    gamma exists: {hasattr(check_model.layer, 'gamma')}")
        print(f"    number of trainable parameters: {trainable:,}")


def plot_results(df: pd.DataFrame, x_col: str, title: str, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax1.plot(x, df["rmse"], marker="o", label="RMSE", color="#1f77b4")
    ax1.set_ylabel("RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df[x_col].astype(str), rotation=30, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, df["ndcg@10"], marker="s", label="NDCG@10", color="#2ca02c")
    ax2.set_ylabel("NDCG@10")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def count_summary(values: pd.Series) -> Dict[str, float]:
    if values.empty:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def print_count_summary(label: str, values: pd.Series) -> None:
    stats = count_summary(values)
    print(
        f"  {label}: min={stats['min']:.0f}, median={stats['median']:.2f}, "
        f"mean={stats['mean']:.2f}, max={stats['max']:.0f}"
    )


def audit_target_split(split: SplitData) -> None:
    target_all = pd.concat(
        [split.target_train, split.target_val, split.target_test],
        ignore_index=True,
    )
    train_items = set(split.target_train.item_idx.astype(int))
    val_items = set(split.target_val.item_idx.astype(int))
    test_items = set(split.target_test.item_idx.astype(int))
    all_target_items = set(int(x) for x in split.target_item_indices)

    print("\nTarget Electronics split audit")
    print("\n1. Basic target statistics")
    print(f"  number of users: {target_all.user_idx.nunique():,}")
    print(f"  number of items: {target_all.item_idx.nunique():,}")
    print(f"  number of interactions: {len(target_all):,}")
    print_count_summary("ratings per user", target_all.groupby("user_idx").size())
    print_count_summary("ratings per item", target_all.groupby("item_idx").size())

    unseen_val_items = val_items - train_items
    unseen_test_items = test_items - train_items
    val_unseen_pct = len(unseen_val_items) / max(len(val_items), 1)
    test_unseen_pct = len(unseen_test_items) / max(len(test_items), 1)
    print("\n2. Train/validation/test item overlap")
    print(f"  number of unique train items: {len(train_items):,}")
    print(f"  number of unique validation items: {len(val_items):,}")
    print(f"  number of unique test items: {len(test_items):,}")
    print(f"  validation items unseen in train: {len(unseen_val_items):,} ({val_unseen_pct:.2%})")
    print(f"  test items unseen in train: {len(unseen_test_items):,} ({test_unseen_pct:.2%})")

    relevance_threshold = 0.0
    test_positive = split.target_test[split.target_test.rating.astype(float) > relevance_threshold]
    positives_by_user = test_positive.groupby("user_idx").size()
    print("\n3. Test positive diagnostics")
    print(f"  relevance threshold used for positive items: rating > {relevance_threshold:.1f}")
    print(f"  number of test positive interactions: {len(test_positive):,}")
    print(f"  number of users with at least one positive test item: {test_positive.user_idx.nunique():,}")
    print_count_summary("positives per user", positives_by_user)

    val_test_items = val_items.union(test_items)
    only_val_test_items = val_test_items - train_items
    no_target_train_items = all_target_items - train_items
    print("\n4. Candidate set diagnostics")
    print(f"  number of target candidate items: {len(all_target_items):,}")
    print(f"  number of candidate items appearing in target train: {len(all_target_items.intersection(train_items)):,}")
    print(f"  number of candidate items appearing only in validation/test: {len(only_val_test_items):,}")
    print(f"  number of candidate items with no target-train interaction: {len(no_target_train_items):,}")

    positive_items = set(test_positive.item_idx.astype(int))
    positive_items_unseen = positive_items - train_items
    unseen_positive_pct = len(positive_items_unseen) / max(len(positive_items), 1)
    print("\n5. Test positive train-visibility warning")
    print(f"  unique positive test items: {len(positive_items):,}")
    print(f"  positive test items unseen in target train: {len(positive_items_unseen):,} ({unseen_positive_pct:.2%})")
    if unseen_positive_pct > 0.20:
        print("  WARNING: more than 20% of test positive items are unseen in target train.")


def iterative_bipartite_kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    filtered = df.copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("raw_user_id").size()
        item_counts = filtered.groupby("raw_item_id").size()
        keep_users = set(user_counts[user_counts >= k].index)
        keep_items = set(item_counts[item_counts >= k].index)
        filtered = filtered[
            filtered.raw_user_id.isin(keep_users) & filtered.raw_item_id.isin(keep_items)
        ].reset_index(drop=True)
        if len(filtered) == before:
            return filtered


def print_interaction_stats(label: str, df: pd.DataFrame) -> None:
    print(f"  {label}:")
    print(f"    users: {df.raw_user_id.nunique():,}")
    print(f"    items: {df.raw_item_id.nunique():,}")
    print(f"    interactions: {len(df):,}")
    print_count_summary("ratings per user", df.groupby("raw_user_id").size())
    print_count_summary("ratings per item", df.groupby("raw_item_id").size())


def audit_kcore_target_split(target: pd.DataFrame, cfg: ExperimentConfig) -> None:
    if target.empty:
        print("  Target split diagnostics skipped: no target interactions remain.")
        return
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(target))
    n_train = int(len(target) * cfg.train_frac)
    n_val = int(len(target) * cfg.val_frac)
    train = target.iloc[perm[:n_train]].reset_index(drop=True)
    val = target.iloc[perm[n_train : n_train + n_val]].reset_index(drop=True)
    test = target.iloc[perm[n_train + n_val :]].reset_index(drop=True)

    train_items = set(train.raw_item_id.astype(str))
    val_items = set(val.raw_item_id.astype(str))
    test_items = set(test.raw_item_id.astype(str))
    unseen_val = val_items - train_items
    unseen_test = test_items - train_items
    print("  Target Electronics 80/10/10 split after k-core:")
    print(f"    train interactions: {len(train):,}")
    print(f"    validation interactions: {len(val):,}")
    print(f"    test interactions: {len(test):,}")
    print(f"    unique train items: {len(train_items):,}")
    print(f"    unique validation items: {len(val_items):,}")
    print(f"    unique test items: {len(test_items):,}")
    print(f"    validation items unseen in train: {len(unseen_val):,} ({len(unseen_val) / max(len(val_items), 1):.2%})")
    print(f"    test items unseen in train: {len(unseen_test):,} ({len(unseen_test) / max(len(test_items), 1):.2%})")

    positive = test[test.rating.astype(float) >= 4.0]
    positives_per_user = positive.groupby("raw_user_id").size()
    print("  Target Electronics test positives with relevance threshold rating >= 4.0:")
    print(f"    number of test positive interactions: {len(positive):,}")
    print(f"    users with at least one positive test item: {positive.raw_user_id.nunique():,}")
    print_count_summary("positives per user", positives_per_user)


def audit_kcore_options(base_dir: Path, cfg: ExperimentConfig) -> None:
    source = load_interactions(base_dir / SOURCE_CSV, BOOKS)
    target = load_interactions(base_dir / TARGET_CSV, ELECTRONICS)
    shared_users = sorted(set(source.raw_user_id).intersection(set(target.raw_user_id)))
    source = source[source.raw_user_id.isin(shared_users)].reset_index(drop=True)
    target = target[target.raw_user_id.isin(shared_users)].reset_index(drop=True)

    print("\nK-core audit after initial shared-user filtering")
    print(f"  initial shared users: {len(shared_users):,}")
    print(f"  initial source interactions: {len(source):,}")
    print(f"  initial target interactions: {len(target):,}")

    for k in [5, 10]:
        print(f"\nIterative bipartite {k}-core audit")
        source_k = iterative_bipartite_kcore(source, k)
        target_k = iterative_bipartite_kcore(target, k)
        shared_after = sorted(set(source_k.raw_user_id).intersection(set(target_k.raw_user_id)))
        source_k = source_k[source_k.raw_user_id.isin(shared_after)].reset_index(drop=True)
        target_k = target_k[target_k.raw_user_id.isin(shared_after)].reset_index(drop=True)
        print(f"  users still shared across both domains: {len(shared_after):,}")
        print_interaction_stats("Source Books", source_k)
        print_interaction_stats("Target Electronics", target_k)
        audit_kcore_target_split(target_k, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-full-ablation", action="store_true")
    parser.add_argument("--debug-semantic-gate", action="store_true")
    parser.add_argument("--debug-ranking-metrics", action="store_true")
    parser.add_argument("--hard-diagnose-model-path", action="store_true")
    parser.add_argument("--audit-target-split", action="store_true")
    parser.add_argument("--audit-kcore-options", action="store_true")
    parser.add_argument("--use-iterative-5core", action="store_true")
    parser.add_argument("--run-overlap-sensitivity", action="store_true")
    parser.add_argument("--run-hyperparameter-sensitivity", action="store_true")
    parser.add_argument("--run-metadata-sensitivity", action="store_true")
    parser.add_argument("--run-variant", choices=["A0", "A1", "A2", "A0R", "A1R", "A2R", "A3R", "A_DEBUG_NoGraph"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    return parser.parse_args()


def run_full_ablation(
    base_dir: Path,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    metadata_coverage: Dict[str, Dict[str, object]],
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    variants = [
        VariantConfig("A0 Full SG-GATv2", True, True, True, 0.1),
        VariantConfig("A1 w/o InfoNCE", True, True, False, 0.0),
        VariantConfig("A2 w/o Semantic Gate", True, False, True, 0.1),
        VariantConfig("A3 w/o LLM Initialization", False, False, True, 0.1),
        VariantConfig("A4 semantic-only baseline", False, False, False, 0.0, semantic_only=True),
    ]
    logs = {
        "config": asdict(cfg),
        "metadata_coverage": metadata_coverage,
        "ablation": [],
        "lambda_sensitivity": [],
    }

    ablation_rows = []
    for variant in variants:
        metrics, log = run_variant(variant, split, item_semantic, user_semantic, cfg, device)
        row = {"variant": variant.name, **asdict(variant), **metrics}
        ablation_rows.append(row)
        logs["ablation"].append(log)
        print(f"  test RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} NDCG@10={metrics['ndcg@10']:.4f}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_path = base_dir / "results_ablation_books_electronics.csv"
    ablation_df.to_csv(ablation_path, index=False)
    plot_results(ablation_df, "variant", "Books -> Electronics Ablation", base_dir / "ablation_rmse_ndcg.png")

    lambda_rows = []
    for lambda_cl in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
        variant = VariantConfig(f"Full SG-GATv2 lambda={lambda_cl}", True, True, True, lambda_cl)
        metrics, log = run_variant(variant, split, item_semantic, user_semantic, cfg, device)
        row = {"lambda_cl": lambda_cl, **metrics}
        lambda_rows.append(row)
        logs["lambda_sensitivity"].append(log)
        print(f"  lambda={lambda_cl} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} NDCG@10={metrics['ndcg@10']:.4f}")

    lambda_df = pd.DataFrame(lambda_rows)
    lambda_path = base_dir / "results_lambda_sensitivity_books_electronics.csv"
    lambda_df.to_csv(lambda_path, index=False)
    plot_results(lambda_df, "lambda_cl", "Lambda Sensitivity", base_dir / "lambda_sensitivity_rmse_ndcg.png")

    save_json(base_dir / "logs_ablation_books_electronics.json", logs)
    print("\nSaved outputs")
    print(f"  {ablation_path}")
    print(f"  {lambda_path}")
    print(f"  {base_dir / 'logs_ablation_books_electronics.json'}")
    print(f"  {base_dir / 'ablation_rmse_ndcg.png'}")
    print(f"  {base_dir / 'lambda_sensitivity_rmse_ndcg.png'}")


def single_variant_spec(variant_id: str, use_iterative_5core: bool) -> Tuple[VariantConfig, str, str]:
    if variant_id == "A0":
        result_name = "results_A0_full_sggatv2_5core.csv" if use_iterative_5core else "results_A0_full_sggatv2.csv"
        return (
            VariantConfig("A0 Full SG-GATv2", True, True, True, 0.1),
            "checkpoint_A0_full_sggatv2.pt",
            result_name,
        )
    if variant_id == "A1":
        if not use_iterative_5core:
            raise SystemExit("A1 single-variant mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A1_w_o_InfoNCE", True, True, False, 0.0),
            "checkpoint_A1_wo_infonce.pt",
            "results_A1_wo_infonce_5core.csv",
        )
    if variant_id == "A2":
        if not use_iterative_5core:
            raise SystemExit("A2 single-variant mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A2_w_o_SemanticGate", True, False, True, 0.1),
            "checkpoint_A2_wo_semantic_gate.pt",
            "results_A2_wo_semantic_gate_5core.csv",
        )
    if variant_id == "A0R":
        if not use_iterative_5core:
            raise SystemExit("A0R single-variant mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A0R Full SG-GATv2 with Residual Fusion", True, True, True, 0.1, residual_fusion=True),
            "checkpoint_A0R_full_sggatv2_residual.pt",
            "results_A0R_full_sggatv2_residual_5core.csv",
        )
    if variant_id == "A1R":
        if not use_iterative_5core:
            raise SystemExit("A1R single-variant mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A1R_w_o_InfoNCE_Residual", True, True, False, 0.0, residual_fusion=True),
            "checkpoint_A1R_wo_infonce_residual.pt",
            "results_A1R_wo_infonce_residual_5core.csv",
        )
    if variant_id == "A2R":
        if not use_iterative_5core:
            raise SystemExit("A2R single-variant mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A2R_w_o_SemanticGate_Residual", True, False, True, 0.1, residual_fusion=True),
            "checkpoint_A2R_wo_semantic_gate_residual.pt",
            "results_A2R_wo_semantic_gate_residual_5core.csv",
        )
    if variant_id == "A3R":
        if not use_iterative_5core:
            raise SystemExit("A3R single-variant mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A3R_w_o_LLMInitialization_Residual", False, False, True, 0.1, residual_fusion=True),
            "checkpoint_A3R_wo_llm_init_residual.pt",
            "results_A3R_wo_llm_init_residual_5core.csv",
        )
    if variant_id == "A_DEBUG_NoGraph":
        if not use_iterative_5core:
            raise SystemExit("A_DEBUG_NoGraph mode is currently supported for --use-iterative-5core.")
        return (
            VariantConfig("A_DEBUG_NoGraph", True, True, True, 0.1, no_graph=True),
            "checkpoint_DEBUG_no_graph.pt",
            "results_DEBUG_no_graph_5core.csv",
        )
    raise SystemExit(f"Unsupported variant: {variant_id}")


def run_single_variant(
    variant_id: str,
    base_dir: Path,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
    use_iterative_5core: bool,
) -> None:
    variant, checkpoint_name, result_name = single_variant_spec(variant_id, use_iterative_5core)
    set_all_seeds(cfg.seed)
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)

    checkpoint_path = base_dir / checkpoint_name
    best_val_rmse = float("inf")
    best_val_metrics: Dict[str, float] = {}
    best_epoch = 0
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
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

        val_metrics = evaluate_graph_model(
            model,
            edge_index,
            edge_user_node_idx,
            edge_item_node_idx,
            split.target_val,
            split.target_train,
            split.target_val.iloc[0:0].copy(),
            split,
            cfg,
            device,
        )
        print(
            f"Epoch {epoch:03d} "
            f"train_loss={float(np.mean(losses)):.6f} "
            f"val_RMSE={val_metrics['rmse']:.6f} "
            f"val_MAE={val_metrics['mae']:.6f} "
            f"val_NDCG@10={val_metrics['ndcg@10']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse - 1e-5:
            best_val_rmse = val_metrics["rmse"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": asdict(variant),
                    "config": asdict(cfg),
                    "best_epoch": best_epoch,
                    "val_metrics": best_val_metrics,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_graph_model(
        model,
        edge_index,
        edge_user_node_idx,
        edge_item_node_idx,
        split.target_test,
        split.target_train,
        split.target_val,
        split,
        cfg,
        device,
    )
    final_gamma = float(model.layer.gamma.detach().cpu().item()) if variant.use_semantic_gate else "NA"
    final_beta_user = float(model.beta_user.detach().cpu().item())
    final_beta_item = float(model.beta_item.detach().cpu().item())
    print(f"\n{variant_id} final test metrics")
    print(f"  best epoch: {best_epoch}")
    print(f"  validation RMSE: {best_val_metrics['rmse']:.6f}")
    print(f"  validation MAE: {best_val_metrics['mae']:.6f}")
    print(f"  validation NDCG@10: {best_val_metrics['ndcg@10']:.6f}")
    print(f"  test RMSE: {test_metrics['rmse']:.6f}")
    print(f"  test MAE: {test_metrics['mae']:.6f}")
    print(f"  test NDCG@10: {test_metrics['ndcg@10']:.6f}")
    if isinstance(final_gamma, float):
        print(f"  final gamma: {final_gamma:.6f}")
    else:
        print(f"  final gamma: {final_gamma}")
    print(f"  final beta_user: {final_beta_user:.6f}")
    print(f"  final beta_item: {final_beta_item:.6f}")

    result = {
        "variant": variant.name,
        "use_llm_init": variant.use_llm_init,
        "use_semantic_gate": variant.use_semantic_gate,
        "use_infonce": variant.use_infonce,
        "lambda_cl": variant.lambda_cl,
        "temperature": cfg.temperature,
        "best_epoch": best_epoch,
        "val_rmse": best_val_metrics["rmse"],
        "val_mae": best_val_metrics["mae"],
        "val_ndcg10": best_val_metrics["ndcg@10"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_ndcg10": test_metrics["ndcg@10"],
        "final_gamma": final_gamma,
        "final_beta_user": final_beta_user,
        "final_beta_item": final_beta_item,
    }
    output_path = base_dir / result_name
    pd.DataFrame([result]).to_csv(output_path, index=False)
    print(f"Saved one-row result CSV: {output_path}")
    print(f"Saved best checkpoint: {checkpoint_path}")


def retained_shared_user_indices(split: SplitData, overlap_ratio: float, seed: int) -> np.ndarray:
    all_users = np.arange(split.num_users, dtype=np.int64)
    if overlap_ratio >= 1.0:
        return all_users
    retain_count = max(1, int(round(len(all_users) * overlap_ratio)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(all_users, size=retain_count, replace=False)).astype(np.int64)


def train_overlap_setting(
    overlap_label: str,
    overlap_ratio: float,
    retained_users_np: np.ndarray,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    variant = VariantConfig(
        f"Overlap {overlap_label} Full SG-GATv2-R",
        use_llm_init=True,
        use_semantic_gate=True,
        use_infonce=True,
        lambda_cl=0.1,
        residual_fusion=True,
    )
    set_all_seeds(cfg.seed)
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    retained_users = torch.tensor(retained_users_np, dtype=torch.long, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    full_edges = build_edge_index(split, device)
    source_edges, target_edges = build_domain_edge_indices(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)

    best_state = None
    best_val_metrics: Dict[str, float] = {}
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            user_z, item_z = model.encode(*full_edges)
            pred = model.predict(train_users[idx], train_items[idx], user_z, item_z)
            rating_loss = F.mse_loss(pred, train_ratings[idx])
            source_user_z, _ = model.encode(*source_edges)
            target_user_z, _ = model.encode(*target_edges)
            cl_loss = model.shared_user_alignment_loss(source_user_z, target_user_z, retained_users)
            loss = rating_loss + variant.lambda_cl * cl_loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_graph_model(
            model,
            *full_edges,
            split.target_val,
            split.target_train,
            split.target_val.iloc[0:0].copy(),
            split,
            cfg,
            device,
        )
        print(
            f"Overlap {overlap_label} epoch {epoch:03d} "
            f"train_loss={float(np.mean(losses)):.6f} "
            f"val_RMSE={val_metrics['rmse']:.6f} "
            f"val_MAE={val_metrics['mae']:.6f} "
            f"val_NDCG@10={val_metrics['ndcg@10']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse - 1e-5:
            best_val_rmse = val_metrics["rmse"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            print(f"Overlap {overlap_label}: early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_graph_model(
        model,
        *full_edges,
        split.target_test,
        split.target_train,
        split.target_val,
        split,
        cfg,
        device,
    )
    final_gamma = float(model.layer.gamma.detach().cpu().item())
    final_beta_user = float(model.beta_user.detach().cpu().item())
    final_beta_item = float(model.beta_item.detach().cpu().item())
    print(
        f"Overlap {overlap_label} final: "
        f"val_RMSE={best_val_metrics['rmse']:.6f} test_RMSE={test_metrics['rmse']:.6f}"
    )
    return {
        "overlap": overlap_label,
        "overlap_ratio": overlap_ratio,
        "retained_shared_users": int(len(retained_users_np)),
        "total_shared_users": int(split.num_users),
        "use_llm_init": variant.use_llm_init,
        "use_semantic_gate": variant.use_semantic_gate,
        "use_infonce": variant.use_infonce,
        "use_residual_fusion": variant.residual_fusion,
        "lambda_cl": variant.lambda_cl,
        "temperature": cfg.temperature,
        "best_epoch": best_epoch,
        "val_rmse": best_val_metrics["rmse"],
        "val_mae": best_val_metrics["mae"],
        "val_ndcg10": best_val_metrics["ndcg@10"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_ndcg10": test_metrics["ndcg@10"],
        "final_gamma": final_gamma,
        "final_beta_user": final_beta_user,
        "final_beta_item": final_beta_item,
    }


def run_overlap_sensitivity(
    base_dir: Path,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    rows = []
    for overlap_label, overlap_ratio in [("100%", 1.0), ("50%", 0.5), ("25%", 0.25), ("10%", 0.10)]:
        retained = retained_shared_user_indices(split, overlap_ratio, seed=42)
        print(
            f"\nRunning overlap sensitivity {overlap_label}: "
            f"{len(retained):,}/{split.num_users:,} shared users retained for InfoNCE alignment"
        )
        row = train_overlap_setting(
            overlap_label,
            overlap_ratio,
            retained,
            split,
            item_semantic,
            user_semantic,
            cfg,
            device,
        )
        rows.append(row)

    output_path = base_dir / "results_overlap_sensitivity_books_electronics_5core.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved overlap sensitivity result CSV: {output_path}")


def train_hyperparameter_setting(
    group: str,
    temperature: float,
    lambda_cl: float,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    variant = VariantConfig(
        "Full SG-GATv2-R hyperparameter sensitivity",
        use_llm_init=True,
        use_semantic_gate=True,
        use_infonce=True,
        lambda_cl=lambda_cl,
        residual_fusion=True,
    )
    run_cfg = ExperimentConfig(
        seed=cfg.seed,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        semantic_proj_dim=cfg.semantic_proj_dim,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        rating_min=cfg.rating_min,
        rating_max=cfg.rating_max,
        temperature=temperature,
        gamma=0.1,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        topk=cfg.topk,
        ndcg_chunk_users=cfg.ndcg_chunk_users,
        encode_batch_size=cfg.encode_batch_size,
        ranking_mode=cfg.ranking_mode,
        relevance_threshold=cfg.relevance_threshold,
        sampled_negatives=cfg.sampled_negatives,
    )
    set_all_seeds(run_cfg.seed)
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, run_cfg, variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)
    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    train_users, train_items, train_ratings = df_to_tensors(split.target_train, device)

    best_state = None
    best_val_metrics: Dict[str, float] = {}
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, run_cfg.epochs + 1):
        model.train()
        perm = torch.randperm(train_users.numel(), device=device)
        losses = []
        for start in range(0, train_users.numel(), run_cfg.batch_size):
            idx = perm[start : start + run_cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)
            pred = model.predict(train_users[idx], train_items[idx], user_z, item_z)
            rating_loss = F.mse_loss(pred, train_ratings[idx])
            cl_loss = model.info_nce_loss(user_z, item_z, train_users[idx], train_items[idx])
            loss = rating_loss + variant.lambda_cl * cl_loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_graph_model(
            model,
            edge_index,
            edge_user_node_idx,
            edge_item_node_idx,
            split.target_val,
            split.target_train,
            split.target_val.iloc[0:0].copy(),
            split,
            run_cfg,
            device,
        )
        print(
            f"{group} temperature={temperature:g} lambda_cl={lambda_cl:g} "
            f"epoch {epoch:03d} train_loss={float(np.mean(losses)):.6f} "
            f"val_RMSE={val_metrics['rmse']:.6f} "
            f"val_MAE={val_metrics['mae']:.6f} "
            f"val_NDCG@10={val_metrics['ndcg@10']:.6f}"
        )
        if val_metrics["rmse"] < best_val_rmse - 1e-5:
            best_val_rmse = val_metrics["rmse"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= run_cfg.patience:
            print(
                f"{group} temperature={temperature:g} lambda_cl={lambda_cl:g}: "
                f"early stopping at epoch {epoch}; best epoch was {best_epoch}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_graph_model(
        model,
        edge_index,
        edge_user_node_idx,
        edge_item_node_idx,
        split.target_test,
        split.target_train,
        split.target_val,
        split,
        run_cfg,
        device,
    )
    final_gamma = float(model.layer.gamma.detach().cpu().item())
    final_beta_user = float(model.beta_user.detach().cpu().item())
    final_beta_item = float(model.beta_item.detach().cpu().item())
    print(
        f"{group} temperature={temperature:g} lambda_cl={lambda_cl:g} final: "
        f"val_RMSE={best_val_metrics['rmse']:.6f} test_RMSE={test_metrics['rmse']:.6f}"
    )
    return {
        "group": group,
        "temperature": temperature,
        "lambda_cl": lambda_cl,
        "best_epoch": best_epoch,
        "val_rmse": best_val_metrics["rmse"],
        "val_mae": best_val_metrics["mae"],
        "val_ndcg10": best_val_metrics["ndcg@10"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_ndcg10": test_metrics["ndcg@10"],
        "final_gamma": final_gamma,
        "final_beta_user": final_beta_user,
        "final_beta_item": final_beta_item,
    }


def run_hyperparameter_sensitivity(
    base_dir: Path,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    output_path = base_dir / "results_hyperparameter_sensitivity_books_electronics_5core.csv"

    def save_completed_rows(rows: List[Dict[str, object]]) -> None:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        print(f"Saved {len(rows)} completed hyperparameter sensitivity row(s): {output_path}")

    rows = []
    for temperature in [0.1, 0.2, 0.5, 1.0]:
        print(f"\nRunning temperature sensitivity: temperature={temperature:g}, lambda_cl=0.1")
        row = train_hyperparameter_setting(
            "temperature_sensitivity",
            temperature,
            0.1,
            split,
            item_semantic,
            user_semantic,
            cfg,
            device,
        )
        rows.append(row)
        save_completed_rows(rows)
    for lambda_cl in [0.01, 0.05, 0.1, 0.2, 0.5]:
        print(f"\nRunning lambda sensitivity: temperature=0.2, lambda_cl={lambda_cl:g}")
        row = train_hyperparameter_setting(
            "lambda_sensitivity",
            0.2,
            lambda_cl,
            split,
            item_semantic,
            user_semantic,
            cfg,
            device,
        )
        rows.append(row)
        save_completed_rows(rows)
    print(f"\nSaved hyperparameter sensitivity result CSV: {output_path}")


def item_indices_by_domain(split: SplitData) -> Tuple[np.ndarray, np.ndarray]:
    source_indices = []
    target_indices = []
    for item_key, idx in split.item_to_idx.items():
        if item_key.startswith(BOOKS + ":"):
            source_indices.append(int(idx))
        elif item_key.startswith(ELECTRONICS + ":"):
            target_indices.append(int(idx))
    return np.array(sorted(source_indices), dtype=np.int64), np.array(sorted(target_indices), dtype=np.int64)


def apply_metadata_retention_mask(
    item_semantic: np.ndarray,
    split: SplitData,
    retention_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, int, int]:
    source_indices, target_indices = item_indices_by_domain(split)
    masked = np.zeros_like(item_semantic)
    rng = np.random.default_rng(seed)

    def retained_indices(indices: np.ndarray) -> np.ndarray:
        if retention_ratio >= 1.0:
            return indices
        if retention_ratio <= 0.0 or len(indices) == 0:
            return np.array([], dtype=np.int64)
        retain_count = int(round(len(indices) * retention_ratio))
        retain_count = min(max(retain_count, 1), len(indices))
        return np.sort(rng.choice(indices, size=retain_count, replace=False)).astype(np.int64)

    retained_source = retained_indices(source_indices)
    retained_target = retained_indices(target_indices)
    retained_all = np.concatenate([retained_source, retained_target])
    if len(retained_all):
        masked[retained_all] = item_semantic[retained_all]
    return masked.astype(np.float32), int(len(retained_source)), int(len(retained_target))


def run_metadata_sensitivity(
    base_dir: Path,
    split: SplitData,
    item_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    output_path = base_dir / "results_metadata_sensitivity_books_electronics_5core.csv"

    def save_completed_rows(rows: List[Dict[str, object]]) -> None:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        print(f"Saved {len(rows)} completed metadata sensitivity row(s): {output_path}")

    rows = []
    profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
    for retention_ratio in [1.0, 0.5, 0.25, 0.0]:
        masked_item_semantic, retained_source, retained_target = apply_metadata_retention_mask(
            item_semantic,
            split,
            retention_ratio,
            seed=42,
        )
        print(
            f"\nRunning metadata sensitivity: retention={retention_ratio:.2f}, "
            f"retained_source_items={retained_source:,}, retained_target_items={retained_target:,}"
        )
        user_semantic = build_user_semantic_profiles(
            profile_interactions,
            masked_item_semantic,
            split.num_users,
        )
        metrics = train_hyperparameter_setting(
            "metadata_sensitivity",
            0.2,
            0.1,
            split,
            masked_item_semantic,
            user_semantic,
            cfg,
            device,
        )
        rows.append(
            {
                "metadata_retention_ratio": retention_ratio,
                "retained_source_items": retained_source,
                "retained_target_items": retained_target,
                "best_epoch": metrics["best_epoch"],
                "val_rmse": metrics["val_rmse"],
                "val_mae": metrics["val_mae"],
                "val_ndcg10": metrics["val_ndcg10"],
                "test_rmse": metrics["test_rmse"],
                "test_mae": metrics["test_mae"],
                "test_ndcg10": metrics["test_ndcg10"],
                "final_gamma": metrics["final_gamma"],
                "final_beta_user": metrics["final_beta_user"],
                "final_beta_item": metrics["final_beta_item"],
            }
        )
        save_completed_rows(rows)
    print(f"\nSaved metadata sensitivity result CSV: {output_path}")


def config_from_checkpoint(checkpoint: Dict[str, object], fallback: ExperimentConfig) -> ExperimentConfig:
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        return fallback
    values = asdict(fallback)
    values.update({key: raw[key] for key in values if key in raw})
    return ExperimentConfig(**values)


def run_debug_ranking_metrics(
    base_dir: Path,
    split: SplitData,
    item_semantic: np.ndarray,
    user_semantic: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> None:
    checkpoint_path = base_dir / "checkpoint_A0_full_sggatv2.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = config_from_checkpoint(checkpoint, cfg)
    variant = VariantConfig("A0 Full SG-GATv2", True, True, True, 0.1)
    semantic_item_t = torch.tensor(item_semantic, dtype=torch.float32, device=device)
    semantic_user_t = torch.tensor(user_semantic, dtype=torch.float32, device=device)
    model = SGGATv2(split.num_users, split.num_items, semantic_item_t, semantic_user_t, model_cfg, variant).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    edge_index, edge_user_node_idx, edge_item_node_idx = build_edge_index(split, device)
    with torch.no_grad():
        user_z, item_z = model.encode(edge_index, edge_user_node_idx, edge_item_node_idx)

    print("\nRating statistics")
    print_rating_stats("train", split.target_train)
    print_rating_stats("validation", split.target_val)
    print_rating_stats("test", split.target_test)

    print("\nObserved-pair prediction statistics")
    print_prediction_stats("validation predictions", model, user_z, item_z, split.target_val, device)
    print_prediction_stats("test predictions", model, user_z, item_z, split.target_test, device)

    val_metrics = evaluate_graph_model(
        model,
        edge_index,
        edge_user_node_idx,
        edge_item_node_idx,
        split.target_val,
        split.target_train,
        split.target_val.iloc[0:0].copy(),
        split,
        model_cfg,
        device,
    )
    test_metrics = evaluate_graph_model(
        model,
        edge_index,
        edge_user_node_idx,
        edge_item_node_idx,
        split.target_test,
        split.target_train,
        split.target_val,
        split,
        model_cfg,
        device,
    )
    print("\nRecomputed checkpoint metrics using current evaluator")
    print(
        f"  validation: RMSE={val_metrics['rmse']:.6f} "
        f"MAE={val_metrics['mae']:.6f} NDCG@10={val_metrics['ndcg@10']:.6f}"
    )
    print(
        f"  test: RMSE={test_metrics['rmse']:.6f} "
        f"MAE={test_metrics['mae']:.6f} NDCG@10={test_metrics['ndcg@10']:.6f}"
    )

    print("\nNDCG@10 debug definition")
    print("  Current relevance definition: observed heldout target-domain ratings are graded relevance values.")
    print("  Current gain formula: gain = 2^rating - 1; ideal ranking sorts heldout observed ratings descending.")
    print("  Positive test item definition for this debug mode: observed test interactions with rating > 0.")
    print("  Train items are excluded from candidate ranking: yes.")
    print("  Full-catalog mode includes validation items in candidates: yes, as non-relevant competitors.")
    print("  Full-catalog mode includes test positives in candidates: yes.")
    print("  Sampled mode includes test positives and sampled unobserved target items; train/test observed negatives are excluded from negatives.")
    print("  Unobserved items are treated as zero relevance only for ranking, never for RMSE/MAE.")

    full_result = ranking_debug_full_catalog(model, user_z, item_z, split, model_cfg, device)
    sampled_result = ranking_debug_sampled(model, user_z, item_z, split, model_cfg, device)
    print_ranking_debug_result(full_result)
    print_ranking_debug_result(sampled_result)


def main() -> None:
    args = parse_args()
    selected_modes = sum(
        bool(flag)
        for flag in (
            args.smoke_test,
            args.run_full_ablation,
            args.debug_semantic_gate,
            args.debug_ranking_metrics,
            args.hard_diagnose_model_path,
            args.audit_target_split,
            args.audit_kcore_options,
            args.run_overlap_sensitivity,
            args.run_hyperparameter_sensitivity,
            args.run_metadata_sensitivity,
            args.run_variant,
        )
    )
    if selected_modes > 1:
        raise SystemExit(
            "Choose only one mode: --smoke-test, --debug-semantic-gate, "
            "--debug-ranking-metrics, --hard-diagnose-model-path, --audit-target-split, "
            "--audit-kcore-options, --run-overlap-sensitivity, --run-hyperparameter-sensitivity, "
            "--run-metadata-sensitivity, "
            "--run-variant A0/A1/A2/A0R/A1R/A2R/A3R/A_DEBUG_NoGraph, or --run-full-ablation."
        )
    cfg = ExperimentConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        temperature=args.temperature,
        gamma=args.gamma,
    )
    if (
        args.use_iterative_5core
        or args.hard_diagnose_model_path
        or args.run_overlap_sensitivity
        or args.run_hyperparameter_sensitivity
        or args.run_metadata_sensitivity
    ):
        cfg.ranking_mode = "sampled"
        cfg.relevance_threshold = 4.0
        cfg.sampled_negatives = 99
    base_dir = args.data_dir.resolve()
    set_all_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.audit_kcore_options:
        audit_kcore_options(base_dir, cfg)
        return

    split = (
        build_iterative_5core_split(base_dir, cfg)
        if (
            args.use_iterative_5core
            or args.hard_diagnose_model_path
            or args.run_overlap_sensitivity
            or args.run_hyperparameter_sensitivity
            or args.run_metadata_sensitivity
        )
        else build_split(base_dir, cfg)
    )

    if args.audit_target_split:
        audit_target_split(split)
        return

    metadata_coverage = validate_metadata_coverage(base_dir)
    validate_embedding_cache(base_dir)

    if (
        not args.smoke_test
        and not args.run_full_ablation
        and not args.debug_semantic_gate
        and not args.debug_ranking_metrics
        and not args.hard_diagnose_model_path
        and not args.audit_target_split
        and not args.audit_kcore_options
        and not args.run_overlap_sensitivity
        and not args.run_hyperparameter_sensitivity
        and not args.run_metadata_sensitivity
        and not args.run_variant
    ):
        print("\nValidation-only mode complete. No training was started.")
        print("Use --smoke-test for a one-epoch A0 runtime check.")
        print("Use --debug-semantic-gate for attention-logit diagnostics.")
        print("Use --debug-ranking-metrics to inspect checkpoint ranking metrics.")
        print("Use --hard-diagnose-model-path to inspect prediction and gradient paths.")
        print("Use --audit-target-split to inspect target train/validation/test overlap.")
        print("Use --audit-kcore-options to inspect stricter iterative k-core filtering.")
        print("Use --run-overlap-sensitivity to train the shared-user overlap sensitivity experiment.")
        print("Use --run-hyperparameter-sensitivity to train the limited hyperparameter sensitivity experiment.")
        print("Use --run-metadata-sensitivity to train the metadata availability sensitivity experiment.")
        print("Use --run-variant A0, A1, A2, A0R, A1R, A2R, A3R, or A_DEBUG_NoGraph to train one supported variant.")
        print("Use --run-full-ablation to run the full experiments.")
        return

    if args.hard_diagnose_model_path:
        validate_embedding_cache(base_dir)
        diag_cfg = ExperimentConfig(
            seed=cfg.seed,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            semantic_proj_dim=cfg.semantic_proj_dim,
            epochs=1,
            patience=1,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=min(cfg.batch_size, 2048),
            rating_min=cfg.rating_min,
            rating_max=cfg.rating_max,
            temperature=0.2,
            gamma=0.1,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            topk=cfg.topk,
            ndcg_chunk_users=cfg.ndcg_chunk_users,
            encode_batch_size=cfg.encode_batch_size,
            ranking_mode=cfg.ranking_mode,
            relevance_threshold=cfg.relevance_threshold,
            sampled_negatives=cfg.sampled_negatives,
        )
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, split)
        profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
        user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, split.num_users)
        run_hard_diagnose_model_path(split, item_semantic, user_semantic, diag_cfg, device)
        return

    if args.smoke_test or args.debug_semantic_gate:
        smoke_cfg = ExperimentConfig(
            seed=cfg.seed,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            semantic_proj_dim=cfg.semantic_proj_dim,
            epochs=1,
            patience=1,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=min(cfg.batch_size, 2048),
            rating_min=cfg.rating_min,
            rating_max=cfg.rating_max,
            temperature=cfg.temperature,
            gamma=cfg.gamma,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            topk=cfg.topk,
            ndcg_chunk_users=cfg.ndcg_chunk_users,
            encode_batch_size=cfg.encode_batch_size,
            ranking_mode=cfg.ranking_mode,
            relevance_threshold=cfg.relevance_threshold,
            sampled_negatives=cfg.sampled_negatives,
        )
        smoke_split = make_smoke_split(split, smoke_cfg)
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, smoke_split)
        profile_interactions = pd.concat([smoke_split.source_train, smoke_split.target_train], ignore_index=True)
        user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, smoke_split.num_users)
        if args.debug_semantic_gate:
            run_semantic_gate_debug(smoke_split, item_semantic, user_semantic, smoke_cfg, device)
        else:
            run_smoke_test(smoke_split, item_semantic, user_semantic, smoke_cfg, device)
        return

    if args.run_variant in {"A0", "A1", "A2", "A0R", "A1R", "A2R", "A3R", "A_DEBUG_NoGraph"}:
        single_cfg = ExperimentConfig(
            seed=cfg.seed,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            semantic_proj_dim=cfg.semantic_proj_dim,
            epochs=30,
            patience=5,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            rating_min=cfg.rating_min,
            rating_max=cfg.rating_max,
            temperature=0.2,
            gamma=0.1,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            topk=cfg.topk,
            ndcg_chunk_users=cfg.ndcg_chunk_users,
            encode_batch_size=cfg.encode_batch_size,
            ranking_mode=cfg.ranking_mode,
            relevance_threshold=cfg.relevance_threshold,
            sampled_negatives=cfg.sampled_negatives,
        )
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, split)
        profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
        user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, split.num_users)
        run_single_variant(
            args.run_variant,
            base_dir,
            split,
            item_semantic,
            user_semantic,
            single_cfg,
            device,
            args.use_iterative_5core,
        )
        return

    if args.run_overlap_sensitivity:
        overlap_cfg = ExperimentConfig(
            seed=42,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            semantic_proj_dim=cfg.semantic_proj_dim,
            epochs=30,
            patience=5,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            rating_min=cfg.rating_min,
            rating_max=cfg.rating_max,
            temperature=0.2,
            gamma=0.1,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            topk=cfg.topk,
            ndcg_chunk_users=cfg.ndcg_chunk_users,
            encode_batch_size=cfg.encode_batch_size,
            ranking_mode="sampled",
            relevance_threshold=4.0,
            sampled_negatives=99,
        )
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, split)
        profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
        user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, split.num_users)
        run_overlap_sensitivity(base_dir, split, item_semantic, user_semantic, overlap_cfg, device)
        return

    if args.run_hyperparameter_sensitivity:
        hyper_cfg = ExperimentConfig(
            seed=42,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            semantic_proj_dim=cfg.semantic_proj_dim,
            epochs=30,
            patience=5,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            rating_min=cfg.rating_min,
            rating_max=cfg.rating_max,
            temperature=0.2,
            gamma=0.1,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            topk=cfg.topk,
            ndcg_chunk_users=cfg.ndcg_chunk_users,
            encode_batch_size=cfg.encode_batch_size,
            ranking_mode="sampled",
            relevance_threshold=4.0,
            sampled_negatives=99,
        )
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, split)
        profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
        user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, split.num_users)
        run_hyperparameter_sensitivity(base_dir, split, item_semantic, user_semantic, hyper_cfg, device)
        return

    if args.run_metadata_sensitivity:
        metadata_cfg = ExperimentConfig(
            seed=42,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            semantic_proj_dim=cfg.semantic_proj_dim,
            epochs=30,
            patience=5,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            rating_min=cfg.rating_min,
            rating_max=cfg.rating_max,
            temperature=0.2,
            gamma=0.1,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            topk=cfg.topk,
            ndcg_chunk_users=cfg.ndcg_chunk_users,
            encode_batch_size=cfg.encode_batch_size,
            ranking_mode="sampled",
            relevance_threshold=4.0,
            sampled_negatives=99,
        )
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, split)
        run_metadata_sensitivity(base_dir, split, item_semantic, metadata_cfg, device)
        return

    if args.debug_ranking_metrics:
        item_semantic = load_semantic_embeddings_from_existing_cache(base_dir, split)
        profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
        user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, split.num_users)
        run_debug_ranking_metrics(base_dir, split, item_semantic, user_semantic, cfg, device)
        return

    item_semantic = load_or_create_semantic_embeddings(base_dir, split, cfg)
    profile_interactions = pd.concat([split.source_train, split.target_train], ignore_index=True)
    user_semantic = build_user_semantic_profiles(profile_interactions, item_semantic, split.num_users)
    run_full_ablation(base_dir, split, item_semantic, user_semantic, metadata_coverage, cfg, device)


if __name__ == "__main__":
    main()
