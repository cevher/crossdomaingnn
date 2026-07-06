"""
Prepare PTUPCDR-compatible Books -> Electronics data from the project split.

This adapter intentionally lives outside the SG-GATv2-R experiment code. It
recreates the iterative 5-core protocol with seed=42 and writes only files
needed by the external PTUPCDR rating-prediction baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SEED = 42
SOURCE_CSV = "source_books_filtered.csv"
TARGET_CSV = "target_electronics_filtered.csv"


Interaction = Tuple[str, str, float]


def read_interactions(path: Path) -> List[Interaction]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        lower_to_col = {name.lower(): name for name in reader.fieldnames or []}
        user_col = lower_to_col.get("user_id") or lower_to_col.get("reviewerid")
        item_col = lower_to_col.get("item_id") or lower_to_col.get("asin")
        rating_col = lower_to_col.get("rating") or lower_to_col.get("overall")
        if not user_col or not item_col or not rating_col:
            raise ValueError(f"Cannot infer columns from {path}: {reader.fieldnames}")
        return [
            (str(row[user_col]), str(row[item_col]), float(row[rating_col]))
            for row in reader
            if row.get(user_col) and row.get(item_col) and row.get(rating_col)
        ]


def keep_users(rows: Sequence[Interaction], users: set[str]) -> List[Interaction]:
    return [row for row in rows if row[0] in users]


def iterative_bipartite_kcore(rows: Sequence[Interaction], k: int) -> List[Interaction]:
    current = list(rows)
    while True:
        before = len(current)
        user_counts = Counter(user for user, _, _ in current)
        item_counts = Counter(item for _, item, _ in current)
        keep_user = {user for user, count in user_counts.items() if count >= k}
        keep_item = {item for item, count in item_counts.items() if count >= k}
        current = [
            (user, item, rating)
            for user, item, rating in current
            if user in keep_user and item in keep_item
        ]
        if len(current) == before:
            return current


def split_target(rows: Sequence[Interaction]) -> Tuple[List[Interaction], List[Interaction], List[Interaction]]:
    rng = random.Random(SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_train = int(len(rows) * 0.8)
    n_val = int(len(rows) * 0.1)
    train = [rows[i] for i in indices[:n_train]]
    val = [rows[i] for i in indices[n_train : n_train + n_val]]
    test = [rows[i] for i in indices[n_train + n_val :]]
    return train, val, test


def write_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_triplets(path: Path, rows: Sequence[Interaction], user_map: Dict[str, int], item_map: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for user, item, rating in rows:
            writer.writerow([user_map[user], item_map[item], rating])


def write_with_history(
    path: Path,
    rows: Sequence[Interaction],
    user_map: Dict[str, int],
    target_item_map: Dict[str, int],
    source_history: Dict[str, List[int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for user, item, rating in rows:
            writer.writerow([user_map[user], target_item_map[item], rating, source_history.get(user, [])])


def summarize(name: str, rows: Sequence[Interaction]) -> None:
    print(
        f"{name}: interactions={len(rows):,}, "
        f"users={len({u for u, _, _ in rows}):,}, items={len({i for _, i, _ in rows}):,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("../.."))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = read_interactions(project_root / SOURCE_CSV)
    target = read_interactions(project_root / TARGET_CSV)
    shared = {user for user, _, _ in source}.intersection({user for user, _, _ in target})
    source = keep_users(source, shared)
    target = keep_users(target, shared)

    source = iterative_bipartite_kcore(source, 5)
    target = iterative_bipartite_kcore(target, 5)
    shared_after = {user for user, _, _ in source}.intersection({user for user, _, _ in target})
    source = keep_users(source, shared_after)
    target = keep_users(target, shared_after)

    users = sorted(shared_after)
    source_items = sorted({item for _, item, _ in source})
    target_items = sorted({item for _, item, _ in target})
    user_map = {user: idx for idx, user in enumerate(users)}
    source_item_map = {item: idx for idx, item in enumerate(source_items)}
    target_item_map = {item: idx + len(source_items) for idx, item in enumerate(target_items)}

    target_train, target_val, target_test = split_target(target)

    source_history: Dict[str, List[int]] = defaultdict(list)
    for user, item, rating in source:
        if rating >= 4.0:
            source_history[user].append(source_item_map[item])
    for user in source_history:
        source_history[user] = sorted(set(source_history[user]))[:20]

    write_triplets(output_dir / "train_src.csv", source, user_map, source_item_map)
    write_triplets(output_dir / "train_tgt.csv", target_train, user_map, target_item_map)
    write_with_history(output_dir / "val.csv", target_val, user_map, target_item_map, source_history)
    write_with_history(output_dir / "test.csv", target_test, user_map, target_item_map, source_history)
    write_with_history(output_dir / "train_meta.csv", target_train, user_map, target_item_map, source_history)

    write_json(output_dir / "user_id_map.json", user_map)
    write_json(output_dir / "source_item_id_map.json", source_item_map)
    write_json(output_dir / "target_item_id_map.json", target_item_map)
    write_json(
        output_dir / "metadata.json",
        {
            "seed": SEED,
            "protocol": "shared users -> iterative 5-core per domain -> shared users -> target 80/10/10 interaction split",
            "uid_all": len(user_map),
            "iid_all": len(source_item_map) + len(target_item_map),
            "source_items": len(source_item_map),
            "target_items": len(target_item_map),
            "source_domain": "Books",
            "target_domain": "Electronics",
        },
    )

    summarize("source Books train", source)
    summarize("target Electronics train", target_train)
    summarize("target Electronics validation", target_val)
    summarize("target Electronics test", target_test)
    print(f"uid_all={len(user_map):,}")
    print(f"iid_all={len(source_item_map) + len(target_item_map):,}")


if __name__ == "__main__":
    main()
