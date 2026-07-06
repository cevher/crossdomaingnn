"""
Prepare filtered item text CSVs for Books -> Electronics experiments.

This script only extracts and validates item text metadata. It does not create
embeddings and does not start model training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SOURCE_CSV = "source_books_filtered.csv"
TARGET_CSV = "target_electronics_filtered.csv"
BOOKS_META = "meta_Books.jsonl.gz"
ELECTRONICS_META = "meta_Electronics.jsonl.gz"
BOOKS_OUTPUT = "books_item_texts_filtered.csv"
ELECTRONICS_OUTPUT = "electronics_item_texts_filtered.csv"
MATCH_KEYS = ("parent_asin", "asin", "item_id")
SEED = 42


def read_unique_item_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path.name} has no header row.")
        item_col = None
        lower_to_col = {name.lower(): name for name in reader.fieldnames}
        for candidate in ("item_id", "itemid", "asin", "parent_asin"):
            if candidate in lower_to_col:
                item_col = lower_to_col[candidate]
                break
        if item_col is None:
            raise ValueError(f"Could not find item_id column in {path.name}: {reader.fieldnames}")
        items = {str(row[item_col]) for row in reader if row.get(item_col)}
    return sorted(items)


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [flatten_text(v) for v in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [flatten_text(v) for v in value.values()]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def build_text(record: Dict[str, object]) -> str:
    parts = []
    for key in ("title", "features", "description", "categories"):
        text = flatten_text(record.get(key))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def scan_metadata(meta_path: Path, item_ids: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, Dict[str, object]]]:
    needed = set(item_ids)
    match_counts = {key: 0 for key in MATCH_KEYS}
    matched_records = {key: {} for key in MATCH_KEYS}

    with gzip.open(meta_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in MATCH_KEYS:
                value = record.get(key)
                if value is None:
                    continue
                item_id = str(value)
                if item_id in needed and item_id not in matched_records[key]:
                    matched_records[key][item_id] = record
                    match_counts[key] += 1

    return match_counts, matched_records


def choose_best_key(match_counts: Dict[str, int]) -> str:
    return max(MATCH_KEYS, key=lambda key: (match_counts[key], -MATCH_KEYS.index(key)))


def write_item_texts(output_path: Path, records_by_item_id: Dict[str, Dict[str, object]]) -> List[Tuple[str, str]]:
    rows = []
    for item_id in sorted(records_by_item_id):
        text = build_text(records_by_item_id[item_id])
        rows.append((item_id, text))

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "text"])
        writer.writerows(rows)
    return rows


def print_samples(label: str, rows: Sequence[Tuple[str, str]], seed: int) -> None:
    print(f"\n5 sample {label} texts")
    if not rows:
        print("  No matched texts.")
        return
    rng = random.Random(seed)
    sample = rng.sample(list(rows), k=min(5, len(rows)))
    for item_id, text in sample:
        preview = text.replace("\n", " ").replace("\r", " ")[:300]
        print(f"  item_id: {item_id}")
        print(f"  text: {preview}")


def process_domain(
    label: str,
    interactions_csv: Path,
    meta_path: Path,
    output_path: Path,
) -> Tuple[int, int, float, List[Tuple[str, str]]]:
    item_ids = read_unique_item_ids(interactions_csv)
    print(f"\nProcessing {label}")
    print(f"  Unique {label} items: {len(item_ids):,}")
    print(f"  Metadata file: {meta_path.name}")

    match_counts, matched_records = scan_metadata(meta_path, item_ids)
    for key in MATCH_KEYS:
        coverage = match_counts[key] / max(len(item_ids), 1)
        print(f"  Matches using {key}: {match_counts[key]:,} ({coverage:.2%})")

    best_key = choose_best_key(match_counts)
    print(f"  Selected metadata key: {best_key}")
    rows = write_item_texts(output_path, matched_records[best_key])
    matched = len(rows)
    coverage = matched / max(len(item_ids), 1)
    print(f"  Saved: {output_path.name}")
    print(f"  Matched {label} item texts: {matched:,}")
    print(f"  {label} coverage: {coverage:.2%}")
    return len(item_ids), matched, coverage, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.data_dir.resolve()

    books_unique, books_matched, books_coverage, books_rows = process_domain(
        "Books",
        base_dir / SOURCE_CSV,
        base_dir / BOOKS_META,
        base_dir / BOOKS_OUTPUT,
    )
    electronics_unique, electronics_matched, electronics_coverage, electronics_rows = process_domain(
        "Electronics",
        base_dir / TARGET_CSV,
        base_dir / ELECTRONICS_META,
        base_dir / ELECTRONICS_OUTPUT,
    )

    print("\nSummary")
    print(f"  unique Books items: {books_unique:,}")
    print(f"  matched Books item texts: {books_matched:,}")
    print(f"  Books coverage percentage: {books_coverage:.2%}")
    print(f"  unique Electronics items: {electronics_unique:,}")
    print(f"  matched Electronics item texts: {electronics_matched:,}")
    print(f"  Electronics coverage percentage: {electronics_coverage:.2%}")
    print_samples("Books", books_rows, SEED)
    print_samples("Electronics", electronics_rows, SEED)


if __name__ == "__main__":
    main()
