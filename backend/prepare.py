"""Audit a labelled CSV, group related messages, then split without group leakage."""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from backend.text import tokenize


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def template_text(text):
    parts = [" entity " if t["protected"] else t["value"] for t in tokenize(canonical_text(text))]
    return " ".join(re.findall(r"[a-z]+", "".join(parts)))


def group_related(texts, threshold=0.85, seed=42):
    """Exact templates + MinHash LSH candidates verified by word-shingle Jaccard.

    This is a documented heuristic, not a guarantee to discover all paraphrases.
    Grouping uses text only; no classifier features or labels are learnt here.
    """
    if not 0 < threshold <= 1:
        raise ValueError("near-threshold must be in (0, 1].")
    parent = list(range(len(texts)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    rng = random.Random(seed)
    prime = 2147483647
    a = np.array([rng.randrange(1, prime) for _ in range(48)], dtype=np.int64)
    b = np.array([rng.randrange(prime) for _ in range(48)], dtype=np.int64)
    templates, buckets, shingles = {}, {}, []
    for i, text in enumerate(texts):
        template = template_text(text)
        words = template.split()
        # Retain empty/non-English strings as separate canonical templates.
        template = template or canonical_text(text)
        grams = {" ".join(words[j:j + 3]) for j in range(len(words) - 2)}
        shingles.append(grams)
        if template in templates:
            union(i, templates[template])
            continue
        templates[template] = i
        if not grams:
            continue
        hashes = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(),
                                         "big") % prime for g in sorted(grams)], dtype=np.int64)
        signature = ((a[:, None] * hashes + b[:, None]) % prime).min(axis=1)
        keys = [(band, tuple(signature[band * 4:band * 4 + 4])) for band in range(12)]
        candidates = set()
        for key in keys:
            candidates.update(buckets.get(key, ()))
        for j in sorted(candidates):
            other = shingles[j]
            if min(len(grams), len(other)) / max(len(grams), len(other)) < threshold:
                continue
            if len(grams & other) / len(grams | other) >= threshold:
                union(i, j)
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return [hashlib.sha256(canonical_text(texts[root(i)]).encode()).hexdigest()[:20]
            for i in range(len(texts))]


def assign_splits(frame, seed=42):
    n_folds = min(20, int(frame.label.value_counts().min()), frame.group_id.nunique())
    if n_folds < 7:
        raise ValueError("Need at least 7 samples per class and 7 independent groups to split.")
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_id = np.empty(len(frame), dtype=int)
    for index, (_, held_out) in enumerate(folds.split(frame.text, frame.label, frame.group_id)):
        fold_id[held_out] = index
    held_out_folds = max(1, round(n_folds * 0.15))
    # Taking the first folds can amplify class imbalance, especially with small
    # or unequal groups. Choose fold combinations using label counts only.
    counts = np.array([[int(((fold_id == i) & frame.label.eq(label)).sum())
                        for label in ("ham", "spam")] for i in range(n_folds)])
    target = np.array([int(frame.label.eq(label).sum()) * 0.15 for label in ("ham", "spam")])

    def choose(available):
        return min(combinations(available, held_out_folds), key=lambda selected: float(
            np.square((counts[list(selected)].sum(axis=0) - target) / np.maximum(target, 1)).sum()
        ))

    test_folds = choose(range(n_folds))
    dev_folds = choose([i for i in range(n_folds) if i not in test_folds])
    frame = frame.copy()
    frame["split"] = np.where(np.isin(fold_id, test_folds), "test",
                              np.where(np.isin(fold_id, dev_folds), "dev", "train"))
    validate_splits(frame)
    return frame


def validate_splits(frame):
    required = {"message_id", "text", "label", "split", "group_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any() or frame.text.str.strip().eq("").any():
        raise ValueError("Split data contains missing/empty values.")
    if frame.message_id.duplicated().any():
        raise ValueError("message_id must be unique.")
    if set(frame.split) != {"train", "dev", "test"}:
        raise ValueError("Expected train, dev and test partitions.")
    if frame.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Related message groups cross partitions.")
    keys = frame.text.map(canonical_text)
    if keys.duplicated().any():
        raise ValueError("Duplicate messages remain in split data.")
    for split, rows in frame.groupby("split"):
        if set(rows.label) != {"ham", "spam"}:
            raise ValueError(f"{split} must contain both ham and spam. Inspect grouping/data size.")


def prepare(input_path, output, text_column="text", label_column="label",
            spam_label="spam", ham_label="ham", seed=42, near_threshold=0.85):
    input_path, output = Path(input_path), Path(output)
    raw = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if text_column not in raw or label_column not in raw:
        raise ValueError(f"Input columns are {list(raw.columns)}; specify text/label columns explicitly.")
    frame = raw[[text_column, label_column]].copy()
    frame.columns = ["text", "label"]
    frame.label = frame.label.str.strip().str.casefold()
    empty = frame.text.str.strip().eq("") | frame.label.eq("")
    frame = frame.loc[~empty].copy()
    label_map = {spam_label.casefold(): "spam", ham_label.casefold(): "ham"}
    if len(label_map) != 2:
        raise ValueError("spam-label and ham-label must differ.")
    unknown = set(frame.label) - set(label_map)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}. Configure --spam-label / --ham-label.")
    frame.label = frame.label.map(label_map)
    frame["key"] = frame.text.map(canonical_text)
    conflicts = frame.groupby("key").label.nunique().gt(1)
    conflicting_keys = set(conflicts[conflicts].index)
    conflicting = frame.key.isin(conflicting_keys)
    conflict_rows = int(conflicting.sum())
    quarantined = frame.loc[conflicting, ["text", "label", "key"]]
    frame = frame.loc[~conflicting].copy()
    before_dedup = len(frame)
    # Sort to make group representatives and splits independent of CSV row order.
    frame = frame.sort_values(["key", "text"]).drop_duplicates("key").reset_index(drop=True)
    duplicate_rows = before_dedup - len(frame)
    if frame.empty:
        raise ValueError("No usable messages after cleaning.")
    frame["message_id"] = frame.key.map(lambda key: hashlib.sha256(key.encode()).hexdigest()[:20])
    frame["group_id"] = group_related(frame.text.tolist(), near_threshold, seed)
    frame = assign_splits(frame.drop(columns="key"), seed)
    frame = frame[["message_id", "text", "label", "split", "group_id"]]
    audit = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_rows": len(raw), "missing_rows_removed": int(empty.sum()),
        "conflicting_rows_quarantined": conflict_rows, "conflicting_keys": len(conflicting_keys),
        "duplicate_rows_removed": duplicate_rows, "usable_rows": len(frame),
        "seed": seed, "near_threshold": near_threshold,
        "split_method": "StratifiedGroupKFold; choose test then dev fold combinations "
                        "minimizing squared relative class-count deviation from 15% targets",
        "grouping": "entity-masked exact templates + 48-signature/12-band MinHash LSH; "
                    "verified 3-word-shingle Jaccard, transitive groups; approximate retrieval",
        "group_count": int(frame.group_id.nunique()),
        "largest_group": int(frame.groupby("group_id").size().max()),
        "counts": {split: {"total": len(rows), "ham": int(rows.label.eq("ham").sum()),
                            "spam": int(rows.label.eq("spam").sum()),
                            "groups": int(rows.group_id.nunique())}
                   for split, rows in frame.groupby("split")},
    }
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "splits.csv", index=False)
    quarantined.to_csv(output / "conflicts.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", required=True, type=Path)
    parser.add_argument("--output", default="data/processed", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--spam-label", default="spam")
    parser.add_argument("--ham-label", default="ham")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
