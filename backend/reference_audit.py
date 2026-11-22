"""Audit supplementary spreadsheet cases for duplicates and main-split overlap.

This is a text-only overlap check, not a labelled evaluation or a certificate of
test independence. No classifier is trained and reference labels are not used.
"""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.models import load_splits
from backend.prepare import canonical_text, template_text


MATCH_TYPES = ("original_exact", "original_template", "modified_exact", "modified_template")
PROVENANCE_FIELDS = ("example_id", "source_workbook", "source_row", "tactic")
REQUIRED_FIELDS = set(PROVENANCE_FIELDS) | {"original_text", "modified_text"}
SPLIT_ORDER = ("train", "dev", "test")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_examples(path):
    # newline='' preserves embedded carriage returns in Excel-derived text.
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_FIELDS.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing example columns: {sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(row.get(key) is None for key in REQUIRED_FIELDS):
            raise ValueError("Malformed supplementary example row.")
        if not all(row[key].strip() for key in REQUIRED_FIELDS):
            raise ValueError("Supplementary examples contain empty required values.")
        if row["example_id"] in seen:
            raise ValueError("Supplementary example_id values must be unique.")
        seen.add(row["example_id"])
    return rows


def build_indexes(frame):
    """Store counts by partition, never model features or main-message text."""
    exact, templates = defaultdict(Counter), defaultdict(Counter)
    for row in frame.itertuples(index=False):
        exact[canonical_text(row.text)][row.split] += 1
        template = template_text(row.text)
        # Empty/non-English templates cannot establish useful overlap; avoid
        # collapsing unrelated messages merely because the ASCII template is ''.
        if template:
            templates[template][row.split] += 1
    return exact, templates


def audit_references(splits="data/processed/splits.csv",
                     examples="data/reference/spstudy/adversarial_examples.csv",
                     output="data/processed/reference_audit"):
    splits, examples, output = Path(splits), Path(examples), Path(output)
    if any(destination.resolve() in {splits.resolve(), examples.resolve()}
           for destination in (output / "overlaps.csv", output / "run.json")):
        raise ValueError("Audit output must not overwrite an input file.")
    hashes = {"splits_sha256": file_hash(splits), "examples_sha256": file_hash(examples)}
    frame = load_splits(splits)  # Includes validate_splits: no duplicate/group leakage.
    cases = read_examples(examples)
    exact_index, template_index = build_indexes(frame)
    original_keys = [canonical_text(row["original_text"]) for row in cases]
    pair_keys = [(original, canonical_text(row["modified_text"]))
                 for row, original in zip(cases, original_keys)]
    original_counts, pair_counts = Counter(original_keys), Counter(pair_keys)
    original_workbooks, pair_workbooks = defaultdict(set), defaultdict(set)
    for row, original, pair in zip(cases, original_keys, pair_keys):
        original_workbooks[original].add(row["source_workbook"])
        pair_workbooks[pair].add(row["source_workbook"])
    overlap_rows = []
    for case, original_key, pair_key in zip(cases, original_keys, pair_keys):
        row = {key: case[key] for key in PROVENANCE_FIELDS}
        row.update(original_duplicate_case_count=original_counts[original_key],
                   original_duplicate_workbook_count=len(original_workbooks[original_key]),
                   pair_duplicate_case_count=pair_counts[pair_key],
                   pair_duplicate_workbook_count=len(pair_workbooks[pair_key]))
        for side, text in (("original", case["original_text"]), ("modified", case["modified_text"])):
            for mode, index, key in (("exact", exact_index, canonical_text(text)),
                                     ("template", template_index, template_text(text))):
                counts = index.get(key, {}) if key else {}
                prefix = f"{side}_{mode}"
                row[prefix + "_count"] = sum(counts.values())
                row[prefix + "_splits"] = ";".join(split for split in SPLIT_ORDER if counts.get(split, 0))
        overlap_rows.append(row)
    any_exact = [row["original_exact_count"] > 0 or row["modified_exact_count"] > 0 for row in overlap_rows]
    any_template = [row["original_template_count"] > 0 or row["modified_template_count"] > 0
                    for row in overlap_rows]
    summary = {
        **hashes, "examples": len(cases), "main_messages": len(frame),
        "unique_canonical_originals": len(original_counts), "unique_canonical_pairs": len(pair_counts),
        "duplicate_original_excess_rows": len(cases) - len(original_counts),
        "duplicate_pair_excess_rows": len(cases) - len(pair_counts),
        "cross_workbook_original_groups": sum(len(books) > 1 for books in original_workbooks.values()),
        "cross_workbook_pair_groups": sum(len(books) > 1 for books in pair_workbooks.values()),
        "cross_workbook_original_cases": sum(len(original_workbooks[key]) > 1 for key in original_keys),
        "cross_workbook_pair_cases": sum(len(pair_workbooks[key]) > 1 for key in pair_keys),
        "tactics": dict(sorted(Counter(row["tactic"] for row in cases).items())),
        "cases_with_matches": {mode: sum(row[mode + "_count"] > 0 for row in overlap_rows)
                               for mode in MATCH_TYPES},
        "cases_with_any_exact_match": sum(any_exact),
        "cases_with_any_template_match": sum(any_template),
        "cases_with_template_match_but_no_exact_match": sum(template and not exact
                                                           for exact, template in zip(any_exact, any_template)),
        "cases_without_detected_match": sum(not (exact or template)
                                             for exact, template in zip(any_exact, any_template)),
        "independence_established": False, "labels_used_for_scoring": False,
        "purpose": "Supplementary reference audit only; not an independent labelled test set.",
        "exact_method": "NFKC + casefold + whitespace collapse, identical to prepare.canonical_text.",
        "template_method": "prepare.template_text: mask supported entities, retain ASCII letter words; skip empty templates.",
        "limitations": ["Template equality is a coarse overlap screen and can create false matches.",
                        "No paraphrase or near-duplicate guarantee; an absent match does not establish independence.",
                        "Workbook examples may have multiple edits and changed meaning; labels require human review.",
                        "No original or modified main-dataset text or inferred class labels are exported."],
        "split_separator": ";", "duplicate_counts_include_current_case": True,
    }
    if hashes != {"splits_sha256": file_hash(splits), "examples_sha256": file_hash(examples)}:
        raise ValueError("Audit inputs changed while the report was being generated; rerun on frozen files.")
    fields = list(PROVENANCE_FIELDS) + ["original_duplicate_case_count", "original_duplicate_workbook_count",
                                       "pair_duplicate_case_count", "pair_duplicate_workbook_count"]
    fields += [mode + suffix for mode in MATCH_TYPES for suffix in ("_count", "_splits")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(overlap_rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "overlaps.csv").write_text(stream.getvalue(), encoding="utf-8")
    (output / "run.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                     encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="data/processed/splits.csv", type=Path)
    parser.add_argument("--examples", default="data/reference/spstudy/adversarial_examples.csv", type=Path)
    parser.add_argument("--output", default="data/processed/reference_audit", type=Path)
    print(json.dumps(audit_references(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
