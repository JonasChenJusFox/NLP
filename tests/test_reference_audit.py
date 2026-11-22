"""Check overlap reports without exposing held-out message text or using labels."""

import csv
import hashlib
import json

import pandas as pd
import pytest

from backend.reference_audit import audit_references


@pytest.fixture
def inputs(tmp_path):
    rows = []
    texts = {"train": ("meet me tomorrow 123456789", "claim free prize today"),
             "dev": ("coffee by the station", "urgent free reward offer"),
             "test": ("private heldout message content", "private prize scam content")}
    for split, pair in texts.items():
        for label, text in zip(("ham", "spam"), pair):
            identity = f"{split}-{label}"
            rows.append({"message_id": identity, "text": text, "label": label,
                         "split": split, "group_id": identity})
    splits = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(splits, index=False)
    examples = tmp_path / "examples.csv"
    cases = [
        {"example_id": "one", "source_workbook": "A.xlsx", "source_row": "3", "tactic": "spacing",
         "original_text": "  CLAIM  FREE prize TODAY ", "modified_text": "claim f r e e prize today"},
        {"example_id": "two", "source_workbook": "B.xlsx", "source_row": "4", "tactic": "spacing",
         "original_text": "claim free prize today", "modified_text": "CLAIM f r e e PRIZE TODAY"},
        {"example_id": "three", "source_workbook": "A.xlsx", "source_row": "5", "tactic": "charswap",
         "original_text": "meet me tomorrow 987654321", "modified_text": "meet me tomorrow 777777777"},
        {"example_id": "four", "source_workbook": "C.xlsx", "source_row": "7", "tactic": "paraphrasing",
         "original_text": "arrange a different lunch meeting", "modified_text": "organize another meal together"},
        {"example_id": "five", "source_workbook": "C.xlsx", "source_row": "8", "tactic": "hybrid",
         "original_text": "unrelated example", "modified_text": "PRIVATE HELDOUT MESSAGE CONTENT"},
    ]
    with examples.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]) + ["label", "review_status"])
        writer.writeheader()
        writer.writerows({**case, "label": "", "review_status": "unreviewed"} for case in cases)
    return splits, examples, tmp_path / "audit"


def test_duplicate_pairs_and_canonical_template_overlap(inputs):
    splits, examples, output = inputs
    summary = audit_references(splits, examples, output)
    rows = pd.read_csv(output / "overlaps.csv", keep_default_na=False)
    assert summary["examples"] == 5
    assert summary["unique_canonical_originals"] == 4
    assert summary["unique_canonical_pairs"] == 4
    assert summary["cross_workbook_original_groups"] == 1
    assert summary["cross_workbook_pair_groups"] == 1
    assert summary["cross_workbook_pair_cases"] == 2
    assert rows.loc[0, "pair_duplicate_case_count"] == 2
    assert rows.loc[0, "pair_duplicate_workbook_count"] == 2
    assert rows.loc[0, "original_exact_count"] == 1
    assert rows.loc[0, "original_exact_splits"] == "train"
    assert rows.loc[2, "original_exact_count"] == 0
    assert rows.loc[2, "original_template_count"] == 1
    assert rows.loc[2, "original_template_splits"] == "train"
    assert rows.loc[4, "modified_exact_count"] == 1
    assert rows.loc[4, "modified_exact_splits"] == "test"
    assert summary["cases_without_detected_match"] == 1
    assert summary["independence_established"] is False
    assert summary["labels_used_for_scoring"] is False
    assert "absent match does not establish independence" in " ".join(summary["limitations"])


def test_reports_hashes_keep_inputs_and_do_not_export_main_text_or_labels(inputs):
    splits, examples, output = inputs
    before = {path: path.read_bytes() for path in (splits, examples)}
    summary = audit_references(splits, examples, output)
    assert summary["splits_sha256"] == hashlib.sha256(before[splits]).hexdigest()
    assert summary["examples_sha256"] == hashlib.sha256(before[examples]).hexdigest()
    assert all(path.read_bytes() == payload for path, payload in before.items())
    serialized = (output / "overlaps.csv").read_text() + (output / "run.json").read_text()
    assert "private heldout message content" not in serialized.casefold()
    assert "claim free prize today" not in serialized.casefold()
    columns = pd.read_csv(output / "overlaps.csv").columns
    assert "label" not in columns and "prediction" not in columns
    assert json.loads((output / "run.json").read_text()) == summary


def test_empty_ascii_templates_do_not_match_unrelated_non_ascii_text(inputs):
    splits, examples, output = inputs
    main = pd.read_csv(splits, keep_default_na=False)
    main.loc[0, "text"] = "你好"
    main.to_csv(splits, index=False)
    cases = pd.read_csv(examples, keep_default_na=False)
    cases.loc[0, "original_text"] = "こんにちは"
    cases.to_csv(examples, index=False)
    audit_references(splits, examples, output)
    rows = pd.read_csv(output / "overlaps.csv", keep_default_na=False)
    assert rows.loc[0, "original_template_count"] == 0


def test_input_output_collision_is_rejected(inputs):
    splits, examples, output = inputs
    colliding = examples.parent / "overlaps.csv"
    colliding.write_bytes(examples.read_bytes())
    with pytest.raises(ValueError, match="must not overwrite an input"):
        audit_references(splits, colliding, colliding.parent)


def test_group_leakage_is_still_rejected(inputs):
    splits, examples, output = inputs
    main = pd.read_csv(splits, keep_default_na=False)
    main.loc[0, "group_id"] = main.loc[4, "group_id"]
    main.to_csv(splits, index=False)
    with pytest.raises(ValueError, match="cross partitions"):
        audit_references(splits, examples, output)
