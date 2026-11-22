"""Hash-bound adjudication preserves held-out rows and source-record accounting."""

import hashlib
import json

import pandas as pd
import pytest

from backend import review


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def save_manifest(decisions, manifest, original_hash):
    manifest.write_text(json.dumps({
        "input_splits_sha256": original_hash,
        "decisions_sha256": digest(decisions.read_bytes()),
        "policy": "Quarantine entire uncertain training groups; preserve all original labels.",
        "reviewer": "Codex review",
    }), encoding="utf-8")


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "SNAPSHOT_ROOT", tmp_path / "snapshots")
    source = tmp_path / "processed"
    source.mkdir()
    rows = [
        ("train", "ham", "sender alpha sent a sticker", "mixed"),
        ("train", "spam", "sender beta sent a sticker", "mixed"),
        ("train", "ham", "coffee meeting tomorrow", "train-ham"),
        ("train", "spam", "claim free reward now", "train-spam"),
        ("dev", "ham", "development normal message", "dev-ham"),
        ("dev", "spam", "development prize offer", "dev-spam"),
        ("test", "ham", "heldout private normal text", "test-ham"),
        ("test", "spam", "heldout private spam text", "test-spam"),
    ]
    frame = pd.DataFrame([
        {"message_id": f"m{i}", "text": text, "label": label, "split": split, "group_id": group}
        for i, (split, label, text, group) in enumerate(rows)
    ])
    (source / "splits.csv").write_bytes(review.csv_bytes(frame))
    split_hash = digest((source / "splits.csv").read_bytes())
    audit = {"input_sha256": "original-source-hash", "input_rows": 12,
             "missing_rows_removed": 1, "conflicting_rows_quarantined": 2,
             "duplicate_rows_removed": 1, "usable_rows": 8,
             "splits_sha256": split_hash, "row_accounting_valid": True,
             "counts": review.partition_counts(frame)}
    (source / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    disposition = pd.DataFrame([
        {"source_record": i + 2, "status": "kept", "message_id": row.message_id,
         "split": row.split, "group_id": row.group_id}
        for i, row in enumerate(frame.itertuples(index=False))
    ] + [
        {"source_record": 10, "status": "missing", "message_id": "", "split": "", "group_id": ""},
        {"source_record": 11, "status": "duplicate", "message_id": "m2", "split": "train", "group_id": "train-ham"},
        {"source_record": 12, "status": "conflict", "message_id": "", "split": "", "group_id": ""},
        {"source_record": 13, "status": "conflict", "message_id": "", "split": "", "group_id": ""},
    ])
    (source / "row_disposition.csv").write_bytes(review.csv_bytes(disposition))
    (source / "groups.csv").write_bytes(review.csv_bytes(review.group_table(frame)))
    for name in review.REMOVAL_FILES:
        (source / name).write_text("source_record,reason\n10,baseline-removal\n", encoding="utf-8")
    (source / "verification.json").write_text('{"baseline": true}', encoding="utf-8")
    for directory in ("profile", "reference_audit"):
        (source / directory).mkdir()
        (source / directory / "run.json").write_text(json.dumps({"split_sha256": split_hash}), encoding="utf-8")
    decisions = tmp_path / "decisions.csv"
    table = pd.DataFrame([
        {"message_id": f"m{i}", "group_id": "mixed", "original_label": label,
         "decision": "quarantine", "reason_code": "missing_consent_context",
         "review_note": "Only sender names differ; consent context is unavailable.",
         "reviewer": "Codex review"}
        for i, label in ((0, "ham"), (1, "spam"))
    ])
    decisions.write_bytes(review.csv_bytes(table))
    manifest = tmp_path / "review.json"
    save_manifest(decisions, manifest, split_hash)
    return source, decisions, manifest, frame, split_hash


def test_review_quarantines_whole_group_preserves_labels_and_heldout_and_snapshots(baseline):
    source, decisions, manifest, before, original_hash = baseline
    original_files = {name: (source / name).read_bytes() for name in review.BASELINE_FILES}
    result = review.apply_review(source, decisions, manifest, source)
    remaining = pd.read_csv(source / "splits.csv", dtype=str, keep_default_na=False, engine="python")
    expected = before.loc[~before.group_id.eq("mixed")].reset_index(drop=True)
    pd.testing.assert_frame_equal(remaining, expected)
    for split in ("dev", "test"):
        pd.testing.assert_frame_equal(
            remaining.loc[remaining.split.eq(split)].reset_index(drop=True),
            before.loc[before.split.eq(split)].reset_index(drop=True))
    assert result["pre_review_usable_rows"] == 8
    assert result["usable_rows"] == 6
    assert result["review_quarantined_rows"] == result["review_quarantined_source_records"] == 2
    assert result["review_quarantined_groups"] == 1
    assert result["mixed_label_groups"] == 0 and result["group_count"] == 6
    assert result["row_accounting_valid"] is True
    assert result["input_rows"] == 1 + 2 + 1 + 2 + 6
    assert result["input_sha256"] == "original-source-hash"
    assert result["counts"]["train"] == {"total": 2, "ham": 1, "spam": 1, "groups": 2}
    assert result["label_review"]["input_splits_sha256"] == original_hash
    assert result["label_review"]["decisions_sha256"] == digest(decisions.read_bytes())
    assert result["splits_sha256"] == digest((source / "splits.csv").read_bytes())
    disposition = pd.read_csv(source / "row_disposition.csv", keep_default_na=False, dtype=str)
    flagged = disposition.loc[disposition.status.eq("review_quarantine")]
    assert flagged.source_record.tolist() == ["2", "3"]
    assert flagged.message_id.tolist() == ["m0", "m1"]
    assert flagged.split.eq("train").all() and flagged.group_id.eq("mixed").all()
    assert disposition.status.value_counts().to_dict() == {
        "kept": 6, "review_quarantine": 2, "conflict": 2, "missing": 1, "duplicate": 1}
    quarantine = pd.read_csv(source / "quarantine_review.csv", keep_default_na=False)
    assert quarantine.text.tolist() == before.text[:2].tolist()
    assert quarantine.label.tolist() == quarantine.original_label.tolist() == ["ham", "spam"]
    assert quarantine.reason_code.eq("missing_consent_context").all()
    assert not (source / "verification.json").exists()
    assert result["derived_reports_require_refresh"] == ["profile", "reference_audit", "verification.json"]
    snapshot = review.SNAPSHOT_ROOT / original_hash
    for name, payload in original_files.items():
        assert (snapshot / name).read_bytes() == payload
    for name in review.REMOVAL_FILES:
        assert (source / name).read_bytes() == original_files[name]
    assert (snapshot / "profile/run.json").exists()
    assert (snapshot / "reference_audit/run.json").exists()
    snapshot_manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
    assert snapshot_manifest["input_splits_sha256"] == original_hash
    assert snapshot_manifest["files"]["splits.csv"] == original_hash


def test_repeat_application_is_idempotent(baseline):
    source, decisions, manifest, _, _ = baseline
    initial = review.apply_review(source, decisions, manifest, source)
    before = {str(path.relative_to(source)): path.read_bytes()
              for path in source.rglob("*") if path.is_file()}
    assert review.apply_review(source, decisions, manifest, source) == initial
    assert before == {str(path.relative_to(source)): path.read_bytes()
                      for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("alteration,match", [
    ("partial", "in full"), ("heldout", "training messages only"),
    ("label", "original_label"), ("group", "group_id"),
    ("unknown", "unknown message"), ("decision", "quarantine decisions"),
    ("reviewer", "reviewers"), ("duplicate_id", "must be unique"),
])
def test_invalid_decisions_fail_before_writing(baseline, alteration, match):
    source, decisions, manifest, _, original_hash = baseline
    table = pd.read_csv(decisions, keep_default_na=False)
    if alteration == "partial":
        table = table.iloc[:1]
    elif alteration == "heldout":
        table = table.iloc[:1].copy()
        table.loc[0, ["message_id", "group_id", "original_label"]] = ["m6", "test-ham", "ham"]
    elif alteration == "label":
        table.loc[0, "original_label"] = "spam"
    elif alteration == "group":
        table.loc[0, "group_id"] = "another-group"
    elif alteration == "unknown":
        table.loc[0, "message_id"] = "unknown"
    elif alteration == "decision":
        table.loc[0, "decision"] = "relabel_spam"
    elif alteration == "reviewer":
        table.loc[0, "reviewer"] = "another reviewer"
    else:
        table.loc[1, "message_id"] = table.loc[0, "message_id"]
    decisions.write_bytes(review.csv_bytes(table))
    save_manifest(decisions, manifest, original_hash)
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    output = source.parent / "out"
    with pytest.raises(ValueError, match=match):
        review.apply_review(source, decisions, manifest, output)
    assert not output.exists() and not review.SNAPSHOT_ROOT.exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


@pytest.mark.parametrize("hash_field", ["input_splits_sha256", "decisions_sha256"])
def test_hash_mismatch_rejected_without_mutating_inputs(baseline, hash_field):
    source, decisions, manifest, _, _ = baseline
    metadata = json.loads(manifest.read_text())
    metadata[hash_field] = "0" * 64
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    before = (source / "splits.csv").read_bytes()
    with pytest.raises(ValueError, match="hash"):
        review.apply_review(source, decisions, manifest, source)
    assert (source / "splits.csv").read_bytes() == before
    assert not review.SNAPSHOT_ROOT.exists()


def test_duplicate_source_reference_in_reviewed_group_is_not_double_counted(baseline):
    source, decisions, manifest, _, _ = baseline
    disposition = pd.read_csv(source / "row_disposition.csv", dtype=str, keep_default_na=False)
    disposition.loc[disposition.status.eq("duplicate"), ["message_id", "group_id"]] = ["m0", "mixed"]
    (source / "row_disposition.csv").write_bytes(review.csv_bytes(disposition))
    with pytest.raises(ValueError, match="duplicate source references"):
        review.apply_review(source, decisions, manifest, source)
    assert not review.SNAPSHOT_ROOT.exists()
    assert not (source / "quarantine_review.csv").exists()


def test_cannot_remove_the_last_training_example_of_a_class(baseline):
    source, decisions, manifest, _, original_hash = baseline
    table = pd.read_csv(decisions, keep_default_na=False)
    extra = table.iloc[[1]].copy()
    extra["message_id"], extra["group_id"] = "m3", "train-spam"
    table = pd.concat([table, extra], ignore_index=True)
    decisions.write_bytes(review.csv_bytes(table))
    save_manifest(decisions, manifest, original_hash)
    with pytest.raises(ValueError, match="both ham and spam"):
        review.apply_review(source, decisions, manifest, source)
    assert not review.SNAPSHOT_ROOT.exists()


def test_different_snapshot_is_not_overwritten(baseline):
    source, decisions, manifest, _, original_hash = baseline
    destination = review.SNAPSHOT_ROOT / original_hash
    destination.mkdir(parents=True)
    (destination / "splits.csv").write_text("different baseline", encoding="utf-8")
    before = (source / "splits.csv").read_bytes()
    with pytest.raises(ValueError, match="different baseline snapshot"):
        review.apply_review(source, decisions, manifest, source)
    assert (source / "splits.csv").read_bytes() == before
    assert (destination / "splits.csv").read_text() == "different baseline"


def test_new_output_directory_leaves_baseline_intact(baseline):
    source, decisions, manifest, _, _ = baseline
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    output = source.parent / "reviewed"
    result = review.apply_review(source, decisions, manifest, output)
    assert result["usable_rows"] == 6
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert (output / "quarantine_review.csv").exists()
    assert not (output / "verification.json").exists()


def test_reprepared_baseline_reuses_snapshot_despite_changed_or_missing_reports(baseline):
    source, decisions, manifest, _, original_hash = baseline
    original_core = {name: (source / name).read_bytes() for name in review.BASELINE_FILES
                     if name != "verification.json"}
    first = review.apply_review(source, decisions, manifest, source)
    snapshot = review.SNAPSHOT_ROOT / original_hash
    frozen_snapshot = {str(path.relative_to(snapshot)): path.read_bytes()
                       for path in snapshot.rglob("*") if path.is_file()}
    # Mimic re-running prepare: restore source-accounted baseline tables, but
    # downstream profile/reference reports still belong to the reviewed run.
    for name, payload in original_core.items():
        (source / name).write_bytes(payload)
    (source / "profile/run.json").write_text('{"post_review_report": true}', encoding="utf-8")
    (source / "reference_audit/run.json").unlink()
    (source / "reference_audit").rmdir()
    assert not (source / "verification.json").exists()
    second = review.apply_review(source, decisions, manifest, source)
    assert second == first
    assert frozen_snapshot == {str(path.relative_to(snapshot)): path.read_bytes()
                               for path in snapshot.rglob("*") if path.is_file()}
    assert (source / "profile/run.json").read_text() == '{"post_review_report": true}'
    assert not (source / "reference_audit").exists()


def test_reused_snapshot_verifies_archived_report_hashes(baseline):
    source, decisions, manifest, _, original_hash = baseline
    original_core = {name: (source / name).read_bytes() for name in review.BASELINE_FILES
                     if name != "verification.json"}
    review.apply_review(source, decisions, manifest, source)
    for name, payload in original_core.items():
        (source / name).write_bytes(payload)
    snapshot = review.SNAPSHOT_ROOT / original_hash
    (snapshot / "profile/run.json").write_text("corrupt archived report", encoding="utf-8")
    original_split = (source / "splits.csv").read_bytes()
    with pytest.raises(ValueError, match="corrupt profile/run.json"):
        review.apply_review(source, decisions, manifest, source)
    assert (source / "splits.csv").read_bytes() == original_split
