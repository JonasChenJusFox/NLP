"""Apply a hash-bound, train-only whole-group label review without resplitting.

Reviewers quarantine uncertain groups rather than inventing labels from a class
majority. Original data, labels and source-record accounting remain recoverable.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from backend.models import load_splits
from backend.prepare import validate_splits


SNAPSHOT_ROOT = Path("artifacts/data_snapshots")
DECISION_FIELDS = {"message_id", "group_id", "original_label", "decision",
                   "reason_code", "review_note", "reviewer"}
BASELINE_FILES = ("splits.csv", "audit.json", "row_disposition.csv", "groups.csv",
                  "missing.csv", "duplicates.csv", "conflicts.csv", "verification.json")
REMOVAL_FILES = ("missing.csv", "duplicates.csv", "conflicts.csv")
DERIVED_REPORTS = ("profile", "reference_audit", "verification.json")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def csv_bytes(frame):
    # CRLF causes CSV quoting to protect embedded LF and isolated CR alike.
    return frame.to_csv(index=False, lineterminator="\r\n").encode("utf-8")


def read_frame(payload):
    return pd.read_csv(io.StringIO(payload.decode("utf-8-sig")), dtype=str,
                       keep_default_na=False, engine="python")


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def snapshot_payloads(input_dir):
    result = {name: (input_dir / name).read_bytes() for name in BASELINE_FILES
              if (input_dir / name).is_file()}
    # Only known report directories are copied; never recursively copy a parent
    # containing the destination snapshot itself.
    for directory in ("profile", "reference_audit"):
        source = input_dir / directory
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    result[str(path.relative_to(input_dir))] = path.read_bytes()
    return result


def reusable_snapshot(snapshot_dir, input_dir, split_hash, split_payload):
    """Validate an existing immutable baseline independently of stale reports."""
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Refusing to reuse a different baseline snapshot: missing snapshot manifest.")
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError("Refusing to reuse a different baseline snapshot: invalid manifest.") from error
    files = metadata.get("files", {})
    if (metadata.get("input_splits_sha256") != split_hash or not isinstance(files, dict)
            or not {"splits.csv", "audit.json", "row_disposition.csv"}.issubset(files)):
        raise ValueError("Refusing to reuse a different baseline snapshot: incomplete manifest.")
    for name, expected_hash in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Refusing to reuse a different baseline snapshot: unsafe manifest path.")
        path = snapshot_dir / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected_hash:
            raise ValueError(f"Refusing to reuse a different baseline snapshot: corrupt {name}.")
    if (snapshot_dir / "splits.csv").read_bytes() != split_payload:
        raise ValueError("Refusing to reuse a different baseline snapshot: split bytes differ.")
    # Re-preparing the same baseline can leave current derived reports missing
    # or describing the post-review split. Those do not redefine the snapshot.
    for name in BASELINE_FILES:
        if name == "verification.json":
            continue
        source = input_dir / name
        if source.is_file() and (name not in files or source.read_bytes() != (snapshot_dir / name).read_bytes()):
            raise ValueError(f"Refusing to reuse a different baseline snapshot: baseline {name} differs.")


def group_table(frame):
    groups = frame.groupby("group_id").agg(total=("message_id", "size"),
                                           split=("split", "first"))
    labels = pd.crosstab(frame.group_id, frame.label).reindex(groups.index, fill_value=0)
    for label in ("ham", "spam"):
        groups[label] = labels[label]
    groups["mixed_labels"] = groups.ham.gt(0) & groups.spam.gt(0)
    return groups[["total", "ham", "spam", "split", "mixed_labels"]].reset_index()


def partition_counts(frame):
    return {split: {"total": len(rows), "ham": int(rows.label.eq("ham").sum()),
                    "spam": int(rows.label.eq("spam").sum()),
                    "groups": int(rows.group_id.nunique())}
            for split, rows in frame.groupby("split")}


def apply_review(input_dir="data/processed", decisions_path="data/reviews/label_review.csv",
                 manifest_path="data/reviews/label_review.json", output="data/processed"):
    input_dir, decisions_path, manifest_path, output = map(
        Path, (input_dir, decisions_path, manifest_path, output))
    split_payload = (input_dir / "splits.csv").read_bytes()
    audit_payload = (input_dir / "audit.json").read_bytes()
    decisions_payload, manifest_payload = decisions_path.read_bytes(), manifest_path.read_bytes()
    split_hash, decision_hash = sha256(split_payload), sha256(decisions_payload)
    manifest = json.loads(manifest_payload.decode("utf-8"))
    audit = json.loads(audit_payload.decode("utf-8"))
    required = {"input_splits_sha256", "decisions_sha256", "policy", "reviewer"}
    if not required.issubset(manifest) or not manifest["policy"] or not manifest["reviewer"]:
        raise ValueError("Review manifest needs input/decision hashes, policy and reviewer.")
    if manifest["decisions_sha256"] != decision_hash:
        raise ValueError("Review decisions hash does not match the manifest.")
    prior = audit.get("label_review", {})
    if prior:
        same_review = (prior.get("decisions_sha256") == decision_hash
                       and prior.get("manifest_sha256") == sha256(manifest_payload)
                       and prior.get("input_splits_sha256") == manifest["input_splits_sha256"]
                       and prior.get("output_splits_sha256") == split_hash
                       and audit.get("splits_sha256") == split_hash)
        if same_review and input_dir.resolve() == output.resolve():
            return audit
        raise ValueError("Input already has a review; use its baseline snapshot for a new review.")
    if manifest["input_splits_sha256"] != split_hash:
        raise ValueError("Input splits hash does not match the reviewed baseline.")
    if audit.get("splits_sha256") != split_hash:
        raise ValueError("Input audit does not describe the current splits file.")
    frame = load_splits(input_dir / "splits.csv")
    decisions = read_frame(decisions_payload)
    if not DECISION_FIELDS.issubset(decisions.columns) or decisions.empty:
        raise ValueError("Review decisions must have all required columns and at least one row.")
    if decisions[list(DECISION_FIELDS)].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Review decision fields must not be empty.")
    if decisions.message_id.duplicated().any():
        raise ValueError("Review message_id values must be unique.")
    if set(decisions.decision) != {"quarantine"}:
        raise ValueError("Only explicit quarantine decisions are supported; labels are never inferred.")
    if not decisions.reviewer.eq(str(manifest["reviewer"])).all():
        raise ValueError("Decision reviewers do not match the manifest reviewer.")
    indexed = frame.set_index("message_id")
    if not set(decisions.message_id).issubset(indexed.index):
        raise ValueError("Review contains unknown message IDs.")
    selected = indexed.loc[decisions.message_id]
    if selected.split.ne("train").any():
        raise ValueError("Review may quarantine training messages only; held-out rows are immutable.")
    if (selected.label.to_numpy() != decisions.original_label.to_numpy()).any():
        raise ValueError("Review original_label does not match the baseline.")
    if (selected.group_id.to_numpy() != decisions.group_id.to_numpy()).any():
        raise ValueError("Review group_id does not match the baseline.")
    for group, rows in decisions.groupby("group_id"):
        if set(rows.message_id) != set(frame.loc[frame.group_id.eq(group), "message_id"]):
            raise ValueError("Every reviewed related group must be quarantined in full.")
    if audit.get("usable_rows") != len(frame):
        raise ValueError("Input audit usable_rows does not match the split data.")
    source_keys = ("input_rows", "missing_rows_removed", "conflicting_rows_quarantined",
                   "duplicate_rows_removed")
    if any(key not in audit for key in source_keys):
        raise ValueError("Input audit is missing source-record accounting counts.")
    if audit["input_rows"] != sum(audit[key] for key in source_keys[1:]) + len(frame):
        raise ValueError("Baseline source-record accounting does not reconcile.")
    disposition_payload = (input_dir / "row_disposition.csv").read_bytes()
    disposition = read_frame(disposition_payload)
    fields = {"source_record", "status", "message_id", "split", "group_id"}
    if not fields.issubset(disposition.columns) or len(disposition) != audit["input_rows"]:
        raise ValueError("Baseline row disposition is missing fields or source records.")
    if disposition.source_record.duplicated().any():
        raise ValueError("Baseline source records must be unique.")
    kept = disposition.loc[disposition.status.eq("kept")]
    if kept.message_id.duplicated().any() or set(kept.message_id) != set(frame.message_id):
        raise ValueError("Baseline kept source records do not match the split messages.")
    provenance = kept.set_index("message_id").loc[frame.message_id]
    if (provenance[["split", "group_id"]].to_numpy() != frame[["split", "group_id"]].to_numpy()).any():
        raise ValueError("Baseline kept source-record provenance does not match splits.")
    ids, groups_selected = set(decisions.message_id), set(decisions.group_id)
    affected = disposition.message_id.isin(ids) | disposition.group_id.isin(groups_selected)
    if disposition.loc[affected, "status"].ne("kept").any():
        # A duplicate reference is already counted as removed. Do not silently
        # double-count it as a newly quarantined usable row.
        raise ValueError("Reviewed groups contain removed/duplicate source references; audit them explicitly first.")
    if int(affected.sum()) != len(decisions):
        raise ValueError("Reviewed groups do not have one kept source record per message.")
    remaining = frame.loc[~frame.message_id.isin(ids)].copy()
    validate_splits(remaining)
    for split in ("dev", "test"):
        if not frame.loc[frame.split.eq(split)].reset_index(drop=True).equals(
                remaining.loc[remaining.split.eq(split)].reset_index(drop=True)):
            raise ValueError("Review changed a held-out partition.")
    quarantine = frame.loc[frame.message_id.isin(ids)].merge(
        decisions.drop(columns="group_id"), on="message_id", how="left", validate="one_to_one")
    disposition.loc[affected, "status"] = "review_quarantine"
    groups = group_table(remaining)
    result = dict(audit)
    split_output = csv_bytes(remaining)
    review_provenance = {"input_splits_sha256": split_hash, "decisions_sha256": decision_hash,
                         "manifest_sha256": sha256(manifest_payload),
                         "output_splits_sha256": sha256(split_output),
                         "policy": manifest["policy"], "reviewer": manifest["reviewer"],
                         "decisions_file": str(decisions_path), "manifest_file": str(manifest_path),
                         "snapshot": str(SNAPSHOT_ROOT / split_hash),
                         "operation": "whole-group training-only quarantine; no relabelling or resplitting"}
    result.update(pre_review_usable_rows=len(frame), usable_rows=len(remaining),
                  review_quarantined_rows=len(quarantine), review_quarantined_groups=len(groups_selected),
                  review_quarantined_source_records=int(affected.sum()),
                  counts=partition_counts(remaining), group_count=len(groups),
                  largest_group=int(groups.total.max()), non_singleton_groups=int(groups.total.gt(1).sum()),
                  mixed_label_groups=int(groups.mixed_labels.sum()), splits_sha256=sha256(split_output),
                  label_review=review_provenance,
                  derived_reports_require_refresh=list(DERIVED_REPORTS))
    result["row_accounting_valid"] = (result["input_rows"] == result["missing_rows_removed"]
        + result["conflicting_rows_quarantined"] + result["duplicate_rows_removed"]
        + result["review_quarantined_rows"] + result["usable_rows"])
    if not result["row_accounting_valid"]:
        raise ValueError("Reviewed source-record accounting does not reconcile.")
    generated = {"splits.csv": split_output, "audit.json": json_bytes(result),
                 "row_disposition.csv": csv_bytes(disposition), "groups.csv": csv_bytes(groups),
                 "quarantine_review.csv": csv_bytes(quarantine)}
    for name in REMOVAL_FILES:
        if (input_dir / name).is_file():
            generated[name] = (input_dir / name).read_bytes()
    snapshot_dir = SNAPSHOT_ROOT / split_hash
    if snapshot_dir.exists():
        reusable_snapshot(snapshot_dir, input_dir, split_hash, split_payload)
        snapshot = {}
    else:
        snapshot = snapshot_payloads(input_dir)
        snapshot["snapshot_manifest.json"] = json_bytes({
            "input_splits_sha256": split_hash,
            "files": {name: sha256(payload) for name, payload in snapshot.items()}})
    if input_dir.resolve() != output.resolve():
        for name, payload in generated.items():
            destination = output / name
            if destination.exists() and (not destination.is_file() or destination.read_bytes() != payload):
                raise ValueError(f"Refusing to overwrite different review output: {destination}")
    # All checks precede snapshot creation or output mutations.
    if ((input_dir / "splits.csv").read_bytes() != split_payload
            or (input_dir / "audit.json").read_bytes() != audit_payload
            or (input_dir / "row_disposition.csv").read_bytes() != disposition_payload
            or decisions_path.read_bytes() != decisions_payload
            or manifest_path.read_bytes() != manifest_payload):
        raise ValueError("Review inputs changed during validation; rerun on frozen inputs.")
    for name, payload in snapshot.items():
        if not (snapshot_dir / name).exists():
            atomic_write(snapshot_dir / name, payload)
    # Write audit last so it never claims a new split before that split exists.
    for name, payload in generated.items():
        if name != "audit.json":
            atomic_write(output / name, payload)
    verification = output / "verification.json"
    if verification.exists():
        verification.unlink()
    atomic_write(output / "audit.json", generated["audit.json"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--decisions", dest="decisions_path", type=Path,
                        default=Path("data/reviews/label_review.csv"))
    parser.add_argument("--manifest", dest="manifest_path", type=Path,
                        default=Path("data/reviews/label_review.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    print(json.dumps(apply_review(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
