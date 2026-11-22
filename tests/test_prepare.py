"""Real-source import constraints and exhaustive cleaning provenance."""

import hashlib

import pandas as pd
import pytest

from backend.prepare import prepare, validate_splits


def test_explicit_encoding_and_record_accounting(tmp_path):
    rows = [{"SMSes": prefix + " " + chr(97 + i) * 7, "Labels": label}
            for prefix, label in (("hamword", "0"), ("spamword", "1"))
            for i in range(14)]
    rows += [
        {"SMSes": "Caf\u00e9\x90\x19 tail", "Labels": "0"},
        {"SMSes": " HAMWORD aaaaaaa ", "Labels": "0"},
        {"SMSes": "conflicting message", "Labels": "0"},
        {"SMSes": "CONFLICTING MESSAGE", "Labels": "1"},
        {"SMSes": "", "Labels": "1"},
        {"SMSes": "unlabelled", "Labels": ""},
    ]
    source = tmp_path / "published.csv"
    pd.DataFrame(rows).to_csv(source, index=False, encoding="latin-1")
    kwargs = dict(text_column="SMSes", label_column="Labels", spam_label="1", ham_label="0")
    with pytest.raises(UnicodeDecodeError):
        prepare(source, tmp_path / "invalid", **kwargs)
    out = tmp_path / "processed"
    audit = prepare(source, out, encoding="latin-1", **kwargs)
    assert audit["input_rows"] == 34
    assert audit["usable_rows"] == 29
    assert audit["missing_rows_removed"] == 2
    assert audit["duplicate_rows_removed"] == 1
    assert audit["conflicting_rows_quarantined"] == 2
    assert audit["row_accounting_valid"] is True
    assert audit["encoding"] == "latin-1"
    assert audit["columns"] == {"text": "SMSes", "label": "Labels"}
    assert audit["label_map"] == {"1": "spam", "0": "ham"}
    assert audit["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["splits_sha256"] == hashlib.sha256((out / "splits.csv").read_bytes()).hexdigest()
    splits = pd.read_csv(out / "splits.csv", keep_default_na=False, engine="python")
    validate_splits(splits)
    assert "Caf\u00e9\x90\x19 tail" in set(splits.text)
    assert set(splits.text).issubset({row["SMSes"] for row in rows})
    disposition = pd.read_csv(out / "row_disposition.csv", keep_default_na=False)
    assert disposition.status.value_counts().to_dict() == {
        "kept": 29, "missing": 2, "conflict": 2, "duplicate": 1}
    assert disposition.source_record.tolist() == list(range(2, 36))
    duplicate = disposition.loc[disposition.status.eq("duplicate")].iloc[0]
    kept = disposition.loc[disposition.status.eq("kept") &
                           disposition.message_id.eq(duplicate.message_id)].iloc[0]
    assert (duplicate.split, duplicate.group_id) == (kept.split, kept.group_id)
    missing = pd.read_csv(out / "missing.csv", keep_default_na=False)
    assert set(missing.reason) == {"missing_text", "missing_label"}
    groups = pd.read_csv(out / "groups.csv")
    assert groups.total.sum() == len(splits)
    assert groups.ham.sum() == splits.label.eq("ham").sum()


def test_blank_csv_record_is_audited(tmp_path):
    source = tmp_path / "blanks.csv"
    text = "text,label\n\n" + "".join(
        f"{label}word {chr(97 + i) * 7},{label}\n"
        for label in ("ham", "spam") for i in range(14))
    source.write_text(text, encoding="utf-8")
    audit = prepare(source, tmp_path / "processed")
    assert audit["input_rows"] == 29
    assert audit["missing_rows_removed"] == 1
    assert audit["usable_rows"] == 28
    missing = pd.read_csv(tmp_path / "processed/missing.csv")
    assert missing.source_record.tolist() == [2]
    assert missing.reason.tolist() == ["missing_text_and_label"]
