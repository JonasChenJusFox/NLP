import hashlib
import json

import pandas as pd
import pytest

from backend.profile import profile


def write_splits(tmp_path, additions=()):
    cases = [
        ("train", "ham", "a"),
        ("train", "ham", "three words here"),
        ("train", "spam", "claim your reward"),
        ("dev", "ham", "ordinary meeting reminder"),
        ("dev", "spam", "another special offer"),
        ("test", "ham", "heldout legitimate message"),
        ("test", "spam", "heldout unwanted offer"),
        *additions,
    ]
    frame = pd.DataFrame([
        {"message_id": f"message-{index}", "group_id": f"group-{index}",
         "split": split, "label": label, "text": text}
        for index, (split, label, text) in enumerate(cases)
    ])
    path = tmp_path / "splits.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    return path, frame


def test_profile_length_statistics_order_and_provenance(tmp_path):
    path, original = write_splits(tmp_path)
    before = path.read_bytes()
    output = tmp_path / "profile"
    summary = profile(path, output)
    ham = summary.query("split == 'train' and label == 'ham'").iloc[0]
    assert ham.n == 2
    assert ham.chars_mean == ham.chars_median == 8.5
    assert ham.chars_p95 == pytest.approx(15.25)
    assert ham.words_mean == ham.words_median == 2
    assert ham.words_p95 == pytest.approx(2.9)
    assert list(zip(summary.split, summary.label)) == [
        (split, label) for split in ("train", "dev", "test") for label in ("ham", "spam")
    ]
    assert summary.n.sum() == len(original)
    assert path.read_bytes() == before
    run = json.loads((output / "run.json").read_text())
    assert run["split_sha256"] == hashlib.sha256(before).hexdigest()
    assert run["input_rows"] == len(original)
    assert run["review_candidates"] == 0
    assert "no model feature fitting" in run["purpose"]
    pd.testing.assert_frame_equal(pd.read_csv(output / "summary.csv"), summary)
    assert pd.read_csv(output / "review_candidates.csv").empty


def test_profile_separates_protected_content_and_uses_presence_fractions(tmp_path):
    path, _ = write_splits(tmp_path, [
        ("dev", "spam", "visit https://example.org/a@b www.example.net now"),
        ("dev", "spam", "contact person@example.org now"),
        ("dev", "spam", "call +1 212-555-0199 please"),
        ("dev", "spam", "reference 12345 and amount 12.50"),
    ])
    summary = profile(path, tmp_path / "profile")
    spam = summary.query("split == 'dev' and label == 'spam'").iloc[0]
    assert spam.n == 5
    assert spam.url_fraction == spam.email_fraction == spam.phone_fraction == spam.numeric_fraction == 0.2
    assert spam.protected_fraction == 0.8
    assert spam.non_ascii_fraction == spam.c0_control_fraction == 0


def test_profile_flags_quality_candidates_but_excludes_test_originals(tmp_path):
    path, original = write_splits(tmp_path, [
        ("train", "spam", "free café reward"),
        ("dev", "ham", "has \x19 unusual control"),
        ("dev", "ham", "plain\tline\nwrap\rhere"),
        ("test", "ham", "heldout café \x02 do not review"),
    ])
    output = tmp_path / "profile"
    summary = profile(path, output)
    review = pd.read_csv(output / "review_candidates.csv", keep_default_na=False,
                         engine="python")
    assert review.text.tolist() == ["free café reward", "has \x19 unusual control"]
    assert review["flags"].tolist() == ["non_ascii", "uncommon_c0_control"]
    assert set(review.split) == {"train", "dev"}
    assert not set(review.message_id).intersection(original.loc[original.split.eq("test"), "message_id"])
    assert "heldout café" not in (output / "review_candidates.csv").read_text()
    dev_ham = summary.query("split == 'dev' and label == 'ham'").iloc[0]
    assert dev_ham.c0_control_fraction == pytest.approx(2 / 3)
    assert dev_ham.uncommon_c0_control_fraction == pytest.approx(1 / 3)
    test_ham = summary.query("split == 'test' and label == 'ham'").iloc[0]
    assert test_ham.non_ascii_fraction == test_ham.uncommon_c0_control_fraction == 0.5


def test_profile_marks_mixed_label_groups_for_review_without_changing_labels(tmp_path):
    path, frame = write_splits(tmp_path, [
        ("train", "spam", "similar topic but another label"),
    ])
    frame.loc[frame.index[-1], "group_id"] = frame.loc[0, "group_id"]
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    output = tmp_path / "profile"
    profile(path, output)
    review = pd.read_csv(output / "review_candidates.csv")
    assert review.text.tolist() == ["a", "similar topic but another label"]
    assert review.label.tolist() == ["ham", "spam"]
    assert review["flags"].eq("mixed_label_group").all()
    assert review.split.eq("train").all()
    assert path.read_bytes() == before
    run = json.loads((output / "run.json").read_text())
    assert run["mixed_label_group_count"] == 1
    assert "not evidence of label error" in run["definition"]["mixed_label_group"]


def test_profile_rejects_invalid_frozen_split_before_writing(tmp_path):
    path, frame = write_splits(tmp_path)
    frame.loc[frame.split.eq("test"), "group_id"] = frame.loc[0, "group_id"]
    frame.to_csv(path, index=False)
    output = tmp_path / "profile"
    with pytest.raises(ValueError, match="cross partitions"):
        profile(path, output)
    assert not output.exists()


def test_profile_rejects_nul_before_writing_or_changing_input(tmp_path):
    path, _ = write_splits(tmp_path, [("train", "ham", "before\x00after")])
    original = path.read_bytes()
    output = tmp_path / "profile"
    with pytest.raises(ValueError, match="NUL"):
        profile(path, output)
    assert not output.exists()
    assert path.read_bytes() == original
