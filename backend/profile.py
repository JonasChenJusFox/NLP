"""Describe frozen SMS splits without changing messages or fitting model features."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd

from backend.prepare import validate_splits
from backend.text import tokenize


SPLIT_ORDER = ("train", "dev", "test")
LABEL_ORDER = ("ham", "spam")
PRESENCE_FIELDS = (
    "url", "email", "phone", "numeric", "protected", "non_ascii",
    "c0_control", "uncommon_c0_control",
)
PHONE_PATTERN = re.compile(r"\+?\d[\d ()-]{5,}\d")


def describe_message(text):
    """Return descriptive counts and presence indicators for unchanged text."""
    values = {field: False for field in PRESENCE_FIELDS}
    values.update(chars=len(text), words=len(text.split()))
    values["non_ascii"] = not text.isascii()
    values["c0_control"] = any(ord(char) < 32 for char in text)
    values["uncommon_c0_control"] = any(
        ord(char) < 32 and char not in "\t\r\n" for char in text
    )
    # Respect the experiment tokenizer's protected spans. Classify each span
    # once, with its existing URL/email/phone/numeric matching precedence.
    for token in tokenize(text):
        if not token["protected"]:
            continue
        values["protected"] = True
        value = token["value"]
        if value.lower().startswith(("http://", "https://", "www.")):
            values["url"] = True
        elif "@" in value:
            values["email"] = True
        elif PHONE_PATTERN.fullmatch(value):
            values["phone"] = True
        else:
            values["numeric"] = True
    return values


def profile(splits_path="data/processed/splits.csv", output="data/processed/profile"):
    """Write aggregate descriptions and train/dev-only review candidates."""
    splits_path, output = Path(splits_path), Path(output)
    source_bytes = splits_path.read_bytes()
    split_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_text = source_bytes.decode("utf-8")
    # Python 3.9's CSV reader rejects NULs, while pandas' C reader can silently
    # truncate fields at NUL. Reject explicitly rather than alter frozen text.
    if "\x00" in source_text:
        raise ValueError("Split data contains NUL characters; cannot parse losslessly on Python 3.9.")
    frame = pd.read_csv(io.StringIO(source_text), dtype=str, keep_default_na=False,
                        engine="python")
    validate_splits(frame)
    observations = pd.DataFrame([describe_message(text) for text in frame.text])
    group_label_counts = frame.groupby("group_id").label.nunique()
    mixed_groups = set(group_label_counts[group_label_counts.gt(1)].index)
    mixed_label_group = frame.group_id.isin(mixed_groups)
    summaries = []
    reviews = []
    for split in SPLIT_ORDER:
        for label in LABEL_ORDER:
            selected = frame.split.eq(split) & frame.label.eq(label)
            values = observations.loc[selected]
            summary = {"split": split, "label": label, "n": len(values)}
            for measure in ("chars", "words"):
                summary[measure + "_mean"] = float(values[measure].mean())
                summary[measure + "_median"] = float(values[measure].median())
                summary[measure + "_p95"] = float(values[measure].quantile(0.95))
            for field in PRESENCE_FIELDS:
                summary[field + "_fraction"] = float(values[field].mean())
            summaries.append(summary)
            if split == "test":
                continue
            candidates = selected & (
                observations.non_ascii | observations.uncommon_c0_control | mixed_label_group
            )
            for index in frame.index[candidates]:
                row = frame.loc[index]
                flags = [field for field in ("non_ascii", "uncommon_c0_control")
                         if observations.at[index, field]]
                if mixed_label_group.at[index]:
                    flags.append("mixed_label_group")
                reviews.append({"message_id": row.message_id, "split": split,
                                "label": label, "flags": ";".join(flags), "text": row.text})
    summary_frame = pd.DataFrame(summaries)
    review_frame = pd.DataFrame(reviews, columns=("message_id", "split", "label", "flags", "text"))
    metadata = {
        "split_sha256": split_sha256,
        "purpose": "Data-quality audit only; no model feature fitting, selection, or text/label changes.",
        "input_encoding": "utf-8",
        "csv_engine": "python; NUL-containing input explicitly rejected to avoid truncating frozen text",
        "split_order": list(SPLIT_ORDER),
        "label_order": list(LABEL_ORDER),
        "definition": {
            "n": "Number of messages in each split/label group.",
            "chars": "Python len(text): Unicode code points in original text, including whitespace/controls.",
            "words": "Whitespace-delimited tokens from original text.split(); no linguistic tokenization.",
            "p95": "95th percentile with pandas' default linear interpolation.",
            "presence_fractions": "Fraction of messages with at least one matching span or character; denominator n.",
            "protected_spans": "Existing backend.text.tokenize protected spans, classified once in URL/email/phone/numeric order.",
            "url": "Protected span beginning with http://, https://, or www., case-insensitive; bare domains excluded.",
            "email": "Protected non-URL span containing @, as recognized by the shared tokenizer.",
            "phone": "Protected non-URL/email span matching \\+?\\d[\\d ()-]{5,}\\d; includes long plain digit strings.",
            "numeric": "Remaining protected numeric span; phone-like spans counted as phone only.",
            "protected": "At least one protected span of any of the four kinds.",
            "non_ascii": "At least one Unicode code point greater than 127; does not imply invalid language or encoding.",
            "c0_control": "At least one character U+0000 through U+001F, including tab, LF, and CR.",
            "uncommon_c0_control": "C0 character other than tab (U+0009), LF (U+000A), or CR (U+000D).",
            "mixed_label_group": "Related-message group contains both labels; similar content can have different labels. Manual review only, not evidence of label error.",
            "review_candidates": "Train/dev messages with non_ascii, uncommon_c0_control, or mixed_label_group; original text retained for manual review only.",
            "test_policy": "Test text contributes only to aggregate summary; test originals never enter review_candidates.csv.",
        },
        "review_candidates": len(review_frame),
        "mixed_label_group_count": len(mixed_groups),
        "input_rows": len(frame),
    }
    # Avoid writing a report attributed to stale bytes if the input was replaced
    # during reading. The profile never writes to the frozen source itself.
    if splits_path.read_bytes() != source_bytes:
        raise ValueError("Split data changed while profiling; retry with frozen splits.")
    output.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output / "summary.csv", index=False, encoding="utf-8")
    review_frame.to_csv(output / "review_candidates.csv", index=False, encoding="utf-8")
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="data/processed/splits.csv", type=Path)
    parser.add_argument("--output", default="data/processed/profile", type=Path)
    args = parser.parse_args()
    print(profile(args.splits, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
