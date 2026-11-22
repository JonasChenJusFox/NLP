"""Source-format, provenance and unlabelled-reference import regressions."""

import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from backend import spstudy


def workbook_bytes(tactic="Hyrbid"):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("xl/workbook.xml", f'''<workbook xmlns="{spstudy.SS_NS}"
            xmlns:r="{spstudy.REL_NS}"><sheets><sheet name="Cases" sheetId="2"
            r:id="caseSheet"/></sheets></workbook>''')
        archive.writestr("xl/_rels/workbook.xml.rels", f'''<Relationships
            xmlns="{spstudy.PACKAGE_REL_NS}"><Relationship Id="caseSheet"
            Target="worksheets/custom.xml"/></Relationships>''')
        archive.writestr("xl/sharedStrings.xml", f'''<sst xmlns="{spstudy.SS_NS}">
            <si><t>Messages (Orignal)</t></si><si><t>Messages (Modified)</t></si>
            <si><r><t>fr</t></r><r><t>ee_x000D_</t></r></si></sst>''')
        archive.writestr("xl/worksheets/custom.xml", f'''<worksheet xmlns="{spstudy.SS_NS}">
            <sheetData><row r="2"><c r="A2" t="s"><v>0</v></c>
            <c r="B2" t="s"><v>1</v></c><c r="C2" t="inlineStr"><is><t>Tactic</t></is></c></row>
            <row r="7"><c r="A7" t="s"><v>2</v></c><c r="B7" t="inlineStr">
            <is><r><t>f.</t></r><r><t>r.e.e</t></r></is></c><c r="C7" t="inlineStr">
            <is><t>{tactic}</t></is></c></row></sheetData></worksheet>''')
    return stream.getvalue()


def test_numeric_labels_and_missing_rows_are_preserved():
    source = b"SMSes,Labels\r\nfree prize,1\r\nhello friend,0\r\nmissing label,\r\n,\r\n"
    payload, info = spstudy.convert_dataset(source)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    assert rows == [{"text": "free prize", "label": "spam"},
                    {"text": "hello friend", "label": "ham"},
                    {"text": "missing label", "label": ""}, {"text": "", "label": ""}]
    assert info["rows"] == 4
    assert info["missing_text_or_label_rows"] == 2


@pytest.mark.parametrize("label", ["spam", "ham", "2", "1.0"])
def test_unexpected_source_labels_are_rejected(label):
    with pytest.raises(ValueError, match="Unknown source label"):
        spstudy.convert_dataset(f"SMSes,Labels\nhello,{label}\n".encode())


def test_latin1_conversion_preserves_source_characters():
    payload, _ = spstudy.convert_dataset(b"SMSes,Labels\ncan\x92t,0\n")
    assert list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))[0]["text"] == "can\u0092t"


@pytest.mark.parametrize("text", ["free\rnow", "free\nnow", "free\r\nnow", "free\r"])
def test_csv_text_newlines_round_trip_without_splitting_records(text):
    payload = spstudy.csv_bytes([{"text": text, "label": "spam"}], ("text", "label"))
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"), newline="")))
    assert rows == [{"text": text, "label": "spam"}]


def test_excel_values_relationships_rich_text_and_unlabelled_provenance():
    path = "Data/adversarial_examples_evaluation/Synthetic.xlsx"
    rows, inventory = spstudy.reference_examples({path: workbook_bytes()})
    assert len(rows) == inventory["examples"] == 1
    row = rows[0]
    assert row["original_text"] == "free\r"
    assert row["modified_text"] == "f.r.e.e"
    assert row["sheet"] == "Cases" and row["source_row"] == 7
    assert row["tactic_raw"] == "Hyrbid" and row["tactic"] == "hybrid"
    assert row["label"] == "" and row["review_status"] == "unreviewed"
    assert spstudy.COMMIT in row["source_url"]
    assert inventory["labels_available"] is False
    assert spstudy.reference_examples({path: workbook_bytes()})[0] == rows


def test_excel_literal_escape_is_not_decoded_twice():
    assert spstudy.excel_text("a_x000D_b") == "a\rb"
    assert spstudy.excel_text("a_x005F_x000D_b") == "a_x000D_b"
    assert spstudy.excel_text("_xD83D__xDE00_") == "\U0001f600"


def test_unknown_tactic_fails_instead_of_inventing_a_class():
    with pytest.raises(ValueError, match="Unknown tactic"):
        spstudy.reference_examples({"Data/test.xlsx": workbook_bytes("unknown attack")})


@pytest.mark.parametrize("path", ["../escape.py", "/absolute.txt",
                                  f"spstudy-{spstudy.COMMIT}/Data/../../escape.py"])
def test_archive_path_traversal_is_rejected(path):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(path, "bad")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        spstudy.read_snapshot(stream.getvalue())


def test_snapshot_verification_and_import_idempotence(tmp_path, monkeypatch):
    source = b"SMSes,Labels\nhello,0\nmissing,\n,\nprize,1\n"
    monkeypatch.setattr(spstudy, "EXPECTED_BLOBS", {spstudy.CSV_PATH: spstudy.git_blob_sha1(source)})
    monkeypatch.setattr(spstudy, "CSV_SHA256", spstudy.sha256(source))
    archive_path = tmp_path / "release.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"spstudy-{spstudy.COMMIT}/{spstudy.CSV_PATH}", source)
    output, references = tmp_path / "raw/spstudy", tmp_path / "references"
    summary = spstudy.import_release(archive_path, output, references)
    assert summary["dataset_rows"] == 4 and summary["missing_rows_preserved"] == 2
    assert summary["reference_examples"] == 0
    assert spstudy.import_release(archive_path, output, references) == summary
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert manifest["files"][0]["git_blob_sha1"] == spstudy.git_blob_sha1(source)
    dataset = Path(summary["dataset"])
    original = dataset.read_bytes()
    dataset.write_text("human changes", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        spstudy.import_release(archive_path, output, references)
    assert dataset.read_text() == "human changes"
    dataset.write_bytes(original)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"spstudy-{spstudy.COMMIT}/{spstudy.CSV_PATH}", source + b"changed,1\n")
    with pytest.raises(ValueError, match="Git blob verification failed"):
        spstudy.import_release(archive_path, output, references)
