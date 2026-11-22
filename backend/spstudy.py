"""Import a pinned Super SMS release and unlabelled adversarial reference cases.

Only data are read: upstream notebooks, scripts, macros and external links are
never executed. The reference workbooks supply tactics, not spam/ham labels.
"""

import argparse
import csv
import hashlib
import io
import json
import posixpath
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as ET


REPOSITORY = "https://github.com/smspamresearch/spstudy"
COMMIT = "a0889f36fb59e23a7f853a6932485d93d18b8afc"
ARCHIVE_URL = f"https://codeload.github.com/smspamresearch/spstudy/zip/{COMMIT}"
SOURCE_ENCODING = "latin-1"
CSV_PATH = "Data/super_sms_dataset.csv"
CSV_SHA256 = "ba825b468b59fb229cc56c15a6c2d14d593a88051d54ea2f402166bf2ab894d3"
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
# Git object IDs from the pinned upstream tree. Verify every imported source,
# including local archives, rather than trusting a filename containing a commit.
EXPECTED_BLOBS = {
    "README.md": "6de1dd48b6df5f4662d67ec1b2ac0fdbeb5ea631",
    "Code/README.md": "ae7e98029953314a9c93b440070b9635c65eec83",
    "Code/ELMo.ipynb": "fff46472c23b6317f8f586ad5d0c299ddafd4084",
    "Code/Ensemble.ipynb": "a03af7f0481d36d1744e82296bf9091cc1737a3e",
    "Code/LSTM_BiLSTM_CNN.ipynb": "5da27ddecadf4b12afe51eae7e131a6a12e2d052",
    "Code/OCR_image_text.ipynb": "edcb8960561893d130544b86e6fbc180d3f4109a",
    "Code/OCSVM_Semantic.ipynb": "8ad37fdba69259ad4b4102a35d5e87b2e8ad035c",
    "Code/OCSVM_TFIDF.ipynb": "cffb853796726f720a80cc359500cb97952acc52",
    "Code/PU_Semantic.ipynb": "5a95608008bd3c794825383801b5a89c26451504",
    "Code/PU_TFIDF.ipynb": "135ef9421458c7b172bef5d9f4cf64f5c65b1dd6",
    "Code/SimpleTransformers_.ipynb": "d9194d1350742edc46e8fa35912498cdf8603174",
    "Code/TCN.ipynb": "9ee7e6beb24c0bdb0df31d9351596a58ea0b10af",
    "Code/TCSVM_Semantic.ipynb": "d93a0ae00ffc8f8ab7c3655960eb5080206bc0aa",
    "Code/TCSVM_TFIDF.ipynb": "11467b00e4af214f3a908a06633acd8d9406030a",
    "Code/fastText.ipynb": "6828a9876f3ec4cb8d748c917a2abb066a978067",
    "Code/feature_analysis.ipynb": "3cae815d633bdbcb42f94d7d3ab9399838949d94",
    "Code/language_filtering_google.py": "5f03a2e62d9aa8fec66d2fd80c4d7f54f49688cc",
    "Code/language_filtering_langdetect.py": "44c8bf8dc645950c5cd4272ae1afdf636c8bd22d",
    "Code/sms_automation.py": "11da7d893fbbb4b2cbda699f88c461fddfc3aef9",
    "Code/textAttack.ipynb": "c6dc5c447a53fec61bd8bb56a2d6bbe5458bdf42",
    "Data/README.md": "f99fa0f07c7932fb953ba176e4b023d6424bffd2",
    CSV_PATH: "0bc76ee5b855e2ee78c0cae9db2106d1482e8bf2",
    "Data/adversarial_examples_evaluation/Key Messages_Examples.xlsx": "8b27358daa3cf7eae7832e0c61c2e0256ba8211a",
    "Data/adversarial_examples_evaluation/OOPSpam_Examples.xlsx": "73a8f5f1ab09a75af1ec4d1680f843ca0b6f9b90",
    "Data/adversarial_examples_evaluation/Plino_Examples.xlsx": "c35ded1b97801f368b3a555f0a0fd6135a6e3040",
    "Data/adversarial_examples_evaluation/SMS Shield_Examples.xlsx": "b103fd1f2a8dec773aea43392eaf7ac70a8e5b49",
    "Data/adversarial_examples_evaluation/TM Check_Examples.xlsx": "3f974e35d327fdb1e1a7ea07f55876477081214f",
    "Data/adversarial_examples_evaluation/VeroSMS_Examples.xlsx": "c8d7b01336114a6660d9e601e3334ab999077542",
    "Data/adversarial_examples_evaluation/Zyla_Examples.xlsx": "d81af3970d531760e5cfd5d91998acbbb0abcab3",
}
SS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
EXAMPLE_FIELDS = ("example_id", "source_workbook", "sheet", "source_row",
                  "original_text", "modified_text", "tactic_raw", "tactic",
                  "label", "review_status", "source_url")
TACTICS = {"spacing": "spacing", "paraphrasing": "paraphrasing",
           "paraphrase": "paraphrasing", "hybrid": "hybrid", "hyrbid": "hybrid",
           "hybird": "hybrid", "charswap": "charswap", "punny": "punny_or_punycode",
           "punycode": "punny_or_punycode"}


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    # A CRLF terminator makes csv.QUOTE_MINIMAL quote fields containing either
    # newline character, including a lone CR decoded from Excel's _x000D_.
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_snapshot(archive_bytes):
    """Validate archive paths and fixed Git blobs before writing any source file."""
    payloads = {}
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        total_size = 0
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Archive symlinks are not accepted: {info.filename}")
            if info.is_dir():
                continue
            total_size += info.file_size
            if total_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Expanded source archive exceeds the size limit.")
            if len(path.parts) < 2 or path.parts[0] != f"spstudy-{COMMIT}":
                raise ValueError("Archive does not have the expected pinned release root.")
            relative = str(PurePosixPath(*path.parts[1:]))
            if relative not in EXPECTED_BLOBS:
                raise ValueError(f"Unexpected file in pinned release: {relative}")
            if relative in payloads:
                raise ValueError(f"Duplicate archive path: {relative}")
            payload = archive.read(info)
            if git_blob_sha1(payload) != EXPECTED_BLOBS[relative]:
                raise ValueError(f"Pinned Git blob verification failed: {relative}")
            payloads[relative] = payload
    missing = set(EXPECTED_BLOBS) - set(payloads)
    if missing:
        raise ValueError(f"Pinned release is incomplete: {sorted(missing)}")
    if sha256(payloads[CSV_PATH]) != CSV_SHA256:
        raise ValueError("Pinned dataset SHA-256 verification failed.")
    return payloads


def convert_dataset(payload):
    """Map the documented numeric labels; preserve missing rows for prepare.py."""
    reader = csv.DictReader(io.StringIO(payload.decode(SOURCE_ENCODING), newline=""))
    if reader.fieldnames != ["SMSes", "Labels"]:
        raise ValueError(f"Unexpected source columns: {reader.fieldnames}")
    rows = []
    for line, source in enumerate(reader, 2):
        if None in source or any(value is None for value in source.values()):
            raise ValueError(f"Malformed source CSV record ending at line {reader.line_num}.")
        label = source["Labels"].strip()
        if label not in {"0", "1", ""}:
            raise ValueError(f"Unknown source label {label!r} at record {line}.")
        rows.append({"text": source["SMSes"], "label": {"0": "ham", "1": "spam", "": ""}[label]})
    return csv_bytes(rows, ("text", "label")), {
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "missing_text_or_label_rows": sum(not row["text"].strip() or not row["label"] for row in rows),
    }


def excel_text(value):
    # One pass is intentional: _x005F_x000D_ represents literal '_x000D_',
    # whereas a direct _x000D_ represents a carriage return in Excel text.
    value = re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), value)
    return value.encode("utf-16", "surrogatepass").decode("utf-16")


def rich_text(element):
    return excel_text("".join(node.text or "" for node in element.iter(f"{{{SS_NS}}}t")))


def read_workbook(payload):
    """Read cell values from worksheet relationships, without evaluating formulas."""
    with ZipFile(io.BytesIO(payload)) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [rich_text(item) for item in root.findall(f"{{{SS_NS}}}si")]
        relations = {}
        root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        for relation in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
            if relation.get("TargetMode") == "External":
                continue
            target = relation.attrib["Target"]
            target = posixpath.normpath(target.lstrip("/") if target.startswith("/")
                                       else posixpath.join("xl", target))
            if not target.startswith("xl/"):
                raise ValueError("Workbook relationship escapes its xl directory.")
            relations[relation.attrib["Id"]] = target
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        result = []
        for sheet in workbook.findall(f"{{{SS_NS}}}sheets/{{{SS_NS}}}sheet"):
            target = relations[sheet.attrib[f"{{{REL_NS}}}id"]]
            root = ET.fromstring(archive.read(target))
            rows = []
            for row in root.findall(f"{{{SS_NS}}}sheetData/{{{SS_NS}}}row"):
                values = {}
                for cell in row.findall(f"{{{SS_NS}}}c"):
                    if cell.find(f"{{{SS_NS}}}f") is not None:
                        raise ValueError("Reference examples must contain literal text, not formulas.")
                    kind = cell.get("t")
                    value = cell.find(f"{{{SS_NS}}}v")
                    if kind == "s":
                        text = shared[int(value.text)]
                    elif kind == "inlineStr":
                        inline = cell.find(f"{{{SS_NS}}}is")
                        text = rich_text(inline) if inline is not None else ""
                    else:
                        text = excel_text(value.text or "") if value is not None else ""
                    if text:
                        column = re.sub(r"\d+$", "", cell.attrib["r"])
                        values[column] = text
                if values:
                    rows.append((int(row.attrib["r"]), values))
            result.append((sheet.attrib["name"], rows))
        return result


def reference_examples(payloads):
    rows, inventory = [], []
    for path, payload in sorted(payloads.items()):
        if not path.endswith(".xlsx"):
            continue
        source_url = f"{REPOSITORY}/blob/{COMMIT}/{path.replace(' ', '%20')}"
        book_rows, sheet_summaries = [], []
        for sheet, sheet_rows in read_workbook(payload):
            header = next((number for number, values in sheet_rows
                           if values.get("A", "").startswith("Messages (Orignal)")
                           and values.get("B") == "Messages (Modified)"), None)
            if header is None:
                raise ValueError(f"Unknown example table header in {path}, {sheet}.")
            examples = []
            for number, values in sheet_rows:
                if number <= header:
                    continue
                if not all(values.get(column, "").strip() for column in ("A", "B", "C")):
                    raise ValueError(f"Incomplete example in {path}, {sheet}, row {number}.")
                raw_tactic = values["C"]
                tactic = TACTICS.get(raw_tactic.strip().casefold())
                if tactic is None:
                    raise ValueError(f"Unknown tactic {raw_tactic!r} in {path}, row {number}.")
                identity = f"{COMMIT}\0{path}\0{sheet}\0{number}"
                example = {"example_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                           "source_workbook": Path(path).name, "sheet": sheet, "source_row": number,
                           "original_text": values["A"], "modified_text": values["B"],
                           "tactic_raw": raw_tactic, "tactic": tactic, "label": "",
                           "review_status": "unreviewed", "source_url": source_url}
                examples.append(example)
            book_rows.extend(examples)
            sheet_summaries.append({"sheet": sheet, "header_row": header, "examples": len(examples),
                                    "tactics": dict(sorted(Counter(row["tactic"] for row in examples).items()))})
        rows.extend(book_rows)
        inventory.append({"file": path, "sha256": sha256(payload), "source_url": source_url,
                          "examples": len(book_rows), "sheets": sheet_summaries})
    return rows, {"commit": COMMIT, "workbooks": inventory, "examples": len(rows),
                  "labels_available": False, "use": "Unreviewed supplementary cases; excluded from main splits.",
                  "punny_or_punycode": "Upstream labels are retained; these names do not establish RFC Punycode encoding.",
                  "tactics": dict(sorted(Counter(row["tactic"] for row in rows).items()))}


def notebook_inventory(payloads):
    result = []
    for path, payload in sorted(payloads.items()):
        if not path.endswith(".ipynb"):
            continue
        notebook = json.loads(payload.decode("utf-8"))
        cells = notebook.get("cells", [])
        result.append({"file": path, "sha256": sha256(payload), "cells": len(cells),
                       "code_cells": sum(cell.get("cell_type") == "code" for cell in cells),
                       "markdown_cells": sum(cell.get("cell_type") == "markdown" for cell in cells),
                       "executed_by_importer": False,
                       "source_url": f"{REPOSITORY}/blob/{COMMIT}/{path}"})
    return {"commit": COMMIT, "notebooks": result, "count": len(result)}


def import_release(archive=None, output="data/raw/spstudy", references="data/reference/spstudy",
                   dataset_output=None):
    if archive is None:
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            archive_bytes = response.read(MAX_ARCHIVE_BYTES + 1)
    else:
        archive_bytes = Path(archive).read_bytes()
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("Source archive exceeds the size limit.")
    payloads = read_snapshot(archive_bytes)
    output, references = Path(output), Path(references)
    dataset_output = Path(dataset_output) if dataset_output else output.parent / "super_sms.csv"
    dataset, dataset_info = convert_dataset(payloads[CSV_PATH])
    examples, inventory = reference_examples(payloads)
    manifest = {"repository": REPOSITORY, "commit": COMMIT, "archive_url": ARCHIVE_URL,
                "archive_sha256": sha256(archive_bytes), "source_encoding": SOURCE_ENCODING,
                "source_csv": CSV_PATH, "source_csv_sha256": CSV_SHA256,
                "source_columns": {"SMSes": "text", "Labels": "label"},
                "label_map": {"0": "ham", "1": "spam", "": ""},
                "missing_policy": "Preserve missing text/labels for prepare.py to audit and remove.",
                "converted_csv_sha256": sha256(dataset), "dataset": dataset_info,
                "files": [{"path": path, "bytes": len(payload), "sha256": sha256(payload),
                           "git_blob_sha1": EXPECTED_BLOBS[path]}
                          for path, payload in sorted(payloads.items())]}
    generated = {output / path: payload for path, payload in payloads.items()}
    generated.update({dataset_output: dataset, output / "source_manifest.json": json_bytes(manifest),
                      references / "adversarial_examples.csv": csv_bytes(examples, EXAMPLE_FIELDS),
                      references / "workbook_inventory.json": json_bytes(inventory),
                      references / "notebook_inventory.json": json_bytes(notebook_inventory(payloads))})
    # Check everything before writing anything. Preserve human changes, including
    # labels or review notes subsequently added to the supplementary table.
    for path, payload in generated.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != payload):
            raise ValueError(f"Refusing to overwrite different existing content: {path}")
    for path, payload in generated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
    return {"commit": COMMIT, "source_directory": str(output), "dataset": str(dataset_output),
            "dataset_rows": dataset_info["rows"], "missing_rows_preserved": dataset_info["missing_text_or_label_rows"],
            "reference_examples": len(examples), "references": str(references),
            "manifest": str(output / "source_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local pinned-release ZIP instead of downloading.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/spstudy"))
    parser.add_argument("--references", type=Path, default=Path("data/reference/spstudy"))
    parser.add_argument("--dataset-output", type=Path)
    print(json.dumps(import_release(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
