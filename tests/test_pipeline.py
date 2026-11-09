import json
import random
import string

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.evaluate import evaluate
from backend.models import DEFAULT_CONFIG, load_bundle, metrics
from backend.prepare import group_related, prepare, validate_splits
from backend.train import train


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """Synthetic integration fixture only; never a research result."""
    root = tmp_path_factory.mktemp("sms-experiment")
    rng = random.Random(19)
    rows = []
    for label in ("ham", "spam"):
        for _ in range(50):
            # Distinct content avoids accidentally making all fixture messages
            # one near-duplicate template; common terms support meaningful TF-IDF.
            unique = ["".join(rng.choices(string.ascii_lowercase, k=7)) for _ in range(5)]
            prefix = "claim free cash prize" if label == "spam" else "coffee meeting hello friend"
            rows.append({"text": prefix + " " + " ".join(unique), "label": label})
    rows += [rows[0].copy(), {"text": "", "label": "spam"},
             {"text": "conflicting record", "label": "ham"},
             {"text": "conflicting record", "label": "spam"}]
    raw = root / "raw.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    audit = prepare(raw, root / "processed")
    splits = root / "processed/splits.csv"
    model = root / "models.joblib"
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(lr_c_values=[1.0], decision_thresholds=[0.5])
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    bundle = train(splits, config_path, model)
    return root, splits, model, bundle, audit


def test_audit_and_grouped_split(experiment):
    _, splits, _, _, audit = experiment
    assert audit["usable_rows"] == 100
    assert audit["duplicate_rows_removed"] == 1
    assert audit["missing_rows_removed"] == 1
    assert audit["conflicting_rows_quarantined"] == 2
    frame = pd.read_csv(splits)
    validate_splits(frame)
    assert frame.groupby("group_id").split.nunique().max() == 1
    for split in ("dev", "test"):
        counts = frame.loc[frame.split.eq(split)].label.value_counts()
        assert abs(counts["ham"] - 7.5) <= 1
        assert abs(counts["spam"] - 7.5) <= 1


def test_related_templates_and_near_duplicates_are_grouped():
    base = "claim your free prize today and call our helpful team now to collect a special reward"
    texts = [base + " 123456789", base + " 987654321", base + " 123456789 please",
             "coffee tomorrow at the station with friends"]
    groups = group_related(texts)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_train_only_vocabulary_and_real_artifact(experiment):
    _, splits, model, bundle, _ = experiment
    frame = pd.read_csv(splits)
    train_text = " ".join(frame.loc[frame.split.eq("train"), "text"])
    dev_test_text = " ".join(frame.loc[~frame.split.eq("train"), "text"])
    held_out_only = set(dev_test_text.split()) - set(train_text.split())
    norm = bundle["models"]["C"].named_steps["normalize"]
    assert held_out_only
    assert not held_out_only.intersection(norm.vocabulary_)
    assert load_bundle(model)["model_id"] == bundle["model_id"]
    assert set(bundle["selected"]) == {"majority", "A", "B", "C"}


def test_full_and_matched_reports_and_no_test_tuning(experiment):
    root, splits, model, bundle, _ = experiment
    selected = json.dumps(bundle["selected"], sort_keys=True)
    report = evaluate(splits, model, "dev", root / "dev")
    assert len(report) == 40  # 4 systems * (4 full + 3 matched clean/modified pairs).
    assert report.query("scope == 'matched'").n.gt(0).all()
    assert report.query("scope == 'matched' and text_version == 'clean'").delta_spam_f1.eq(0).all()
    for _, group in report.query("scope == 'matched'").groupby(["system", "change_type"]):
        assert group.n.nunique() == 1
        assert group.n_ham.nunique() == 1
    coverage = pd.read_csv(root / "dev/coverage.csv")
    assert len(coverage) == 6
    assert coverage.coverage.eq(1).all()
    modified = pd.read_csv(root / "dev/modified.csv")
    assert modified.changed_word_count.le(1).all()
    assert (root / "dev/errors.csv").exists()
    assert (root / "dev/manual_review.csv").exists()
    # Synthetic held-out smoke test verifies the path, not project performance.
    evaluate(splits, model, "test", root / "test")
    assert json.dumps(load_bundle(model)["selected"], sort_keys=True) == selected


def test_group_leakage_is_rejected(experiment):
    _, splits, _, _, _ = experiment
    frame = pd.read_csv(splits)
    frame.loc[frame.split.eq("test"), "group_id"] = frame.loc[frame.split.eq("train"), "group_id"].iloc[0]
    with pytest.raises(ValueError, match="cross partitions"):
        validate_splits(frame)


def test_metrics_empty_and_false_positives():
    values = metrics(["ham", "ham", "spam"], ["spam", "ham", "spam"])
    assert values["ham_fpr"] == 0.5
    assert (values["tn"], values["fp"], values["fn"], values["tp"]) == (1, 1, 0, 1)
    assert metrics([], [])["spam_f1"] is None
    assert metrics(["spam"], ["spam"])["ham_fpr"] is None


def test_api_and_served_frontend(experiment):
    _, _, model, bundle, _ = experiment
    with TestClient(create_app(model)) as client:
        assert client.get("/api/health").json()["models_ready"]
        assert 'id="C-modified"' in client.get("/").text
        response = client.post("/api/analyze", json={"text": "claim free prize", "method": "substitution"})
        assert response.status_code == 200
        result = response.json()
        assert result["modified"]["changed_word_count"] == 1
        assert set(result["predictions"]) == {"majority", "A", "B", "C"}
        assert result["normalized"]["text"] == bundle["models"]["C"].named_steps["normalize"].transform(
            [result["modified"]["text"]])[0]
        for system, values in result["predictions"].items():
            assert 0 <= values["modified"]["score"] <= 1
            assert values["modified"]["label"] in {"ham", "spam"}
        for invalid in ({"text": " "}, {"text": "a" * 2001}, {"text": "free", "method": "unknown"},
                        {"text": "free", "strength": 1}):
            assert client.post("/api/analyze", json=invalid).status_code == 422


def test_missing_models_do_not_fabricate_predictions(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert not client.get("/api/health").json()["models_ready"]
        assert client.post("/api/analyze", json={"text": "free"}).status_code == 503


def test_changed_split_file_is_rejected(experiment):
    root, splits, model, _, _ = experiment
    altered = root / "altered.csv"
    altered.write_bytes(splits.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after training"):
        evaluate(altered, model, output=root / "altered-report")


def test_reports_when_no_message_can_be_changed(experiment, tmp_path):
    _, splits, _, _, _ = experiment
    frame = pd.read_csv(splits)
    dev_indices = frame.index[frame.split.eq("dev")]
    for i, index in enumerate(dev_indices):
        frame.loc[index, "text"] = f"{string.ascii_lowercase[i // 26]}{string.ascii_lowercase[i % 26]} zz"
    no_edits = tmp_path / "no-edits.csv"
    frame.to_csv(no_edits, index=False)
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(lr_c_values=[1.0], decision_thresholds=[0.5])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    model_path = tmp_path / "models.joblib"
    train(no_edits, config_path, model_path)
    report = evaluate(no_edits, model_path, output=tmp_path / "report")
    matched = report.query("scope == 'matched'")
    assert matched.n.eq(0).all()
    assert matched.spam_f1.isna().all()
    assert matched.ham_fpr.isna().all()
    assert pd.read_csv(tmp_path / "report/coverage.csv").coverage.eq(0).all()
    assert pd.read_csv(tmp_path / "report/manual_review.csv").empty
