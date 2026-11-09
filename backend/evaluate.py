"""Matched and full-set robustness metrics, coverage, CSV interfaces and dev errors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.models import (file_hash, load_bundle, load_splits, metrics,
                            predictions, spam_scores)
from backend.text import METHODS, message_seed, perturb


MEASURES = ("spam_precision", "spam_recall", "spam_f1", "ham_fpr")


def conditions(rows, seed):
    result = {"clean": {"texts": rows.text.tolist(), "changed": np.zeros(len(rows), dtype=bool),
                        "details": []}}
    for method in METHODS:
        changed = [perturb(row.text, method, message_seed(seed, row.message_id))
                   for row in rows.itertuples()]
        result[method] = {"texts": [entry["text"] for entry in changed],
                          "changed": np.array([entry["changed"] for entry in changed], dtype=bool),
                          "details": changed}
    return result


def metric_row(system, condition, scope, version, labels, predicted, reference=None):
    values = metrics(labels, predicted)
    row = {"system": system, "change_type": condition, "scope": scope,
           "text_version": version, **values}
    for measure in MEASURES:
        before = reference[measure] if reference else values[measure]
        row["delta_" + measure] = (values[measure] - before
                                   if values[measure] is not None and before is not None else None)
    return row


def evaluate(splits_path, model_path, split="dev", output=None):
    if split not in {"dev", "test"}:
        raise ValueError("Evaluate only dev or test.")
    bundle = load_bundle(model_path)
    if file_hash(splits_path) != bundle["split_sha256"]:
        raise ValueError("Split data changed after training; use the frozen splits or retrain.")
    rows = load_splits(splits_path)
    rows = rows.loc[rows.split.eq(split)].reset_index(drop=True)
    labels = rows.label.to_numpy()
    variants = conditions(rows, bundle["config"]["seed"])
    metric_rows, prediction_rows, variant_rows, coverage_rows, error_rows = [], [], [], [], []
    all_predictions = {}
    for system, model in bundle["models"].items():
        threshold = bundle["selected"][system]["threshold"]
        all_predictions[system] = {}
        for condition, data in variants.items():
            scores = spam_scores(model, system, data["texts"])
            predicted = predictions(scores, threshold)
            all_predictions[system][condition] = predicted
            full_clean = metrics(labels, all_predictions[system]["clean"])
            metric_rows.append(metric_row(system, condition, "full", "clean" if condition == "clean"
                                          else "modified", labels, predicted, full_clean))
            mask = data["changed"]
            if condition != "clean":
                paired_clean = metrics(labels[mask], all_predictions[system]["clean"][mask])
                metric_rows.append(metric_row(system, condition, "matched", "clean", labels[mask],
                                              all_predictions[system]["clean"][mask]))
                metric_rows.append(metric_row(system, condition, "matched", "modified", labels[mask],
                                              predicted[mask], paired_clean))
            if split == "dev" and system == "C":
                # Hold C's vectorizer/classifier fixed and bypass only normalization
                # to identify changes directly attributable to this preprocessing.
                features = model.named_steps["tfidf"].transform(data["texts"])
                clf = model.named_steps["classifier"]
                raw_scores = clf.predict_proba(features)[:, list(clf.classes_).index("spam")]
                without_normalization = predictions(raw_scores, threshold)
            for i, row in enumerate(rows.itertuples()):
                prediction_rows.append({"message_id": row.message_id, "split": split,
                                        "label": row.label, "system": system, "change_type": condition,
                                        "prediction": predicted[i], "score": float(scores[i]),
                                        "changed": bool(mask[i]), "threshold": threshold,
                                        "model_id": bundle["model_id"]})
                if split == "dev":
                    harmful = system == "C" and without_normalization[i] == row.label and predicted[i] != row.label
                    if predicted[i] != row.label or harmful:
                        normalized = bundle["models"]["C"].named_steps["normalize"].normalize(data["texts"][i])
                        reasons = sorted({r["reason"] for r in normalized["records"]})
                        error_rows.append({"message_id": row.message_id, "system": system,
                                           "change_type": condition, "label": row.label,
                                           "prediction": predicted[i], "score": float(scores[i]),
                                           "error_type": "false_positive" if row.label == "ham" else "false_negative",
                                           "text": data["texts"][i], "normalized_text": normalized["text"],
                                           "normalization_harmed": bool(harmful),
                                           "C_without_normalization_prediction": without_normalization[i] if system == "C" else "",
                                           "normalization_reasons": ";".join(reasons),
                                           "normalization_records": json.dumps(normalized["records"], ensure_ascii=False)})
    for condition in METHODS:
        data = variants[condition]
        for label in ("ham", "spam"):
            mask = labels == label
            changed_count = int(data["changed"][mask].sum())
            coverage_rows.append({"split": split, "change_type": condition, "label": label,
                                  "n": int(mask.sum()), "changed": changed_count,
                                  "coverage": changed_count / int(mask.sum())})
        for i, row in enumerate(rows.itertuples()):
            detail = data["details"][i]
            edits = [{"original": t["value"], "output": t["output"], "start": t["start"]}
                     for t in detail["tokens"] if t["changed"]]
            variant_rows.append({"message_id": row.message_id, "split": split, "label": row.label,
                                 "change_type": condition, "text": row.text, "modified_text": detail["text"],
                                 "changed": detail["changed"], "eligible_word_count": detail["eligible_word_count"],
                                 "changed_word_count": detail["changed_word_count"],
                                 "edits": json.dumps(edits, ensure_ascii=False)})
    output = Path(output or Path("reports") / split)
    output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(metric_rows)
    report.insert(0, "split", split)
    report.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(output / "modified.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "coverage.csv", index=False)
    if split == "dev":
        error_columns = ["message_id", "system", "change_type", "label", "prediction", "score", "error_type",
                         "text", "normalized_text", "normalization_harmed", "C_without_normalization_prediction",
                         "normalization_reasons", "normalization_records"]
        pd.DataFrame(error_rows, columns=error_columns).to_csv(output / "errors.csv", index=False)
        changed = pd.DataFrame(variant_rows).query("changed == True")
        samples = [group.sample(min(5, len(group)), random_state=bundle["config"]["seed"])
                   for _, group in changed.groupby(["change_type", "label"])]
        review = pd.concat(samples, ignore_index=True) if samples else changed.copy()
        for column in ("readable", "label_preserved", "notes"):
            review[column] = ""
        review.to_csv(output / "manual_review.csv", index=False)
    metadata = {"model_id": bundle["model_id"], "split": split,
                "split_sha256": bundle["split_sha256"],
                "implementation_sha256": bundle["implementation_sha256"],
                "config": bundle["config"], "selected": bundle["selected"],
                "versions": bundle["versions"], "confusion_order": [["tn", "fp"], ["fn", "tp"]],
                "delta_definition": "modified minus clean; matched uses the same actually changed IDs",
                "empty_scope": "n=0: precision/recall/F1/FPR undefined (blank); absent ham: FPR blank"}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="data/processed/splits.csv", type=Path)
    parser.add_argument("--model", default="artifacts/models.joblib", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.splits, args.model, args.split, args.output)
    print(report.loc[report.scope.eq("full"), ["system", "change_type", "spam_f1", "ham_fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()
