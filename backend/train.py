"""Fit on clean train only; choose C and thresholds on clean/modified dev only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.evaluate import conditions
from backend.models import (DEFAULT_CONFIG, SYSTEMS, build_model, file_hash,
                            implementation_hash, load_config, load_splits, metrics,
                            model_input, predictions, save_bundle, spam_scores, versions)


def train(splits_path, config_path=DEFAULT_CONFIG, output="artifacts/models.joblib"):
    config = load_config(config_path)
    frame = load_splits(splits_path)
    train_rows = frame.loc[frame.split.eq("train")]
    dev = frame.loc[frame.split.eq("dev")].reset_index(drop=True)
    dev_conditions = conditions(dev, config["seed"])
    train_texts, train_labels = train_rows.text.tolist(), train_rows.label.tolist()
    labels = dev.label.to_numpy()
    models, settings, search = {}, {}, []
    for system in SYSTEMS:
        best_key = None
        for c in ([None] if system == "majority" else config["lr_c_values"]):
            model = build_model(system, config, c)
            model.fit(model_input(system, train_texts), train_labels)
            scores = {name: spam_scores(model, system, data["texts"])
                      for name, data in dev_conditions.items()}
            for threshold in ([0.5] if system == "majority" else config["decision_thresholds"]):
                results = [metrics(labels, predictions(value, threshold)) for value in scores.values()]
                mean_f1 = float(np.mean([r["spam_f1"] for r in results]))
                mean_fpr = float(np.mean([r["ham_fpr"] for r in results]))
                candidate = {"system": system, "C": c, "threshold": threshold,
                             "dev_mean_spam_f1": mean_f1, "dev_mean_ham_fpr": mean_fpr}
                search.append(candidate)
                # F1 first; tie-break on fewer ham errors, then proximity to 0.5.
                key = (mean_f1, -mean_fpr, -abs(threshold - 0.5))
                if best_key is None or key > best_key:
                    best_key = key
                    models[system] = model
                    settings[system] = candidate
        print(f"Selected {system}: {settings[system]}")
    provenance = {"config": config, "split_sha256": file_hash(splits_path),
                  "implementation_sha256": implementation_hash(), "versions": versions(),
                  "selected": settings}
    model_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    bundle = {"format_version": 1, "model_id": model_id, **provenance,
              "selection": "mean spam F1 over clean and three full dev conditions; "
                           "ties: lower mean ham FPR, then threshold closest to 0.5, then grid order",
              "training_rows": len(train_rows), "development_rows": len(dev),
              "search": search, "models": models}
    save_bundle(bundle, output)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="data/processed/splits.csv", type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--output", default="artifacts/models.joblib", type=Path)
    args = parser.parse_args()
    bundle = train(args.splits, args.config, args.output)
    print(f"Saved frozen model {bundle['model_id']} to {args.output}")


if __name__ == "__main__":
    main()
