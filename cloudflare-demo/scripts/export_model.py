"""Export this project's trusted fitted TF-IDF/Logistic Regression parameters.

The JSON is data, not a pickle, and does not contain raw training messages.
Feature vocabularies and document-frequency evidence are fitted public-corpus
statistics. No training or parameter selection is performed by this exporter.
"""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded_floats(values):
    raw = np.asarray(values, dtype="<f8").tobytes()
    return base64.b64encode(raw).decode("ascii"), sha256(raw)


def export(project_root, output, fixtures):
    project_root, output, fixtures = map(Path, (project_root, output, fixtures))
    sys.path.insert(0, str(project_root))
    from backend.models import load_bundle

    source = project_root / "artifacts/models.joblib"
    bundle = load_bundle(source)
    fitted_normalizer = bundle["models"]["C"].named_steps["normalize"]
    majority = bundle["models"]["majority"]
    classes = list(majority.classes_)
    spam_index = classes.index("spam")
    data = {
        "format_version": 1,
        "model_id": bundle["model_id"],
        "implementation_sha256": bundle["implementation_sha256"],
        "split_sha256": bundle["split_sha256"],
        "config": bundle["config"],
        "selected": bundle["selected"],
        "versions": bundle["versions"],
        "majority": {"strategy": majority.strategy,
                     "spam_score": float(majority.predict_proba(np.zeros((1, 1)))[0, spam_index])},
        "normalizer": {"vocabulary": dict(sorted(fitted_normalizer.vocabulary_.items())),
                       "min_df": fitted_normalizer.min_df,
                       "protect_content": fitted_normalizer.protect_content},
        "models": {},
    }
    parameter_hashes = {}
    for system in ("A", "B", "C"):
        pipeline = bundle["models"][system]
        vectorizer, classifier = pipeline.named_steps["tfidf"], pipeline.named_steps["classifier"]
        expected = {"lowercase": True, "strip_accents": None, "stop_words": None,
                    "binary": False, "sublinear_tf": True, "norm": "l2", "use_idf": True}
        parameters = vectorizer.get_params()
        for key, value in expected.items():
            if parameters[key] != value:
                raise ValueError(f"Unsupported {system} vectorizer option {key}: {parameters[key]!r}")
        if parameters["analyzer"] not in {"word", "char"}:
            raise ValueError("Only word/character analyzers can be exported.")
        if parameters["analyzer"] == "word" and parameters["token_pattern"] != r"(?u)\b\w\w+\b":
            raise ValueError("Unexpected word tokenizer; export the corresponding implementation first.")
        if list(classifier.classes_) != ["ham", "spam"] or classifier.coef_.shape[0] != 1:
            raise ValueError("Expected a binary ham/spam Logistic Regression in fixed class order.")
        features = vectorizer.get_feature_names_out().tolist()
        assert len(features) == len(vectorizer.vocabulary_) == len(vectorizer.idf_)
        assert all(vectorizer.vocabulary_[feature] == index for index, feature in enumerate(features))
        idf, idf_hash = encoded_floats(vectorizer.idf_)
        coefficients, coefficient_hash = encoded_floats(classifier.coef_[0])
        data["models"][system] = {
            "feature_count": len(features), "features": features,
            "analyzer": parameters["analyzer"], "ngram_range": list(parameters["ngram_range"]),
            "idf_b64": idf, "coef_b64": coefficients,
            "intercept": float(classifier.intercept_[0]),
        }
        parameter_hashes[system] = {"idf_float64_le_sha256": idf_hash,
                                    "coefficient_float64_le_sha256": coefficient_hash,
                                    "feature_count": len(features)}
    payload = (json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)

    # Use training rows plus deliberately varied synthetic inputs, never tune
    # exported parameters using fixture labels or resulting model scores.
    synthetic = [
        "", " ", "unknownwordzzzz", "free free free prize", "f.r.e.e fr€e freee",
        "1st 2nd 3rd 4th 101st class", "Your BOOK is HERE", "coffee tomorrow at 3pm",
        "https://example.com/free hello@example.org +1 212-555-0199", "___ __proto__ constructor",
        "free\nprize", "free\tprize", "free\r\nprize", "free\x1cprize", "free\x1c\x1fprize",
        "free\x85prize", "free\x85\u00a0prize", "free\u2003prize", "free\u2003\u2009prize",
        "𝒇𝒓𝒆𝒆 prize 😀 café", "İI ıſ KKK", "ΟΣ ΟΣΑ Σσς", "mañana über café", "你好 免费 prize",
        "Ⅻ ² Ⅳ1 ₁2", "___free___ _a a_", "a a a a", "free!prize?hello", "Ააბგ ქართული",
        "😀😀fre€eee😀", "foo\ufeff\ufeffbar", "a\u2028b\u2029c", "free\v\fprize",
        "free\u1c89prize", "free\U00010d40prize",
    ]
    frame = pd.read_csv(project_root / "data/processed/splits.csv", dtype=str,
                        keep_default_na=False, engine="python")
    train = frame.loc[frame.split.eq("train"), "text"]
    texts = synthetic + train.sample(100 - len(synthetic), random_state=109).tolist()
    checks = []
    for system in ("majority", "A", "B", "C"):
        pipeline = bundle["models"][system]
        inputs = texts if system != "C" else fitted_normalizer.transform(texts)
        if system == "majority":
            expected_scores = majority.predict_proba(np.zeros((len(texts), 1)))[:, spam_index]
        else:
            features = pipeline.named_steps["tfidf"].transform(inputs)
            expected_scores = pipeline.named_steps["classifier"].predict_proba(features)[:, 1]
        for index, (text, input_text, score) in enumerate(zip(texts, inputs, expected_scores)):
            checks.append({"case_id": index, "system": system, "original_text": text,
                           "model_input": input_text, "score": float(score),
                           "prediction": "spam" if score >= bundle["selected"][system]["threshold"] else "ham"})
    fixture_payload = {"model_id": bundle["model_id"], "cases": 100, "rows": checks,
                       "scope": "Training and synthetic inputs only; C inputs are already normalized for pure classifier math parity."}
    fixtures.parent.mkdir(parents=True, exist_ok=True)
    fixtures.write_text(json.dumps(fixture_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"model_id": bundle["model_id"], "source_joblib_sha256": sha256(source.read_bytes()),
                "export_sha256": sha256(payload), "bytes": len(payload), "parameters": parameter_hashes,
                "normalizer_vocabulary_words": len(data["normalizer"]["vocabulary"]),
                "fixture_rows": len(checks), "fixtures_sha256": sha256(fixtures.read_bytes()),
                "float_format": "IEEE754 Float64 little-endian, base64", "new_training": False}
    output.with_name("model-export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "edge/model-data.json")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "edge/model-fixtures.json")
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
