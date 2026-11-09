"""Model construction, frozen artifacts and shared scoring."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from backend.text import ConservativeNormalizer


DEFAULT_CONFIG = Path(__file__).with_name("config.json")
SYSTEMS = ("majority", "A", "B", "C")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hash():
    # Freeze all experiment logic as well as fitted parameters.
    digest = hashlib.sha256()
    for name in ("text.py", "models.py", "prepare.py", "train.py", "evaluate.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["max_changed_words"] != 1:
        raise ValueError("The main experiment requires max_changed_words = 1.")
    if not config["lr_c_values"] or any(c <= 0 for c in config["lr_c_values"]):
        raise ValueError("lr_c_values must contain positive values.")
    if not config["decision_thresholds"] or any(not 0 < t < 1 for t in config["decision_thresholds"]):
        raise ValueError("decision_thresholds must be between 0 and 1.")
    return config


def build_model(system, config, c=1.0):
    if system == "majority":
        return DummyClassifier(strategy="most_frequent")
    steps = []
    if system == "C":
        steps.append(("normalize", ConservativeNormalizer(config["normalizer_min_df"],
                                                         config["protect_content"])))
    if system == "B":
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=tuple(config["char_ngram_range"]),
                                    max_features=config["char_max_features"],
                                    min_df=config["tfidf_min_df"], sublinear_tf=True)
    elif system in {"A", "C"}:
        vectorizer = TfidfVectorizer(ngram_range=tuple(config["word_ngram_range"]),
                                    max_features=config["word_max_features"],
                                    min_df=config["tfidf_min_df"], sublinear_tf=True)
    else:
        raise ValueError(f"Unknown system: {system}")
    steps += [("tfidf", vectorizer), ("classifier", LogisticRegression(
        C=c, max_iter=config["max_iter"], solver="liblinear", random_state=config["seed"]
    ))]
    return Pipeline(steps)


def model_input(system, texts):
    return np.zeros((len(texts), 1)) if system == "majority" else texts


def spam_scores(model, system, texts):
    if len(texts) == 0:
        return np.array([], dtype=float)
    index = list(model.classes_).index("spam")
    return model.predict_proba(model_input(system, texts))[:, index]


def predictions(scores, threshold):
    return np.where(np.asarray(scores) >= threshold, "spam", "ham")


def metrics(labels, predicted):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=["ham", "spam"]).ravel()
    if len(labels):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted, labels=["spam"], zero_division=0
        )
    else:
        precision = recall = f1 = [None]
    return {"n": len(labels), "n_ham": int(tn + fp), "n_spam": int(tp + fn),
            "spam_precision": float(precision[0]) if precision[0] is not None else None,
            "spam_recall": float(recall[0]) if recall[0] is not None else None,
            "spam_f1": float(f1[0]) if f1[0] is not None else None,
            "ham_fpr": float(fp / (tn + fp)) if tn + fp else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def load_splits(path):
    from backend.prepare import validate_splits
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_splits(frame)
    return frame


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    manifest = {key: value for key, value in bundle.items() if key != "models"}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_bundle(path):
    # joblib is a pickle format. This path must point to our own training output.
    bundle = joblib.load(path)
    if bundle.get("format_version") != 1 or tuple(bundle["models"]) != SYSTEMS:
        raise ValueError("Unsupported model artifact; retrain with this backend.")
    if bundle["implementation_sha256"] != implementation_hash():
        raise ValueError("Experiment code changed after training; retrain before use.")
    if bundle["versions"]["scikit_learn"] != sklearn.__version__:
        raise ValueError("scikit-learn version differs from training; use the recorded version or retrain.")
    return bundle


def versions():
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}
