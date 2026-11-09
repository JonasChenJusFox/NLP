# SMS Spam Robustness

**NLP Final Project: Spam Detection under Character-Level Text Changes**

Can conservative text normalization recover modified SMS spam without increasing
false alarms on legitimate messages? We compare word and character TF-IDF with
the same Logistic Regression classifier. Examples such as `fr€e`, `f.r.e.e` and
`freee` motivate the study; improved performance is a hypothesis to test.

This README follows **SMS_Spam_Robustness_Proposal(2).pdf**. The repository now
contains data preparation, four systems, robustness evaluation and an API-backed
demo. The main dataset and trained artifacts are **not bundled**. Automated tests
use synthetic fixtures; their scores are not experimental results. Dataset audit,
manual review, final experiments, plots and the paper remain to be completed.

## Dataset and leakage prevention

The main study uses the [Super SMS Dataset](https://github.com/smspamresearch/spstudy)
from Salman, Ikram and Kaafar (2024). The proposal reports **67,018 messages**
(26,181 spam / 40,837 ham) before our cleaning. Actual usable counts must come from
`data/processed/audit.json`, rather than assuming these reported counts survive.
The smaller [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
is suitable for early pipeline checks. Because Super SMS combines earlier
collections, UCI is not an independent external test set until overlap is checked.

Preparation in `backend/prepare.py`:

1. Read a labelled CSV; remove empty text/labels; reject unrecognized labels.
2. Compare NFKC-normalized, case-folded, whitespace-collapsed text for duplicates.
   Quarantine **all** copies of conflicting labels in `conflicts.csv`; retain one
   copy of other duplicates. Preserve the retained original text for modelling.
3. Group entity-masked exact templates and near duplicates before splitting.
   Near-duplicate candidates use 48-signature, 12-band MinHash LSH, verified with
   3-word-shingle Jaccard similarity >= 0.85. Connected messages stay in one group.
   This text-only heuristic is approximate: inspect groups and possible missed
   overlap before freezing the study; it does not guarantee paraphrase detection.
4. Use `StratifiedGroupKFold`: normally 20 folds, assigning 14/3/3 folds to
   train/dev/test for approximately **70%/15%/15%**. Test/dev fold combinations
   minimize class-count deviation from the 15% targets, using labels only to
   stratify the split. Smaller datasets use fewer folds.
   Group integrity takes priority over exact proportions; report final
   class counts and group sizes from the audit. Splits lacking either class fail.
5. Generate modified messages **after splitting**. Every variant inherits its
   original `message_id` and partition. Only training text fits vocabulary, IDF,
   normalizer evidence and classifier parameters. Dev selects settings and rules;
   test is reserved for the frozen final evaluation.

Entity masking is used for grouping only. Model A/B inputs remain original text.
The [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
explains why fitting preprocessing inside training pipelines matters.

## Backend algorithm

| System | Pipeline | Purpose |
| --- | --- | --- |
| Majority | Most frequent training label | Minimum baseline |
| A: Word | Word unigram/bigram TF-IDF + Logistic Regression | Standard baseline |
| B: Character | Character 3–5 gram TF-IDF + Logistic Regression | Robust representation baseline |
| C: Normalized word | Conservative normalizer + word unigram/bigram TF-IDF + Logistic Regression | Proposed system |

The shared experiment settings are in [`backend/config.json`](backend/config.json):
seed 42, TF-IDF minimum document frequency 2, at most 30,000 word or 60,000
character features, sublinear TF scaling, and Logistic Regression with L2
regularization, `liblinear`, and at most 1,000 iterations. All main models train
on **clean training messages**, without augmentation.

The training script searches the same `C = {0.5, 1, 2}` and decision thresholds
`{0.4, 0.5, 0.6}` for A/B/C. It maximizes mean spam F1 over clean and three
modified **full dev** conditions. Ties prefer lower mean ham false-positive rate,
then a threshold closer to 0.5, then grid order. This is an initial, explicit
selection policy; review the F1/FPR trade-off on dev before accepting final rules.
It does not impose a maximum acceptable ham FPR. Majority has no tuning.
Test scores never select parameters.

### Conservative normalization

`backend/text.py` is shared by training, evaluation and the demo:

1. Build word document-frequency evidence from **both ham and spam training text**.
   The vocabulary is frozen after fitting and never learns from later input.
2. Tokenize protected spans first: URLs with `http(s)://` or `www.`, email
   addresses, phone numbers and numeric strings. Preserve them exactly.
3. Preserve known words and conservatively treat any uppercase/title-case/mixed-case
   token, apostrophes, multiple digits or unsupported digits as possible names,
   abbreviations or identifiers. This may also preserve obfuscated uppercase spam;
   measure that cost during error analysis.
4. For an unknown eligible token, apply NFKC and the explicit map
   `4→a`, `€→e`, `1→i`, `0→o`, `$→s`; remove dots only from recognized
   single-letter chains such as `f.r.e.e`; generate candidates by deleting at most
   one adjacent repeated letter. Unicode normalization alone is insufficient.
5. Replace only when **exactly one** candidate appears in at least two training
   documents. Multiple plausible candidates or no evidence leave the token unchanged.

For a suitable training vocabulary, `fr€e → free`, `f.r.e.e → free` and
`freee → free`; `free`, phone numbers and ambiguous abbreviations remain unchanged.
These are rule examples, not measured findings. Model C embeds this fitted
normalizer in its pipeline, so the **same preprocessing runs at training and
prediction time**, on both clean and modified input.

### Controlled modifications

| Condition | Rule | Example |
| --- | --- | --- |
| Substitution | Replace one supported character, preferring `e`, then `a/i/o/s` | `free → fr€e` |
| Punctuation | Put dots between a word's letters | `free → f.r.e.e` |
| Repetition | Append one copy of the last letter | `free → freee` |

Each condition selects **at most one eligible word per message**. Eligible words
contain at least three English letters; substitution additionally needs a mapped
character. All conditions preserve protected spans and use identical,
label-independent rules and budgets for spam and ham. Batch seeds derive from
SHA-256 of `(seed, message_id)`, making selection independent of row ordering.
No eligible word means `changed=false`, which is included in coverage statistics.
Review sampled dev variants for readability and label preservation; generation
does not automatically establish either property.

## Run locally

Use Python **3.9+**. Run commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
mkdir -p data/raw
```

Obtain and inspect Super SMS from its authors' repository, then provide a UTF-8
CSV at `data/raw/super_sms.csv` with columns `text,label`. The two expected label
values are `spam` and `ham`. An illustrative CSV row is:

```csv
text,label
claim your free prize,spam
```

One row is not sufficient to train: the split requires enough independent groups
and both classes. For other source columns/labels, specify their mapping explicitly:

```sh
python -m backend.prepare --input data/raw/super_sms.csv
# Example for a source using Message and Class, with 1=spam and 0=ham:
# python -m backend.prepare --input data/raw/source.csv --text-column Message --label-column Class --spam-label 1 --ham-label 0

python -m backend.train
python -m backend.evaluate --split dev
```

Inspect `audit.json`, dev metrics, errors and `manual_review.csv`. Revise rules or
config using train/dev only, then prepare/train again as needed. The saved model
contains selected thresholds, config, data/code hashes and package versions.
After rules, settings and data are frozen, run the final held-out evaluation:

```sh
python -m backend.evaluate --split test
```

Evaluation rejects a changed split file or changed experiment code after training.
Use the recorded scikit-learn version when loading artifacts. Use only trusted
artifacts produced by this project: `joblib` uses pickle serialization.

Start the integrated demo:

```sh
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>; API docs are at <http://127.0.0.1:8000/docs>.
`MODEL_PATH` can select another trusted artifact; restart the API after retraining.
Without a compatible trained model, analysis returns HTTP 503. Predictions are
real model outputs once training is complete; no scores are hard-coded.

```sh
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"claim free prize", "method":"substitution", "seed":42}'
python -m pytest -q
```

The demo uses the existing HTML/JavaScript frontend with a FastAPI backend.
Its one-word budget matches the study. The API uses the supplied seed directly;
batch experiments additionally derive the seed from `message_id`.
Streamlit remains optional in the proposal and is not needed to run this demo.

## Evaluation and collaboration files

Report spam precision/recall/F1, **ham FPR = FP / (FP + TN)**, confusion matrices
`[[TN, FP], [FN, TP]]`, and clean-to-modified deltas. Deltas always mean
**modified minus clean**, so an F1 decrease is negative and an FPR increase positive.

Primary robustness comparisons use **matched** original/modified messages that
were actually changed. `metrics.csv` also contains **full-set** results, including
unchanged messages. Matched clean and modified rows share the same IDs; the clean
reference is specific to each change type. Coverage is reported by class and edit
type. Empty matched groups have `n=0` and blank metrics; FPR is blank without ham.

| Output | Contents |
| --- | --- |
| `data/processed/splits.csv` | `message_id,text,label,split,group_id` |
| `data/processed/audit.json` | Cleaning counts, input hash, group summary, split/class counts |
| `data/processed/conflicts.csv` | Conflicting labelled rows for review |
| `artifacts/models.joblib` / `models.json` | Four fitted models / readable frozen manifest and dev search |
| `reports/{dev,test}/modified.csv` | `message_id,change_type,modified_text,changed`, plus labels, original text and edit records |
| `reports/{dev,test}/predictions.csv` | `message_id,system,change_type,prediction,score`, plus split, label, threshold and model version |
| `reports/{dev,test}/metrics.csv` | Full/matched metrics, confusion counts and deltas |
| `reports/{dev,test}/coverage.csv` | Changed-message fraction by class and edit type |
| `reports/{dev,test}/run.json` | Frozen experiment provenance and report conventions |
| `reports/dev/errors.csv` | False positives/negatives, normalization reasons and harmful corrections |
| `reports/dev/manual_review.csv` | Up to five changed examples per class/edit, with blank human-review fields |

For harmful normalization cases, hold C's fitted vectorizer/classifier fixed and
compare its prediction with and without normalization at inference. This isolates
the preprocessing effect more directly than comparing A to C. The errors file
includes the before-normalization prediction, corrected text, ambiguity and
protected-content records. Manual reviewers should categorize errors, check labels
and document rule changes **before** the final test run.

## Database choice: SQL

**Recommendation: use relational storage if persistence becomes necessary.**
For the current coursework pipeline, CSV + JSON + `joblib` already provide the
required shared interfaces and reproducible experiments; no database server is
required or connected. SMS text does not by itself require MongoDB.

| Stage / option | Recommendation |
| --- | --- |
| Offline experiments | Keep the existing CSV/JSON interfaces and model files |
| Local demo history | SQLite if saving history is required; no separate server |
| Shared deployed backend | PostgreSQL for concurrent access and experiment queries |
| MongoDB | Consider when heterogeneous nested documents and document-oriented access become the main workload |

The main entities have stable relationships: messages → variants → predictions,
and experiment runs → model versions → metrics. SQL foreign keys, uniqueness
constraints and joins suit comparisons across systems, edits and runs. Suggested
future tables are `messages`, `variants`, `experiment_runs`, `model_versions`,
`predictions` and `metrics`. Keep fitted models in files/object storage and store
their path, version and hash in the database. Training must read frozen split
snapshots, rather than automatically ingesting demo history.

Variable normalization logs and experiment settings can live in PostgreSQL
[`jsonb` columns](https://www.postgresql.org/docs/current/datatype-json.html),
alongside typed relational fields. [SQLite's usage guidance](https://www.sqlite.org/whentouse.html)
supports local application storage and explains when a client/server database is
more suitable. [MongoDB's schema design guidance](https://www.mongodb.com/docs/manual/data-modeling/schema-design-process/)
describes its flexible document model; that flexibility is not currently a
requirement of this project. Database migration/history storage is a future option,
not an implemented dependency of the algorithms.

## Team and deliverables

| Team member | Lead responsibility | Independent contribution |
| --- | --- | --- |
| [Jonas Chen](https://github.com/JonasChenJusFox) | Evaluation and frontend | Evaluation scripts/tables/plots, error-analysis format, optional demo |
| [Aarron Chen](https://github.com/kc5358-cpu) | Data and backend | Audit/splits, modifications, normalization, TF-IDF and classifiers |
| [Ophelia Liang](https://github.com/yl10837) | Literature and paper | Related work/methods, manual checks, results/discussion/references |

All members review experiments and co-author the final paper. The CSV schemas
above allow independent work and integration through GitHub pull requests.
Deliverables: reproducible code, audited splits and transformation rules, result
tables/figures, dev error analysis, the paper and an eight-minute presentation.

The minimum study is the four systems, clean evaluation and three separate edits.
Combinations, training augmentation and protection-rule ablation are extensions.
For a protection ablation, copy the config with `protect_content=false`, train a
separate artifact and evaluate on the **same splits and variants**; do not replace
the main protected-model artifact. No augmentation or combined-edit study is
claimed by the current implementation.

## Related work

1. Han & Baldwin (2011), [Lexical Normalisation of Short Text Messages](https://aclanthology.org/P11-1038/).
2. Eger et al. (2019), [Text Processing Like Humans Do](https://aclanthology.org/N19-1165/).
3. Pruthi, Dhingra & Lipton (2019), [Combating Adversarial Misspellings with Robust Word Recognition](https://aclanthology.org/P19-1561/).
4. Bitton, Pavlova & Evtimov (2022), [Adversarial Text Normalization](https://aclanthology.org/2022.naacl-industry.30/).
5. Wang et al. (2021), [TextFlint](https://aclanthology.org/2021.acl-demo.41/).
6. Salman, Ikram & Kaafar (2024), [Investigating Evasive Techniques in SMS Spam Filtering](https://doi.org/10.1109/ACCESS.2024.3364671).
