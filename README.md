# SMS Spam Robustness

**NLP Final Project: Spam Detection under Character-Level Text Changes**

Live demonstration: **https://nlpsms.cc**

This project compares four SMS classifiers on original messages and three
controlled one-word edits, with a reproducible Python pipeline and a Cloudflare
demonstration.

## Team

| Member |  Contributions |
| --- | --- |
| [Aarron Chen](https://github.com/kc5358-cpu) | Backend algorithm & Model testing |
| [Ophelia Liang](https://github.com/yl10837) | Backend algorithm & Data evaluation |
| [Jonas Chen](https://github.com/JonasChenJusFox) | Backend algorithm, frontend development, data cleaning, data analysis, algorithm optimization, and deployment |

## Project structure

| Path | Purpose |
| --- | --- |
| `backend/` | Data processing, algorithms, training, evaluation and local API |
| `frontend/` | Local demonstration interface |
| `cloudflare-demo/` | Website and model inference deployed to Cloudflare |
| `data/` | Source data, frozen splits, cleaning audits and review decisions |
| `artifacts/` | Fitted models and model metadata |
| `reports/` | Experiment metrics, review evidence and deployment checks |
| `tests/` | Automated tests with temporary synthetic datasets |

## Models

The original test results use the same 10,040 held-out messages for all models.

| System | Algorithm | Accuracy | Spam F1 | Ham FPR |
| --- | --- | ---: | ---: | ---: |
| Majority | Always predicts the most common training class: ham | 61.01% | 0.00% | 0.00% |
| A | Word 1–2-gram TF-IDF + Logistic Regression | 98.46% | 98.00% | 0.69% |
| B | Character 3–5-gram TF-IDF + Logistic Regression | 99.05% | 98.78% | 0.65% |
| C | Conservative normalization + word TF-IDF + Logistic Regression | 98.47% | 98.02% | 0.69% |

A/B/C use L2-regularized Logistic Regression with `C=2`. Development-selected
spam thresholds are A/C `0.5` and B `0.4`. C uses training-vocabulary evidence
for supported word recovery. A and C are fitted separately, so their difference
does not isolate normalization's causal effect. Settings are in
[`backend/config.json`](backend/config.json).

## Dataset and evaluation

Data come from Salman, Ikram and Kaafar's (2024)
[Super SMS study](https://github.com/smspamresearch/spstudy), pinned to commit
`a0889f36fb59e23a7f853a6932485d93d18b8afc`.
Of 67,010 source records, cleaning removes 2 missing records and 95 duplicates,
and quarantines one uncertain 37-message training group. The main study retains
**66,876 messages**, with retained text and labels unchanged.

| Partition | Ham | Spam | Total | Use |
| --- | ---: | ---: | ---: | --- |
| Training | 28,568 | 18,229 | 46,797 | Fit features and model parameters |
| Development | 6,126 | 3,913 | 10,039 | Select settings, rules and thresholds |
| Test | 6,125 | 3,915 | 10,040 | Evaluate the frozen final system |
| **Total** | **40,819** | **26,057** | **66,876** | All retained messages |

Only training text fits vocabulary, IDF, normalization evidence and model weights.
Development data guide model choices; test data provide the final independent
evaluation. Detected related-message groups stay together, and each edited
variant inherits its original's partition. Cleaning records and review decisions
are retained in `data/processed/` and `data/reviews/`.

Each robustness condition changes at most one eligible word:

| Condition | Example |
| --- | --- |
| Character substitution | `free -> fr€e` |
| Punctuation insertion | `free -> f.r.e.e` |
| Character repetition | `free -> freee` |

URLs, emails and numeric content are protected. `full` metrics include unchanged
inputs; `matched` metrics compare the same actually changed messages before and
after an edit. Deltas mean modified minus original. Results and provenance are
in `reports/{dev,test}/` and `reports/freeze.json`; they apply to these controlled
edits and do not establish universal robustness or statistical significance.

## Run locally

Use Python 3.9+ from the project root. The frozen experiment uses scikit-learn 1.6.1.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Raw/processed datasets, fitted Python models and generated reports are excluded
from Git. On a fresh clone, prepare the pinned dataset, apply the recorded review
and train before starting the local API:

```sh
python -m backend.spstudy
python -m backend.prepare --input data/raw/super_sms.csv
python -m backend.review
python -m backend.train
```

```sh
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000; API docs are at `/docs`. `/api/health` checks model
readiness, and `POST /api/analyze` returns text changes and four-model predictions.
Inputs are limited to 2,000 characters. Spam scores are estimated probabilities,
not per-message accuracy.

## Reproduce and test

Keep new outputs separate from existing frozen models and results:

```sh
mkdir -p /tmp/sms-study-reproduction
python -m backend.train --output /tmp/sms-study-reproduction/models.joblib
python -m backend.evaluate --model /tmp/sms-study-reproduction/models.joblib --split dev --output /tmp/sms-study-reproduction/dev
python -m backend.evaluate --model /tmp/sms-study-reproduction/models.joblib --split test --output /tmp/sms-study-reproduction/test
python -m pytest -q
```

`backend.prepare`, `backend.review`, `backend.profile` and
`backend.reference_audit` provide cleaning and audit tools; use `--help` for
their output options. Recorded review decisions apply only to the recorded
baseline. Evaluation checks the fitted artifact's data and implementation hashes.

## Cloudflare deployment

`cloudflare-demo/` serves the interface and exported-model inference directly in
Cloudflare Workers. It runs independently of the local Python API.

```sh
cd cloudflare-demo
npm run build
npm run validate
npm run test:api
npx --yes wrangler@4.166.0 login
npm run deploy:cloudflare
```

Deployment requires the Cloudflare account that owns `nlpsms.cc`. See
[`cloudflare-demo/README.md`](cloudflare-demo/README.md) for model-parity checks
and deployment details.

## Dataset reference

Salman, Ikram and Kaafar (2024).
[Investigating Evasive Techniques in SMS Spam Filtering](https://doi.org/10.1109/ACCESS.2024.3364671).
