# SMS Spam Robustness — Cloudflare Deployment

Live demo: https://nlpsms.cc

This directory deploys the SMS demonstration directly to Cloudflare Workers
through Wrangler. The page and prediction API share one origin and are publicly
accessible without ChatGPT authentication. Predictions run entirely in the
Worker; no Python server, external prediction service, database or runtime
secret is required. The application processes submitted messages in memory.

## Frozen Models

`edge/model-data.json` contains parameters exported from the parent project's
fitted `artifacts/models.joblib`, model ID `9439f39e3b9e7f45`. The export preserves
the existing parameters and does not retrain or select thresholds.

| System | Representation | Classifier | Spam threshold |
| --- | --- | --- | ---: |
| Majority | Majority-class baseline | Always predicts ham | 0.5 |
| A | Word TF-IDF, 1–2 grams | Logistic Regression | 0.5 |
| B | Character TF-IDF, 3–5 grams | Logistic Regression | 0.4 |
| C | Conservative normalization, then word TF-IDF | Logistic Regression | 0.5 |

`edge/model.mjs` implements TF-IDF and classification from the exported weights.
`edge/text.mjs` preserves protected content, the one-word edit budget, seeded
changes and normalization logs. `edge/unicode.mjs` pins the experiment's Python
Unicode 13 behavior. Normalization evidence comes only from training text.
`edge/model-export-manifest.json` records source and exported-parameter hashes.

## Build and Validate

Run these commands from this directory using Node.js with built-in
`fetch`, `Request` and `Response` support. The build needs no npm dependencies.

```sh
node scripts/build.mjs
node scripts/validate-artifact.mjs
node --test tests/worker.test.mjs
node edge/test-model.mjs
node tests/text-parity.mjs
```

The build produces the self-contained Worker at `dist/server/index.js` and
`dist/build-manifest.json`, which records model and Worker hashes. Fixed weights
initialize during Worker startup. Test fixtures and Python packages are not
included in the deployed Worker. `dist/` and `.wrangler/` are disposable local
build/cache directories; run the build before artifact or API checks.

The retained regression fixtures cover 60 synthetic API responses, 400 model
predictions and 3,736 text comparisons, including Unicode and protected content.
`tests/text-fixtures.json` is intentionally larger because it preserves the
reference cases required by the text-parity test. These Node checks run without
Python or downloaded experiment reports.

The `scripts/export_model.py`, `export-unicode.py`, `export-api-fixtures.py` and
`export-text-fixtures.py` scripts regenerate model parameters or references when
the parent project's Python dependencies and frozen artifacts are available.
They are development tools, not runtime dependencies.

## Routes

- `/`: input, three edit methods, normalization logs and before/after predictions.
- `POST /api/analyze`: accepts a non-empty message of at most 2,000 Unicode
  characters, an edit method and an unsigned 32-bit integer seed; uses the same
  response schema as the Python API.
- `GET /api/health`: readiness and the frozen model ID.

Spam scores estimate the spam class probability. They are not per-message
accuracy; evaluation metrics require labelled data.

## Deploy

Authenticate to your own Cloudflare account, then deploy:

```sh
npx --yes wrangler@4.166.0 login
npm run deploy:cloudflare
```

`wrangler.jsonc` builds the Worker, binds `nlpsms.cc` and retains the existing
`workers.dev` alias. Credentials stay in Wrangler's authentication store outside
this project. Existing valid authentication can be reused.

To check the published page, assets, health endpoint, reference predictions and
input boundaries without sending authentication headers or cookies:

```sh
node tests/online-smoke.mjs https://nlpsms.cc
```
