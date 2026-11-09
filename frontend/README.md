# SMS Spam Robustness frontend

The plain HTML/CSS/JavaScript demo uses the Python backend for transformations,
training-vocabulary normalization and all four model predictions. No browser-side
vocabulary or placeholder prediction is used.

Follow the root [README](../README.md) to prepare data and train the models, then run
from the repository root:

```sh
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Serving `frontend/` with a separate static server or
opening `index.html` directly does not provide the required API.

- Three separate edit types; at most one eligible word per message; seed 42.
- URLs, emails, phone numbers and numeric strings are preserved by perturbations.
- Original, modified and normalized text, highlighted changes, normalization log.
- Majority, A, B and C predictions with estimated spam scores.
- C applies its fitted normalizer to both original and modified model inputs.
- Missing/incompatible models return HTTP 503; no predictions are fabricated.
- Input changes cancel pending requests and clear stale results.
- User text and backend records are rendered using `textContent`.

API: `POST /api/analyze` with `{ "text": "claim free prize", "method": "substitution", "seed": 42 }`.
The response includes `original_text`, `modified`, `normalized`, `predictions`,
`seed` and `model_id`. `GET /api/health` reports whether models are ready.

The demo seed acts directly on the entered message. Batch experiments derive a
stable per-message seed from the experiment seed and `message_id`, so changing CSV
row order does not change the chosen word. The transformation rules and one-word
budget are identical in both paths.
