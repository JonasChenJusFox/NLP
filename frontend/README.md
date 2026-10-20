# SMS Spam Robustness frontend

Open `index.html` in a browser. No dependencies are required.

Alternatively, run from the repository root:

```sh
python3 -m http.server 8000 --directory frontend
```

Visit http://localhost:8000.

## Features

- Spam, ham, and protected-content examples; custom message input.
- Character substitution, punctuation insertion, and character repetition.
- Modification of 25%, 50%, or 100% of eligible words, rounded up, with seed 42.
- URL, email, and phone number protection.
- Original, modified, and normalized text with highlighted changes.
- Normalization log using a small built-in vocabulary.

## Backend integration

Models are not connected. Prediction cells remain unavailable until actual models are loaded.

Replace the browser implementations of `perturb()` and `normalize()` with backend processing and a training-set vocabulary. Update `run()` to fetch text results, modification records, and predictions for models A, B, and C. Model C must apply the same normalization used during training to both original and modified input.

Input changes clear previous results. User text is rendered with `textContent`.
