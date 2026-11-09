"""Serve the existing demo and real, frozen model predictions from one origin."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from backend.models import load_bundle, spam_scores
from backend.text import perturb


ROOT = Path(__file__).resolve().parents[1]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    text: str = Field(min_length=1, max_length=2000)
    method: Literal["substitution", "punctuation", "repetition"] = "substitution"
    seed: int = Field(default=42, ge=0, le=4294967295)


def create_app(model_path=None):
    artifact = Path(model_path or os.environ.get("MODEL_PATH", ROOT / "artifacts/models.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        app.state.model_error = None
        try:
            app.state.bundle = load_bundle(artifact)
        except (FileNotFoundError, ValueError) as error:
            app.state.model_error = str(error)
        yield

    app = FastAPI(title="SMS Spam Robustness", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        bundle = app.state.bundle
        return {"status": "ok" if bundle else "models_unavailable", "models_ready": bundle is not None,
                "model_id": bundle["model_id"] if bundle else None}

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest):
        if not request.text.strip():
            raise HTTPException(422, "Enter a non-empty English SMS message.")
        bundle = app.state.bundle
        if bundle is None:
            raise HTTPException(503, "Models are unavailable. Train compatible models, then restart the backend.")
        modified = perturb(request.text, request.method, request.seed)
        normalizer = bundle["models"]["C"].named_steps["normalize"]
        normalized = normalizer.normalize(modified["text"])
        results = {}
        for system, model in bundle["models"].items():
            scores = spam_scores(model, system, [request.text, modified["text"]])
            threshold = bundle["selected"][system]["threshold"]
            results[system] = {
                name: {"label": "spam" if score >= threshold else "ham", "score": float(score)}
                for name, score in zip(("original", "modified"), scores)
            }
            results[system]["threshold"] = threshold
        return {"original_text": request.text, "modified": modified, "normalized": normalized,
                "predictions": results, "seed": request.seed, "model_id": bundle["model_id"]}

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()
