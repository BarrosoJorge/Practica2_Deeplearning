"""Servidor FastAPI de inferencia (Etapa 6).

  uv run uvicorn api.app:app --host 0.0.0.0 --port 8000   # local
  docker compose up -d api                                # contenedor :8000

Contrato (kW reales, sin escala):
  GET  /health
  POST /predict  {"series": [24 horas de consumo kW]} -> {"prediction": kW}
"""

import json
import os
import pathlib

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from models.factory import create_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINTS_DIR = ROOT / "artifacts" / "checkpoints"
REGISTRY_CACHE = ROOT / "artifacts" / "registry"


def load_artifacts(checkpoint_path=None, scaler_path=None, tracking_uri=None):
    """Carga (model, scaler, origen) para el endpoint."""
    scaler_path = pathlib.Path(scaler_path or DATA_DIR / "scaler.json")
    scaler = json.loads(scaler_path.read_text())

    if checkpoint_path is None:
        from api.model_loader import resolve_checkpoint

        fallback = sorted(
            (CHECKPOINTS_DIR / "fase3-lstm").glob("best.pt"),
        )
        tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000",
        )
        checkpoint_path, source = resolve_checkpoint(
            tracking_uri=tracking_uri,
            registry_name=os.environ.get("REGISTRY_NAME", "p2dl-forecasting"),
            cache_dir=REGISTRY_CACHE,
            fallback=fallback[0] if fallback else None,
        )
    else:
        checkpoint_path = pathlib.Path(checkpoint_path)
        source = f"{checkpoint_path} (checkpoint local)"

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = create_model(ckpt["arch"], ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, scaler, source


def create_app(checkpoint_path=None, scaler_path=None, tracking_uri=None):
    """App de FastAPI con el modelo ya cargado al construirse.

    Tests pasan checkpoint_path/scaler_path explícitos (determinista, sin
    red); en producción se resuelven vía Registry/fallback.
    """
    model, scaler, source = load_artifacts(
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        tracking_uri=tracking_uri,
    )
    window = int(scaler["window"])
    mean = float(scaler["mean"])
    std = float(scaler["std"])

    app = FastAPI(
        title="P2_DL — Inference API (RNN/LSTM Forecasting)",
        version="1.0.0",
        description="POST /predict con ventana de 24 h horarias → siguiente hora (kW).",
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Estado del servidor y modelo cargado",
    )
    def health():
        return HealthResponse(
            status="ok",
            model=model.__class__.__name__,
            source=source,
            window=window,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        summary="Predice la hora siguiente (kW) a partir de las últimas 24 h",
    )
    def predict(request: PredictRequest):
        values = request.series

        if len(values) != window:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"series debe contener exactamente {window} valores horarios "
                    f"(recibidos: {len(values)})"
                ),
            )
        if any(v < 0 for v in values):
            raise HTTPException(
                status_code=422,
                detail="el consumo eléctrico no puede ser negativo (kW)",
            )

        window_series = np.asarray(values, dtype=np.float32)
        x = (
            torch.from_numpy(
                (window_series - mean) / std,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # (1, window, 1)

        with torch.no_grad():
            prediction = model(x).item()

        prediction_kw = float(max(prediction * std + mean, 0.0))
        return PredictResponse(prediction=prediction_kw, model=source)

    return app


app = create_app()
