"""Prueba 3 (Inferencia): servidor FastAPI operacional y respuesta correcta."""

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app
from models.factory import create_model

WINDOW = 24


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Servidor con un checkpoint sintético (determinista, sin red)."""
    tmp_path = tmp_path_factory.mktemp("api")
    cfg = {
        "data": {"dir": "data/processed", "window": WINDOW},
        "model": {"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
    }
    model = create_model("lstm", cfg)
    torch.save(
        {
            "arch": "lstm",
            "state_dict": model.state_dict(),
            "best_val_loss": 999.0,
            "config": cfg,
            "device": "cpu",
        },
        tmp_path / "best.pt",
    )
    (tmp_path / "scaler.json").write_text(
        json.dumps(
            {
                "mean": 1.0,
                "std": 0.5,
                "window": WINDOW,
                "target": "Global_active_power",
                "freq": "1h",
            }
        )
    )

    app = create_app(
        checkpoint_path=tmp_path / "best.pt",
        scaler_path=tmp_path / "scaler.json",
    )
    return TestClient(app)


def payload(values):
    return {"series": values}


def default_series():
    rng = np.random.default_rng(7)
    return list(np.maximum(rng.normal(0.9, 0.6, WINDOW), 0.0).round(4))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["window"] == WINDOW
    assert "LSTM" in body["model"]


def test_predict_ok_and_structure(client):
    resp = client.post("/predict", json=payload(default_series()))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"prediction", "model"}
    assert isinstance(body["prediction"], float)
    assert body["prediction"] >= 0.0
    assert torch.isfinite(torch.tensor(body["prediction"]))


def test_predict_deterministic(client):
    serie = default_series()
    r1 = client.post("/predict", json=payload(serie)).json()
    r2 = client.post("/predict", json=payload(serie)).json()
    assert r1 == r2


def test_predict_cero_consumo(client):
    """24 h de 0 kW → respuesta válida y predicción nunca negativa."""
    resp = client.post("/predict", json=payload([0.0] * WINDOW))
    assert resp.status_code == 200
    assert resp.json()["prediction"] >= 0.0


def test_predict_longitud_incorrecta(client):
    for n in (WINDOW - 1, WINDOW + 1):
        resp = client.post("/predict", json=payload([0.5] * n))
        assert resp.status_code == 422
        assert "horas" in resp.json()["detail"] or "series" in resp.json()["detail"]


def test_predict_consumo_negativo(client):
    serie = default_series()
    serie[0] = -0.3
    resp = client.post("/predict", json=payload(serie))
    assert resp.status_code == 422
    assert "negativo" in resp.json()["detail"]


def test_predict_valor_nan_rechazado(client):
    """Pydantic (allow_inf_nan=False) debe rechazar NaN con 422."""
    raw = json.dumps({"series": default_series()[:-1] + [None]})
    resp = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert resp.status_code == 422
