"""Prueba 2 (Modelos): forward pass de RNN/LSTM, métricas y early stopping."""

import numpy as np
import pytest
import torch

from callbacks.early_stopping import EarlyStopping
from engine.metrics import compute_metrics
from models.factory import create_model

CFG = {
    "model": {
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
    }
}


@pytest.mark.parametrize("arch", ["rnn", "lstm"])
def test_forward_shapes(arch):
    """(N, window, features) -> (N, 1) para ambas arquitecturas."""
    model = create_model(arch, CFG)
    x = torch.randn(4, 24, 1)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()
    assert sum(p.numel() for p in model.parameters()) > 0


def test_factory_invalid_name():
    with pytest.raises(ValueError):
        create_model("gru", CFG)


def test_metrics_perfect_prediction():
    """Si pred == true, todas las métricas son 0."""
    mean, std = 1.0857, 0.9269
    y_true = np.array([0.5, 1.0, 2.0])
    metrics = compute_metrics(y_true, y_true, mean, std)
    assert metrics["MAE_kW"] == pytest.approx(0.0)
    assert metrics["MSE_kW2"] == pytest.approx(0.0)
    assert metrics["RMSE_kW"] == pytest.approx(0.0)
    assert metrics["MAPE_pct"] == pytest.approx(0.0)


def test_metrics_units_and_zero_guard():
    """Métricas están en kW (des-escaladas) y MAPE ignora targets ~0."""
    mean, std = 0.0, 2.0
    y_true = np.array([0.0, 1.0])  # kW: 0 y 2
    y_pred = np.array([0.5, 1.5])  # kW: 1 y 3 → error fijo de +1 kW

    metrics = compute_metrics(y_true, y_pred, mean, std)

    assert metrics["MAE_kW"] == pytest.approx(1.0)
    assert metrics["RMSE_kW"] == pytest.approx(1.0)
    assert metrics["MAPE_pct"] == pytest.approx(50.0)  # sólo el target no-nulo
    assert metrics["MAPE_coverage"] == pytest.approx(0.5)

    # WAPE = Σ|err| / Σ|y| = 2 / 2 = 100%
    assert metrics["WAPE_pct"] == pytest.approx(100.0)

    # sMAPE = media( 2|err| / (|y|+|yhat|) ) = 2·1/(0+1) y 2·1/(2+3) → 200% y 40%
    assert metrics["sMAPE_pct"] == pytest.approx((200.0 + 40.0) / 2.0)


def test_wape_is_volume_weighted():
    """WAPE pondera por volumen: un error en consumo alto pesa proporcional
    a su magnitud, no se dispara como en MAPE con bases pequeñas."""
    mean, std = 0.0, 1.0
    y_true = np.array([10.0, 0.01])  # kW
    y_pred = np.array([10.5, 0.51])  # error fijo de +0.5 kW en ambos

    metrics = compute_metrics(y_true, y_pred, mean, std)

    # MAPE explota en la base 0.01 (5000%); WAPE no: 1.0/10.01 ≈ 10%
    assert metrics["MAPE_pct"] > 2000.0
    assert metrics["WAPE_pct"] == pytest.approx(1.0 / 10.01 * 100.0, rel=1e-3)


def test_early_stopping_stops_and_restores():
    model = torch.nn.Linear(2, 1)
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    with torch.no_grad():
        model.weight.fill_(1.0)

    # mejora inicial: guarda estado
    assert not stopper.step(1.0, model)
    with torch.no_grad():
        model.weight.fill_(0.0)

    # 2 épocas empeorando seguidas con patience=2 → para
    assert not stopper.step(1.5, model)
    assert stopper.step(1.4, model)

    stopper.restore(model)
    assert torch.all(model.weight == 1.0)
