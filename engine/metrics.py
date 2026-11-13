import numpy as np


def descale(values, mean, std):
    """Revirtiendo el z-score para volver a unidades de kW."""
    return values * std + mean


def compute_metrics(y_true_scaled, y_pred_scaled, mean, std):
    """Métricas de regresión calculadas en kW reales (des-escalados).

    - MAE/MSE/RMSE: errores absolutos/cuadráticos en kW.
    - MAPE: error porcentual medio. Excluye puntos con y_true ≈ 0 para no
      inflar el porcentaje por división por una base minúscula (el consumo
      horario tiene muchas horas cercanas a 0 kW). Reporta la cobertura
      de esa máscara. Ver "Nota sobre MAPE" en el reporte comparativo.
    - WAPE: Σ|err| / Σ|y_true|. Robusto ante bases pequeñas; es el error
      porcentual "pesado por volumen" y la referencia más honesta cuando
      hay horas de consumo muy bajo.
    - sMAPE: 2|err| / (|y_true|+|y_pred|), acotada en [0%, 200%].
    """
    y_true = np.asarray(descale(np.asarray(y_true_scaled, dtype=np.float64), mean, std))
    y_pred = np.asarray(descale(np.asarray(y_pred_scaled, dtype=np.float64), mean, std))

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    mse = float((err**2).mean())
    rmse = float(np.sqrt(mse))

    mask = np.abs(y_true) > 1e-3
    if mask.any():
        mape = float((abs_err[mask] / np.abs(y_true[mask])).mean() * 100.0)
        mape_coverage = float(mask.mean())
    else:
        mape = float("nan")
        mape_coverage = 0.0

    total_true = float(np.abs(y_true).sum())
    wape = float(abs_err.sum() / total_true * 100.0) if total_true > 0 else float("nan")

    denom = np.abs(y_true) + np.abs(y_pred)
    positive = denom > 1e-12  # solo ambas 0 kW simultáneamente son 0/0
    smape = float((2.0 * abs_err[positive] / denom[positive]).mean() * 100.0)

    return {
        "MAE_kW": mae,
        "MSE_kW2": mse,
        "RMSE_kW": rmse,
        "MAPE_pct": mape,
        "MAPE_coverage": mape_coverage,
        "WAPE_pct": wape,
        "sMAPE_pct": smape,
    }
