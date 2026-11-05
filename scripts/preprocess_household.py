"""Driver de preprocesamiento (Fase 2).

Serie minutal -> resample '1h' -> limpia NaN -> split temporal 70/15/15 ->
z-score (fit solo en train) -> guarda .npy + scaler.json en data/processed/,
graficas de reporte en artifacts/ y log en MLflow.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.household import download_household, load_household

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("artifacts/reports/preprocess_household")
RUN_NAME = "fase2-preprocesamiento"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
WINDOW = 24


def construir_serie_horaria(fechas, valores):
    """Serie minutal -> horaria (media por hora) + NaN interpolados."""
    serie = pd.Series(valores, index=fechas)
    serie = serie.resample("1h").mean()
    serie = serie.interpolate(method="time")
    serie = serie.ffill().bfill()
    return serie


def split_temporal(serie):
    n = len(serie)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    return {
        "train_end": train_end,
        "val_end": val_end,
        "splits": {
            "train": serie.iloc[:train_end],
            "val": serie.iloc[train_end:val_end],
            "test": serie.iloc[val_end:],
        },
    }


def escalar(splits):
    """Z-score: media y desviación calculadas SOLO con train."""
    train = splits["splits"]["train"]

    mean = float(train.mean())
    std = float(train.std())

    escalados = {
        name: ((split.values - mean) / std).astype(np.float32)
        for name, split in splits["splits"].items()
    }
    return escalados, mean, std


def sanity_checks(serie, escalados):
    """Invariantes: sin NaN, 1-D, índice temporal monótono y único."""
    assert serie.index.is_monotonic_increasing, "indice no monótono"
    assert not serie.index.has_duplicates, "índices temporales duplicados"

    for name, arr in escalados.items():
        assert not np.isnan(arr).any(), f"NaN en {name}"
        assert arr.ndim == 1, f"{name} no es unidimensional"


def escalar_files(escalados, mean, std, splits):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in escalados.items():
        np.save(PROCESSED_DIR / f"{name}.npy", arr)

    scaler = {
        "mean": mean,
        "std": std,
        "target": "Global_active_power",
        "freq": "1h",
        "window": WINDOW,
        "split": {
            "train_end": int(splits["train_end"]),
            "val_end": int(splits["val_end"]),
        },
    }
    (PROCESSED_DIR / "scaler.json").write_text(
        json.dumps(scaler, indent=2),
        encoding="utf-8",
    )


def guardar_plots(splits, escalados, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)

    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for name, split in splits["splits"].items():
        ax.plot(split.index, split.values, lw=0.4, color=colors[name], label=name)
    ax.set_title("Consumo horario (kW) — corte train/val/test")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(report_dir / "1_serie_completa.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(escalados["train"][: 24 * 10], lw=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("train escalado (z-score) — primeras 10 días")
    fig.tight_layout()
    fig.savefig(report_dir / "2_train_escalado.png", dpi=130)
    plt.close(fig)


def escribir_resumen(md_path, len_serie, escalados, splits, mean, std):
    n = {k: len(v) for k, v in escalados.items()}
    s = splits["splits"]
    info = (
        "# Preproceso — Household Power horario\n\n"
        "| Métrica | Valor |\n| :--- | :--- |\n"
        f"| Horas totales | {len_serie:,} |\n"
        f"| Split train/val/test | {n['train']:,} / {n['val']:,} / {n['test']:,} |\n"
        f"| Z-score (fit train) | mean={mean:.4f} kW · std={std:.4f} |\n"
        f"| Ventana | {WINDOW} h → t+1 |\n\n"
        f"- train: {s['train'].index[0]} → {s['train'].index[-1]}\n"
        f"- val: {s['val'].index[0]} → {s['val'].index[-1]}\n"
        f"- test: {s['test'].index[0]} → {s['test'].index[-1]}\n"
    )
    md_path.write_text(info, encoding="utf-8")


def registrar_mlflow(params, metrics, report_dir):
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PROCESSED_DIR / "scaler.json"), artifact_path="processed")
        mlflow.log_artifact(str(report_dir / "resumen.md"), artifact_path="report")
        for png in report_dir.glob("*.png"):
            mlflow.log_artifact(str(png), artifact_path="report")


def main():
    txt_path = download_household("data/raw")

    print("Cargando serie minutal...")
    fechas, valores = load_household(txt_path)

    print("Resampleo a 1h...")
    serie = construir_serie_horaria(fechas, valores)

    print("Split 70/15/15 + z-score (fit train)...")
    splits = split_temporal(serie)
    escalados, mean, std = escalar(splits)

    sanity_checks(serie, escalados)
    escalar_files(escalados, mean, std, splits)

    print("Reportes...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    guardar_plots(splits, escalados, REPORT_DIR)
    escribir_resumen(REPORT_DIR / "resumen.md", len(serie), escalados, splits, mean, std)

    params = {
        "freq": "1h",
        "window": f"{WINDOW} h -> t+1",
        "split": "70/15/15 temporal",
        "scaler": "z-score (fit train)",
        "target": "Global_active_power (kW)",
    }
    metrics = {
        "horas_totales": float(len(serie)),
        "train_horas": float(len(escalados["train"])),
        "val_horas": float(len(escalados["val"])),
        "test_horas": float(len(escalados["test"])),
        "kW_mean_train": mean,
        "kW_std_train": std,
    }
    try:
        registrar_mlflow(params, metrics, REPORT_DIR)
        print("Run 'fase2-preprocesamiento' registrado en http://localhost:5000")
    except Exception as exc:
        print(f"MLflow no disponible ({exc}); .npy y reportes ya guardados.")


if __name__ == "__main__":
    main()
