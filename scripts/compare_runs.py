"""Etapa 5 — comparativa RNN vs LSTM + selección del mejor modelo.

  uv run python scripts/compare_runs.py          # reporte + registro del modelo

Qué hace:
  1. Reevalua cada checkpoint (best.pt) de artifacts/checkpoints/fase3-* sobre
     val y test (modelos restaurados exactamente como terminó el training).
  2. Regenera la tabla MAE/MSE/RMSE/MAPE (kW) val y test.
  3. Produce artifacts/reports/compare/rnn_vs_lstm.md + gráficas (val/test).
  4. Registra el mejor modelo (por val_loss) en el MLflow Registry.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import yaml

from datasets.sequence import get_loaders
from engine.train import evaluate
from models.factory import create_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "artifacts" / "checkpoints"
REPORT_DIR = ROOT / "artifacts" / "reports" / "compare"


def load_checkpoint_results(cfg, device):
    """Reevalúa cada checkpoint entrenado sobre val y test."""
    data_cfg = cfg["data"]
    loaders, _ = get_loaders(
        data_cfg["dir"],
        batch_size=cfg["train"]["batch_size"],
        window=data_cfg["window"],
    )
    scaler = json.loads((ROOT / data_cfg["dir"] / "scaler.json").read_text())

    results = []
    for run_dir in sorted(CHECKPOINTS_DIR.glob("fase3-*")):
        if run_dir.name.endswith("-quick"):
            continue
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        arch = ckpt["arch"]

        model = create_model(arch, cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])

        val_loss, val_metrics = evaluate(
            model, loaders["val"], device, scaler["mean"], scaler["std"]
        )
        test_loss, test_metrics = evaluate(
            model, loaders["test"], device, scaler["mean"], scaler["std"]
        )
        results.append(
            {
                "run_name": run_dir.name,
                "arch": arch,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "n_params": ckpt.get("n_params"),
            }
        )
    return results


def table_rows(results):
    """Filas de la tabla: una por modelo, con métricas val y test (kW)."""
    rows = []
    for r in results:
        rows.append(
            {
                "modelo": r["run_name"],
                "val_MAE": r["val_metrics"]["MAE_kW"],
                "val_RMSE": r["val_metrics"]["RMSE_kW"],
                "val_MAPE": r["val_metrics"]["MAPE_pct"],
                "val_WAPE": r["val_metrics"]["WAPE_pct"],
                "val_sMAPE": r["val_metrics"]["sMAPE_pct"],
                "test_MAE": r["test_metrics"]["MAE_kW"],
                "test_RMSE": r["test_metrics"]["RMSE_kW"],
                "test_MAPE": r["test_metrics"]["MAPE_pct"],
                "test_WAPE": r["test_metrics"]["WAPE_pct"],
                "test_sMAPE": r["test_metrics"]["sMAPE_pct"],
                "val_loss": r["val_loss"],
            }
        )
    return rows


def markdown_table(rows):
    """Tabla markdown de comparativa (orden por val_MAE)."""
    headers = [
        "Modelo",
        "MAE val (kW)",
        "RMSE val (kW)",
        "MAPE val (%)",
        "WAPE val (%)",
        "sMAPE val (%)",
        "MAE test (kW)",
        "RMSE test (kW)",
        "MAPE test (%)",
        "WAPE test (%)",
        "sMAPE test (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" --- "] * len(headers)) + "|",
    ]
    for row in sorted(rows, key=lambda x: x["val_MAE"]):
        cells = [
            row["modelo"],
            f"{row['val_MAE']:.3f}",
            f"{row['val_RMSE']:.3f}",
            f"{row['val_MAPE']:.2f}",
            f"{row['val_WAPE']:.2f}",
            f"{row['val_sMAPE']:.2f}",
            f"{row['test_MAE']:.3f}",
            f"{row['test_RMSE']:.3f}",
            f"{row['test_MAPE']:.2f}",
            f"{row['test_WAPE']:.2f}",
            f"{row['test_sMAPE']:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_predictions(results, loaders, scaler, device, cfg, horizon=72):
    """Curvas reales vs predicha (72 h) del segmento test para cada modelo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(10, 3 * len(results)),
        sharex=True,
    )
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results, strict=False):
        run_dir = CHECKPOINTS_DIR / r["run_name"]
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model = create_model(r["arch"], cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        test_series = loaders["test"].dataset.series.numpy()
        window = cfg["data"]["window"]

        preds = []
        with torch.no_grad():
            for i in range(horizon):
                x = (
                    torch.as_tensor(
                        test_series[i : i + window],
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(device)
                )
                preds.append(model(x).item())

        true = test_series[window : window + horizon]
        preds = np.array(preds)
        true_kw = true * scaler["std"] + scaler["mean"]
        preds_kw = preds * scaler["std"] + scaler["mean"]

        ax.plot(true_kw, label="real (kW)", color="black")
        ax.plot(preds_kw, label="predicción (kW)", color="tab:red", alpha=0.8)
        ax.set_title(r["run_name"])
        ax.set_ylabel("kW")
        ax.legend()

    axes[-1].set_xlabel("horas (segmento inicial del test)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "predicciones_72h.png", dpi=120)
    plt.close(fig)


def register_best(results, cfg):
    """Registra el mejor modelo al Registry de MLflow (por val_loss)."""
    import os

    import mlflow
    from mlflow.exceptions import MlflowException

    best = min(results, key=lambda r: r["val_loss"])
    best_run_id = _run_ids[best["run_name"]]

    # env var tiene prioridad (compose usa http://mlflow:5000)
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]),
    )
    client = mlflow.MlflowClient()
    name = cfg["mlflow"]["registry_name"]

    if not client.search_registered_models(f"name='{name}'"):
        client.create_registered_model(name)

    source = f"runs:/{best_run_id}/model"
    try:
        version = client.create_model_version(
            name=name,
            source=source,
            run_id=best_run_id,
            tags={"selection_metric": "val_loss", "selector": "compare_runs.py"},
        )
        print(f"  Registry: {name} versión {version.version} <= {best['run_name']}")
    except MlflowException as exc:
        print(f"  Registry: no se pudo crear versión ({exc})")
    return best


# run ids de los runs fase3 (para el Registry)
_run_ids = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--no-registry", action="store_true", help="no registrar en el Model Registry"
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run ids desde el resumen del training (registry necesita run_id)
    resumen_path = CHECKPOINTS_DIR / "resumen.json"
    if resumen_path.exists():
        for entry in json.loads(resumen_path.read_text()):
            _run_ids[entry["run_name"]] = entry["run_id"]

    print("=== Etapa 5: comparativa RNN vs LSTM ===\n")
    print(f"  device: {device}")
    print("  reevaluando checkpoints sobre val y test...")

    results = load_checkpoint_results(cfg, device)
    rows = table_rows(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # gráficas val (curvas de training ya generadas en fase 4) + test predictions
    loaders, _ = get_loaders(
        cfg["data"]["dir"],
        batch_size=cfg["train"]["batch_size"],
        window=cfg["data"]["window"],
    )
    scaler = json.loads((ROOT / cfg["data"]["dir"] / "scaler.json").read_text())
    plot_predictions(results, loaders, scaler, device, cfg)

    best = min(results, key=lambda r: r["val_loss"])

    report = [
        "# Comparativa RNN vs LSTM (Etapa 5)",
        "",
        f"_Config: {args.config} · window {cfg['data']['window']}h → t+1 · "
        f"hidden {cfg['model']['hidden_size']} · seed {cfg['train']['seed']}_",
        "",
        "Métricas de selección: val (elección del mejor modelo). Test es la",
        "medición final sin sesgo. Todas en kW reales.",
        "",
        markdown_table(rows),
        "",
        "## Nota sobre MAPE",
        "",
        "El MAPE test (~51-54%) se ve alto para un modelo con MAE ~0.35 kW",
        "frente a consumo medio ~1.1 kW. La causa es la definición del MAPE",
        "con series horarias de consumo bajo: en madrugada el target puede",
        "ser 0.03-0.1 kW y predicciones razonables de 0.15-0.3 kW generan",
        "errores porcentuales de cientos por ciento sobre bases minúsculas.",
        "Es un artefacto del denominador, no del error absoluto.",
        "",
        "Para la lectura volumétrica se reporta **WAPE** (Σ|err|/Σ|y|),",
        "que pondera por consumo real del periodo, y **sMAPE** (acotada en",
        "[0, 200%]). El error típico por hora es ~0.35 kW, consistente con",
        "el WAPE test reportado en la tabla.",
        "",
        "## Curvas de predicción (test, 72 h)",
        "",
        "![predicciones](predicciones_72h.png)",
        "",
    ]

    if args.no_registry:
        report.append("_Registro en Model Registry omitido (--no-registry)._")
    else:
        report.append(
            f"## Selección\n\nMejor modelo por val_loss: **{best['run_name']}**"
            f" ({best['val_loss']:.4f})."
        )

    report_path = REPORT_DIR / "rnn_vs_lstm.md"
    report_path.write_text("\n".join(report) + "\n")

    (REPORT_DIR / "comparativa.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "best": best["run_name"],
                "criteria": "val_loss",
            },
            indent=2,
        )
        + "\n"  # end-of-file-fixer exige newline final
    )

    print("\n" + markdown_table(rows))
    print(f"\n  mejor por val_loss: {best['run_name']}")
    print("  reporte   => artifacts/reports/compare/rnn_vs_lstm.md")

    if not args.no_registry:
        register_best(results, cfg)


if __name__ == "__main__":
    main()
