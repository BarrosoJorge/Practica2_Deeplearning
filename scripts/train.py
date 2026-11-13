"""Driver de entrenamiento RNN/LSTM para la Etapa 4.

  uv run python scripts/train.py                          # ambas arquitecturas, config completa
  uv run python scripts/train.py --archs lstm             # sólo una
  uv run python scripts/train.py --quick --no-mlflow      # sanity local (2 epocas, subset)
  uv run python scripts/train.py --quick                  # sanity registrando en MLflow

Registra en MLflow: hiperparámetros, métricas por época (train_loss,
val_loss, val_MAE/RMSE/MAPE en kW) y artefactos (curvas + checkpoint best).
Test NO se toca en epoch: se evalua al final de cada run.
"""

import argparse
import json
import os
import pathlib
import random
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import yaml

from datasets.sequence import get_loaders
from engine.train import evaluate, train_model
from models.factory import create_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "artifacts" / "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_make_loaders(cfg, quick):
    """Devuelve loaders (train con shuffle) y recorta train si --quick."""
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loaders, _ = get_loaders(
        data_cfg["dir"],
        batch_size=train_cfg["batch_size"],
        window=data_cfg["window"],
        shuffle_train=True,
    )
    device = pick_device()

    if quick:
        cap = train_cfg["quick"]["max_train_samples"]
        dataset = loaders["train"].dataset
        loaders["train"] = DataLoader(
            Subset(dataset, list(range(min(cap, len(dataset))))),
            batch_size=train_cfg["batch_size"],
            shuffle=True,
        )

    return loaders, device


def train_one_arch(arch, cfg, quick, no_mlflow, epochs_override=None):
    import mlflow

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    early_cfg = cfg["early_stopping"]

    set_seed(train_cfg["seed"])
    torch.use_deterministic_algorithms(False)

    scaler = json.loads((ROOT / data_cfg["dir"] / "scaler.json").read_text())
    mean = scaler["mean"]
    std = scaler["std"]

    if scaler["window"] != data_cfg["window"]:
        raise ValueError(
            f"window del scaler ({scaler['window']}) != window del config "
            f"({data_cfg['window']})"
        )

    loaders, device = build_make_loaders(cfg, quick)
    epochs_cfg = cfg["train"]["epochs"]
    if quick:
        epochs_cfg = cfg["train"]["quick"]["epochs"]
    if epochs_override:
        epochs_cfg = epochs_override

    model = create_model(arch, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    run_name = f"{cfg['mlflow']['run_prefix']}-{arch}"
    if quick:
        run_name += "-quick"

    print(f"\n=== {run_name} ===")
    print(
        f"  device {device} · params {n_params:,} · epochs {epochs_cfg} · "
        f"hidden {model_cfg['hidden_size']} · window {data_cfg['window']}"
    )

    history = []

    def on_epoch(epoch, record):
        history.append(record)
        mlflow.log_metrics(
            {
                "train_loss": record["train_loss"],
                "val_loss": record["val_loss"],
                "val_MAE": record["MAE_kW"],
                "val_RMSE": record["RMSE_kW"],
            },
            step=epoch,
        )

    if not no_mlflow:
        # env var tiene prioridad (compose usa http://mlflow:5000)
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]),
        )
        mlflow.set_experiment(cfg["mlflow"]["experiment"])

    context = (
        mlflow.start_run(run_name=run_name) if not no_mlflow else tempfile.TemporaryDirectory()
    )
    with context as run:
        run_id = None
        if not no_mlflow:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "arch": arch,
                    "window": data_cfg["window"],
                    "hidden_size": model_cfg["hidden_size"],
                    "num_layers": model_cfg["num_layers"],
                    "dropout": model_cfg["dropout"],
                    "batch_size": train_cfg["batch_size"],
                    "epochs": epochs_cfg,
                    "lr": train_cfg["lr"],
                    "weight_decay": train_cfg["weight_decay"],
                    "seed": train_cfg["seed"],
                    "patience": early_cfg["patience"],
                    "quick": quick,
                    "num_params": n_params,
                }
            )

        history, stopper = train_model(
            model,
            loaders,
            device,
            mean,
            std,
            epochs=epochs_cfg,
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            patience=early_cfg["patience"],
            min_delta=early_cfg["min_delta"],
            on_epoch=on_epoch if not no_mlflow else None,
        )
        test_loss, test_metrics = evaluate(
            model,
            loaders["test"],
            device,
            mean,
            std,
        )
        print(
            f"  TEST (kW): MAE {test_metrics['MAE_kW']:.3f} · "
            f"RMSE {test_metrics['RMSE_kW']:.3f} · MAPE {test_metrics['MAPE_pct']:.2f}%"
        )

        if not no_mlflow:
            mlflow.log_metrics(
                {
                    "test_MAE": test_metrics["MAE_kW"],
                    "test_RMSE": test_metrics["RMSE_kW"],
                    "test_MSE": test_metrics["MSE_kW2"],
                    "test_MAPE": test_metrics["MAPE_pct"],
                    "best_val_loss": stopper.best,
                    "epochs_run": len(history),
                }
            )
            plot_curves(history, run_name)
            save_checkpoint(model, arch, cfg, run_name, stopper, history, device)
            mlflow.log_artifacts(str(CHECKPOINTS_DIR / run_name), artifact_path="model")

    return {
        "run_name": run_name,
        "run_id": run_id,
        "history": history,
        "test_metrics": test_metrics,
        "best_val_loss": stopper.best,
        "num_params": n_params,
    }


def plot_curves(history, run_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    out_dir = CHECKPOINTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [r["train_loss"] for r in history], label="train_loss")
    ax.plot(epochs, [r["val_loss"] for r in history], label="val_loss")
    ax.set_xlabel("epoca")
    ax.set_ylabel("MSE (z-score)")
    ax.set_title(f"{run_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curvas.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [r["MAE_kW"] for r in history], label="val_MAE_kW")
    ax.plot(epochs, [r["RMSE_kW"] for r in history], label="val_RMSE_kW")
    ax.set_xlabel("epoca")
    ax.set_ylabel("kW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metricas_kW.png", dpi=120)
    plt.close(fig)


def save_checkpoint(model, arch, cfg, run_name, stopper, history, device):
    out_dir = CHECKPOINTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "state_dict": model.state_dict(),
            "best_val_loss": stopper.best,
            "config": cfg,
            "device": str(device),
        },
        out_dir / "best.pt",
    )
    (out_dir / "historial.json").write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--archs", nargs="*", choices=["rnn", "lstm"])
    parser.add_argument("--quick", action="store_true", help="sanity: 2 epocas + subset")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override del numero de epocas (con o sin --quick)",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="sin tracking")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    archs = args.archs if args.archs else cfg["archs"]

    results = [
        train_one_arch(arch, cfg, args.quick, args.no_mlflow, args.epochs) for arch in archs
    ]

    print("\n=== resumen ===")
    for res in results:
        m = res["test_metrics"]
        print(
            f"  {res['run_name']:22s}  val {res['best_val_loss']:.4f}  "
            f"test MAE {m['MAE_kW']:.3f} kW · RMSE {m['RMSE_kW']:.3f} kW · "
            f"MAPE {m['MAPE_pct']:.2f}%"
        )

    out_dir = CHECKPOINTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / ("resumen_quick.json" if args.quick else "resumen.json")
    summary_path.write_text(
        json.dumps(
            [
                {
                    "run_name": r["run_name"],
                    "run_id": r["run_id"],
                    "best_val_loss": r["best_val_loss"],
                    "test_metrics": r["test_metrics"],
                }
                for r in results
            ],
            indent=2,
        )
    )
    print(f"  resumen => {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
