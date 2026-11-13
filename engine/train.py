import numpy as np
import torch
import torch.nn as nn

from engine.metrics import compute_metrics


def run_epoch(model, loader, device, criterion, optimizer=None):
    """Una epoca. Si optimizer no es None, hace backward/step (modo train).

    Devuelve (loss_promedio, predicciones, targets) como numpy escalado.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_pred = []
    all_true = []

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch = y.size(0)
            total_loss += loss.item() * batch
            total_samples += batch
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred).ravel()
    true_np = np.concatenate(all_true).ravel()
    return total_loss / total_samples, pred_np, true_np


def evaluate(model, loader, device, mean, std):
    """Evalua en loader y devuelve (loss_escalado dict, métricas en kW)."""
    criterion = nn.MSELoss()
    loss, pred, true = run_epoch(model, loader, device, criterion, optimizer=None)
    metrics = compute_metrics(
        true,
        pred,
        mean,
        std,
    )
    return loss, metrics


def train_model(
    model,
    loaders,
    device,
    mean,
    std,
    *,
    epochs=30,
    lr=1e-3,
    weight_decay=0.0,
    patience=5,
    min_delta=0.0,
    on_epoch=None,
):
    """Entrena model con loaders['train'] y valida con loaders['val'].

    Early stopping sobre val_loss (MSE escalado). Al aplicar, restaura el
    mejor estado. Devuelve history (list de dicts) y EarlyStopping.

    on_epoch(epoch, epoch_record) se consulta tras cada epoca (p. ej. para
    registrar en MLflow con step sincronizado).
    """
    from callbacks.early_stopping import EarlyStopping

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, _, _ = run_epoch(
            model,
            loaders["train"],
            device,
            criterion,
            optimizer,
        )
        val_loss, val_metrics = evaluate(
            model,
            loaders["val"],
            device,
            mean,
            std,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(record)

        if on_epoch is not None:
            on_epoch(epoch, record)

        print(
            f"  epoch {epoch:3d}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_MAE {val_metrics['MAE_kW']:.3f} kW"
        )

        if stopper.step(val_loss, model):
            print(f"  early stopping en epoch {epoch} (mejor val_loss {stopper.best:.4f})")
            break

    stopper.restore(model)
    return history, stopper
