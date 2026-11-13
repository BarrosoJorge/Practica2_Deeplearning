"""Prueba 1 (Datos y Preprocesamiento): ventanas, loaders y scaler."""

import json

import numpy as np
import pytest
import torch

from datasets.sequence import SequenceDataset, get_loaders, load_processed

WINDOW = 6


def _scaffold_processed(tmp_path, sizes=(200, 60, 60), mean=1.0, std=0.5):
    rng = np.random.default_rng(0)
    for name, size in zip(("train", "val", "test"), sizes, strict=False):
        series = (rng.normal(mean, std, size)).astype(np.float32)
        np.save(tmp_path / f"{name}.npy", series)
    (tmp_path / "scaler.json").write_text(json.dumps({"mean": mean, "std": std}))


def test_dataset_shapes():
    """X e y tienen shapes (window, 1) y (1,); len = N - window."""
    window = 7
    series = torch.arange(100, dtype=torch.float32)
    dataset = SequenceDataset(series, window=window)

    assert len(dataset) == 100 - window

    x, y = dataset[0]
    assert x.shape == (window, 1)
    assert y.shape == (1,)


def test_dataset_sliding_window():
    """La ventana deslizante avanza 1 paso por item."""
    series = torch.arange(50, dtype=torch.float32)
    dataset = SequenceDataset(series, window=4)

    x0, _ = dataset[0]
    x1, _ = dataset[1]
    assert torch.equal(x0[1:], x1[:-1])

    _, y = dataset[10]
    assert y.item() == float(10 + 4)


def test_load_processed_nan_guard(tmp_path):
    np.save(tmp_path / "train.npy", np.arange(10, dtype=np.float32))
    np.save(tmp_path / "val.npy", np.arange(5, dtype=np.float32))
    np.save(tmp_path / "test.npy", np.arange(5, dtype=np.float32))

    splits = load_processed(tmp_path)
    for values in splits.values():
        assert np.isfinite(values).all()


def test_get_loaders(tmp_path):
    _scaffold_processed(tmp_path)
    loaders, scaler = get_loaders(tmp_path, batch_size=8, window=WINDOW)

    assert set(loaders) == {"train", "val", "test"}
    assert scaler == {"mean": 1.0, "std": 0.5}

    x, y = next(iter(loaders["train"]))
    assert x.shape == (8, WINDOW, 1)
    assert y.shape == (8, 1)
    assert torch.isfinite(x).all()


def test_get_loaders_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_processed(tmp_path / "no_existe")
