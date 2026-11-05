from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

WINDOW = 24


class SequenceDataset(Dataset):
    """Ventanas deslizantes sobre una serie escalada.

    X = últimos `window` pasos, y = paso t+1.
    shapes: X (window, 1) · y (1,)
    """

    def __init__(
        self,
        series,
        window=WINDOW,
    ):
        self.series = torch.as_tensor(
            series,
            dtype=torch.float32,
        )
        self.window = window

    def __len__(self):
        return len(self.series) - self.window

    def __getitem__(self, index):
        x = self.series[index : index + self.window]
        y = self.series[index + self.window]

        # (window,) -> (window, 1)
        x = x.unsqueeze(-1)
        # escalar -> (1,)
        y = y.unsqueeze(0)

        return x, y


def load_processed(data_dir):
    """Carga los splits horarios guardados por el driver de preproceso."""
    data_dir = Path(data_dir)
    return {
        "train": np.load(data_dir / "train.npy"),
        "val": np.load(data_dir / "val.npy"),
        "test": np.load(data_dir / "test.npy"),
    }


def get_loaders(
    data_dir,
    batch_size=64,
    window=WINDOW,
):
    """DataLoaders de train/val/test a partir de los .npy versionados."""
    data_dir = Path(data_dir)
    splits = load_processed(data_dir)

    loaders = {}
    for name, values in splits.items():
        dataset = SequenceDataset(
            values,
            window,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    scaler = None
    scaler_path = data_dir / "scaler.json"
    if scaler_path.exists():
        import json

        scaler = json.loads(scaler_path.read_text())

    return loaders, scaler
