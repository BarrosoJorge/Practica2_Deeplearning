import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM (PyTorch) para forecasting de 1 paso.

    Entrada: (N, window, features) · Salida: (N, 1) escalado.
    """

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
