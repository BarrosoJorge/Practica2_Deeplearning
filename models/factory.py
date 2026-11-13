from models.lstm import LSTMForecaster
from models.rnn import RNNForecaster

MODELS = {
    "rnn": RNNForecaster,
    "lstm": LSTMForecaster,
}


def create_model(name, cfg):
    """Construye un modelo por nombre ('rnn' | 'lstm') según su config."""
    if name not in MODELS:
        raise ValueError(f"Arquitectura desconocida: {name!r} (disponibles: {sorted(MODELS)})")

    model_cfg = cfg["model"]
    return MODELS[name](
        input_size=1,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )
