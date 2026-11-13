import copy


class EarlyStopping:
    """Detiene el entrenamiento cuando la métrica vigilada deja de mejorar.

    Vigila una métrica a minimizar (p. ej. val_loss). Guarda copia del mejor
    estado del modelo para restaurarlo al final del entrenamiento.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None

    @property
    def improved_last(self):
        return self.counter == 0

    def step(self, value, model):
        """Evalua mejora. Devuelve True si hay que parar.

        Sólo considera mejora si value < best - min_delta (evita parar en
        episodios artificialmente fléxibles con cambio mínimo).
        """
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        """Carga el mejor estado observado en el modelo."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
