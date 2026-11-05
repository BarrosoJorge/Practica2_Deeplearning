# P2_DL — Forecasting de Series Temporales (RNN vs LSTM) + Pipeline MLOps

Predicción de la variable de **consumo eléctrico** a partir de series históricas,
comparando dos arquitecturas de redes recurrentes (**RNN** y **LSTM**), con un
pipeline MLOps completo: versionado de datos (DVC), seguimiento de experimentos
(MLflow), entrenamiento contenerizado con GPU (Docker) y servidor de inferencia
(FastAPI).

## Stack

| Tecnología  | Rol                                          |
| ----------- | -------------------------------------------- |
| Python 3.12 | Desarrollo (entorno gestionado con `uv`)      |
| PyTorch     | Implementación de RNN y LSTM                  |
| DVC         | Versionamiento de datasets y artefactos       |
| MLflow      | Tracking de experimentos y registro de modelos|
| Docker      | Contenedor de entrenamiento (GPU)             |
| FastAPI     | Servidor de inferencia (`POST /predict`)      |
| Pre-commit   | Calidad de código en cada commit             |

## Estructura

```
data/            datos crudos y procesados (versionados con DVC)
configs/         hiperparámetros de entrenamiento (YAML)
datasets/        carga + preprocesamiento + ventanas temporales
models/          arquitecturas RNN, LSTM y factory
engine/          loops de entrenamiento y métricas
callbacks/       early stopping
api/             servidor FastAPI de inferencia
tests/           pruebas de datos, modelos e inferencia
```

## Cómo correr (resumen)

Entorno local:

```bash
uv sync                  # crea .venv con Python 3.12
uv run pre-commit run --all-files
```

Entrenamiento (contenedor con GPU) y servicios:

```bash
docker compose up -d mlflow
docker compose run --build train        # entrenamiento RNN/LSTM (perfil dev)
docker compose up -d api                # servidor de inferencia :8000
```

Ver `documento_del_proyecto.md` para la propuesta completa.
