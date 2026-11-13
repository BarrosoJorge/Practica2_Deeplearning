# P2_DL — Forecasting de Series Temporales (RNN vs LSTM) + Pipeline MLOps

Predicción de la variable de **consumo eléctrico** a partir de series históricas,
comparando dos arquitecturas de redes recurrentes (**RNN** y **LSTM**), con un
pipeline MLOps completo: versionado de datos (DVC), seguimiento de experimentos
(MLflow), entrenamiento contenerizado con GPU (Docker) y servidor de inferencia
(FastAPI).

- **Tarea**: dado el consumo horario de las últimas **24 h**, predecir la hora
  siguiente (`window=24 → t+1`).
- **Dataset**: [Individual Household Electric Power Consumption (UCI)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
  re-muestreado a frecuencia horaria.
- **Selección de modelo**: por `val_loss`; el ganador se registra en el
  **MLflow Model Registry** (`p2dl-forecasting`) y la API lo descarga de ahí.

## Stack

| Tecnología  | Rol                                           |
| ----------- | --------------------------------------------- |
| Python 3.12 | Desarrollo (entorno gestionado con `uv`)       |
| PyTorch     | Implementación de RNN y LSTM                   |
| DVC         | Versionamiento de datasets y artefactos        |
| MLflow      | Tracking de experimentos y registro de modelos |
| Docker      | Contenedorización de entrenamiento e inferencia|
| FastAPI     | Servidor de inferencia (`POST /predict`)       |
| Pre-commit   | Calidad de código en cada commit              |

## Estructura

```
data/            datos crudos y procesados (versionados con DVC)
configs/         hiperparámetros de entrenamiento (YAML)
datasets/        carga + preprocesamiento + ventanas temporales
models/          arquitecturas RNN, LSTM y factory
engine/          loops de entrenamiento, métricas (MAE/RMSE/MAPE/WAPE/sMAPE)
callbacks/       early stopping
scripts/         train.py y compare_runs.py (drivers CLI)
api/             servidor FastAPI de inferencia
docker/          Dockerfile.train y Dockerfile.api
tests/           las 3 pruebas del proyecto (datos, modelos, inferencia)
artifacts/       checkpoints y reportes (checkpoints ignorados por Git)
```

## Cómo correr

### 1. Entorno y calidad de código

```bash
uv sync                              # crea .venv con Python 3.12
uv run pre-commit run --all-files    # hooks ruff/ruff-format + checks yaml
```

### 2. Pipeline de datos (DVC)

```bash
dvc repro                            # ingesta + preproceso (scaler + splits)
```

### 3. Entrenamiento

Local (con GPU CUDA disponible):

```bash
docker compose up -d mlflow          # servidor de tracking :5000
uv run python scripts/train.py --config configs/train.yaml
```

Flags útiles:

```bash
uv run python scripts/train.py --quick      # 1 época por modelo (smoke test)
uv run python scripts/train.py --no-mlflow  # sin tracking (debug)
```

En contenedor:

```bash
docker compose run --build train     # perfil "dev", MLflow en la red de compose
```

### 4. Comparativa y registro del mejor modelo

```bash
uv run python scripts/compare_runs.py --config configs/train.yaml
```

Genera `artifacts/reports/compare/rnn_vs_lstm.md` (tabla con
MAE/RMSE/MAPE/WAPE/sMAPE + curvas 72 h), el PNG comparativo y registra el mejor
modelo (por `val_loss`) como versión nueva en el registry `p2dl-forecasting`.
Con `--no-registry` solo evalúa y genera reportes.

### 5. Servidor de inferencia

```bash
docker compose --profile inference up -d api    # :8000
# o local: uv run uvicorn api.app:app --port 8000
```

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H "content-type: application/json" \
     -d '{"series": [0.8, 0.9, 0.85, ... 24 valores horarios en kW ...]}'
# → {"prediction": 0.47, "model": "p2dl-forecasting v1 (MLflow Registry)"}
```

El servidor baja el checkpoint del **MLflow Model Registry** (cache en
`artifacts/registry/`); si MLflow no está disponible usa el checkpoint local
más reciente de `artifacts/checkpoints/fase3-lstm/best.pt`.

### 6. Pruebas (3 suites del proyecto)

```bash
uv run pytest -q
# 19 tests: test_sequence.py (datos/preproceso),
#           test_models.py (forward/métricas),
#           test_inference.py (API: /health, /predict, validaciones 422)
```

## Resultados (resumen)

| Modelo | MAE test (kW) | RMSE test (kW) | WAPE test (%) |
| ------ | ------------- | -------------- | ------------- |
| fase3-lstm | 0.355 | 0.491 | 37.2 |
| fase3-rnn  | 0.351 | 0.495 | 36.8 |

Seleccionado **fase3-lstm** (mejor `val_loss`) → `p2dl-forecasting` v1.
Detalle completo en `artifacts/reports/compare/rnn_vs_lstm.md`; la nota sobre
por qué el MAPE (51-54%) infla el error y WAPE es la referencia honesta está
incluida en el reporte.

Ver `documento_del_proyecto.md` para la propuesta completa.
