# COMANDOS — P2_DL (chuleta de operación)

Todos los comandos se ejecutan desde la raíz del proyecto (`P2_DL/`).
Estado normal: los contenedores **apagados**; se levantan solo al usarlos.

## Atajos: `./dev.sh` (equivalente ejecutable de esta chuleta)

```bash
./dev.sh lint             # pre-commit
./dev.sh test             # las 3 pruebas (pytest)
./dev.sh train-quick      # entrenamiento EN CONTENEDOR ~2 min (demo)
./dev.sh train-container  # entrenamiento completo EN CONTENEDOR
./dev.sh train-local      # entrenamiento local (uv run + GPU host)
./dev.sh compare          # comparativa + registro del mejor modelo
./dev.sh up / down        # levantar / apagar MLflow + API
./dev.sh demo             # POST /predict con la ventana real del test set
./dev.sh commit "msg"     # gate: pre-commit + pruebas -> commit
./dev.sh push             # publicar a GitHub
./dev.sh build            # construir imágenes docker
./dev.sh clean            # liberar espacio (cache de build, sin tocar imágenes)
./dev.sh status           # contenedores, disco y docker
./dev.sh help             # ayuda completa
```

---

## 0. Estado de contenedores

```bash
docker ps                                    # qué está corriendo
docker compose --profile inference --profile dev down   # apagar TODO lo nuestro
docker compose up -d mlflow                  # levantar solo MLflow (:5000)
docker compose --profile inference up -d api # levantar MLflow + API (:8000)
```

---

## 1. Entorno y calidad de código

```bash
uv sync                                      # (re)crear .venv con Python 3.12
uv run pre-commit run --all-files            # hooks ruff/ruff-format + yaml
uv run ruff check --fix .                    # linter con auto-fix
uv run ruff format .                         # formateo
```

---

## 2. Datos (DVC)

```bash
dvc status                                   # datos procesados al día
dvc repro                                    # regenerar ingesta + preproceso
git add data/*.dvc && git commit -m "data: actualizar dataset procesado"
```

---

## 3. Pruebas (las 3 del proyecto — deben pasar para commitear)

```bash
uv run pytest -q                             # 19 tests: datos, modelos, inferencia
uv run pytest tests/test_sequence.py -q      # prueba 1: datos y preproceso
uv run pytest tests/test_models.py -q        # prueba 2: modelos y métricas
uv run pytest tests/test_inference.py -q     # prueba 3: servidor de inferencia
```

---

## 4. Entrenamiento

**Local (rápido, usa .venv con CUDA):**

```bash
docker compose up -d mlflow                  # el tracking necesita el server
uv run python scripts/train.py --config configs/train.yaml
uv run python scripts/train.py --quick       # ~2 min, 1-2 épocas por modelo (demo)
uv run python scripts/train.py --no-mlflow   # sin tracking (debug)
uv run python scripts/train.py --archs rnn   # solo una arquitectura
```

**En contenedor (infra como código — lo que pide la práctica):**

```bash
docker compose up -d mlflow
docker compose run --rm train                # entrenamiento completo en contenedor
docker compose run --rm train /env/bin/python scripts/train.py --quick   # versión demo
```

El primer `docker compose build train` descarga ~12 GB (torch cu128) — ya está
cacheado en esta máquina.

---

## 5. Comparativa y registro del mejor modelo

```bash
uv run python scripts/compare_runs.py --config configs/train.yaml
uv run python scripts/compare_runs.py --no-registry   # solo evalúa y reporta
```

Genera `artifacts/reports/compare/`: tabla (MAE/RMSE/MAPE/WAPE/sMAPE), PNG de
predicciones 72h, y registra el mejor (por val_loss) como nueva versión en el
Registry `p2dl-forecasting`.

---

## 6. Servidor de inferencia

```bash
docker compose --profile inference up -d api          # contenedor :8000
uv run uvicorn api.app:app --port 8000                # o local

curl http://localhost:8000/health                     # modelo cargado y origen
```

**Inferencia con la ventana real de la demo** (24h reales del test set):

```bash
jq -c '{series: .series}' artifacts/demo_payload.json \
  | curl -X POST http://localhost:8000/predict \
        -H 'content-type: application/json' -d @-
# → {"prediction": 2.75, "model": "p2dl-forecasting v1 (MLflow Registry)"}
# valor real siguiente: 2.59 kW (campo _meta.real_next_kw del json)
```

**O sin terminal — Swagger UI (el "front" de la demo):**
abrir `http://localhost:8000/docs` → `POST /predict` → *Try it out* → pegar el
JSON (los 24 valores de `series` de `artifacts/demo_payload.json`) → *Execute*.

**Regenerar el payload con otra ventana:**

```bash
uv run python scripts/make_demo_payload.py --index 5090   # ventana elegida
uv run python scripts/make_demo_payload.py                # por defecto: el pico
```

**Validaciones en vivo (status 422):** serie de 23 o 25 valores, valor
negativo, o NaN — la API las rechaza con detalle explicativo.

---

## 7. MLflow (UI de experimentos y Registry)

```bash
docker compose up -d mlflow                  # si está apagado
# abrir http://localhost:5000  → experimento P2_DL, runs fase3-rnn/fase3-lstm
# Registry → modelo p2dl-forecasting (v1, fuente del checkpoint de la API)
docker stop p2dl-mlflow                      # apagarlo al terminar
```

---

## 8. Git y GitHub

```bash
git status --short                           # qué cambió
git add <archivos>                           # siempre revisar ANTES de add
git commit -m "tipo: descripción corta"      # pre-commit corre en cada commit:
                                             # si las pruebas fallan, no commitea
git push origin                              # publicar a GitHub

git log --oneline -6                         # historial
git remote -v                                # github.com/BarrosoJorge/Practica2
```

Convención de mensajes en este repo: `feat:` (funcionalidad nueva),
`fix:` (corrección), `chore:` (mantenimiento), `data:` (dataset/versionado).

---

## 9. Guion de demostración para la presentación (~10 min)

```bash
# 0) preparar
docker compose --profile inference up -d api
uv run pytest -q

# 1) mostrar repo y trazabilidad
git log --oneline && dvc status

# 2) las 3 pruebas en vivo (ya corrieron en el paso 0)

# 3) infra y observabilidad
docker ps                                    # mlflow + api arriba
# abrir http://localhost:5000 (MLflow UI: runs, métricas, Registry)

# 4) inferencia (la estrella)
# abrir http://localhost:8000/docs → Swagger UI
#   /health → modelo del Registry cargado
#   /predict → pegar {"series": [24 valores de demo_payload.json]} → predicción
# comparar con _meta.real_next_kw (2.59 kW)
# payload con valor negativo o 23 valores → 422 en vivo

# 5) resultados
# abrir artifacts/reports/compare/rnn_vs_lstm.md + predicciones_72h.png
# (WAPE test ~37%; nota sobre por qué MAPE se infla)

# 6) si pide entrenar en vivo
docker compose run --rm train /env/bin/python scripts/train.py --quick

# 7) cerrar
docker compose --profile inference --profile dev down
```

---

## 10. Solución de problemas

```bash
docker compose build train api               # reconstruir imágenes
docker builder prune -f                      # liberar ~15 GB de cache de build
docker compose logs api 2>&1 | tail -20      # ver logs del servidor
docker compose run --rm train /env/bin/python -c "import torch; print(torch.cuda.is_available())"
                                             # verificar GPU en contenedor
```

- **"Invalid Host header" al hablar con MLflow**: ya resuelto con
  `--allowed-hosts` en el compose (si se toca, no quitar ese flag).
- **La API no encuentra el modelo**: requiere MLflow arriba la primera vez
  (descarga del Registry); si MLflow está caído usa el checkpoint local de
  `artifacts/checkpoints/fase3-lstm/best.pt`.
- **"no space left on device"**: `docker builder prune -f` y reintentar.
