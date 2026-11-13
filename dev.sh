#!/usr/bin/env bash
# dev.sh — chuleta ejecutable del proyecto P2_DL
# Uso: ./dev.sh <comando> [args]   (sin argumentos: muestra la ayuda)

set -euo pipefail

CONTAINER_PY="/env/bin/python"
ERR=0

banner() { echo -e "\n\033[1m==> $*\033[0m"; }

die() { echo -e "\033[31mERROR: $*\033[0m" >&2; exit 1; }

check_venv() {
  [ -d .venv ] || die "no hay .venv — corre primero: uv sync"
}

mlflow_up() {
  docker compose up -d mlflow
  sleep 4
}

cmd_lint() {
  banner "pre-commit (hooks: ruff, ruff-format, yaml, whitespace...)"
  uv run pre-commit run --all-files
}

cmd_test() {
  banner "las 3 pruebas (19 tests: datos, modelos, inferencia)"
  uv run pytest -q
}

cmd_train_local() {
  check_venv
  mlflow_up
  banner "entrenamiento LOCAL (uv run, GPU del host)"
  uv run python scripts/train.py --config configs/train.yaml "$@"
}

cmd_train_container() {
  mlflow_up
  banner "entrenamiento EN CONTENEDOR (docker, GPU visible en Docker)"
  docker compose run --rm train "$CONTAINER_PY" scripts/train.py \
    --config configs/train.yaml "$@"
}

cmd_train_quick() {
  banner "entrenamiento EN CONTENEDOR --quick (~2 min, demo)"
  mlflow_up
  docker compose run --rm train "$CONTAINER_PY" scripts/train.py --quick "$@"
}

cmd_compare() {
  mlflow_up
  banner "comparativa RNN vs LSTM + registro del mejor en el Registry"
  uv run python scripts/compare_runs.py --config configs/train.yaml "$@"
}

cmd_up() {
  banner "levantando MLflow + API (:5000 y :8000)"
  docker compose --profile inference up -d
  sleep 5
  curl -s http://localhost:8000/health | head -c 200; echo
}

cmd_down() {
  banner "apagando TODO lo de este proyecto"
  docker compose --profile inference --profile dev down
}

cmd_demo() {
  banner "POST /predict con la ventana real de la demo (idx 5090 del test)"
  curl -sf http://localhost:8000/health > /dev/null ||
    die "API no responde en :8000 — levántala con: ./dev.sh up"
  local real
  real=$(python3 -c "import json; print(json.load(open('artifacts/demo_payload.json'))['_meta']['real_next_kw'])")
  jq -c '{series: .series}' artifacts/demo_payload.json |
    curl -s -X POST http://localhost:8000/predict \
      -H 'content-type: application/json' -d @- ; echo
  echo "valor real siguiente: ${real} kW"
}

cmd_commit() {
  [ $# -ge 1 ] || die "uso: ./dev.sh commit \"mensaje del commit\""
  banner "gate de pre-commit + pruebas antes del commit"
  cmd_lint
  cmd_test
  banner "staging + commit: $1"
  git add -A
  git commit -m "$1"
  echo
  git log --oneline -1
  echo "(para publicar: ./dev.sh push)"
}

cmd_push() {
  banner "push a GitHub"
  git push origin
  git log --oneline -3
}

cmd_clean() {
  banner "limpieza segura: cache de build + capas colgadas (NO toca imágenes)"
  docker builder prune -f
  docker image prune -f
  docker system df
}

cmd_build() {
  banner "construyendo imágenes train + api (largo la primera vez, ~15 min)"
  docker compose build train api
}

cmd_status() {
  banner "estado"
  echo "-- contenedores --"
  docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep -E "p2dl|NAMES"
  echo "-- disco --"
  df -h / | tail -1
  docker system df | head -3
}

help() {
  cat <<'EOF'
P2_DL — dev.sh: atajos de operación

  ./dev.sh lint                  pre-commit sobre todo el repo (ruff, yaml...)
  ./dev.sh test                  las 3 pruebas (pytest, 19 tests)
  ./dev.sh train-quick           entrenamiento EN CONTENEDOR rápido (~2 min, demo)
  ./dev.sh train-container       entrenamiento completo EN CONTENEDOR
  ./dev.sh train-local           entrenamiento local (uv run + GPU del host)
  ./dev.sh compare               comparativa + registro del mejor modelo
  ./dev.sh up                    levantar MLflow + API (docker)
  ./dev.sh down                  apagar todo lo del proyecto
  ./dev.sh demo                  POST /predict con la ventana real del test set
  ./dev.sh commit "mensaje"      gate: pre-commit + pruebas -> git add -A + commit
  ./dev.sh push                  git push origin
  ./dev.sh build                 construir imágenes docker (train + api)
  ./dev.sh clean                 liberar espacio: cache de build + capas colgadas
  ./dev.sh status                contenedores, disco y uso de docker

Ejemplos:
  ./dev.sh train-quick
  ./dev.sh train-local --quick
  ./dev.sh compare --no-registry
  ./dev.sh commit "feat: nueva métrica"
EOF
}

case "${1:-help}" in
  lint) shift; cmd_lint ;;
  test) shift; cmd_test ;;
  train-local) shift; cmd_train_local "$@" ;;
  train-container) shift; cmd_train_container "$@" ;;
  train-quick) shift; cmd_train_quick "$@" ;;
  compare) shift; cmd_compare "$@" ;;
  up) shift; cmd_up ;;
  down) shift; cmd_down ;;
  demo) shift; cmd_demo ;;
  commit) shift; cmd_commit "$@" ;;
  push) shift; cmd_push ;;
  build) shift; cmd_build ;;
  clean) shift; cmd_clean ;;
  status) shift; cmd_status ;;
  help|-h|--help) help ;;
  *) help; die "comando desconocido: $1" ;;
esac
