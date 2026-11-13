"""Resolución del checkpoint del modelo para el servidor de inferencia.

Orden de preferencia:
  1. Model Registry de MLflow (última versión del modelo seleccionado).
  2. Checkpoint local especificado (artifacts/checkpoints/fase3-*/best.pt).
"""

import pathlib

import mlflow
from mlflow.tracking import MlflowClient

CKPT_NAME = "best.pt"


def resolve_checkpoint(tracking_uri, registry_name, cache_dir, fallback=None):
    """Devuelve (ruta_a_best.pt, descripción_origen).

    Descarga la última versión del registry a cache_dir y la reutiliza si ya
    existe (idempotente). Si no hay registry alcanzable, usa `fallback`.
    """
    cache_dir = pathlib.Path(cache_dir)

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        versions = list(client.search_model_versions(f"name='{registry_name}'"))
        if not versions:
            raise LookupError("registry vacío")

        latest = max(versions, key=lambda v: int(v.version))
        dst = cache_dir / f"v{latest.version}"

        candidates = list(dst.rglob(CKPT_NAME))
        if not candidates:
            uri = client.get_model_version_download_uri(
                registry_name,
                str(latest.version),
            )
            mlflow.artifacts.download_artifacts(
                artifact_uri=uri,
                dst_path=str(dst),
                tracking_uri=tracking_uri,
            )
            candidates = list(dst.rglob(CKPT_NAME))
            if not candidates:
                raise FileNotFoundError(f"{CKPT_NAME} no está en el artefacto")

        return candidates[0], f"{registry_name} v{latest.version} (MLflow Registry)"
    except Exception as exc:
        print(f"[model_loader] Registry no disponible ({exc}); usando fallback")

    if fallback and pathlib.Path(fallback).exists():
        return pathlib.Path(fallback), f"{fallback} (checkpoint local)"

    raise FileNotFoundError(
        f"No hay modelo: ni MLflow Registry alcanzable ni checkpoint local ({fallback})"
    )
