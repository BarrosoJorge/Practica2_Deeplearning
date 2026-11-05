"""EDA del dataset UCI Household Electric Power (Fase 1).

Genera perfil estadístico + plots + resumen en markdown, y registra el
paso de ingesta en MLflow (run: fase1-data-ingesta).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.household import download_household, load_household

REPORT_DIR = Path("artifacts/reports/eda_household")
RUN_NAME = "fase1-data-ingesta"


def perfil(fechas, valores):
    total = len(valores)
    n_missing = int(np.isnan(valores).sum())

    gaps = np.diff(fechas.values).astype("timedelta64[m]").astype(np.int64)
    n_gaps = int((gaps > 1).sum())
    gap_minutes = gaps[gaps > 1]

    stats = {
        "n_filas": total,
        "n_missing": n_missing,
        "pct_missing": 100.0 * n_missing / total,
        "fecha_ini": fechas[0],
        "fecha_fin": fechas[-1],
        "n_gaps": n_gaps,
        "gap_max_h": (gap_minutes.max() / 60.0) if gap_minutes.size else 0.0,
        "media": float(np.nanmean(valores)),
        "mediana": float(np.nanmedian(valores)),
        "p95": float(np.nanpercentile(valores, 95)),
        "max": float(np.nanmax(valores)),
    }
    return stats, (
        fechas,
        pd.Series(valores, index=fechas),
        gaps,
    )


def escribir_resumen(md_path, stats, serie_h, resample="1h"):
    media_hora = serie_h.groupby(serie_h.index.hour).mean()
    media_dia = serie_h.groupby(serie_h.index.dayofweek).mean()
    hora_pico = media_hora.idxmax()
    dia_pico = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][media_dia.idxmax()]

    contenido = [
        "# EDA — UCI Household Electric Power",
        "",
        "| Métrica | Valor |",
        "| :--- | :--- |",
        f"| Filas (1 min) | {stats['n_filas']:,} |",
        f"| Rango temporal | {stats['fecha_ini']} → {stats['fecha_fin']} |",
        f"| Faltantes (?→NaN) | {stats['n_missing']:,} ({stats['pct_missing']:.2f}%) |",
        f"| Huecos >1 min | {stats['n_gaps']:,} (máx {stats['gap_max_h']:.1f} h) |",
        f"| Media/Mediana kW | {stats['media']:.3f} / {stats['mediana']:.3f} |",
        f"| P95/Max kW | {stats['p95']:.3f} / {stats['max']:.3f} |",
        "",
        f"Hora pico de consumo: **{hora_pico}:00 h** · Día pico: **{dia_pico}**",
        "",
        "## decisiones para Fase 2",
        "Target: Global_active_power (kW) · resampleo a 1 h (media) ·",
        "huecos: interpolación temporal + ffill · splits train/val/test 70/15/15",
    ]

    md_path.write_text("\n".join(contenido), encoding="utf-8")
    return media_hora, media_dia


def guardar_plots(
    report_dir,
    serie_min,
    media_hora,
    media_dia,
):
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primera_semana = serie_min.dropna().index[0]
    semana = serie_min.loc[primera_semana : primera_semana + pd.Timedelta(days=7)]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(semana.index, semana.values, lw=0.6)
    ax.set_title("Global_active_power — primera semana (min)")
    fig.tight_layout()
    fig.savefig(report_dir / "1_semana.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(media_hora.index, media_hora.values)
    ax.set_title("Consumo medio por hora del día (todas las muestras)")
    ax.set_xlabel("Hora")
    fig.tight_layout()
    fig.savefig(report_dir / "2_hora_dia.png", dpi=130)
    plt.close(fig)

    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(dias, media_dia.values)
    ax.set_title("Consumo medio por día de la semana (todas las muestras)")
    fig.tight_layout()
    fig.savefig(report_dir / "3_dia_semana.png", dpi=130)
    plt.close(fig)

    valores_validos = serie_min.dropna().values
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(valores_validos, bins=80)
    ax.set_title("Distribución de Global_active_power (kW)")
    fig.tight_layout()
    fig.savefig(report_dir / "4_distribucion.png", dpi=130)
    plt.close(fig)


def registrar_mlflow(stats, report_dir):
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params(
            {
                "fuente": "UCI id 235",
                "periodo": f"{stats['fecha_ini']} → {stats['fecha_fin']}",
                "freq_raw": "1min",
                "target": "Global_active_power",
            }
        )
        mlflow.log_metrics(
            {
                "n_filas": stats["n_filas"],
                "pct_missing": stats["pct_missing"],
                "n_gaps": stats["n_gaps"],
                "kW_media": stats["media"],
            }
        )
        mlflow.log_artifact(str(report_dir / "resumen.md"), artifact_path="eda")
        for png in report_dir.glob("*.png"):
            mlflow.log_artifact(str(png), artifact_path="eda")


def main():
    txt_path = download_household("data/raw")

    print("Cargando 2M filas...")
    fechas, valores = load_household(txt_path)

    stats, (fechas, serie_min, _gaps) = perfil(fechas, valores)
    serie_h = serie_min.resample("1h").mean()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    media_hora, media_dia = escribir_resumen(REPORT_DIR / "resumen.md", stats, serie_h)
    guardar_plots(REPORT_DIR, serie_min, media_hora, media_dia)

    print(
        f"filas={stats['n_filas']:,} · missing={stats['pct_missing']:.2f}% · "
        f"gaps>1min={stats['n_gaps']:,} (máx {stats['gap_max_h']:.1f}h)"
    )
    print(f"Reportes en {REPORT_DIR}")

    try:
        registrar_mlflow(stats, REPORT_DIR)
        print("Run 'fase1-data-ingesta' registrado en http://localhost:5000")
    except Exception as exc:
        print(f"MLflow no disponible ({exc}); reportes guardados igualmente.")


if __name__ == "__main__":
    main()
