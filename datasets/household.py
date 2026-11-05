import csv
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


def download_household(data_dir):
    """Descarga el dataset de consumo eléctrico (UCI id 235) a data_dir.

    El zip original (~20MB) contiene household_power_consumption.txt,
    con muestreo de 1 minuto entre 2006-12 y 2010-11.

    Es idempotente: si el archivo ya está descargado, no vuelve a descargarlo.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_path = data_dir / "household_power_consumption.txt"
    if txt_path.exists():
        return txt_path

    zip_path = data_dir / "household_power_consumption.zip"

    if not zip_path.exists():
        print("Downloading UCI Household Power...")
        urllib.request.urlretrieve(
            DATA_URL,
            zip_path,
        )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zip_file:
        zip_file.extractall(data_dir)

    return txt_path


def load_household(txt_path):
    """Carga la serie de consumo eléctrico (Global_active_power, kW).

    Devuelve (fechas, valores), donde:
      - fechas: np.datetime64[m] con dia+hora de cada muestra
      - valores: np.float32, con NaN donde el dataset usa '?' (valor faltante)
    """
    fechas = []
    valores = []

    with open(
        txt_path,
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            fechas.append(
                pd.to_datetime(
                    row["Date"] + " " + row["Time"],
                    format="%d/%m/%Y %H:%M:%S",
                )
            )
            raw = row["Global_active_power"]
            valores.append(np.nan if raw == "?" else float(raw))

    fechas = pd.DatetimeIndex(fechas)
    valores = np.asarray(
        valores,
        dtype=np.float32,
    )

    return fechas, valores
