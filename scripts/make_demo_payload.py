"""Genera artifacts/demo_payload.json para demostrar POST /predict en vivo.

Extrae una ventana real del split de test (convertida a kW con el scaler),
eligiendo la cuyo valor siguiente es el pico de consumo — la predicción más
interesante para mostrar. El archivo incluye el valor real para comparar.
"""

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "demo_payload.json"))
    parser.add_argument("--test-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument(
        "--index",
        default=None,
        help="posición del valor siguiente en test; por defecto el pico",
    )
    args = parser.parse_args()

    scaler = json.loads((pathlib.Path(args.test_dir) / "scaler.json").read_text())
    mean, std, window = scaler["mean"], scaler["std"], scaler["window"]

    test = np.load(pathlib.Path(args.test_dir) / "test.npy")  # z-score, serie 1-D
    y_true = test * std + mean  # kW reales por hora

    # ventana cuyo siguiente valor es el pico de consumo del test
    if args.index is not None:
        idx = int(args.index)
    else:
        idx = int(np.argmax(y_true[window:])) + window
    series_kw = [round(float(v), 4) for v in (test[idx - window : idx] * std + mean)]
    real_next_kw = round(float(y_true[idx]), 4)

    payload = {
        "series": series_kw,
        "_meta": {
            "origen": f"data/processed/test.npy (posicion {idx - window})",
            "real_next_kw": real_next_kw,
            "nota": "envia solo 'series' al endpoint; real_next_kw es para comparar",
        },
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"payload => {out}")
    print(f"  ventana t-{window}..t-1 (kW): {series_kw[:3]} ... {series_kw[-3:]}")
    print(f"  valor real siguiente: {real_next_kw} kW")


if __name__ == "__main__":
    main()
