"""
Script para entrenar el modelo y persistirlo en disco.
Usage:
    python -m challenge.train
"""
from pathlib import Path

import pandas as pd

from challenge.model import ARTIFACT_PATH, ReplenishmentModel


def main():
    model = ReplenishmentModel()

    movimientos = pd.read_csv("dataset/movimientos.csv")
    features, target = model.preprocess(data=movimientos, target_column="cantidad")
    model.fit(features=features, target=target)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ARTIFACT_PATH))
    print(f"Modelo entrenado y guardado en {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
