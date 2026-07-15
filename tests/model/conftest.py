import sys
import traceback

import pandas as pd
import pytest

from challenge.model import ARTIFACT_PATH, ReplenishmentModel


@pytest.fixture(scope="session", autouse=True)
def train_artifact_before_tests():
    print("\n[conftest] Iniciando entrenamiento del artefacto...", file=sys.stderr)
    try:
        data = pd.read_csv("dataset/movimientos.csv")
        model = ReplenishmentModel()
        features, target = model.preprocess(data=data, target_column="cantidad")
        model.fit(features=features, target=target)

        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ARTIFACT_PATH))

        print(f"[conftest] Artefacto guardado en {ARTIFACT_PATH}", file=sys.stderr)
        print(f"[conftest] ¿Existe? {ARTIFACT_PATH.exists()}", file=sys.stderr)
    except Exception:
        print("[conftest] ERROR AL ENTRENAR:", file=sys.stderr)
        traceback.print_exc()
        raise

    yield
