from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from challenge.model import ReplenishmentModel

app = FastAPI(title="CENABAST - Replenishment API")

# El modelo se carga UNA vez al iniciar la app (via ARTIFACT_PATH.exists()
# dentro de __init__), no en cada request. Evita reentrenar/recargar
# repetidamente bajo carga concurrente (ver make stress-test).
_model = ReplenishmentModel()

_PRODUCTOS_PATH = Path(__file__).resolve().parent.parent / "dataset" / "productos.csv"
_KNOWN_GTINS = (
    set(pd.read_csv(_PRODUCTOS_PATH)["gtin"].astype(str))
    if _PRODUCTOS_PATH.exists()
    else set()
)


class ProductRequest(BaseModel):
    gtin: str
    fecha: str


class PredictRequest(BaseModel):
    products: List[ProductRequest]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    feature_rows = []

    for item in request.products:
        if item.gtin not in _KNOWN_GTINS:
            raise HTTPException(status_code=400, detail=f"gtin desconocido: {item.gtin}")

        try:
            fecha = pd.Timestamp(datetime.strptime(item.fecha, "%Y-%m-%d"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"fecha invalida: {item.fecha}")

        try:
            gtin_int = int(item.gtin)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"gtin invalido: {item.gtin}")

        try:
            feature_rows.append(_model.build_point_features(gtin_int, fecha))
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"sin historial suficiente para gtin: {item.gtin}",
            )

    features = pd.concat(feature_rows, ignore_index=True)
    return {"predict": _model.predict(features)}
