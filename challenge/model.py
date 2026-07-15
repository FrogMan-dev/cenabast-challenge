import pickle
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACT_PATH = BASE_DIR / "challenge" / "artifacts" / "model.pkl"

EXTENDED_LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14]
DECISION_THRESHOLD = 0.5


class ReplenishmentModel:
    def __init__(self):
        self._clf = None
        self._reg = None
        self._model = None  # alias de compatibilidad con tests
        self._feature_columns: List[str] = []
        self._history: dict = {}
        self._stock_history: dict = {}
        self._gtin_encoder: dict = {}
        self._is_fitted = False

        if ARTIFACT_PATH.exists():
            self.load(str(ARTIFACT_PATH))

    def _densify(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for gtin, g in df.groupby("gtin"):
            idx = pd.date_range(g["fecha"].min(), g["fecha"].max(), freq="D")
            g = g.set_index("fecha").reindex(idx, fill_value=0)
            g["gtin"] = gtin
            g.index.name = "fecha"
            frames.append(g.reset_index())
        return pd.concat(frames, ignore_index=True)

    def _dias_desde_ultima_venta(self, s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        tuvo_venta = (shifted > 0).astype(int)
        grupo = (tuvo_venta == 1).cumsum()
        dias = shifted.groupby(grupo).cumcount()
        dias = dias.where(tuvo_venta.cumsum() > 0, other=len(s))
        return dias

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["dia"] = df["fecha"].dt.day
        df["mes"] = df["fecha"].dt.month
        df["dia_semana"] = df["fecha"].dt.dayofweek

        for lag in EXTENDED_LAGS:
            df[f"lag_{lag}"] = df.groupby("gtin")["cantidad"].shift(lag)

        for w in ROLLING_WINDOWS:
            grouped = df.groupby("gtin")["cantidad"]
            df[f"roll_mean_{w}"] = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"roll_std_{w}"] = grouped.transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).std()
            )

        df["ewm_mean"] = df.groupby("gtin")["cantidad"].transform(
            lambda s: s.shift(1).ewm(halflife=3, min_periods=1).mean()
        )
        df["dias_desde_ultima_venta"] = df.groupby("gtin")["cantidad"].transform(
            self._dias_desde_ultima_venta
        )

        indicador_venta = (df["cantidad"] > 0).astype(int)
        df["pct_dias_con_venta_14"] = indicador_venta.groupby(df["gtin"]).transform(
            lambda s: s.shift(1).rolling(14, min_periods=1).mean()
        )

        feat_cols = [
            c for c in df.columns
            if c.startswith(("lag_", "roll_", "ewm_", "dias_desde", "pct_dias"))
        ]
        df[feat_cols] = df[feat_cols].fillna(0)
        return df

    def _merge_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        stock_path = DATASET_DIR / "stock.csv"
        if not stock_path.exists():
            df["stock"] = 0.0
            df["dias_cobertura"] = 0.0
            return df

        stock = pd.read_csv(stock_path)
        stock["fecha"] = pd.to_datetime(stock["fecha"])
        df = df.merge(stock[["gtin", "fecha", "stock"]], on=["gtin", "fecha"], how="left")
        df["stock"] = df.groupby("gtin")["stock"].ffill()
        df["stock"] = df["stock"].fillna(0.0)

        denom = df["roll_mean_7"].replace(0, np.nan)
        df["dias_cobertura"] = (df["stock"] / denom).fillna(df["stock"]).clip(upper=365)
        return df

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        df = data.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])

        if "tipo_movimiento" in df.columns:
            df = df[df["tipo_movimiento"] == "S"]

        df = df.groupby(["gtin", "fecha"], as_index=False)["cantidad"].sum()
        df = self._densify(df)
        df = df.sort_values(["gtin", "fecha"]).reset_index(drop=True)
        df = self._add_features(df)
        df = self._merge_stock(df)
        df["gtin"] = df["gtin"].astype("category")

        target = None
        if target_column:
            target = df[["cantidad"]].rename(columns={"cantidad": target_column}).copy()

        feature_cols = [c for c in df.columns if c != "cantidad"]
        features = df[feature_cols].copy()

        return (features, target) if target is not None else features

    def _encode_gtin_for_model(self, X: pd.DataFrame, fit_encoder: bool) -> pd.DataFrame:
        """
        HistGradientBoosting espera que las columnas categoricas contengan
        codigos ordinales pequenos (< 255). Los GTIN son numeros de 13
        digitos, muy por encima de ese limite, asi que se remapean aqui a
        codigos 0..n-1 justo antes de entrenar/predecir. El resto del
        sistema (historial, API, tests) sigue usando el GTIN real.
        """
        X = X.copy()
        if fit_encoder:
            unique_gtins = sorted(X["gtin"].unique())
            self._gtin_encoder = {g: i for i, g in enumerate(unique_gtins)}

        X["gtin"] = X["gtin"].map(self._gtin_encoder)
        # gtin no visto en entrenamiento (edge case): cae en codigo 0 en vez
        # de romper la prediccion.
        X["gtin"] = X["gtin"].fillna(0).astype(int)
        return X

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X = features.drop(columns=["fecha"])
        y = target.values.ravel()

        X_encoded = self._encode_gtin_for_model(X, fit_encoder=True)
        self._feature_columns = list(X.columns)
        cat_idx = [X_encoded.columns.get_loc("gtin")]

        self._clf = HistGradientBoostingClassifier(
            categorical_features=cat_idx,
            random_state=42,
            max_iter=300,
            learning_rate=0.05,
        )
        self._clf.fit(X_encoded, (y > 0).astype(int))

        mask_nz = y > 0
        self._reg = HistGradientBoostingRegressor(
            categorical_features=cat_idx,
            random_state=42,
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=10,
            l2_regularization=0.3,
            loss="poisson",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        self._reg.fit(X_encoded[mask_nz], y[mask_nz])

        self._model = self._reg
        self._is_fitted = True

        hist_df = features[["gtin", "fecha"]].copy()
        hist_df["cantidad"] = y
        self._history = {
            gtin: g.set_index("fecha")["cantidad"].sort_index()
            for gtin, g in hist_df.groupby("gtin")
        }

        stock_path = DATASET_DIR / "stock.csv"
        if stock_path.exists():
            stock_raw = pd.read_csv(stock_path)
            stock_raw["fecha"] = pd.to_datetime(stock_raw["fecha"])
            self._stock_history = {
                gtin: g.set_index("fecha")["stock"].sort_index()
                for gtin, g in stock_raw.groupby("gtin")
            }
        else:
            self._stock_history = {}

    def predict(self, features: pd.DataFrame) -> List[dict]:
        if not self._is_fitted:
            raise RuntimeError(
                "El modelo no esta entrenado. Llama fit() o carga un "
                "artefacto con load() antes de predecir."
            )

        X = features[self._feature_columns]
        X_encoded = self._encode_gtin_for_model(X, fit_encoder=False)

        prob_venta = self._clf.predict_proba(X_encoded)[:, 1]
        magnitud = np.clip(self._reg.predict(X_encoded), a_min=0, a_max=None)
        preds = np.where(prob_venta > DECISION_THRESHOLD, magnitud, 0.0)

        result = []
        for i, pred in enumerate(preds):
            result.append({
                "fecha": str(features.iloc[i]["fecha"].date()),
                "cantidad": float(pred)
            })
        return result

    def build_point_features(self, gtin, fecha: pd.Timestamp) -> pd.DataFrame:
        if gtin not in self._history:
            raise KeyError(f"gtin desconocido para el modelo: {gtin}")

        serie = self._history[gtin]

        def value_at(offset_days: int) -> float:
            return float(serie.get(fecha - pd.Timedelta(days=offset_days), 0.0))

        row = {
            "gtin": gtin,
            "fecha": fecha,
            "dia": fecha.day,
            "mes": fecha.month,
            "dia_semana": fecha.dayofweek,
        }
        for lag in EXTENDED_LAGS:
            row[f"lag_{lag}"] = value_at(lag)
        for w in ROLLING_WINDOWS:
            vals = [value_at(d) for d in range(1, w + 1)]
            row[f"roll_mean_{w}"] = float(np.mean(vals)) if vals else 0.0
            row[f"roll_std_{w}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        ventana_ewm = [value_at(d) for d in range(1, 15)]
        pesos = [0.5 ** (d / 3) for d in range(1, 15)]
        row["ewm_mean"] = (
            float(np.average(ventana_ewm, weights=pesos)) if ventana_ewm else 0.0
        )

        dias_transcurridos = 0
        for d in range(1, 366):
            if value_at(d) > 0:
                break
            dias_transcurridos += 1
        row["dias_desde_ultima_venta"] = float(dias_transcurridos)

        ventana_venta = [1.0 if value_at(d) > 0 else 0.0 for d in range(1, 15)]
        row["pct_dias_con_venta_14"] = (
            float(np.mean(ventana_venta)) if ventana_venta else 0.0
        )

        stock_serie = self._stock_history.get(gtin)
        if stock_serie is not None and len(stock_serie) > 0:
            stock_hasta_fecha = stock_serie[stock_serie.index <= fecha]
            stock_val = float(stock_hasta_fecha.iloc[-1]) if len(stock_hasta_fecha) else 0.0
        else:
            stock_val = 0.0
        row["stock"] = stock_val
        row["dias_cobertura"] = min(
            stock_val / row["roll_mean_7"] if row["roll_mean_7"] > 0 else stock_val, 365
        )

        return pd.DataFrame([row])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self._clf,
                "reg": self._reg,
                "feature_columns": self._feature_columns,
                "history": self._history,
                "stock_history": self._stock_history,
                "gtin_encoder": self._gtin_encoder,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._clf = state["clf"]
        self._reg = state["reg"]
        self._model = self._reg
        self._feature_columns = state["feature_columns"]
        self._history = state["history"]
        self._stock_history = state.get("stock_history", {})
        self._gtin_encoder = state.get("gtin_encoder", {})
        self._is_fitted = True
