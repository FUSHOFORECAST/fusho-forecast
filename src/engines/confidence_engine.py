import numpy as np
import pandas as pd

from src.engines.base_engine import BaseEngine


class ConfidenceEngine(BaseEngine):
    def __init__(self):
        super().__init__("confidence_engine_v1")

        self.pressure_score = {
            "LOW_PRESSURE": 1.00,
            "MEDIUM_PRESSURE": 0.75,
            "HIGH_PRESSURE": 0.45,
        }

        self.weights = {
            "conf_business": 0.20,
            "conf_volatility": 0.25,
            "conf_momentum": 0.20,
            "conf_health": 0.20,
            "conf_pressure": 0.15,
        }

    def predict(self, df):
        df = df.copy()

        df["conf_business"] = (
            1 - np.clip(np.abs(df["business_acceleration"]) / 0.15, 0, 1)
        )

        df["conf_volatility"] = (
            1 - np.clip((df["volatility_ratio_14_60"] - 0.5) / 1.5, 0, 1)
        )

        df["conf_momentum"] = np.clip(
            1 - np.abs(df["business_momentum_30_90"] - 1),
            0,
            1,
        )

        df["conf_health"] = np.clip(
            df["restaurant_health_index"] / 100,
            0,
            1,
        )

        df["conf_pressure"] = (
            df["market_pressure"]
            .map(self.pressure_score)
            .fillna(0.80)
        )

        df["forecast_confidence"] = sum(
            df[col] * weight
            for col, weight in self.weights.items()
        )

        df["forecast_confidence_pct"] = (
            df["forecast_confidence"] * 100
        ).round(2)

        df["forecast_confidence_label"] = df["forecast_confidence_pct"].apply(
            self._label
        )

        return df

    def _label(self, x):
        if x >= 90:
            return "VERY_HIGH"
        if x >= 80:
            return "HIGH"
        if x >= 70:
            return "MEDIUM"
        if x >= 60:
            return "LOW"
        return "VERY_LOW"
