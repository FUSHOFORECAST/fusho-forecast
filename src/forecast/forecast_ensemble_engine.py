import os
import pandas as pd

from src.engines.confidence_engine import ConfidenceEngine
from src.engines.adaptive_weight_engine import AdaptiveWeightEngine


class ForecastEnsembleEngine:
    def __init__(
        self,
        feature_store="data/processed/feature_store_state.csv",
        output_file="data/processed/forecast_ensemble_v3.csv",
    ):
        self.feature_store = feature_store
        self.output_file = output_file
        self.confidence_engine = ConfidenceEngine()
        self.weight_engine = AdaptiveWeightEngine()

    def _candidate_forecasts(self, df, row):
        date = row["date"]

        memory_short = row["total_rolling_7"]
        memory_medium = row["total_rolling_30"]
        memory_long = row["total_rolling_90"]

        adaptive_memory = (
            memory_short * 0.35
            + memory_medium * 0.35
            + memory_long * 0.20
            + row["total_rolling_365"] * 0.10
        )

        past = df[df["date"] < date]

        state_hist = past[past["restaurant_state"] == row["restaurant_state"]]
        if len(state_hist) >= 5:
            state_forecast = state_hist["total"].median()
        else:
            state_forecast = memory_medium

        calendar_hist = past[
            (past["month"] == row["month"])
            & (past["dayofweek"] == row["dayofweek"])
        ]

        if len(calendar_hist) >= 3:
            calendar_forecast = calendar_hist["total"].median()
        else:
            calendar_forecast = memory_medium

        similarity_forecast = (
            state_forecast * 0.50
            + calendar_forecast * 0.30
            + memory_medium * 0.20
        )

        return {
            "memory_short": memory_short,
            "memory_medium": memory_medium,
            "memory_long": memory_long,
            "adaptive_memory": adaptive_memory,
            "state_forecast": state_forecast,
            "calendar_forecast": calendar_forecast,
            "similarity_forecast": similarity_forecast,
        }

    def predict_backtest(self, start_date="2026-06-01", end_date="2026-06-14"):
        df = pd.read_csv(self.feature_store)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df = self.confidence_engine.predict(df)

        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        test = df[mask].copy()

        rows = []

        for _, row in test.iterrows():
            candidates = self._candidate_forecasts(df, row)
            weights = self.weight_engine.get_weights_for_row(row)

            confidence = row["forecast_confidence"]

            weighted_sum = 0
            weight_total = 0

            out = {
                "date": row["date"],
                "real_total": row["total"],
                "confidence_pct": row["forecast_confidence_pct"],
                "confidence_label": row["forecast_confidence_label"],
                "restaurant_state": row["restaurant_state"],
                "growth_state": row.get("growth_state", ""),
                "volatility_state": row.get("volatility_state", ""),
                "restaurant_temperature": row.get("restaurant_temperature", ""),
                "delivery_state": row.get("delivery_state", ""),
                "cash_state": row.get("cash_state", ""),
                "market_pressure": row.get("market_pressure", ""),
                "commercial_season": row["commercial_season"],
            }

            for name, pred in candidates.items():
                weight = weights.get(name, 0)
                final_weight = weight * confidence

                weighted_sum += pred * final_weight
                weight_total += final_weight

                out[name] = round(float(pred), 2)
                out[f"weight_{name}"] = round(float(weight), 4)

            forecast = (
                weighted_sum / weight_total
                if weight_total
                else row["total_rolling_30"]
            )

            out["ensemble_forecast"] = round(float(forecast), 2)
            out["error"] = round(float(forecast - row["total"]), 2)
            out["abs_error"] = round(float(abs(forecast - row["total"])), 2)
            out["pct_error"] = round(
                float(abs(forecast - row["total"]) / row["total"] * 100),
                2,
            )

            rows.append(out)

        result = pd.DataFrame(rows)

        os.makedirs("data/processed", exist_ok=True)
        result.to_csv(self.output_file, index=False)

        return result
