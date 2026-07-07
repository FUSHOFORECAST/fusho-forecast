import numpy as np

from src.forecast_candidates.base_candidate import BaseCandidate


class StateForecastCandidate(BaseCandidate):
    name = "state_forecast"

    def build(self, df, target="total"):
        values = []

        for _, row in df.iterrows():
            hist = df[
                (df["date"] < row["date"])
                & (df["restaurant_state"] == row["restaurant_state"])
            ].tail(20)

            if len(hist) < 3:
                values.append(row.get("memory_medium", np.nan))
            else:
                values.append(hist[target].mean())

        return values


class CalendarForecastCandidate(BaseCandidate):
    name = "calendar_forecast"

    def build(self, df, target="total"):
        values = []

        for _, row in df.iterrows():
            hist = df[
                (df["date"] < row["date"])
                & (df["dayofweek"] == row["dayofweek"])
                & (df["commercial_season"] == row["commercial_season"])
            ].tail(20)

            if len(hist) < 3:
                values.append(row.get("memory_medium", np.nan))
            else:
                values.append(hist[target].mean())

        return values


class SimilarityForecastCandidate(BaseCandidate):
    name = "similarity_forecast"

    def build(self, df, target="total"):
        values = []

        for _, row in df.iterrows():
            tmp = df[df["date"] < row["date"]].copy()

            if len(tmp) < 30:
                values.append(row.get("memory_medium", np.nan))
                continue

            tmp["distance"] = (
                (tmp["business_momentum_30_90"] - row["business_momentum_30_90"]).abs()
                + (tmp["delivery_strength"] - row["delivery_strength"]).abs()
                + (tmp["restaurant_health_index"] - row["restaurant_health_index"]).abs() / 100
            )

            best = tmp.nsmallest(10, "distance")
            values.append(best[target].mean())

        return values
