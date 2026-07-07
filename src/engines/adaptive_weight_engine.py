import json
import pandas as pd

from src.engines.base_engine import BaseEngine


class AdaptiveWeightEngine(BaseEngine):
    def __init__(
        self,
        weights_file="reports/adaptive_weight_engine_v3/learned_selective_weights.json",
    ):
        super().__init__("adaptive_weight_engine_v3_learned_selective")
        self.weights_file = weights_file
        self.data = self._load_weights()

    def _load_weights(self):
        with open(self.weights_file, "r") as f:
            return json.load(f)

    def get_global_weights(self):
        return self.data["global"]

    def get_weights_for_row(self, row):
        weights_list = [self.get_global_weights()]

        contexts = self.data.get("contexts", {})

        context_priority = [
            "restaurant_state",
            "growth_state",
            "volatility_state",
            "restaurant_temperature",
            "delivery_state",
            "cash_state",
            "market_pressure",
            "commercial_season",
            "weekday",
            "month",
        ]

        for context in context_priority:
            value = row.get(context, None)

            if value is None:
                continue

            value = str(value)

            if context in contexts and value in contexts[context]:
                context_data = contexts[context][value]

                if "weights" in context_data:
                    weights_list.append(context_data["weights"])

        averaged = self._average_many(weights_list)
        normalized = self._normalize(averaged)

        return normalized

    def _average_many(self, weights_list):
        keys = set()

        for weights in weights_list:
            keys.update(weights.keys())

        result = {}

        for key in keys:
            values = [
                weights.get(key, 0)
                for weights in weights_list
            ]

            result[key] = sum(values) / len(values)

        return result

    def _normalize(self, weights):
        total = sum(weights.values())

        if total == 0:
            return weights

        return {
            key: round(value / total, 4)
            for key, value in weights.items()
        }

    def predict(self, df):
        df = df.copy()

        rows = []

        for _, row in df.iterrows():
            weights = self.get_weights_for_row(row)

            out = {
                "date": row.get("date"),
                "restaurant_state": row.get("restaurant_state"),
                "growth_state": row.get("growth_state"),
                "volatility_state": row.get("volatility_state"),
                "restaurant_temperature": row.get("restaurant_temperature"),
                "delivery_state": row.get("delivery_state"),
                "cash_state": row.get("cash_state"),
                "market_pressure": row.get("market_pressure"),
                "commercial_season": row.get("commercial_season"),
                "weekday": row.get("weekday"),
                "month": row.get("month"),
            }

            for key, value in weights.items():
                out[f"weight_{key}"] = value

            rows.append(out)

        return pd.DataFrame(rows)
