import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from src.forecast_candidates.base_candidate import BaseCandidate


class ExternalContextCandidate(BaseCandidate):
    name = "external_context_forecast"

    def build(self, df, target="total"):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        base_col = "memory_short"

        if base_col not in df.columns:
            df[base_col] = df[target].shift(1).rolling(7).mean()

        numeric_features = [
            base_col,
            "month",
            "dayofweek",
            "dayofyear",
            "week_of_month",
            "business_momentum_7_30",
            "business_momentum_30_90",
            "business_acceleration",
            "volatility_ratio_14_60",
            "restaurant_health_index",
            "is_public_holiday",
            "is_preholiday",
            "is_postholiday",
            "is_bridge_candidate",
            "is_summer_period",
            "is_christmas_period",
            "days_to_next_holiday",
            "days_from_previous_holiday",
            "external_event_count",
            "external_positive_strength",
            "external_negative_strength",
            "external_mixed_strength",
            "external_total_strength",
            "event_channel_delivery_strength",
            "event_channel_total_strength",
        ]

        categorical_features = [
            "weekday",
            "commercial_season",
            "growth_state",
            "volatility_state",
            "restaurant_temperature",
            "delivery_state",
            "cash_state",
            "market_pressure",
            "restaurant_state",
        ]

        available_numeric = [c for c in numeric_features if c in df.columns]
        available_categorical = [c for c in categorical_features if c in df.columns]

        work = df.dropna(subset=[target, base_col]).copy()
        work["correction_target"] = work[target] - work[base_col]

        if len(work) < 220:
            return df[base_col]

        x_num = work[available_numeric].fillna(0)

        x_cat = pd.get_dummies(
            work[available_categorical].fillna("UNKNOWN"),
            prefix=available_categorical,
        ).astype(int)

        X = pd.concat([x_num, x_cat], axis=1)
        y = work["correction_target"]

        predictions = pd.Series(index=df.index, dtype=float)

        split_start = 220

        for i in range(len(work)):
            if i < split_start:
                original_idx = work.index[i]
                predictions.loc[original_idx] = work.loc[original_idx, base_col]
                continue

            train_start = max(0, i - 365)
            train_idx = work.index[train_start:i]
            test_idx = work.index[i]

            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test = X.loc[[test_idx]]

            model = ExtraTreesRegressor(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=8,
            )

            model.fit(X_train, y_train)

            correction = model.predict(X_test)[0]
            base_value = work.loc[test_idx, base_col]

            predictions.loc[test_idx] = base_value + correction

        predictions = predictions.reindex(df.index)
        predictions = predictions.fillna(df[base_col])

        return predictions
