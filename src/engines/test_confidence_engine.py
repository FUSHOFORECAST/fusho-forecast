import pandas as pd

from src.engines.confidence_engine import ConfidenceEngine

INPUT = "data/processed/feature_store_state.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

engine = ConfidenceEngine()

result = engine.predict(df)

print("=" * 90)
print("TEST CONFIDENCE ENGINE CLASS")
print("=" * 90)

print(
    result[
        [
            "date",
            "total",
            "forecast_confidence_pct",
            "forecast_confidence_label",
            "business_momentum_30_90",
            "volatility_ratio_14_60",
            "restaurant_health_index",
            "market_pressure",
        ]
    ].tail(20).to_string(index=False)
)
