import pandas as pd

from src.engines.adaptive_weight_engine import AdaptiveWeightEngine

INPUT = "data/processed/feature_store_state.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

engine = AdaptiveWeightEngine()

result = engine.predict(df.tail(20))

print("=" * 90)
print("TEST ADAPTIVE WEIGHT ENGINE CLASS")
print("=" * 90)

print(result.to_string(index=False))
