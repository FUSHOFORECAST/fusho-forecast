import os
import pandas as pd

from src.forecast_candidates.registry import CANDIDATE_REGISTRY, get_candidate_names

INPUT = "data/processed/feature_store_state.csv"
OUTPUT_DIR = "reports/adaptive_weight_engine"

TARGETS = [
    "total",
    "delivery",
    "digital",
    "cash",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_for_target(base_df, target):
    df = base_df.copy()

    print("\n" + "=" * 90)
    print(f"BUILDING FORECAST CANDIDATES — TARGET: {target.upper()}")
    print("=" * 90)

    if target not in df.columns:
        raise ValueError(f"Target non trovato nel dataset: {target}")

    for candidate in CANDIDATE_REGISTRY:
        df[candidate.name] = candidate.build(df, target=target)
        print("Built:", candidate.name)

    candidates = get_candidate_names()

    for candidate in candidates:
        df[f"{candidate}_error"] = (df[candidate] - df[target]).abs()
        df[f"{candidate}_ape"] = df[f"{candidate}_error"] / df[target] * 100

    output_file = f"{OUTPUT_DIR}/forecast_candidates_backtest_{target}.csv"
    df.to_csv(output_file, index=False)

    print("\nCANDIDATE MAPE")
    for candidate in candidates:
        print(f"{candidate:25}", round(df[f"{candidate}_ape"].mean(), 2))

    print("\nSalvato:")
    print(output_file)

    return output_file


def main():
    base_df = pd.read_csv(INPUT)
    base_df["date"] = pd.to_datetime(base_df["date"])
    base_df = base_df.sort_values("date").reset_index(drop=True)

    outputs = {}

    for target in TARGETS:
        outputs[target] = build_for_target(base_df, target)

    print("\n" + "=" * 90)
    print("ALL TARGET CANDIDATES BUILT")
    print("=" * 90)

    for target, path in outputs.items():
        print(f"{target}: {path}")


if __name__ == "__main__":
    main()
