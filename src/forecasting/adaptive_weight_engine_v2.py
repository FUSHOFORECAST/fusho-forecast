import os
import json
import pandas as pd

INPUT = "reports/adaptive_weight_engine/forecast_candidates_backtest.csv"

OUTPUT_DIR = "reports/adaptive_weight_engine_v2"
OUTPUT_WEIGHTS = f"{OUTPUT_DIR}/learned_adaptive_weights.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATES = [
    "memory_short",
    "memory_medium",
    "memory_long",
    "adaptive_memory",
    "state_forecast",
    "calendar_forecast",
    "similarity_forecast",
]

CONTEXTS = [
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


def make_weights(group):
    scores = {}

    for c in CANDIDATES:
        ape_col = f"{c}_ape"
        if ape_col in group.columns:
            scores[c] = group[ape_col].mean()

    inv = {
        k: 1 / max(v, 0.01)
        for k, v in scores.items()
    }

    total = sum(inv.values())

    return {
        k: round(v / total, 4)
        for k, v in inv.items()
    }


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    learned = {
        "engine": "adaptive_weight_engine_v2_learned",
        "method": "contextual_inverse_mape_weights",
        "min_samples_per_context": 10,
        "candidates": CANDIDATES,
        "global": make_weights(df),
        "contexts": {},
    }

    for context in CONTEXTS:
        if context not in df.columns:
            continue

        learned["contexts"][context] = {}

        for value, group in df.groupby(context):
            if len(group) < 10:
                continue

            learned["contexts"][context][str(value)] = {
                "samples": int(len(group)),
                "weights": make_weights(group),
                "best_candidate_wins": group["best_candidate"].value_counts().to_dict(),
            }

    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(learned, f, indent=2)

    print("=" * 90)
    print("ADAPTIVE WEIGHT ENGINE V2 - LEARNED")
    print("=" * 90)

    print("\nGLOBAL WEIGHTS")
    print(json.dumps(learned["global"], indent=2))

    print("\nCONTEXTS CREATED")
    for context, values in learned["contexts"].items():
        print(context, ":", len(values), "valori")

    print("\nEsempio growth_state:")
    print(json.dumps(learned["contexts"].get("growth_state", {}), indent=2)[:3000])

    print("\nSalvato:")
    print(OUTPUT_WEIGHTS)


if __name__ == "__main__":
    main()
