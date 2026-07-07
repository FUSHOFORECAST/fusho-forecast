import os
import json
import pandas as pd

from src.forecast_candidates.registry import get_candidate_names

INPUT = "reports/adaptive_weight_engine/forecast_candidates_backtest_total.csv"

OUTPUT_DIR = "reports/adaptive_weight_engine_v3"
OUTPUT_WEIGHTS = f"{OUTPUT_DIR}/learned_selective_weights.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATES = get_candidate_names()

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

MIN_SAMPLES = 10
SHARPNESS = 2.2
WIN_RATE_POWER = 1.4


def make_weights(group):
    raw_scores = {}
    total_rows = len(group)
    wins = group["best_candidate"].value_counts().to_dict()

    for candidate in CANDIDATES:
        ape_col = f"{candidate}_ape"

        if ape_col not in group.columns:
            continue

        valid_ape = group[ape_col].dropna()

        if len(valid_ape) == 0:
            continue

        mape = valid_ape.mean()
        win_rate = wins.get(candidate, 0) / total_rows

        score = (1 / max(mape, 0.01)) * ((win_rate + 0.05) ** WIN_RATE_POWER)
        raw_scores[candidate] = score

    if not raw_scores:
        return {key: round(1 / len(CANDIDATES), 4) for key in CANDIDATES}

    sharpened = {key: value ** SHARPNESS for key, value in raw_scores.items()}
    total = sum(sharpened.values())

    return {key: round(value / total, 4) for key, value in sharpened.items()}


def summarize_group(group):
    return {
        "samples": int(len(group)),
        "weights": make_weights(group),
        "best_candidate_wins": group["best_candidate"].value_counts().to_dict(),
        "candidate_mape": {
            candidate: round(float(group[f"{candidate}_ape"].mean()), 2)
            for candidate in CANDIDATES
            if f"{candidate}_ape" in group.columns
            and group[f"{candidate}_ape"].notna().any()
        },
    }


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    ape_cols = [f"{c}_ape" for c in CANDIDATES if f"{c}_ape" in df.columns]

    valid_mask = df[ape_cols].notna().any(axis=1)
    df = df[valid_mask].copy()

    df["best_candidate"] = (
        df[ape_cols]
        .idxmin(axis=1)
        .str.replace("_ape", "", regex=False)
    )

    df["best_ape"] = df[ape_cols].min(axis=1)

    learned = {
        "engine": "adaptive_weight_engine_v3_registry_based",
        "input": INPUT,
        "method": "mape_plus_winrate_sharpened_weights",
        "min_samples_per_context": MIN_SAMPLES,
        "sharpness": SHARPNESS,
        "win_rate_power": WIN_RATE_POWER,
        "candidates": CANDIDATES,
        "rows_used": int(len(df)),
        "global": make_weights(df),
        "contexts": {},
    }

    for context in CONTEXTS:
        if context not in df.columns:
            continue

        learned["contexts"][context] = {}

        for value, group in df.groupby(context):
            if len(group) < MIN_SAMPLES:
                continue

            learned["contexts"][context][str(value)] = summarize_group(group)

    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(learned, f, indent=2)

    print("=" * 90)
    print("ADAPTIVE WEIGHT ENGINE V3 - REGISTRY BASED")
    print("=" * 90)

    print("\nINPUT:", INPUT)

    print("\nCANDIDATES")
    print(CANDIDATES)

    print("\nROWS USED:", len(df))

    print("\nGLOBAL WEIGHTS")
    print(json.dumps(learned["global"], indent=2))

    print("\nGLOBAL WINS")
    print(df["best_candidate"].value_counts().to_string())

    print("\nSalvato:")
    print(OUTPUT_WEIGHTS)


if __name__ == "__main__":
    main()
