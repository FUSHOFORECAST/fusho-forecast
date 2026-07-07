import os
import json
import pandas as pd

INPUT = "data/processed/feature_store_state.csv"

OUTPUT_DIR = "reports/adaptive_weight_engine"
OUTPUT_CANDIDATES = f"{OUTPUT_DIR}/forecast_candidates_backtest.csv"
OUTPUT_WEIGHTS = f"{OUTPUT_DIR}/adaptive_weights.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_mape(real, pred):
    if real == 0:
        return None
    return abs(pred - real) / real * 100


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    rows = []

    # partiamo da righe dove ci sono abbastanza rolling
    test_df = df.dropna(subset=[
        "total",
        "total_rolling_7",
        "total_rolling_30",
        "total_rolling_90",
        "total_rolling_365",
        "business_momentum_30_90",
    ]).copy()

    for _, row in test_df.iterrows():
        real = row["total"]

        memory_short = row["total_rolling_7"]
        memory_medium = row["total_rolling_30"]
        memory_long = row["total_rolling_90"]
        memory_seasonal = row["total_rolling_365"]

        adaptive_memory = (
            memory_short * 0.35
            + memory_medium * 0.35
            + memory_long * 0.20
            + memory_seasonal * 0.10
        )

        state_mask = (
            (df["date"] < row["date"])
            & (df["restaurant_state"] == row["restaurant_state"])
        )

        state_history = df[state_mask]

        if len(state_history) >= 5:
            state_forecast = state_history["total"].median()
        else:
            state_forecast = memory_medium

        calendar_mask = (
            (df["date"] < row["date"])
            & (df["month"] == row["month"])
            & (df["dayofweek"] == row["dayofweek"])
        )

        calendar_history = df[calendar_mask]

        if len(calendar_history) >= 3:
            calendar_forecast = calendar_history["total"].median()
        else:
            calendar_forecast = memory_medium

        similarity_forecast = (
            state_forecast * 0.50
            + calendar_forecast * 0.30
            + memory_medium * 0.20
        )

        candidates = {
            "memory_short": memory_short,
            "memory_medium": memory_medium,
            "memory_long": memory_long,
            "adaptive_memory": adaptive_memory,
            "state_forecast": state_forecast,
            "calendar_forecast": calendar_forecast,
            "similarity_forecast": similarity_forecast,
        }

        row_out = {
            "date": row["date"],
            "real_total": real,
            "weekday": row.get("weekday", ""),
            "month": row["month"],
            "commercial_season": row.get("commercial_season", ""),
            "growth_state": row.get("growth_state", ""),
            "volatility_state": row.get("volatility_state", ""),
            "restaurant_temperature": row.get("restaurant_temperature", ""),
            "delivery_state": row.get("delivery_state", ""),
            "cash_state": row.get("cash_state", ""),
            "market_pressure": row.get("market_pressure", ""),
            "restaurant_state": row.get("restaurant_state", ""),
        }

        for name, pred in candidates.items():
            row_out[name] = round(float(pred), 2)
            row_out[f"{name}_ape"] = safe_mape(real, pred)

        best_model = min(
            candidates.keys(),
            key=lambda name: safe_mape(real, candidates[name])
        )

        row_out["best_candidate"] = best_model
        row_out["best_ape"] = row_out[f"{best_model}_ape"]

        rows.append(row_out)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CANDIDATES, index=False)

    candidate_cols = [
        "memory_short",
        "memory_medium",
        "memory_long",
        "adaptive_memory",
        "state_forecast",
        "calendar_forecast",
        "similarity_forecast",
    ]

    global_scores = {}

    for c in candidate_cols:
        global_scores[c] = {
            "mape": round(float(out[f"{c}_ape"].mean()), 2),
            "mae": round(float((out[c] - out["real_total"]).abs().mean()), 2),
            "wins": int((out["best_candidate"] == c).sum()),
        }

    weights = {}

    # pesi globali inversi al MAPE
    inv = {
        k: 1 / max(v["mape"], 0.01)
        for k, v in global_scores.items()
    }

    total_inv = sum(inv.values())

    weights["global"] = {
        k: round(v / total_inv, 4)
        for k, v in inv.items()
    }

    # pesi per stato ristorante
    weights["by_restaurant_state"] = {}

    for state, group in out.groupby("restaurant_state"):
        if len(group) < 10:
            continue

        scores = {}

        for c in candidate_cols:
            scores[c] = group[f"{c}_ape"].mean()

        inv_state = {
            k: 1 / max(v, 0.01)
            for k, v in scores.items()
        }

        total_inv_state = sum(inv_state.values())

        weights["by_restaurant_state"][state] = {
            k: round(v / total_inv_state, 4)
            for k, v in inv_state.items()
        }

    # pesi per stagione commerciale
    weights["by_commercial_season"] = {}

    for season, group in out.groupby("commercial_season"):
        if len(group) < 10:
            continue

        scores = {}

        for c in candidate_cols:
            scores[c] = group[f"{c}_ape"].mean()

        inv_season = {
            k: 1 / max(v, 0.01)
            for k, v in scores.items()
        }

        total_inv_season = sum(inv_season.values())

        weights["by_commercial_season"][season] = {
            k: round(v / total_inv_season, 4)
            for k, v in inv_season.items()
        }

    summary = {
        "engine": "adaptive_weight_engine_v1",
        "input": INPUT,
        "rows": int(len(out)),
        "candidate_scores": global_scores,
        "weights": weights,
    }

    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("ADAPTIVE WEIGHT ENGINE V1")
    print("=" * 90)

    print("\nGLOBAL CANDIDATE SCORES")
    print(json.dumps(global_scores, indent=2))

    print("\nGLOBAL WEIGHTS")
    print(json.dumps(weights["global"], indent=2))

    print("\nBEST CANDIDATE WINS")
    print(out["best_candidate"].value_counts().to_string())

    print("\nSalvato:")
    print(OUTPUT_CANDIDATES)
    print(OUTPUT_WEIGHTS)


if __name__ == "__main__":
    main()
