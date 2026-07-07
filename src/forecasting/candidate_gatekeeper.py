import os
import json
import pandas as pd

INPUT = "reports/adaptive_weight_engine/forecast_candidates_backtest_total.csv"

OUTPUT_DIR = "reports/candidate_gatekeeper"
OUTPUT_CSV = f"{OUTPUT_DIR}/candidate_gatekeeper_scores.csv"
OUTPUT_JSON = f"{OUTPUT_DIR}/candidate_gatekeeper_summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATES = [
    "memory_short",
    "memory_medium",
    "memory_long",
    "adaptive_memory",
    "state_forecast",
    "calendar_forecast",
    "similarity_forecast",
    "external_context_forecast",
]

LOOKBACK_WINDOWS = [30, 60, 120]

MIN_SCORE_TO_PASS = 0.70
MAX_CANDIDATES = 5


def safe_score(value, good, bad):
    """
    value basso = meglio.
    good = valore ottimo
    bad = valore scarso
    """
    if pd.isna(value):
        return 0

    if value <= good:
        return 1

    if value >= bad:
        return 0

    return 1 - ((value - good) / (bad - good))


def evaluate_candidate(df, candidate):
    rows = []

    valid = df.dropna(subset=[candidate, "total"]).copy()

    valid["error"] = valid[candidate] - valid["total"]
    valid["abs_error"] = valid["error"].abs()
    valid["pct_error"] = valid["abs_error"] / valid["total"] * 100

    latest_date = valid["date"].max()

    result = {
        "candidate": candidate,
        "rows": int(len(valid)),
        "latest_date": str(latest_date.date()),
    }

    for window in LOOKBACK_WINDOWS:
        recent = valid[valid["date"] > latest_date - pd.Timedelta(days=window)]

        result[f"mape_{window}"] = round(float(recent["pct_error"].mean()), 2)
        result[f"mae_{window}"] = round(float(recent["abs_error"].mean()), 2)
        result[f"bias_pct_{window}"] = round(
            float(recent["error"].sum() / recent["total"].sum() * 100),
            2,
        )
        result[f"volatility_ratio_{window}"] = round(
            float(recent[candidate].std() / recent["total"].std())
            if recent["total"].std() != 0
            else 0,
            4,
        )

    # score principali
    score_mape_30 = safe_score(result["mape_30"], good=10, bad=25)
    score_mape_60 = safe_score(result["mape_60"], good=10, bad=25)
    score_mape_120 = safe_score(result["mape_120"], good=10, bad=25)

    # bias assoluto: meglio vicino a 0
    score_bias_60 = safe_score(abs(result["bias_pct_60"]), good=0, bad=12)

    # volatilità forecast simile alla reale: ratio ideale circa 1
    volatility_distance = abs(result["volatility_ratio_60"] - 1)
    score_volatility_60 = safe_score(volatility_distance, good=0.15, bad=0.75)

    # trend errore: se 30 giorni è peggio di 120, penalizza
    error_trend = result["mape_30"] - result["mape_120"]
    score_error_trend = safe_score(error_trend, good=-3, bad=6)

    final_score = (
        score_mape_30 * 0.25
        + score_mape_60 * 0.25
        + score_mape_120 * 0.15
        + score_bias_60 * 0.15
        + score_volatility_60 * 0.10
        + score_error_trend * 0.10
    )

    result["score_mape_30"] = round(score_mape_30, 4)
    result["score_mape_60"] = round(score_mape_60, 4)
    result["score_mape_120"] = round(score_mape_120, 4)
    result["score_bias_60"] = round(score_bias_60, 4)
    result["score_volatility_60"] = round(score_volatility_60, 4)
    result["score_error_trend"] = round(score_error_trend, 4)
    result["gatekeeper_score"] = round(float(final_score), 4)

    return result


def main():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    rows = []

    for candidate in CANDIDATES:
        if candidate not in df.columns:
            continue

        rows.append(evaluate_candidate(df, candidate))

    result = pd.DataFrame(rows)
    result = result.sort_values("gatekeeper_score", ascending=False)

    selected = result[
        result["gatekeeper_score"] >= MIN_SCORE_TO_PASS
    ].head(MAX_CANDIDATES)

    # fallback: se nessuno passa, prendi comunque i migliori 3
    if selected.empty:
        selected = result.head(3)

    result["selected"] = result["candidate"].isin(selected["candidate"]).astype(int)

    result.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "engine": "candidate_gatekeeper_v1",
        "input": INPUT,
        "min_score_to_pass": MIN_SCORE_TO_PASS,
        "max_candidates": MAX_CANDIDATES,
        "candidates_evaluated": int(len(result)),
        "selected_candidates": selected["candidate"].tolist(),
        "scores": result.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("CANDIDATE GATEKEEPER V1")
    print("=" * 90)

    print("\nSCORES")
    print(
        result[
            [
                "candidate",
                "gatekeeper_score",
                "selected",
                "mape_30",
                "mape_60",
                "mape_120",
                "bias_pct_60",
                "volatility_ratio_60",
            ]
        ].to_string(index=False)
    )

    print("\nSELECTED CANDIDATES")
    print(selected["candidate"].tolist())

    print("\nSalvato:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
