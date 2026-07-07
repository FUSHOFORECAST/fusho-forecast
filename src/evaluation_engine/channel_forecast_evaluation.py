import os
import json
import pandas as pd

INPUT_DIR = "reports/adaptive_weight_engine"
OUTPUT_DIR = "reports/channel_forecast_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = [
    "total",
    "delivery",
    "digital",
    "cash",
]

CANDIDATES = [
    "memory_short",
    "memory_medium",
    "memory_long",
    "adaptive_memory",
    "state_forecast",
    "calendar_forecast",
    "similarity_forecast",
]


def evaluate_target(target):
    input_file = f"{INPUT_DIR}/forecast_candidates_backtest_{target}.csv"

    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    rows = []

    for candidate in CANDIDATES:
        pred_col = candidate
        real_col = target

        valid = df.dropna(subset=[pred_col, real_col]).copy()

        valid["error"] = valid[pred_col] - valid[real_col]
        valid["abs_error"] = valid["error"].abs()
        valid["pct_error"] = valid["abs_error"] / valid[real_col] * 100

        row = {
            "target": target,
            "candidate": candidate,
            "rows": int(len(valid)),
            "mae": round(float(valid["abs_error"].mean()), 2),
            "mape": round(float(valid["pct_error"].mean()), 2),
            "total_forecast": round(float(valid[pred_col].sum()), 2),
            "total_real": round(float(valid[real_col].sum()), 2),
            "total_error": round(float(valid[pred_col].sum() - valid[real_col].sum()), 2),
            "total_error_pct": round(
                float(
                    (valid[pred_col].sum() - valid[real_col].sum())
                    / valid[real_col].sum()
                    * 100
                ),
                2,
            ),
            "forecast_volatility": round(float(valid[pred_col].std()), 2),
            "real_volatility": round(float(valid[real_col].std()), 2),
        }

        rows.append(row)

    result = pd.DataFrame(rows).sort_values("mape")

    return result


def main():
    all_results = []

    print("=" * 90)
    print("CHANNEL FORECAST EVALUATION V1")
    print("=" * 90)

    for target in TARGETS:
        result = evaluate_target(target)

        output_file = f"{OUTPUT_DIR}/{target}_candidate_evaluation.csv"
        result.to_csv(output_file, index=False)

        all_results.append(result)

        print("\n" + "=" * 70)
        print(target.upper())
        print("=" * 70)
        print(result.to_string(index=False))

    summary = pd.concat(all_results, ignore_index=True)

    summary_file = f"{OUTPUT_DIR}/channel_candidate_evaluation_summary.csv"
    summary.to_csv(summary_file, index=False)

    best = (
        summary.sort_values("mape")
        .groupby("target")
        .head(1)
        .reset_index(drop=True)
    )

    best_file = f"{OUTPUT_DIR}/best_candidate_by_target.csv"
    best.to_csv(best_file, index=False)

    summary_json = {
        "engine": "channel_forecast_evaluation_v1",
        "targets": TARGETS,
        "best_candidate_by_target": best.to_dict(orient="records"),
    }

    json_file = f"{OUTPUT_DIR}/channel_forecast_evaluation_summary.json"

    with open(json_file, "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "=" * 90)
    print("BEST CANDIDATE BY TARGET")
    print("=" * 90)
    print(best.to_string(index=False))

    print("\nSalvato:")
    print(summary_file)
    print(best_file)
    print(json_file)


if __name__ == "__main__":
    main()
