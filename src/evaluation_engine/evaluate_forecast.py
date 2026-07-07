import json
import os
import pandas as pd

FORECAST_FILE = "data/processed/forecast_adaptive_memory_v2.csv"
REAL_FILE = "data/processed/june_real_totals.csv"

DATE_COL = "date"
PRED_COL = "total_pred"
REAL_COL = "real_total"

OUTPUT_DIR = "reports/evaluation"


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    forecast = pd.read_csv(FORECAST_FILE)
    real = pd.read_csv(REAL_FILE)

    forecast[DATE_COL] = pd.to_datetime(forecast[DATE_COL])
    real[DATE_COL] = pd.to_datetime(real[DATE_COL])

    df = forecast.merge(real, on=DATE_COL, how="inner")

    df["error"] = df[PRED_COL] - df[REAL_COL]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["abs_error"] / df[REAL_COL] * 100
    df["squared_error"] = df["error"] ** 2
    df["weekday"] = df[DATE_COL].dt.day_name()
    df["month"] = df[DATE_COL].dt.month

    mae = df["abs_error"].mean()
    mape = df["pct_error"].mean()
    rmse = df["squared_error"].mean() ** 0.5
    median_ae = df["abs_error"].median()
    median_ape = df["pct_error"].median()

    bias_eur = df["error"].sum()
    bias_pct = safe_div(bias_eur, df[REAL_COL].sum()) * 100

    total_pred = df[PRED_COL].sum()
    total_real = df[REAL_COL].sum()
    total_error = total_pred - total_real
    total_error_pct = safe_div(total_error, total_real) * 100

    accuracy = max(0, 100 - mape)

    wape = safe_div(df["abs_error"].sum(), df[REAL_COL].sum()) * 100

    smape = (
        (df["abs_error"] / ((df[PRED_COL].abs() + df[REAL_COL].abs()) / 2))
        .replace([float("inf"), -float("inf")], 0)
        .mean()
        * 100
    )

    forecast_volatility = df[PRED_COL].std()
    real_volatility = df[REAL_COL].std()

    error_std = df["error"].std()
    max_error = df["abs_error"].max()

    error_distribution = {
        "under_5_pct": int((df["pct_error"] < 5).sum()),
        "between_5_10_pct": int(((df["pct_error"] >= 5) & (df["pct_error"] < 10)).sum()),
        "between_10_20_pct": int(((df["pct_error"] >= 10) & (df["pct_error"] < 20)).sum()),
        "over_20_pct": int((df["pct_error"] >= 20).sum()),
    }

    metrics = {
        "period": {
            "start_date": str(df[DATE_COL].min().date()),
            "end_date": str(df[DATE_COL].max().date()),
            "days": int(len(df)),
        },
        "totals": {
            "forecast_total": round(float(total_pred), 2),
            "real_total": round(float(total_real), 2),
            "total_error_eur": round(float(total_error), 2),
            "total_error_pct": round(float(total_error_pct), 2),
        },
        "accuracy": {
            "mae": round(float(mae), 2),
            "mape": round(float(mape), 2),
            "rmse": round(float(rmse), 2),
            "median_absolute_error": round(float(median_ae), 2),
            "median_absolute_percentage_error": round(float(median_ape), 2),
            "wape": round(float(wape), 2),
            "smape": round(float(smape), 2),
            "forecast_accuracy_score": round(float(accuracy), 2),
        },
        "bias": {
            "bias_eur": round(float(bias_eur), 2),
            "bias_pct": round(float(bias_pct), 2),
        },
        "volatility": {
            "forecast_volatility": round(float(forecast_volatility), 2),
            "real_volatility": round(float(real_volatility), 2),
            "volatility_ratio": round(float(safe_div(forecast_volatility, real_volatility)), 4),
            "error_std": round(float(error_std), 2),
            "max_abs_error": round(float(max_error), 2),
        },
        "error_distribution_days": error_distribution,
    }

    weekday_analysis = (
        df.groupby("weekday")
        .agg(
            days=(DATE_COL, "count"),
            mae=("abs_error", "mean"),
            mape=("pct_error", "mean"),
            bias=("error", "sum"),
        )
        .reset_index()
    )

    month_analysis = (
        df.groupby("month")
        .agg(
            days=(DATE_COL, "count"),
            mae=("abs_error", "mean"),
            mape=("pct_error", "mean"),
            bias=("error", "sum"),
        )
        .reset_index()
    )

    worst_days = df.sort_values("pct_error", ascending=False).head(20)
    best_days = df.sort_values("pct_error", ascending=True).head(20)

    df.to_csv(f"{OUTPUT_DIR}/daily_errors.csv", index=False)
    weekday_analysis.to_csv(f"{OUTPUT_DIR}/weekday_analysis.csv", index=False)
    month_analysis.to_csv(f"{OUTPUT_DIR}/month_analysis.csv", index=False)
    worst_days.to_csv(f"{OUTPUT_DIR}/worst_days.csv", index=False)
    best_days.to_csv(f"{OUTPUT_DIR}/best_days.csv", index=False)

    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 90)
    print("FORECAST EVALUATION ENGINE")
    print("=" * 90)

    print(json.dumps(metrics, indent=2))

    print("\nWORST DAYS")
    print(
        worst_days[
            [DATE_COL, PRED_COL, REAL_COL, "error", "abs_error", "pct_error"]
        ].to_string(index=False)
    )

    print("\nSalvato in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
