import json
import os
import pandas as pd

INPUT_FILE = "data/processed/master_dataset_full.csv"
OUTPUT_FILE = "reports/restaurant_dna_universal.json"


def pct_change(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return 0.0
    return round(((a - b) / b) * 100, 2)


def main():
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["total"] > 0].copy()
    df = df.sort_values("date")

    df["delivery_share"] = df["delivery"] / df["total"]
    df["digital_share"] = df["digital"] / df["total"]
    df["cash_share"] = df["cash"] / df["total"]

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    avg_total = df["total"].mean()
    median_total = df["total"].median()
    std_total = df["total"].std()
    cv_total = std_total / avg_total

    weekend_avg = df[df["is_weekend"] == 1]["total"].mean()
    weekday_avg = df[df["is_weekend"] == 0]["total"].mean()

    delivery_share = df["delivery_share"].mean()
    digital_share = df["digital_share"].mean()
    cash_share = df["cash_share"].mean()

    yearly_delivery = df.groupby("year")["delivery_share"].mean()
    yearly_cash = df.groupby("year")["cash_share"].mean()
    yearly_total = df.groupby("year")["total"].mean()

    weekday_avg_total = df.groupby("dayofweek")["total"].mean()
    month_avg_total = df.groupby("month")["total"].mean()

    dna = {
        "schema_version": "1.0",

        "dataset": {
            "start_date": str(df["date"].min().date()),
            "end_date": str(df["date"].max().date()),
            "days": int(len(df)),
        },

        "scale": {
            "avg_daily_revenue": round(float(avg_total), 2),
            "median_daily_revenue": round(float(median_total), 2),
            "std_daily_revenue": round(float(std_total), 2),
            "coefficient_of_variation": round(float(cv_total), 4),
        },

        "channel_mix": {
            "delivery_share": round(float(delivery_share), 4),
            "digital_share": round(float(digital_share), 4),
            "cash_share": round(float(cash_share), 4),
        },

        "trend": {
            "revenue_trend_first_to_last_year_pct": pct_change(
                yearly_total.iloc[-1],
                yearly_total.iloc[0],
            ),
            "delivery_share_trend_first_to_last_year_pct": pct_change(
                yearly_delivery.iloc[-1],
                yearly_delivery.iloc[0],
            ),
            "cash_share_trend_first_to_last_year_pct": pct_change(
                yearly_cash.iloc[-1],
                yearly_cash.iloc[0],
            ),
        },

        "weekday_behavior": {
            "best_weekday": int(weekday_avg_total.idxmax()),
            "worst_weekday": int(weekday_avg_total.idxmin()),
            "weekday_spread_pct": pct_change(
                weekday_avg_total.max(),
                weekday_avg_total.min(),
            ),
            "avg_total_by_weekday": {
                str(k): round(float(v), 2)
                for k, v in weekday_avg_total.items()
            },
        },

        "seasonality": {
            "best_month": int(month_avg_total.idxmax()),
            "worst_month": int(month_avg_total.idxmin()),
            "month_spread_pct": pct_change(
                month_avg_total.max(),
                month_avg_total.min(),
            ),
            "avg_total_by_month": {
                str(k): round(float(v), 2)
                for k, v in month_avg_total.items()
            },
        },

        "weekend": {
            "weekend_effect_pct": pct_change(weekend_avg, weekday_avg),
        },

        "business_tags": {
            "delivery_driven": bool(delivery_share >= 0.60),
            "cash_light": bool(cash_share <= 0.15),
            "volatile": bool(cv_total >= 0.25),
            "seasonal": bool(pct_change(month_avg_total.max(), month_avg_total.min()) >= 25),
            "weekday_sensitive": bool(pct_change(weekday_avg_total.max(), weekday_avg_total.min()) >= 15),
            "delivery_growing": bool(yearly_delivery.iloc[-1] > yearly_delivery.iloc[0]),
            "cash_declining": bool(yearly_cash.iloc[-1] < yearly_cash.iloc[0]),
        }
    }

    os.makedirs("reports", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dna, f, indent=2)

    print("=" * 80)
    print("UNIVERSAL RESTAURANT DNA")
    print("=" * 80)
    print(json.dumps(dna, indent=2))
    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
