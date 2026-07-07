import os
import json
import pandas as pd

INPUT_DIR = "reports/adaptive_weight_engine"
OUTPUT_DIR = "reports/channel_ensemble_forecast"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNELS = [
    "delivery",
    "digital",
    "cash",
]

CANDIDATE_BY_TARGET = {
    "delivery": "memory_short",
    "digital": "calendar_forecast",
    "cash": "memory_short",
}


def load_channel_forecast(channel):
    input_file = f"{INPUT_DIR}/forecast_candidates_backtest_{channel}.csv"

    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])

    candidate = CANDIDATE_BY_TARGET[channel]

    out = df[["date", channel, candidate]].copy()
    out = out.rename(columns={
        channel: f"real_{channel}",
        candidate: f"pred_{channel}",
    })

    return out


def main():
    merged = None

    for channel in CHANNELS:
        channel_df = load_channel_forecast(channel)

        if merged is None:
            merged = channel_df
        else:
            merged = merged.merge(channel_df, on="date", how="inner")

    merged["real_total_from_channels"] = (
        merged["real_delivery"]
        + merged["real_digital"]
        + merged["real_cash"]
    )

    merged["pred_total_from_channels"] = (
        merged["pred_delivery"]
        + merged["pred_digital"]
        + merged["pred_cash"]
    )

    merged["error"] = (
        merged["pred_total_from_channels"]
        - merged["real_total_from_channels"]
    )

    merged["abs_error"] = merged["error"].abs()

    merged["pct_error"] = (
        merged["abs_error"]
        / merged["real_total_from_channels"]
        * 100
    )

    merged["delivery_error"] = merged["pred_delivery"] - merged["real_delivery"]
    merged["digital_error"] = merged["pred_digital"] - merged["real_digital"]
    merged["cash_error"] = merged["pred_cash"] - merged["real_cash"]

    summary = {
        "engine": "channel_ensemble_forecast_v1",
        "channels": CHANNELS,
        "candidate_by_target": CANDIDATE_BY_TARGET,
        "rows": int(len(merged)),
        "mae": round(float(merged["abs_error"].mean()), 2),
        "mape": round(float(merged["pct_error"].mean()), 2),
        "total_forecast": round(float(merged["pred_total_from_channels"].sum()), 2),
        "total_real": round(float(merged["real_total_from_channels"].sum()), 2),
        "total_error": round(
            float(
                merged["pred_total_from_channels"].sum()
                - merged["real_total_from_channels"].sum()
            ),
            2,
        ),
        "total_error_pct": round(
            float(
                (
                    merged["pred_total_from_channels"].sum()
                    - merged["real_total_from_channels"].sum()
                )
                / merged["real_total_from_channels"].sum()
                * 100
            ),
            2,
        ),
        "channel_mae": {
            "delivery": round(float(merged["delivery_error"].abs().mean()), 2),
            "digital": round(float(merged["digital_error"].abs().mean()), 2),
            "cash": round(float(merged["cash_error"].abs().mean()), 2),
        },
    }

    output_csv = f"{OUTPUT_DIR}/channel_ensemble_forecast.csv"
    output_json = f"{OUTPUT_DIR}/channel_ensemble_summary.json"

    merged.to_csv(output_csv, index=False)

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("CHANNEL ENSEMBLE FORECAST V1")
    print("=" * 90)

    print(json.dumps(summary, indent=2))

    print("\nULTIMI 20 GIORNI")
    print(
        merged[
            [
                "date",
                "real_total_from_channels",
                "pred_total_from_channels",
                "error",
                "pct_error",
                "real_delivery",
                "pred_delivery",
                "real_digital",
                "pred_digital",
                "real_cash",
                "pred_cash",
            ]
        ].tail(20).to_string(index=False)
    )

    print("\nSalvato:")
    print(output_csv)
    print(output_json)


if __name__ == "__main__":
    main()
