import os

import pandas as pd

from src.pipeline.config import RestaurantConfig
from src.pipeline.model import MODEL_FACTORIES


def walk_forward_backtest(df: pd.DataFrame, config: RestaurantConfig, feature_cols: list[str]) -> pd.DataFrame:
    backtest_days = config.forecast.backtest_days
    start = max(0, len(df) - backtest_days)

    records = []

    for i in range(start, len(df)):
        train = df.iloc[:i]
        test = df.iloc[[i]]

        for channel in config.channels:
            for model_type in config.model.candidates:
                model = MODEL_FACTORIES[model_type]()
                model.fit(train[feature_cols], train[channel])
                predicted = model.predict(test[feature_cols])[0]
                actual = test[channel].values[0]

                records.append({
                    "date": test["date"].values[0],
                    "channel": channel,
                    "model_type": model_type,
                    "actual": actual,
                    "predicted": predicted,
                    "abs_error": abs(predicted - actual),
                    "pct_error": abs(predicted - actual) / actual * 100 if actual else None,
                })

    return pd.DataFrame(records)


def summarize_backtest(results: pd.DataFrame) -> dict:
    per_channel_model = {}

    for (channel, model_type), group in results.groupby(["channel", "model_type"]):
        per_channel_model.setdefault(channel, {})[model_type] = {
            "mape": round(float(group["pct_error"].mean()), 2),
            "mae": round(float(group["abs_error"].mean()), 2),
        }

    return {"backtest_days": int(results["date"].nunique()), "per_channel_model": per_channel_model}


def compute_total_metrics(results: pd.DataFrame, winners: dict[str, str]) -> dict:
    frames = []
    for channel, model_type in winners.items():
        sub = results[(results["channel"] == channel) & (results["model_type"] == model_type)][["date", "actual", "predicted"]]
        sub = sub.rename(columns={"actual": f"actual_{channel}", "predicted": f"predicted_{channel}"}).set_index("date")
        frames.append(sub)

    merged = pd.concat(frames, axis=1)
    merged["actual_total"] = merged[[f"actual_{c}" for c in winners]].sum(axis=1)
    merged["predicted_total"] = merged[[f"predicted_{c}" for c in winners]].sum(axis=1)

    abs_error = (merged["predicted_total"] - merged["actual_total"]).abs()
    mape = (abs_error / merged["actual_total"]).mean() * 100

    return {"mape": round(float(mape), 2), "mae": round(float(abs_error.mean()), 2)}


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config
    from src.pipeline.model import get_feature_columns, select_best_model_per_channel

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    df = pd.read_csv(config.processed_path("model_dataset_full.csv"))
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = get_feature_columns(df, config)
    results = walk_forward_backtest(df, config, feature_cols)

    os.makedirs(config.processed_dir, exist_ok=True)
    results.to_csv(config.processed_path("backtest_results.csv"), index=False)

    summary = summarize_backtest(results)
    winners = select_best_model_per_channel(config, summary)
    summary["total"] = compute_total_metrics(results, winners)
    summary["winners"] = winners

    print(summary)


if __name__ == "__main__":
    main()
