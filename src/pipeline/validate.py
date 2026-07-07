import json

import pandas as pd

from src.pipeline.config import RestaurantConfig
from src.pipeline.forecast import build_historical_calendar_weather_events, recursive_forecast
from src.pipeline.model import get_feature_columns, load_final_models


def forecast_period(config: RestaurantConfig, start_date: str, end_date: str) -> pd.DataFrame:
    """Genera un forecast ricorsivo per [start_date, end_date] riusando i modelli finali
    gia' allenati (salvati in models/{restaurant_id}/) e meteo storico reale (il periodo
    e' nel passato, quindi il meteo e' noto per davvero, non una stima)."""
    feat_df = pd.read_csv(config.processed_path("model_dataset_full.csv"))
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    with open(config.reports_path("pipeline_summary.json")) as f:
        summary = json.load(f)
    winners = summary["backtest"]["winners"]

    with open(config.reports_path("restaurant_profile.json")) as f:
        profile = json.load(f)

    models = load_final_models(config, winners)
    feature_cols = get_feature_columns(feat_df, config)

    future_meta = build_historical_calendar_weather_events(config, start_date, end_date)

    return recursive_forecast(feat_df, config, models, feature_cols, profile, future_meta=future_meta)


def compare_to_actuals(forecast_df: pd.DataFrame, actuals_df: pd.DataFrame, config: RestaurantConfig) -> pd.DataFrame:
    """actuals_df deve avere colonne: date, total, e una per ogni canale in config.channels."""
    actuals = actuals_df.copy()
    actuals["date"] = pd.to_datetime(actuals["date"])

    merged = forecast_df.merge(actuals, on="date", how="inner")

    for channel in config.channels + ["total"]:
        merged[f"{channel}_error_pct"] = (
            (merged[f"{channel}_pred"] - merged[channel]).abs() / merged[channel] * 100
        ).round(2)

    return merged


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--actuals-csv", default=None, help="CSV con colonne date,total,<canali...>")
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    forecast_df = forecast_period(config, args.start_date, args.end_date)

    if args.actuals_csv:
        actuals = pd.read_csv(args.actuals_csv)
        result = compare_to_actuals(forecast_df, actuals, config)
        print(result.to_string(index=False))
    else:
        print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()
