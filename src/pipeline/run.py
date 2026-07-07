import argparse
import dataclasses
import json
import os
from datetime import datetime

import pandas as pd

from src.pipeline.backtest import compute_total_metrics, summarize_backtest, walk_forward_backtest
from src.pipeline.calendar_features import add_calendar_features
from src.pipeline.clean import clean
from src.pipeline.config import RestaurantConfig, load_restaurant_config
from src.pipeline.events import merge_events
from src.pipeline.features import build_features
from src.pipeline.forecast import recursive_forecast
from src.pipeline.ingest import ingest
from src.pipeline.model import get_feature_columns, select_best_model_per_channel, train_final_models
from src.pipeline.profile import build_profile, profile_features
from src.pipeline.weather import fetch_historical_weather, merge_weather


def run_pipeline(config: RestaurantConfig) -> dict:
    os.makedirs(config.processed_dir, exist_ok=True)
    os.makedirs(config.reports_dir, exist_ok=True)
    os.makedirs(config.models_dir, exist_ok=True)

    print(f"[1/8] Ingestione dati grezzi per {config.display_name}...")
    master_df, audit_df = ingest(config)

    print("[2/8] Pulizia dataset (audit tolerance)...")
    clean_df = clean(master_df, audit_df, config)

    print("[3/8] Meteo storico...")
    weather_hist = fetch_historical_weather(
        config, clean_df["date"].min().strftime("%Y-%m-%d"), clean_df["date"].max().strftime("%Y-%m-%d")
    )
    df = merge_weather(clean_df, weather_hist)
    df = add_calendar_features(df, config)
    df = merge_events(df, config)
    df.to_csv(config.processed_path("master_dataset_full.csv"), index=False)

    print("[4/8] Profilo ristorante (DNA, stagionalita', pattern)...")
    profile = build_profile(df, config)
    with open(config.reports_path("restaurant_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    print("[5/8] Feature engineering (lag/rolling/trend/profilo)...")
    feat_df = build_features(df, config)
    feat_df = profile_features(feat_df, profile, config)
    feat_df.to_csv(config.processed_path("model_dataset_full.csv"), index=False)

    feature_cols = get_feature_columns(feat_df, config)

    print(f"[6/8] Backtest walk-forward ({config.forecast.backtest_days} giorni x {len(config.channels)} canali x {len(config.model.candidates)} modelli)...")
    backtest_results = walk_forward_backtest(feat_df, config, feature_cols)
    backtest_results.to_csv(config.processed_path("backtest_results.csv"), index=False)
    backtest_summary = summarize_backtest(backtest_results)
    winners = select_best_model_per_channel(config, backtest_summary)
    backtest_summary["winners"] = winners
    backtest_summary["total"] = compute_total_metrics(backtest_results, winners)

    print("[7/8] Training modelli finali (su tutto lo storico)...")
    final_models = train_final_models(feat_df, config, feature_cols, winners)

    print(f"[8/8] Forecast ricorsivo a {config.forecast.horizon_days} giorni...")
    forecast_df = recursive_forecast(feat_df, config, final_models, feature_cols, profile)
    forecast_path = config.reports_path(f"forecast_{config.forecast.horizon_days}d.csv")
    forecast_df.to_csv(forecast_path, index=False)

    summary = {
        "restaurant_id": config.restaurant_id,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": {
            "start_date": str(feat_df["date"].min()),
            "end_date": str(feat_df["date"].max()),
            "rows_ingested": int(len(master_df)),
            "rows_after_cleaning": int(len(clean_df)),
            "rows_for_modeling": int(len(feat_df)),
        },
        "channels": config.channels,
        "backtest": backtest_summary,
        "forecast_file": str(forecast_path),
    }

    with open(config.reports_path("pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETATA per {config.display_name}")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

    return {"summary": summary, "forecast": forecast_df}


def main():
    parser = argparse.ArgumentParser(description="Pipeline di forecast parametrizzata per ristorante")
    parser.add_argument("--config", required=True, help="Percorso del file YAML di configurazione ristorante")
    parser.add_argument("--backtest-days", type=int, default=None, help="Override giorni di backtest")
    parser.add_argument("--horizon-days", type=int, default=None, help="Override orizzonte forecast in giorni")
    args = parser.parse_args()

    config = load_restaurant_config(args.config)

    if args.backtest_days is not None:
        config = dataclasses.replace(config, forecast=dataclasses.replace(config.forecast, backtest_days=args.backtest_days))
    if args.horizon_days is not None:
        config = dataclasses.replace(config, forecast=dataclasses.replace(config.forecast, horizon_days=args.horizon_days))

    run_pipeline(config)


if __name__ == "__main__":
    main()
