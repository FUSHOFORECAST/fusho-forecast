from datetime import timedelta

import pandas as pd

from src.pipeline.calendar_features import add_calendar_features
from src.pipeline.config import RestaurantConfig
from src.pipeline.events import merge_events
from src.pipeline.features import add_calendar_derived_fields
from src.pipeline.profile import profile_features
from src.pipeline.weather import fetch_forecast_weather, fetch_historical_weather, merge_weather


def build_future_calendar_weather_events(config: RestaurantConfig, last_date: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    future_df = pd.DataFrame({"date": future_dates})

    future_df = add_calendar_features(future_df, config)

    weather_fc = fetch_forecast_weather(config, horizon_days)
    future_df = merge_weather(future_df, weather_fc)

    future_df = merge_events(future_df, config)

    return future_df


def build_historical_calendar_weather_events(config: RestaurantConfig, start_date: str, end_date: str) -> pd.DataFrame:
    """Come build_future_calendar_weather_events ma con meteo storico reale (API archive)
    invece che previsto. Utile per validare il forecast ricorsivo su un periodo gia'
    trascorso (le date sono nel passato, quindi il meteo e' noto per davvero, non stimato).
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    meta_df = pd.DataFrame({"date": dates})

    meta_df = add_calendar_features(meta_df, config)

    weather_hist = fetch_historical_weather(config, start_date, end_date)
    meta_df = merge_weather(meta_df, weather_hist)

    meta_df = merge_events(meta_df, config)

    return meta_df


def recursive_forecast(
    history_df: pd.DataFrame,
    config: RestaurantConfig,
    models: dict[str, object],
    feature_cols: list[str],
    profile: dict,
    future_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    channel_total_cols = config.channels + ["total"]
    history = history_df[["date"] + channel_total_cols].copy()
    history["date"] = pd.to_datetime(history["date"])

    last_date = history["date"].max()
    lags = config.features.lags
    rolling_windows = config.features.rolling_windows
    short_window, long_window = config.features.trend_windows

    if future_meta is None:
        future_meta = build_future_calendar_weather_events(config, last_date, config.forecast.horizon_days)

    horizon_days = len(future_meta)

    results = []

    for i in range(horizon_days):
        meta_row = future_meta.iloc[[i]].copy()
        future_date = meta_row["date"].iloc[0]

        meta_row = add_calendar_derived_fields(meta_row)
        new_row = meta_row.iloc[0].to_dict()

        new_row["year"] = future_date.year
        new_row["month"] = future_date.month
        new_row["day"] = future_date.day

        for channel in channel_total_cols:
            for lag in lags:
                new_row[f"{channel}_lag_{lag}"] = history[channel].iloc[-lag]
            for window in rolling_windows:
                new_row[f"{channel}_rolling_{window}"] = history[channel].tail(window).mean()
                new_row[f"{channel}_rolling_median_{window}"] = history[channel].tail(window).median()
            new_row[f"{channel}_trend"] = (
                new_row[f"{channel}_rolling_{short_window}"] / new_row[f"{channel}_rolling_{long_window}"]
            )

        for channel in config.channels:
            new_row[f"{channel}_share_lag"] = new_row[f"{channel}_lag_7"] / new_row["total_lag_7"]

        x_row = pd.DataFrame([new_row])
        x_row = profile_features(x_row, profile, config)
        x_row = x_row[feature_cols]

        channel_preds = {channel: float(models[channel].predict(x_row)[0]) for channel in config.channels}
        total_pred = sum(channel_preds.values())

        results.append({
            "date": future_date,
            "total_pred": round(total_pred, 2),
            **{f"{channel}_pred": round(channel_preds[channel], 2) for channel in config.channels},
        })

        new_history_row = {"date": future_date, "total": total_pred}
        new_history_row.update(channel_preds)
        history = pd.concat([history, pd.DataFrame([new_history_row])], ignore_index=True)

    return pd.DataFrame(results)
