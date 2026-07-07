from datetime import timedelta

import pandas as pd

from src.pipeline.calendar_features import add_calendar_features
from src.pipeline.config import RestaurantConfig
from src.pipeline.events import merge_events
from src.pipeline.features import add_calendar_derived_fields
from src.pipeline.profile import profile_features
from src.pipeline.weather import (
    MAX_FORECAST_DAYS,
    estimate_seasonal_weather,
    fetch_forecast_weather,
    fetch_historical_weather,
    merge_weather,
)


def build_future_calendar_weather_events(
    config: RestaurantConfig,
    last_date: pd.Timestamp,
    horizon_days: int,
    historical_weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Costruisce calendario/meteo/eventi per i prossimi horizon_days giorni
    dopo last_date. last_date puo' essere molto indietro rispetto ad oggi
    (es. dati compilati una volta al mese): il meteo va quindi sorgente in
    modo diverso a seconda di quanto ogni data e' lontana da oggi --
    l'endpoint forecast di Open-Meteo restituisce sempre il meteo a partire
    da oggi reale, non da last_date, quindi non si puo' assumere che le
    prime horizon_days coincidano con l'oggi dell'API.
      - date gia' passate (rispetto ad oggi reale): meteo storico vero (archive)
      - da oggi ai prossimi MAX_FORECAST_DAYS giorni: meteo previsto vero (forecast)
      - oltre: climatologia stagionale (nessun fornitore da' previsioni reali li')
    """
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    future_df = pd.DataFrame({"date": future_dates})

    future_df = add_calendar_features(future_df, config)

    today = pd.Timestamp.now().normalize()

    past_dates = [d for d in future_dates if d < today]
    forecast_dates = [d for d in future_dates if today <= d < today + timedelta(days=MAX_FORECAST_DAYS)]
    far_dates = [d for d in future_dates if d >= today + timedelta(days=MAX_FORECAST_DAYS)]

    weather_parts = []

    if past_dates:
        weather_parts.append(
            fetch_historical_weather(config, past_dates[0].strftime("%Y-%m-%d"), past_dates[-1].strftime("%Y-%m-%d"))
        )

    if forecast_dates:
        weather_fc = fetch_forecast_weather(config, len(forecast_dates))
        weather_parts.append(weather_fc[weather_fc["date"].isin(forecast_dates)])

    if far_dates:
        if historical_weather_df is None:
            raise ValueError(
                f"{len(far_dates)} giorni dell'orizzonte sono oltre i {MAX_FORECAST_DAYS} giorni di previsione "
                f"meteo reale: serve historical_weather_df per stimarli via climatologia stagionale."
            )
        weather_parts.append(estimate_seasonal_weather(historical_weather_df, far_dates))

    weather_combined = pd.concat(weather_parts, ignore_index=True)
    future_df = merge_weather(future_df, weather_combined)

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
        future_meta = build_future_calendar_weather_events(
            config, last_date, config.forecast.horizon_days, historical_weather_df=history_df
        )

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
