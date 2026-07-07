import os

import pandas as pd

from src.pipeline.config import RestaurantConfig

REQUIRED_EVENT_COLUMNS = {"start_date", "end_date", "event_name", "event_type", "impact"}


def load_manual_events(config: RestaurantConfig) -> pd.DataFrame | None:
    path = config.events.manual_events_path
    if not path or not os.path.exists(path):
        return None

    events = pd.read_csv(path)

    missing = REQUIRED_EVENT_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(
            f"Il file eventi '{path}' esiste ma non ha lo schema atteso "
            f"({sorted(REQUIRED_EVENT_COLUMNS)}). Colonne mancanti: {sorted(missing)}."
        )

    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    return events


def build_events_calendar(dates: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.to_datetime(dates.unique())})

    for event_type in events["event_type"].unique():
        calendar[f"event_{event_type}"] = 0

    calendar["big_event"] = 0
    calendar["event_count"] = 0
    calendar["event_impact"] = 0

    for _, row in events.iterrows():
        mask = (calendar["date"] >= row["start_date"]) & (calendar["date"] <= row["end_date"])

        calendar.loc[mask, f"event_{row['event_type']}"] = 1
        calendar.loc[mask, "big_event"] = 1
        calendar.loc[mask, "event_count"] += 1
        calendar.loc[mask, "event_impact"] += row["impact"]

    return calendar


def merge_events(df: pd.DataFrame, config: RestaurantConfig) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    events = load_manual_events(config)
    if events is None:
        return out

    events_calendar = build_events_calendar(out["date"], events)

    out = out.merge(events_calendar, on="date", how="left")

    event_cols = [c for c in out.columns if c.startswith("event_") or c == "big_event"]
    for col in event_cols:
        out[col] = out[col].fillna(0).astype(int)

    return out


def main():
    import argparse

    from src.pipeline.calendar_features import add_calendar_features
    from src.pipeline.clean import clean
    from src.pipeline.config import load_restaurant_config
    from src.pipeline.ingest import ingest
    from src.pipeline.weather import fetch_historical_weather, merge_weather

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    master_df, audit_df = ingest(config)
    clean_df = clean(master_df, audit_df, config)
    weather_df = fetch_historical_weather(
        config, clean_df["date"].min().strftime("%Y-%m-%d"), clean_df["date"].max().strftime("%Y-%m-%d")
    )
    df = merge_weather(clean_df, weather_df)
    df = add_calendar_features(df, config)
    df = merge_events(df, config)

    os.makedirs(config.processed_dir, exist_ok=True)
    df.to_csv(config.processed_path("master_dataset_full.csv"), index=False)

    event_cols = [c for c in df.columns if c.startswith("event_") or c == "big_event"]
    print("Colonne evento:", event_cols)
    if event_cols:
        print(df[event_cols].sum())
    print("Salvato:", config.processed_path("master_dataset_full.csv"))


if __name__ == "__main__":
    main()
