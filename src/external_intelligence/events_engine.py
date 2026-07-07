import os
import pandas as pd

EVENTS_FILE = "data/external/events_manual.csv"
CALENDAR_FILE = "reports/external_intelligence/calendar_features.csv"
OUTPUT = "reports/external_intelligence/events_features.csv"


def ensure_events_file():
    os.makedirs("data/external", exist_ok=True)

    if os.path.exists(EVENTS_FILE):
        return

    sample = pd.DataFrame([
        {
            "start_date": "2026-06-01",
            "end_date": "2026-06-03",
            "event_name": "ponte_2_giugno",
            "event_type": "bridge",
            "city": "Milano",
            "area": "citywide",
            "expected_impact": "negative",
            "impact_strength": 2,
            "target_channel": "total",
        },
        {
            "start_date": "2026-06-02",
            "end_date": "2026-06-02",
            "event_name": "festa_repubblica",
            "event_type": "holiday",
            "city": "Milano",
            "area": "citywide",
            "expected_impact": "negative",
            "impact_strength": 3,
            "target_channel": "total",
        },
    ])

    sample.to_csv(EVENTS_FILE, index=False)


def run_events_engine():
    os.makedirs("reports/external_intelligence", exist_ok=True)
    ensure_events_file()

    calendar = pd.read_csv(CALENDAR_FILE)
    calendar["date"] = pd.to_datetime(calendar["date"])

    events = pd.read_csv(EVENTS_FILE)
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    df = calendar[["date"]].copy()

    df["external_event_count"] = 0
    df["external_positive_strength"] = 0
    df["external_negative_strength"] = 0
    df["external_mixed_strength"] = 0
    df["external_total_strength"] = 0

    for event_type in sorted(events["event_type"].dropna().unique()):
        df[f"event_type_{event_type}"] = 0

    for channel in sorted(events["target_channel"].dropna().unique()):
        df[f"event_channel_{channel}_strength"] = 0

    for _, event in events.iterrows():
        mask = (
            (df["date"] >= event["start_date"])
            & (df["date"] <= event["end_date"])
        )

        strength = int(event["impact_strength"])
        direction = event["expected_impact"]
        event_type = event["event_type"]
        channel = event["target_channel"]

        df.loc[mask, "external_event_count"] += 1
        df.loc[mask, f"event_type_{event_type}"] = 1
        df.loc[mask, f"event_channel_{channel}_strength"] += strength

        if direction == "positive":
            df.loc[mask, "external_positive_strength"] += strength
            df.loc[mask, "external_total_strength"] += strength
        elif direction == "negative":
            df.loc[mask, "external_negative_strength"] += strength
            df.loc[mask, "external_total_strength"] -= strength
        else:
            df.loc[mask, "external_mixed_strength"] += strength

    df.to_csv(OUTPUT, index=False)

    return OUTPUT


if __name__ == "__main__":
    path = run_events_engine()
    print("Events features salvate:", path)
