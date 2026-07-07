import os
import pandas as pd

OUTPUT = "reports/external_intelligence/calendar_features.csv"

START_DATE = "2024-01-01"
END_DATE = "2026-12-31"

HOLIDAYS_IT = {
    "2024-01-01": "capodanno",
    "2024-01-06": "epifania",
    "2024-04-01": "pasquetta",
    "2024-04-25": "liberazione",
    "2024-05-01": "lavoro",
    "2024-06-02": "repubblica",
    "2024-08-15": "ferragosto",
    "2024-11-01": "ognissanti",
    "2024-12-08": "immacolata",
    "2024-12-25": "natale",
    "2024-12-26": "santo_stefano",

    "2025-01-01": "capodanno",
    "2025-01-06": "epifania",
    "2025-04-21": "pasquetta",
    "2025-04-25": "liberazione",
    "2025-05-01": "lavoro",
    "2025-06-02": "repubblica",
    "2025-08-15": "ferragosto",
    "2025-11-01": "ognissanti",
    "2025-12-08": "immacolata",
    "2025-12-25": "natale",
    "2025-12-26": "santo_stefano",

    "2026-01-01": "capodanno",
    "2026-01-06": "epifania",
    "2026-04-06": "pasquetta",
    "2026-04-25": "liberazione",
    "2026-05-01": "lavoro",
    "2026-06-02": "repubblica",
    "2026-08-15": "ferragosto",
    "2026-11-01": "ognissanti",
    "2026-12-08": "immacolata",
    "2026-12-25": "natale",
    "2026-12-26": "santo_stefano",
}


def commercial_season(date):
    m = date.month
    d = date.day

    if m == 8:
        return "AUGUST_EMPTY"
    if m == 7 and d <= 15:
        return "EARLY_JULY"
    if m == 7 and d > 15:
        return "LATE_JULY"
    if m == 9 and d <= 15:
        return "BACK_TO_CITY"
    if m == 12 and d >= 20:
        return "CHRISTMAS_PERIOD"
    if m == 1 and d <= 7:
        return "NEW_YEAR_PERIOD"
    if m in [3, 4, 5]:
        return "SPRING"
    if m == 6:
        return "EARLY_SUMMER"

    return "NORMAL"


def run_calendar_engine():
    os.makedirs("reports/external_intelligence", exist_ok=True)

    df = pd.DataFrame({
        "date": pd.date_range(START_DATE, END_DATE, freq="D")
    })

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekday"] = df["date"].dt.day_name()
    df["dayofyear"] = df["date"].dt.dayofyear
    df["week_of_month"] = ((df["day"] - 1) // 7) + 1
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    holiday_dates = {pd.to_datetime(k): v for k, v in HOLIDAYS_IT.items()}

    df["holiday_name"] = df["date"].map(holiday_dates).fillna("")
    df["is_public_holiday"] = (df["holiday_name"] != "").astype(int)

    df["is_preholiday"] = df["date"].apply(
        lambda x: int((x + pd.Timedelta(days=1)) in holiday_dates)
    )

    df["is_postholiday"] = df["date"].apply(
        lambda x: int((x - pd.Timedelta(days=1)) in holiday_dates)
    )

    df["is_bridge_candidate"] = (
        (df["is_preholiday"] == 1)
        | (df["is_postholiday"] == 1)
    ).astype(int)

    df["is_august"] = (df["month"] == 8).astype(int)
    df["is_summer_period"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_christmas_period"] = (
        ((df["month"] == 12) & (df["day"] >= 20))
        | ((df["month"] == 1) & (df["day"] <= 7))
    ).astype(int)

    df["commercial_season"] = df["date"].apply(commercial_season)

    holidays = sorted(holiday_dates.keys())

    df["days_to_next_holiday"] = df["date"].apply(
        lambda x: min([(h - x).days for h in holidays if h >= x], default=999)
    )

    df["days_from_previous_holiday"] = df["date"].apply(
        lambda x: min([(x - h).days for h in holidays if h <= x], default=999)
    )

    df.to_csv(OUTPUT, index=False)

    return OUTPUT


if __name__ == "__main__":
    path = run_calendar_engine()
    print("Calendar features salvate:", path)
