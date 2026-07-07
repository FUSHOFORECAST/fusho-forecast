import os
import pandas as pd

INPUT_FILE = "reports/anomaly_profile.csv"
OUTPUT_FILE = "reports/business_calendar_patterns.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["weekday"] = df["date"].dt.day_name()
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

# settimana del mese: 1,2,3,4,5
df["week_of_month"] = ((df["day"] - 1) // 7) + 1

# weekend
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# periodi commerciali semplici
def commercial_season(row):
    m = row["month"]
    d = row["day"]

    if m == 8:
        return "AUGUST_EMPTY"

    if m == 7 and d <= 15:
        return "EARLY_JULY_ESCAPE"

    if m == 7 and d > 15:
        return "LATE_JULY_ESCAPE"

    if m == 9 and d <= 15:
        return "BACK_TO_CITY"

    if m == 12 and d >= 20:
        return "CHRISTMAS_PERIOD"

    if m == 1 and d <= 7:
        return "NEW_YEAR_PERIOD"

    if m in [3, 4, 5]:
        return "SPRING_STRONG"

    if m in [6]:
        return "EARLY_SUMMER"

    return "NORMAL_SEASON"

df["commercial_season"] = df.apply(commercial_season, axis=1)

# cluster utile: mese + settimana mese + weekend
df["calendar_pattern"] = (
    "M" + df["month"].astype(str)
    + "_WOM" + df["week_of_month"].astype(str)
    + "_" + df["weekday"]
)

# sintesi per pattern
summary = (
    df.groupby("calendar_pattern")
    .agg(
        days=("date", "count"),
        avg_total=("total", "mean"),
        avg_vs_baseline_pct=("vs_baseline_pct", "mean"),
        low_rate=("day_regime", lambda x: (x == "LOW").mean()),
        high_rate=("day_regime", lambda x: (x == "HIGH").mean()),
        normal_rate=("day_regime", lambda x: (x == "NORMAL").mean()),
    )
    .reset_index()
)

summary = summary[summary["days"] >= 2].copy()

summary["low_rate"] = (summary["low_rate"] * 100).round(1)
summary["high_rate"] = (summary["high_rate"] * 100).round(1)
summary["normal_rate"] = (summary["normal_rate"] * 100).round(1)
summary["avg_total"] = summary["avg_total"].round(2)
summary["avg_vs_baseline_pct"] = summary["avg_vs_baseline_pct"].round(2)

# sintesi per stagione commerciale
season_summary = (
    df.groupby("commercial_season")
    .agg(
        days=("date", "count"),
        avg_total=("total", "mean"),
        avg_vs_baseline_pct=("vs_baseline_pct", "mean"),
        low_rate=("day_regime", lambda x: (x == "LOW").mean()),
        high_rate=("day_regime", lambda x: (x == "HIGH").mean()),
        normal_rate=("day_regime", lambda x: (x == "NORMAL").mean()),
    )
    .reset_index()
)

season_summary["low_rate"] = (season_summary["low_rate"] * 100).round(1)
season_summary["high_rate"] = (season_summary["high_rate"] * 100).round(1)
season_summary["normal_rate"] = (season_summary["normal_rate"] * 100).round(1)
season_summary["avg_total"] = season_summary["avg_total"].round(2)
season_summary["avg_vs_baseline_pct"] = season_summary["avg_vs_baseline_pct"].round(2)

# pattern più rischiosi
top_low_patterns = summary.sort_values(
    ["low_rate", "avg_vs_baseline_pct"],
    ascending=[False, True]
).head(25)

top_high_patterns = summary.sort_values(
    ["high_rate", "avg_vs_baseline_pct"],
    ascending=[False, False]
).head(25)

os.makedirs("reports", exist_ok=True)

summary.to_csv(OUTPUT_FILE, index=False)
season_summary.to_csv("reports/business_season_patterns.csv", index=False)

print("=" * 80)
print("BUSINESS CALENDAR ANALYZER")
print("=" * 80)

print("\n=== COMMERCIAL SEASONS ===")
print(season_summary.sort_values("avg_vs_baseline_pct").to_string(index=False))

print("\n=== TOP LOW PATTERNS ===")
print(top_low_patterns.to_string(index=False))

print("\n=== TOP HIGH PATTERNS ===")
print(top_high_patterns.to_string(index=False))

print("\nSalvato:")
print(OUTPUT_FILE)
print("reports/business_season_patterns.csv")
