import pandas as pd
import os

INPUT_FILE = "data/processed/master_dataset_clean.csv"
OUTPUT_FILE = "data/processed/model_dataset.csv"

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ==========================
# CALENDAR
# ==========================

df["dayofweek"] = df["date"].dt.dayofweek
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["dayofyear"] = df["date"].dt.dayofyear

df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# ==========================
# STANDARD LAG
# ==========================

for col in ["total","delivery","digital","cash"]:

    for lag in [1,7,14,21,28,30]:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df[f"{col}_rolling_7"] = df[col].shift(1).rolling(7).mean()
    df[f"{col}_rolling_14"] = df[col].shift(1).rolling(14).mean()
    df[f"{col}_rolling_30"] = df[col].shift(1).rolling(30).mean()

# ==========================
# STESSO GIORNO ANNO PRECEDENTE
# ==========================

for col in ["total","delivery","digital","cash"]:

    previous = (
        df[["date",col]]
        .copy()
    )

    previous["date"] = previous["date"] + pd.DateOffset(years=1)

    previous.columns = ["date",f"{col}_last_year"]

    df = df.merge(previous,on="date",how="left")

# ==========================
# TREND
# ==========================

df["total_trend"] = df["total_rolling_7"] / df["total_rolling_30"]
df["delivery_trend"] = df["delivery_rolling_7"] / df["delivery_rolling_30"]
df["digital_trend"] = df["digital_rolling_7"] / df["digital_rolling_30"]
df["cash_trend"] = df["cash_rolling_7"] / df["cash_rolling_30"]

# ==========================
# SHARE
# ==========================

df["delivery_share_lag"] = df["delivery_lag_7"] / df["total_lag_7"]
df["digital_share_lag"] = df["digital_lag_7"] / df["total_lag_7"]
df["cash_share_lag"] = df["cash_lag_7"] / df["total_lag_7"]

# ==========================
# CLEAN
# ==========================

df = df.replace([float("inf"),-float("inf")],pd.NA)
df = df.dropna()

os.makedirs("data/processed",exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("="*60)
print("MODEL DATASET V3")
print("="*60)
print("Rows:",len(df))
print("Columns:",len(df.columns))
print(df.tail())
