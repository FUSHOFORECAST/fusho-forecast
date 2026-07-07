import pandas as pd
import numpy as np

INPUT = "data/processed/master_dataset_full.csv"
OUTPUT = "data/processed/master_dataset_full.csv"

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df["dayofyear"] = df["date"].dt.dayofyear

##########################################################
# MEDIA STORICA DEL GIORNO DELL'ANNO
##########################################################

seasonality = (
    df.groupby("dayofyear")["total"]
    .mean()
    .rename("seasonality_total")
)

df = df.merge(
    seasonality,
    on="dayofyear",
    how="left"
)

##########################################################
# DELIVERY
##########################################################

seasonality = (
    df.groupby("dayofyear")["delivery"]
    .mean()
    .rename("seasonality_delivery")
)

df = df.merge(
    seasonality,
    on="dayofyear",
    how="left"
)

##########################################################
# DIGITAL
##########################################################

seasonality = (
    df.groupby("dayofyear")["digital"]
    .mean()
    .rename("seasonality_digital")
)

df = df.merge(
    seasonality,
    on="dayofyear",
    how="left"
)

##########################################################
# CASH
##########################################################

seasonality = (
    df.groupby("dayofyear")["cash"]
    .mean()
    .rename("seasonality_cash")
)

df = df.merge(
    seasonality,
    on="dayofyear",
    how="left"
)

df.to_csv(OUTPUT, index=False)

print("="*60)
print("FEATURE STAGIONALI AGGIUNTE")
print("="*60)

print(df[
    [
        "date",
        "seasonality_total",
        "seasonality_delivery"
    ]
].tail())

print("\nColonne totali:", len(df.columns))
