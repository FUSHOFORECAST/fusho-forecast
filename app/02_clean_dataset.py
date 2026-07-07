import pandas as pd
import os

df = pd.read_csv("data/processed/master_dataset.csv")
audit = pd.read_csv("data/processed/audit_totals.csv")

df["date"] = pd.to_datetime(df["date"])
audit["date"] = pd.to_datetime(audit["date"])

bad_dates = audit[audit["difference"].abs() > 1]["date"]

df_clean = df[~df["date"].isin(bad_dates)].copy()

os.makedirs("data/processed", exist_ok=True)
df_clean.to_csv("data/processed/master_dataset_clean.csv", index=False)

print("Righe originali:", len(df))
print("Righe pulite:", len(df_clean))
print("Date escluse:")
print(bad_dates)
print("\nDelivery share pulita:", df_clean["delivery_share"].mean())
