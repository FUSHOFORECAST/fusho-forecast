import pandas as pd

df = pd.read_csv("data/processed/master_dataset_clean.csv")
df["date"] = pd.to_datetime(df["date"])

df = df[df["total"] > 0].copy()

df["cash_share"] = df["cash"] / df["total"]

print("=== CASH SHARE ===")
print(df["cash_share"].describe())

print("\nMedia cash share:", round(df["cash_share"].mean(), 4))
print("Mediana cash share:", round(df["cash_share"].median(), 4))

print("\nCash share per anno:")
print(df.groupby(df["date"].dt.year)["cash_share"].mean())

print("\nCash share per giorno settimana:")
print(df.groupby(df["date"].dt.day_name())["cash_share"].mean().sort_values())
