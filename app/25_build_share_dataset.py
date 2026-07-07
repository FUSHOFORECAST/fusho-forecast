import pandas as pd

INPUT_FILE = "data/processed/model_dataset_full.csv"
OUTPUT_FILE = "data/processed/model_dataset_share.csv"

df = pd.read_csv(INPUT_FILE)

print("Righe iniziali:", len(df))

# usiamo solo giorni con fatturato maggiore di zero
df = df[df["total"] > 0].copy()

# creiamo le quote
df["delivery_share_target"] = df["delivery"] / df["total"]
df["digital_share_target"] = df["digital"] / df["total"]
df["cash_share_target"] = df["cash"] / df["total"]

# pulizia SOLO sulle 3 quote, non su tutto il dataset
df = df.replace([float("inf"), -float("inf")], pd.NA)

df = df.dropna(
    subset=[
        "delivery_share_target",
        "digital_share_target",
        "cash_share_target",
    ]
)

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("DATASET SHARE CREATO")
print("=" * 60)

print("Righe finali:", len(df))

print(
    df[
        [
            "delivery_share_target",
            "digital_share_target",
            "cash_share_target",
        ]
    ].describe()
)

print("\nSalvato:")
print(OUTPUT_FILE)
