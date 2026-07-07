import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

DATA = "data/processed/model_dataset.csv"

TARGETS = ["total", "delivery", "digital", "cash"]

df = pd.read_csv(DATA)

EXCLUDE = [
    "date",
    "source_file",
    "sheet",
    "total",
    "delivery",
    "digital",
    "cash",
    "delivery_share",
    "digital_share",
    "cash_share",
    "pos",
    "ticket",
    "satispay",
    "uber",
    "just_eat",
    "glovo",
    "deliveroo",
]

FEATURES = [c for c in df.columns if c not in EXCLUDE]

os.makedirs("models", exist_ok=True)

for target in TARGETS:
    X = df[FEATURES]
    y = df[target]

    model = RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )

    model.fit(X, y)

    joblib.dump(
        {"model": model, "features": FEATURES},
        f"models/{target}_model.pkl"
    )

    print(f"Modello salvato: models/{target}_model.pkl")

print("\nTRAINING COMPLETATO")
print("Numero feature:", len(FEATURES))
