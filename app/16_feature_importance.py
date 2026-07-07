import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA = "data/processed/model_dataset_calendar.csv"

TARGET = "total"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

EXCLUDE = [
    "date",
    "source_file",
    "sheet",
    "holiday_name",

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

X = df[FEATURES]
y = df[TARGET]

model = RandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

model.fit(X, y)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n")
print("="*70)
print("FEATURE IMPORTANCE")
print("="*70)

print(
    importance.head(40).to_string(index=False)
)

importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print("\nSalvato:")
print("data/processed/feature_importance.csv")
