import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/model_dataset.csv"

TARGETS = ["total", "delivery", "digital", "cash"]

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

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

TEST_DAYS = 30

train = df.iloc[:-TEST_DAYS].copy()
test = df.iloc[-TEST_DAYS:].copy()

results = pd.DataFrame()
results["date"] = test["date"]

print("=" * 60)
print("BACKTEST RANDOM FOREST V2")
print("=" * 60)
print("Numero feature:", len(FEATURES))

for target in TARGETS:
    X_train = train[FEATURES]
    y_train = train[target]

    X_test = test[FEATURES]
    y_test = test[target]

    model = RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    mape = (
        (abs(y_test - pred) / y_test.replace(0, pd.NA))
        .dropna()
        .mean()
        * 100
    )

    print(f"\n{target.upper()}")
    print(f"MAE  : {mae:.2f} €")
    print(f"MAPE : {mape:.2f} %")

    results[f"{target}_real"] = y_test.values
    results[f"{target}_pred"] = pred
    results[f"{target}_error"] = pred - y_test.values

results.to_csv("data/processed/backtest_results.csv", index=False)

print("\n" + "=" * 60)
print("BACKTEST COMPLETATO")
print("=" * 60)
print("File salvato:")
print("data/processed/backtest_results.csv")
