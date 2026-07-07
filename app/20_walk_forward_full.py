import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/model_dataset_full.csv"
TARGETS = ["total", "delivery", "digital", "cash"]

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

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

TEST_DAYS = 30
start_test = len(df) - TEST_DAYS

print("=" * 65)
print("WALK FORWARD - FULL MODEL")
print("=" * 65)
print("Numero feature:", len(FEATURES))

results = pd.DataFrame()
results["date"] = df.iloc[start_test:]["date"].values

for target in TARGETS:

    real = []
    pred = []

    for i in range(start_test, len(df)):

        train = df.iloc[:i]

        test = df.iloc[[i]]

        X_train = train[FEATURES]
        y_train = train[target]

        X_test = test[FEATURES]

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        model.fit(X_train, y_train)

        p = model.predict(X_test)[0]

        pred.append(p)
        real.append(test[target].values[0])

    real = pd.Series(real)
    pred = pd.Series(pred)

    mae = mean_absolute_error(real, pred)

    mape = (
        (abs(real - pred) / real.replace(0, pd.NA))
        .dropna()
        .mean()
        * 100
    )

    print(f"\n{target.upper()}")
    print(f"MAE  : {mae:.2f} €")
    print(f"MAPE : {mape:.2f} %")

    results[f"{target}_real"] = real
    results[f"{target}_pred"] = pred

results.to_csv(
    "data/processed/walk_forward_full.csv",
    index=False,
)

print("\nSalvato:")
print("data/processed/walk_forward_full.csv")
