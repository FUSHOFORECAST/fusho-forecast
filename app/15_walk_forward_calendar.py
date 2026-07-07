import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/model_dataset_calendar.csv"
TARGETS = ["total", "delivery", "digital", "cash"]

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

EXCLUDE = [
    "date", "source_file", "sheet", "holiday_name",
    "total", "delivery", "digital", "cash",
    "delivery_share", "digital_share", "cash_share",
    "pos", "ticket", "satispay", "uber", "just_eat", "glovo", "deliveroo",
]

FEATURES = [c for c in df.columns if c not in EXCLUDE]

TEST_DAYS = 30
start_test_index = len(df) - TEST_DAYS

results = pd.DataFrame()
results["date"] = df.iloc[start_test_index:]["date"].values

print("=" * 60)
print("WALK-FORWARD RF + METEO + CALENDARIO")
print("=" * 60)
print("Feature:", len(FEATURES))

for target in TARGETS:
    real_values = []
    pred_values = []

    for i in range(start_test_index, len(df)):
        train = df.iloc[:i].copy()
        test_row = df.iloc[[i]].copy()

        X_train = train[FEATURES]
        y_train = train[target]

        X_test = test_row[FEATURES]
        y_test = test_row[target].values[0]

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        real_values.append(y_test)
        pred_values.append(pred)

    real_series = pd.Series(real_values)
    pred_series = pd.Series(pred_values)

    mae = mean_absolute_error(real_series, pred_series)

    mape = (
        (abs(real_series - pred_series) / real_series.replace(0, pd.NA))
        .dropna()
        .mean()
        * 100
    )

    print(f"\n{target.upper()}")
    print(f"MAE  : {mae:.2f} €")
    print(f"MAPE : {mape:.2f} %")

    results[f"{target}_real"] = real_values
    results[f"{target}_pred"] = pred_values
    results[f"{target}_error"] = pred_series - real_series

results.to_csv("data/processed/walk_forward_calendar_results.csv", index=False)

print("\nSalvato: data/processed/walk_forward_calendar_results.csv")
