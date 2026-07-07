import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/model_dataset_full.csv"

TARGET = "total"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

df["weekday"] = df["date"].dt.dayofweek

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

predictions = []

for i in range(start_test, len(df)):

    test_row = df.iloc[[i]]

    weekday = test_row["weekday"].iloc[0]

    train = df.iloc[:i]
    train = train[train["weekday"] == weekday]

    if len(train) < 50:
        continue

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test_row[FEATURES]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)[0]

    predictions.append({
        "date": test_row["date"].iloc[0],
        "weekday": weekday,
        "real": test_row[TARGET].iloc[0],
        "pred": pred,
    })

pred = pd.DataFrame(predictions)

pred["abs_error"] = abs(pred["pred"] - pred["real"])
pred["pct_error"] = pred["abs_error"] / pred["real"] * 100

mae = pred["abs_error"].mean()
mape = pred["pct_error"].mean()

print("=" * 60)
print("7 MODELLI - UNO PER GIORNO")
print("=" * 60)

print(f"MAE  : {mae:.2f} €")
print(f"MAPE : {mape:.2f} %")

print("\nPer giorno:")

print(
    pred.groupby("weekday")["pct_error"]
    .mean()
)

pred.to_csv(
    "data/processed/weekday_models_results.csv",
    index=False,
)

print("\nSalvato:")
print("data/processed/weekday_models_results.csv")
