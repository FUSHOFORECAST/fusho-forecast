import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/model_dataset_profile.csv"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGETS = ["total", "delivery", "digital", "cash"]

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
    "delivery_share_target",
    "digital_share_target",
    "cash_share_target",
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
START = len(df) - TEST_DAYS

print("=" * 65)
print("BACKTEST PROFILE FEATURES")
print("=" * 65)
print("Numero feature:", len(FEATURES))

for target in TARGETS:

    real = []
    pred = []

    for i in range(START, len(df)):

        train = df.iloc[:i]
        test = df.iloc[[i]]

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        model.fit(train[FEATURES], train[target])

        prediction = model.predict(test[FEATURES])[0]

        real.append(test[target].values[0])
        pred.append(prediction)

    real = pd.Series(real)
    pred = pd.Series(pred)

    mae = mean_absolute_error(real, pred)
    mape = (
        abs(real - pred) /
        real.replace(0, pd.NA)
    ).dropna().mean() * 100

    print()
    print(target.upper())
    print(f"MAE  : {mae:.2f} €")
    print(f"MAPE : {mape:.2f} %")
