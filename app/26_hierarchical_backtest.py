import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = "data/processed/model_dataset_share.csv"

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

EXCLUDE = [
    "date", "source_file", "sheet", "holiday_name",
    "total", "delivery", "digital", "cash",
    "delivery_share", "digital_share", "cash_share",
    "delivery_share_target", "digital_share_target", "cash_share_target",
    "pos", "ticket", "satispay", "uber", "just_eat", "glovo", "deliveroo",
]

FEATURES = [c for c in df.columns if c not in EXCLUDE]

TEST_DAYS = 30
START = len(df) - TEST_DAYS

results = []

print("=" * 65)
print("BACKTEST GERARCHICO FAST")
print("=" * 65)
print("Feature:", len(FEATURES))

for i in range(START, len(df)):
    print(f"Giorno test {i - START + 1}/{TEST_DAYS}")

    train = df.iloc[:i]
    test = df.iloc[[i]]

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
    )

    # totale
    model.fit(X_train, train["total"])
    total_pred = model.predict(X_test)[0]

    # quote
    model.fit(X_train, train["delivery_share_target"])
    delivery_share = model.predict(X_test)[0]

    model.fit(X_train, train["digital_share_target"])
    digital_share = model.predict(X_test)[0]

    model.fit(X_train, train["cash_share_target"])
    cash_share = model.predict(X_test)[0]

    # evita quote negative o strane
    delivery_share = max(0, delivery_share)
    digital_share = max(0, digital_share)
    cash_share = max(0, cash_share)

    share_sum = delivery_share + digital_share + cash_share

    if share_sum == 0:
        delivery_share = 0.62
        digital_share = 0.25
        cash_share = 0.13
        share_sum = 1

    delivery_share /= share_sum
    digital_share /= share_sum
    cash_share /= share_sum

    results.append({
        "date": test["date"].iloc[0],
        "total_real": test["total"].iloc[0],
        "total_pred": total_pred,
        "delivery_real": test["delivery"].iloc[0],
        "delivery_pred": total_pred * delivery_share,
        "digital_real": test["digital"].iloc[0],
        "digital_pred": total_pred * digital_share,
        "cash_real": test["cash"].iloc[0],
        "cash_pred": total_pred * cash_share,
    })

results = pd.DataFrame(results)

print("\n" + "=" * 65)
print("RISULTATI")
print("=" * 65)

for target in ["total", "delivery", "digital", "cash"]:
    mae = mean_absolute_error(results[f"{target}_real"], results[f"{target}_pred"])
    mape = (
        abs(results[f"{target}_real"] - results[f"{target}_pred"])
        / results[f"{target}_real"]
    ).mean() * 100

    print(f"\n{target.upper()}")
    print(f"MAE  : {mae:.2f} €")
    print(f"MAPE : {mape:.2f}%")

results.to_csv("data/processed/hierarchical_results.csv", index=False)

print("\nSalvato:")
print("data/processed/hierarchical_results.csv")
