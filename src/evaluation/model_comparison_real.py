import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

DATA_FILE = "data/processed/model_dataset_share.csv"
REAL_FILE = "data/processed/june_real_totals.csv"

EXCLUDE = [
    "date", "source_file", "sheet", "holiday_name",
    "total", "delivery", "digital", "cash",
    "delivery_share", "digital_share", "cash_share",
    "delivery_share_target", "digital_share_target", "cash_share_target",
    "pos", "ticket", "satispay", "uber", "just_eat", "glovo", "deliveroo",
]

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

real = pd.read_csv(REAL_FILE)
real["date"] = pd.to_datetime(real["date"])
real = real.dropna(subset=["real_total"]).copy()

features = [c for c in df.columns if c not in EXCLUDE]

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        random_state=42,
        learning_rate=0.04,
        max_depth=3,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=42,
    ),
}

results = []

train = df.copy()

future_rows = []

# usa le ultime righe del dataset come base per creare righe future semplici
history = df.copy()

for _, real_row in real.iterrows():
    target_date = real_row["date"]

    last = history.iloc[-1].copy()
    new_row = last.copy()
    new_row["date"] = target_date

    new_row["year"] = target_date.year
    new_row["month"] = target_date.month
    new_row["day"] = target_date.day
    new_row["dayofweek"] = target_date.dayofweek
    new_row["week"] = target_date.isocalendar().week
    new_row["dayofyear"] = target_date.dayofyear
    new_row["is_weekend"] = int(target_date.dayofweek in [5, 6])
    new_row["is_month_start"] = int(target_date.is_month_start)
    new_row["is_month_end"] = int(target_date.is_month_end)

    for col in ["total", "delivery", "digital", "cash"]:
        new_row[f"{col}_lag_1"] = history[col].iloc[-1]
        new_row[f"{col}_lag_7"] = history[col].iloc[-7]
        new_row[f"{col}_lag_14"] = history[col].iloc[-14]
        new_row[f"{col}_lag_21"] = history[col].iloc[-21]
        new_row[f"{col}_lag_28"] = history[col].iloc[-28]
        new_row[f"{col}_lag_30"] = history[col].iloc[-30]

        new_row[f"{col}_rolling_7"] = history[col].tail(7).mean()
        new_row[f"{col}_rolling_14"] = history[col].tail(14).mean()
        new_row[f"{col}_rolling_30"] = history[col].tail(30).mean()

    new_row["total_trend"] = new_row["total_rolling_7"] / new_row["total_rolling_30"]
    new_row["delivery_trend"] = new_row["delivery_rolling_7"] / new_row["delivery_rolling_30"]
    new_row["digital_trend"] = new_row["digital_rolling_7"] / new_row["digital_rolling_30"]
    new_row["cash_trend"] = new_row["cash_rolling_7"] / new_row["cash_rolling_30"]

    new_row["delivery_share_lag"] = new_row["delivery_lag_7"] / new_row["total_lag_7"]
    new_row["digital_share_lag"] = new_row["digital_lag_7"] / new_row["total_lag_7"]
    new_row["cash_share_lag"] = new_row["cash_lag_7"] / new_row["total_lag_7"]

    future_rows.append(new_row)

    # per confronto multi-step usiamo la riga prevista come storico provvisorio
    # qui mettiamo il reale solo dopo la validazione? No: per test forecast puro non usiamo il reale.
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

future = pd.DataFrame(future_rows)

print("=" * 80)
print("MODEL COMPARISON REAL JUNE")
print("=" * 80)

for name, model in models.items():
    model.fit(train[features], train["total"])

    pred = model.predict(future[features])

    temp = pd.DataFrame({
        "date": real["date"].values,
        "model": name,
        "total_pred": pred,
        "real_total": real["real_total"].values,
    })

    temp["error"] = temp["total_pred"] - temp["real_total"]
    temp["abs_error"] = temp["error"].abs()
    temp["pct_error"] = temp["abs_error"] / temp["real_total"] * 100

    mae = mean_absolute_error(temp["real_total"], temp["total_pred"])
    mape = temp["pct_error"].mean()
    total_error_pct = (
        (temp["total_pred"].sum() - temp["real_total"].sum())
        / temp["real_total"].sum()
        * 100
    )
    volatility = temp["total_pred"].std()

    print()
    print(name)
    print("MAE:", round(mae, 2))
    print("MAPE:", round(mape, 2), "%")
    print("Errore totale:", round(total_error_pct, 2), "%")
    print("Volatilità forecast:", round(volatility, 2))
    print(temp[["date", "total_pred", "real_total", "pct_error"]].to_string(index=False))

    results.append(temp)

final = pd.concat(results)
final.to_csv("data/processed/model_comparison_real.csv", index=False)

print("\nSalvato: data/processed/model_comparison_real.csv")
