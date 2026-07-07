import pandas as pd

from sklearn.metrics import mean_absolute_error

from src.models.random_forest_model import create_model as RF

from src.models.catboost_model import create_model as CAT

DATASET = "data/processed/model_dataset_share.csv"

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

df = pd.read_csv(DATASET)

df["date"] = pd.to_datetime(df["date"])

train = df.iloc[:-30]

test = df.iloc[-30:]

features = [c for c in df.columns if c not in EXCLUDE]

models = {
    "Random Forest": RF(),
    "CatBoost": CAT(),
}

print("=" * 70)
print("MODEL BENCHMARK")
print("=" * 70)

results = []

for name, model in models.items():

    model.fit(train[features], train["total"])

    pred = model.predict(test[features])

    mae = mean_absolute_error(test["total"], pred)

    mape = (
        (abs(pred - test["total"]) / test["total"])
        .mean()
        * 100
    )

    print()

    print(name)

    print("MAE :", round(mae, 2))

    print("MAPE:", round(mape, 2), "%")

    results.append(
        {
            "Model": name,
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
        }
    )

print()

print(pd.DataFrame(results))
