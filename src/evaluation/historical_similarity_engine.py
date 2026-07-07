import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

HISTORY_FILE = "data/processed/model_dataset_share.csv"
FORECAST_FILE = "data/processed/forecast_adaptive_memory_v2.csv"
OUTPUT_FILE = "reports/historical_similarity_forecast.csv"

history = pd.read_csv(HISTORY_FILE)
forecast = pd.read_csv(FORECAST_FILE)

history["date"] = pd.to_datetime(history["date"])
forecast["date"] = pd.to_datetime(forecast["date"])

def add_calendar_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["day"] = df["date"].dt.day
    df["week_of_month"] = ((df["day"] - 1) // 7) + 1
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

history = add_calendar_features(history)
forecast = add_calendar_features(forecast)

SIMILARITY_FEATURES = [
    "month",
    "dayofweek",
    "dayofyear",
    "week_of_month",
    "is_weekend",
]

history_data = history.dropna(subset=SIMILARITY_FEATURES + ["total"]).copy()
forecast_data = forecast.copy()

scaler = StandardScaler()

X_history = scaler.fit_transform(history_data[SIMILARITY_FEATURES])
X_forecast = scaler.transform(forecast_data[SIMILARITY_FEATURES])

rows = []
TOP_N = 10

for i, frow in forecast_data.iterrows():
    sims = cosine_similarity([X_forecast[i]], X_history)[0]

    temp = history_data.copy()
    temp["similarity"] = sims

    # evita confronto con date future impossibili
    temp = temp[temp["date"] < frow["date"]]

    top = temp.sort_values("similarity", ascending=False).head(TOP_N).copy()

    similar_avg_total = top["total"].mean()
    similar_median_total = top["total"].median()

    base_forecast = frow["total_pred"]

    correction_factor = similar_median_total / base_forecast if base_forecast != 0 else 1

    # correzione prudente
    corrected_forecast = base_forecast * (0.70 + 0.30 * correction_factor)

    rows.append({
        "date": frow["date"],
        "base_forecast": round(base_forecast, 2),
        "similar_avg_total": round(similar_avg_total, 2),
        "similar_median_total": round(similar_median_total, 2),
        "correction_factor": round(correction_factor, 4),
        "corrected_forecast": round(corrected_forecast, 2),
        "avg_similarity": round(top["similarity"].mean(), 4),
        "top_match_1_date": top.iloc[0]["date"],
        "top_match_1_total": round(top.iloc[0]["total"], 2),
        "top_match_1_similarity": round(top.iloc[0]["similarity"], 4),
        "top_match_2_date": top.iloc[1]["date"],
        "top_match_2_total": round(top.iloc[1]["total"], 2),
        "top_match_2_similarity": round(top.iloc[1]["similarity"], 4),
        "top_match_3_date": top.iloc[2]["date"],
        "top_match_3_total": round(top.iloc[2]["total"], 2),
        "top_match_3_similarity": round(top.iloc[2]["similarity"], 4),
    })

result = pd.DataFrame(rows)

os.makedirs("reports", exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

print("=" * 80)
print("HISTORICAL SIMILARITY ENGINE V1")
print("=" * 80)

print("Feature usate:")
print(SIMILARITY_FEATURES)

print()
print(result.to_string(index=False))

print("\nTOTALE BASE:", round(result["base_forecast"].sum(), 2))
print("TOTALE CORRETTO:", round(result["corrected_forecast"].sum(), 2))
print("\nSalvato:", OUTPUT_FILE)
