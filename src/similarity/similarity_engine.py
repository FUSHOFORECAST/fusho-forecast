import json
import math
import os

INPUT_FILE = "reports/restaurant_dna_universal.json"
OUTPUT_FILE = "reports/restaurant_similarity_vector.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0.0

    value = max(min_value, min(value, max_value))
    return round((value - min_value) / (max_value - min_value), 4)


def main():
    dna = load_json(INPUT_FILE)

    vector = {
        "scale_avg_revenue": normalize(
            dna["scale"]["avg_daily_revenue"],
            0,
            10000,
        ),
        "scale_volatility": normalize(
            dna["scale"]["coefficient_of_variation"],
            0,
            0.80,
        ),
        "delivery_share": normalize(
            dna["channel_mix"]["delivery_share"],
            0,
            1,
        ),
        "digital_share": normalize(
            dna["channel_mix"]["digital_share"],
            0,
            1,
        ),
        "cash_share": normalize(
            dna["channel_mix"]["cash_share"],
            0,
            1,
        ),
        "revenue_trend": normalize(
            dna["trend"]["revenue_trend_first_to_last_year_pct"],
            -80,
            120,
        ),
        "delivery_trend": normalize(
            dna["trend"]["delivery_share_trend_first_to_last_year_pct"],
            -80,
            120,
        ),
        "cash_trend": normalize(
            dna["trend"]["cash_share_trend_first_to_last_year_pct"],
            -80,
            120,
        ),
        "weekday_spread": normalize(
            dna["weekday_behavior"]["weekday_spread_pct"],
            0,
            80,
        ),
        "seasonality_spread": normalize(
            dna["seasonality"]["month_spread_pct"],
            0,
            120,
        ),
        "weekend_effect": normalize(
            dna["weekend"]["weekend_effect_pct"],
            -50,
            50,
        ),
    }

    tags = dna["business_tags"]

    tag_vector = {
        f"tag_{k}": 1 if v else 0
        for k, v in tags.items()
    }

    output = {
        "restaurant_id": "current_restaurant",
        "schema_version": dna["schema_version"],
        "similarity_vector": vector,
        "tag_vector": tag_vector,
    }

    os.makedirs("reports", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("SIMILARITY VECTOR")
    print("=" * 80)
    print(json.dumps(output, indent=2))
    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
