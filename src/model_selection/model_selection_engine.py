import json
import os

DNA_FILE = "reports/restaurant_dna_universal.json"
BUSINESS_PROFILE_FILE = "reports/business_model_profile.json"
OUTPUT_FILE = "reports/model_selection.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    dna = load_json(DNA_FILE)

    business_profile = None
    if os.path.exists(BUSINESS_PROFILE_FILE):
        business_profile = load_json(BUSINESS_PROFILE_FILE)

    tags = dna["business_tags"]

    cv = dna["scale"]["coefficient_of_variation"]
    seasonality = dna["seasonality"]["month_spread_pct"]
    weekday_spread = dna["weekday_behavior"]["weekday_spread_pct"]
    delivery_share = dna["channel_mix"]["delivery_share"]
    cash_share = dna["channel_mix"]["cash_share"]

    strategy = "weighted_hierarchical_forecast"
    reasons = []

    if tags["delivery_driven"] or delivery_share >= 0.55:
        strategy = "weighted_hierarchical_forecast"
        reasons.append("High delivery share: hierarchical channel forecast recommended.")

    if tags["cash_light"] or cash_share <= 0.15:
        reasons.append("Cash-light business: cash should be modeled as share, not absolute revenue.")

    if tags["volatile"] or cv >= 0.25:
        reasons.append("Volatile business: confidence/risk layer required.")

    if tags["seasonal"] or seasonality >= 25:
        reasons.append("Strong seasonality: business calendar and seasonal profile required.")

    if tags["weekday_sensitive"] or weekday_spread >= 15:
        reasons.append("Weekday-sensitive business: weekday-specific features required.")

    if business_profile:
        recommended = business_profile.get("business_model", {}).get(
            "recommended_forecast_strategy"
        )

        if recommended == "regime_aware_forecast":
            strategy = "weighted_hierarchical_with_regime_warning"
            reasons.append("Regime-sensitive business: add regime warning and anomaly detection.")

    output = {
        "selected_strategy": strategy,
        "required_modules": [
            "data_quality_engine",
            "restaurant_dna",
            "similarity_engine",
            "weighted_hierarchical_forecast",
            "confidence_engine",
        ],
        "optional_modules": [],
        "reasons": reasons,
    }

    if tags["seasonal"]:
        output["required_modules"].append("business_calendar_engine")

    if tags["volatile"]:
        output["required_modules"].append("volatility_diagnostics")

    if business_profile:
        if business_profile.get("pattern_sensitivity", {}).get("requires_regime_model"):
            output["required_modules"].append("adaptive_regime_engine")
            output["optional_modules"].append("regime_aware_forecast")

    os.makedirs("reports", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print("MODEL SELECTION ENGINE")
    print("=" * 80)
    print(json.dumps(output, indent=2))
    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
