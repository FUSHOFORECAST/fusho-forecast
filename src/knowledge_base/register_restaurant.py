import json
import os
import shutil
from datetime import datetime

RESTAURANT_ID = "fusho_plinio"

SOURCES = {
    "dna": "reports/restaurant_dna_universal.json",
    "similarity_vector": "reports/restaurant_similarity_vector.json",
    "business_profile": "reports/business_model_profile.json",
    "model_selection": "reports/model_selection.json",
    "adaptive_regime": "reports/adaptive_regime_profile.json",
}

BASE_DIR = f"knowledge_base/restaurants/{RESTAURANT_ID}"


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    registry = {
        "restaurant_id": RESTAURANT_ID,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "sources": {},
    }

    for name, path in SOURCES.items():
        if os.path.exists(path):
            target = os.path.join(BASE_DIR, f"{name}.json")
            shutil.copy(path, target)

            registry["sources"][name] = target
        else:
            registry["sources"][name] = None

    dna = load_json(SOURCES["dna"])
    business_profile = load_json(SOURCES["business_profile"])
    model_selection = load_json(SOURCES["model_selection"])

    summary = {
        "restaurant_id": RESTAURANT_ID,
        "updated_at": datetime.now().isoformat(),
        "business_tags": dna.get("business_tags", {}) if dna else {},
        "selected_strategy": (
            model_selection.get("selected_strategy")
            if model_selection else None
        ),
        "business_model_tags": (
            business_profile.get("business_model", {}).get("tags", [])
            if business_profile else []
        ),
        "recommended_forecast_strategy": (
            business_profile.get("business_model", {}).get(
                "recommended_forecast_strategy"
            )
            if business_profile else None
        ),
    }

    with open(os.path.join(BASE_DIR, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

    with open(os.path.join(BASE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("RESTAURANT REGISTERED IN KNOWLEDGE BASE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\nCartella:")
    print(BASE_DIR)


if __name__ == "__main__":
    main()
