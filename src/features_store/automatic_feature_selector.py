import json
import os
import pandas as pd

IMPORTANCE_DIR = "reports/feature_importance"
OUTPUT_FILE = "reports/selected_features.json"

TARGETS = {
    "delivery": {
        "file": "delivery_importance_no_leakage.csv",
        "top_n": 25,
    },
    "digital": {
        "file": "digital_importance_no_leakage.csv",
        "top_n": 25,
    },
    "cash": {
        "file": "cash_importance_no_leakage.csv",
        "top_n": 20,
    },
    "total": {
        "file": "total_importance_no_leakage.csv",
        "top_n": 30,
    },
}

MIN_IMPORTANCE = 0.0001


def load_top_features(path, top_n):
    df = pd.read_csv(path)

    df = df[df["importance"] >= MIN_IMPORTANCE].copy()

    df = df.sort_values("importance", ascending=False)

    return df.head(top_n)["feature"].tolist()


def main():
    selected = {
        "version": "1.0",
        "method": "random_forest_importance_no_leakage",
        "min_importance": MIN_IMPORTANCE,
        "targets": {},
    }

    for target, config in TARGETS.items():
        path = os.path.join(IMPORTANCE_DIR, config["file"])

        if not os.path.exists(path):
            raise FileNotFoundError(f"File non trovato: {path}")

        features = load_top_features(path, config["top_n"])

        selected["targets"][target] = {
            "top_n": config["top_n"],
            "n_selected": len(features),
            "features": features,
        }

    os.makedirs("reports", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(selected, f, indent=2)

    print("=" * 80)
    print("AUTOMATIC FEATURE SELECTOR")
    print("=" * 80)

    for target, data in selected["targets"].items():
        print()
        print(target.upper())
        print("Feature selezionate:", data["n_selected"])
        for i, feature in enumerate(data["features"], start=1):
            print(f"{i:02d}. {feature}")

    print("\nSalvato:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
