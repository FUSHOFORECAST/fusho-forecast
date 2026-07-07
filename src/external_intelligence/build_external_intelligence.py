import os
import pandas as pd

from src.external_intelligence.registry import EXTERNAL_ENGINES

OUTPUT = "reports/external_intelligence/external_features.csv"


def main():
    os.makedirs("reports/external_intelligence", exist_ok=True)

    paths = []

    print("=" * 90)
    print("BUILD EXTERNAL INTELLIGENCE")
    print("=" * 90)

    for engine in EXTERNAL_ENGINES:
        path = engine()
        paths.append(path)
        print("Built:", path)

    merged = None

    for path in paths:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        if merged is None:
            merged = df
        else:
            duplicate_cols = [
                c for c in df.columns
                if c in merged.columns and c != "date"
            ]

            df = df.drop(columns=duplicate_cols)

            merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    merged.to_csv(OUTPUT, index=False)

    print("\nExternal features salvate:")
    print(OUTPUT)
    print("Righe:", len(merged))
    print("Colonne:", len(merged.columns))

    print("\nEsempio giugno 2026:")
    print(
        merged[
            (merged["date"] >= "2026-06-01")
            & (merged["date"] <= "2026-06-14")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
