import os

import pandas as pd

from src.pipeline.config import RestaurantConfig


def clean(master_df: pd.DataFrame, audit_df: pd.DataFrame, config: RestaurantConfig, persist: bool = True) -> pd.DataFrame:
    df = master_df.copy()
    audit = audit_df.copy()

    df["date"] = pd.to_datetime(df["date"])
    audit["date"] = pd.to_datetime(audit["date"])

    bad_dates = audit[audit["difference"].abs() > config.audit.tolerance]["date"]

    clean_df = df[~df["date"].isin(bad_dates)].copy().sort_values("date").reset_index(drop=True)

    if persist:
        os.makedirs(config.processed_dir, exist_ok=True)
        clean_df.to_csv(config.processed_path("master_dataset_clean.csv"), index=False)

    return clean_df


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config
    from src.pipeline.ingest import ingest

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    master_df, audit_df = ingest(config)
    clean_df = clean(master_df, audit_df, config)

    print("Righe originali:", len(master_df))
    print("Righe pulite:", len(clean_df))


if __name__ == "__main__":
    main()
