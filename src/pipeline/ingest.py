import pandas as pd

from src.pipeline.adapters.common import slugify  # re-export: model.py importa slugify da qui
from src.pipeline.adapters.registry import get_adapter
from src.pipeline.config import RestaurantConfig

__all__ = ["ingest", "slugify"]


def ingest(config: RestaurantConfig, persist: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    return get_adapter(config)(config, persist=persist)


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    df, audit_df = ingest(config)

    print("MASTER DATASET CREATO")
    print("Formato:", config.data_source.format)
    print("Righe:", len(df))
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print("Canali:", config.channels)
    print("Differenza audit media:", round(audit_df["difference"].mean(), 2))


if __name__ == "__main__":
    main()
