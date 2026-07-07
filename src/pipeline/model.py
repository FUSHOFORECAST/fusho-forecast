import os

import joblib
import pandas as pd

from src.models.calibrated_model import create_calibrated_model
from src.models.catboost_model import create_model as create_catboost_model
from src.models.random_forest_model import create_model as create_random_forest_model
from src.models.seasonal_naive_model import create_model as create_seasonal_naive_model
from src.pipeline.config import RestaurantConfig
from src.pipeline.ingest import slugify

BASE_MODEL_FACTORIES = {
    "random_forest": create_random_forest_model,
    "catboost": create_catboost_model,
    "seasonal_naive": create_seasonal_naive_model,
}

# Ogni modello base ottiene automaticamente una variante "_calibrated": stesso
# modello, ma con bias sistematico auto-misurato e corretto (vedi CalibratedModel).
# L'automl (backtest.py) sceglie quale variante usare in base ai dati reali del
# ristorante, canale per canale -- nessuna scelta manuale.
MODEL_FACTORIES = dict(BASE_MODEL_FACTORIES)
for _name, _factory in BASE_MODEL_FACTORIES.items():
    MODEL_FACTORIES[f"{_name}_calibrated"] = create_calibrated_model(_factory)

EXCLUDE_COLUMNS_BASE = ["date", "source_file", "sheet", "holiday_name"]


def get_feature_columns(df: pd.DataFrame, config: RestaurantConfig) -> list[str]:
    channels = config.channels
    raw_columns = [slugify(k) for k in config.channel_map.keys()]

    exclude = set(EXCLUDE_COLUMNS_BASE)
    exclude.update(channels)
    exclude.add("total")
    exclude.update(f"{c}_share" for c in channels)
    exclude.update(f"{c}_share_target" for c in channels)
    exclude.update(raw_columns)

    return [c for c in df.columns if c not in exclude]


def train_channel_model(df: pd.DataFrame, target: str, feature_cols: list[str], model_type: str):
    model = MODEL_FACTORIES[model_type]()
    model.fit(df[feature_cols], df[target])
    return model


def select_best_model_per_channel(config: RestaurantConfig, backtest_summary: dict) -> dict[str, str]:
    winners = {}
    for channel in config.channels:
        per_model = backtest_summary["per_channel_model"][channel]
        winners[channel] = min(per_model, key=lambda model_type: per_model[model_type]["mape"])
    return winners


def _save_model(model, model_type: str, channel: str, config: RestaurantConfig):
    os.makedirs(config.models_dir, exist_ok=True)
    if model_type == "catboost":
        path = config.models_path(f"{channel}_catboost.cbm")
        model.save_model(str(path))
    else:
        path = config.models_path(f"{channel}_{model_type}.joblib")
        joblib.dump(model, path)
    return path


def train_final_models(df: pd.DataFrame, config: RestaurantConfig, feature_cols: list[str], winners: dict[str, str]) -> dict[str, object]:
    models = {}
    for channel in config.channels:
        model_type = winners[channel]
        model = train_channel_model(df, channel, feature_cols, model_type)
        _save_model(model, model_type, channel, config)
        models[channel] = model
    return models


def load_model(channel: str, model_type: str, config: RestaurantConfig):
    if model_type == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(config.models_path(f"{channel}_catboost.cbm")))
        return model

    return joblib.load(config.models_path(f"{channel}_{model_type}.joblib"))


def load_final_models(config: RestaurantConfig, winners: dict[str, str]) -> dict[str, object]:
    return {channel: load_model(channel, model_type, config) for channel, model_type in winners.items()}
