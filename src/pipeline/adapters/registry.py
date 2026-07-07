from typing import Callable

import pandas as pd

from src.pipeline.adapters import excel_monthly, wide_table
from src.pipeline.config import RestaurantConfig

ADAPTER_REGISTRY: dict[str, Callable[[RestaurantConfig, bool], tuple[pd.DataFrame, pd.DataFrame]]] = {
    "excel_monthly_it": excel_monthly.extract,
    "wide_table": wide_table.extract,
}


def get_adapter(config: RestaurantConfig) -> Callable:
    fmt = config.data_source.format
    try:
        return ADAPTER_REGISTRY[fmt]
    except KeyError:
        raise ValueError(
            f"Formato data_source sconosciuto: '{fmt}'. Formati disponibili: {sorted(ADAPTER_REGISTRY)}"
        )
