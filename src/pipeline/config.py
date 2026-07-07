from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class LocationConfig:
    city: str
    country_code: str
    latitude: float
    longitude: float
    timezone: str


@dataclasses.dataclass(frozen=True)
class DataSourceConfig:
    raw_path: str
    format: str = "excel_monthly_it"
    file_pattern: str = "*.xlsx"
    year_regex: str = r"20\d{2}"
    remote_folder_id: str | None = None


@dataclasses.dataclass(frozen=True)
class AuditConfig:
    total_row_label: str = "TOTALE INCASSO"
    tolerance: float = 1.0


@dataclasses.dataclass(frozen=True)
class EventsConfig:
    manual_events_path: str | None = None


@dataclasses.dataclass(frozen=True)
class NotifyConfig:
    recipient_email: str | None = None


@dataclasses.dataclass(frozen=True)
class FeaturesConfig:
    lags: list[int] = dataclasses.field(default_factory=lambda: [1, 7, 14, 21, 28, 30])
    rolling_windows: list[int] = dataclasses.field(default_factory=lambda: [7, 14, 30])
    trend_windows: list[int] = dataclasses.field(default_factory=lambda: [7, 30])


@dataclasses.dataclass(frozen=True)
class ForecastConfig:
    horizon_days: int = 7
    backtest_days: int = 60


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    candidates: list[str] = dataclasses.field(default_factory=lambda: ["random_forest", "catboost", "seasonal_naive"])
    random_state: int = 42


@dataclasses.dataclass(frozen=True)
class RestaurantConfig:
    restaurant_id: str
    display_name: str
    location: LocationConfig
    language: str
    data_source: DataSourceConfig
    channel_map: dict[str, str]
    audit: AuditConfig
    events: EventsConfig
    notify: NotifyConfig
    features: FeaturesConfig
    forecast: ForecastConfig
    model: ModelConfig

    @property
    def channels(self) -> list[str]:
        """Nomi canale unici, derivati da channel_map -- generalizza a qualsiasi numero di canali."""
        return sorted(set(self.channel_map.values()))

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_source.raw_path)

    @property
    def processed_dir(self) -> Path:
        return Path(f"data/{self.restaurant_id}/processed")

    @property
    def reports_dir(self) -> Path:
        return Path(f"reports/{self.restaurant_id}")

    @property
    def models_dir(self) -> Path:
        return Path(f"models/{self.restaurant_id}")

    def processed_path(self, name: str) -> Path:
        return self.processed_dir / name

    def reports_path(self, name: str) -> Path:
        return self.reports_dir / name

    def models_path(self, name: str) -> Path:
        return self.models_dir / name


def load_restaurant_config(path: str | Path) -> RestaurantConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    location = LocationConfig(**raw["location"])
    data_source = DataSourceConfig(**raw.get("data_source", {}))
    audit = AuditConfig(**raw.get("audit", {}))
    events = EventsConfig(**raw.get("events", {}))
    notify = NotifyConfig(**raw.get("notify", {}))
    features = FeaturesConfig(**raw.get("features", {}))
    forecast = ForecastConfig(**raw.get("forecast", {}))
    model = ModelConfig(**raw.get("model", {}))

    trend_windows = features.trend_windows
    if len(trend_windows) != 2 or trend_windows[0] >= trend_windows[1]:
        raise ValueError(
            f"features.trend_windows deve avere esattamente [finestra_corta, finestra_lunga] "
            f"in ordine crescente, ricevuto: {trend_windows}"
        )

    if forecast.horizon_days > 16:
        raise ValueError(
            f"forecast.horizon_days={forecast.horizon_days} supera il limite di 16 giorni "
            f"dell'endpoint forecast di Open-Meteo."
        )

    return RestaurantConfig(
        restaurant_id=raw["restaurant_id"],
        display_name=raw["display_name"],
        location=location,
        language=raw["locale"]["language"],
        data_source=data_source,
        channel_map=raw["channel_map"],
        audit=audit,
        events=events,
        notify=notify,
        features=features,
        forecast=forecast,
        model=model,
    )


def list_restaurant_configs(config_dir: str | Path = "config/restaurants") -> list[Path]:
    return sorted(
        p for p in Path(config_dir).glob("*.yaml") if not p.name.startswith("_")
    )
