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
    address: str | None = None


@dataclasses.dataclass(frozen=True)
class FoodCostConfig:
    ddt_folder_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DeliveryCostsConfig:
    glovo_folder_id: str | None = None


@dataclasses.dataclass(frozen=True)
class RestaurantConfig:
    restaurant_id: str
    display_name: str
    location: LocationConfig
    food_cost: FoodCostConfig
    delivery_costs: DeliveryCostsConfig

    @property
    def raw_ddt_dir(self) -> Path:
        return Path(f"data/{self.restaurant_id}/raw_ddt")

    @property
    def raw_glovo_dir(self) -> Path:
        return Path(f"data/{self.restaurant_id}/raw_glovo")

    @property
    def processed_dir(self) -> Path:
        return Path(f"data/{self.restaurant_id}/processed")

    def processed_path(self, name: str) -> Path:
        return self.processed_dir / name


def load_restaurant_config(path: str | Path) -> RestaurantConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    location = LocationConfig(**raw["location"])
    food_cost = FoodCostConfig(**raw.get("food_cost", {}))
    delivery_costs = DeliveryCostsConfig(**raw.get("delivery_costs", {}))

    return RestaurantConfig(
        restaurant_id=raw["restaurant_id"],
        display_name=raw["display_name"],
        location=location,
        food_cost=food_cost,
        delivery_costs=delivery_costs,
    )


def list_restaurant_configs(config_dir: str | Path = "config/restaurants") -> list[Path]:
    return sorted(
        p for p in Path(config_dir).glob("*.yaml") if not p.name.startswith("_")
    )
