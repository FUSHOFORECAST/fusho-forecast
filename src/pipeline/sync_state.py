import json
from pathlib import Path

STATE_DIR = Path("state")


def load_last_signature(restaurant_id: str) -> dict | None:
    path = STATE_DIR / f"{restaurant_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_signature(restaurant_id: str, signature: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{restaurant_id}.json"
    with open(path, "w") as f:
        json.dump(signature, f, indent=2)
