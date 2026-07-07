"""Wizard interattivo per registrare un nuovo ristorante nel sistema.

Uso: PYTHONPATH=. .venv/bin/python -m src.pipeline.onboard_restaurant

Presume lo stesso formato foglio/canali gia' validato su FUSHO PLINIO (tutti
i ristoranti del prodotto usano lo stesso modello, per ora). Se un futuro
ristorante avesse un formato diverso, il file generato va rifinito a mano
(o richiede una nuova config/adapter dedicata).
"""

import re
from pathlib import Path

import yaml

DEFAULT_CHANNEL_MAP = {
    "POS": "digital",
    "TICKET": "digital",
    "SATISPAY": "digital",
    "SUMUP": "digital",
    "JUST EAT": "delivery",
    "GLOVO senza cash": "delivery",
    "DELIVEROO senza cash": "delivery",
    "GLOVO cash": "cash",
    "DELIVEROO cash": "cash",
    "Incasso cont. negozio": "cash",
    "cont. tot giorno": "cash",
}
DEFAULT_DUPLICATE_ROW_GROUPS = [["Incasso cont. negozio", "cont. tot giorno"]]
DEFAULT_TOTAL_ROW_LABEL = "Totale incasso"
DEFAULT_AUDIT_TOLERANCE = 350.0


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def slugify_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def build_config(
    restaurant_id: str,
    display_name: str,
    city: str,
    country_code: str,
    latitude: float,
    longitude: float,
    timezone: str,
    remote_folder_id: str,
    recipient_email: str,
) -> dict:
    return {
        "restaurant_id": restaurant_id,
        "display_name": display_name,
        "location": {
            "city": city,
            "country_code": country_code,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
        },
        "locale": {"language": "it"},
        "data_source": {
            "format": "excel_monthly_it",
            "raw_path": f"data/{restaurant_id}/raw",
            "file_pattern": "*.xlsx",
            "year_regex": r"20\d{2}",
            "remote_folder_id": remote_folder_id,
        },
        "channel_map": DEFAULT_CHANNEL_MAP,
        "duplicate_row_groups": DEFAULT_DUPLICATE_ROW_GROUPS,
        "audit": {"total_row_label": DEFAULT_TOTAL_ROW_LABEL, "tolerance": DEFAULT_AUDIT_TOLERANCE},
        "events": {"manual_events_path": None},
        "notify": {"recipient_email": recipient_email or None},
        "features": {"lags": [1, 7, 14, 21, 28, 30], "rolling_windows": [7, 14, 30], "trend_windows": [7, 30]},
        "forecast": {"horizon_days": 31, "backtest_days": 60},
        "model": {
            "candidates": ["random_forest", "catboost", "seasonal_naive", "weekday_shrinkage"],
            "random_state": 42,
        },
    }


def main():
    print("=== Onboarding nuovo ristorante ===\n")

    display_name = ask("Nome del ristorante (es. 'Trattoria Da Mario')")
    restaurant_id = ask("ID interno (minuscolo, senza spazi, usato nei percorsi)", slugify_id(display_name))

    city = ask("Citta'", "Milano")
    country_code = ask("Codice paese (ISO, es. IT)", "IT")
    latitude = float(ask("Latitudine", "45.4642"))
    longitude = float(ask("Longitudine", "9.1900"))
    timezone = ask("Timezone", "Europe/Rome")

    print(
        "\nID della cartella Google Drive del ristorante (deve essere gia' "
        "condivisa con l'email del service account, permesso Visualizzatore)."
    )
    remote_folder_id = ask("ID cartella Drive")

    recipient_email = ask("Email per il report giornaliero (vuoto = nessun report)", "")

    config = build_config(
        restaurant_id, display_name, city, country_code, latitude, longitude, timezone,
        remote_folder_id, recipient_email,
    )

    output_path = Path("config/restaurants") / f"{restaurant_id}.yaml"

    if output_path.exists():
        overwrite = ask(f"\n{output_path} esiste gia', sovrascrivere? (s/n)", "n")
        if not overwrite.lower().startswith("s"):
            print("Annullato.")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\nCreato: {output_path}")
    print("\nProssimi passi:")
    print(f"  1. Controlla il file generato, correggi se qualcosa non torna: {output_path}")
    print(f"  2. git add {output_path} && git commit -m 'Onboard {display_name}' && git push")
    print("  3. Nessun nuovo secret GitHub necessario -- riusa le credenziali Google/Gmail gia' configurate")
    print("  4. Verifica che la cartella Drive sia condivisa col service account (permesso Visualizzatore)")


if __name__ == "__main__":
    main()
