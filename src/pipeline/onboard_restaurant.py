"""Wizard interattivo per registrare un nuovo ristorante nel sistema.

Uso: PYTHONPATH=. .venv/bin/python -m src.pipeline.onboard_restaurant
"""

import re
from pathlib import Path

import yaml


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
    address: str,
    ddt_folder_id: str,
    glovo_folder_id: str,
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
            "address": address or None,
        },
        "food_cost": {"ddt_folder_id": ddt_folder_id or None},
        "delivery_costs": {"glovo_folder_id": glovo_folder_id or None},
    }


def main():
    print("=== Onboarding nuovo ristorante ===\n")

    display_name = ask("Nome del ristorante (es. 'Trattoria Da Mario')")
    restaurant_id = ask("ID interno (minuscolo, senza spazi, usato nei percorsi)", slugify_id(display_name))

    city = ask("Citta'", "Milano")
    address = ask("Indirizzo completo (via e numero civico, opzionale)", "")
    country_code = ask("Codice paese (ISO, es. IT)", "IT")
    latitude = float(ask("Latitudine", "45.4642"))
    longitude = float(ask("Longitudine", "9.1900"))
    timezone = ask("Timezone", "Europe/Rome")

    print(
        "\nID cartella Google Drive con le foto DDT (fornitori), vuoto per "
        "disattivare il food cost. Deve essere condivisa (Visualizzatore) con "
        "l'email del service account."
    )
    ddt_folder_id = ask("ID cartella DDT", "")

    print(
        "\nID cartella Google Drive con le fatture Glovo, vuoto per "
        "disattivare i costi delivery. Stessa condivisione di sopra."
    )
    glovo_folder_id = ask("ID cartella Glovo", "")

    config = build_config(
        restaurant_id, display_name, city, country_code, latitude, longitude, timezone,
        address, ddt_folder_id, glovo_folder_id,
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
    print("  3. Nessun nuovo secret GitHub necessario -- riusa le credenziali Google gia' configurate")
    print("  4. Verifica che le cartelle Drive siano condivise col service account (permesso Visualizzatore)")


if __name__ == "__main__":
    main()
