"""Estrazione dati dai DDT fotografati (vedi ddt_capture/) e aggregazione in
un food cost mensile. Fase informativa, non e' un input al forecast:
sincronizza le foto da Drive, le legge con Gemini (capisce documenti di
layout diversi senza parser per-fornitore), e aggrega i totali per mese.

Se config.food_cost.ddt_folder_id non e' impostato, ogni funzione qui fa
skip silenzioso -- stesso pattern di sync_from_drive per remote_folder_id,
zero impatto sui ristoranti che non hanno ancora attivato questa fase.
"""

import csv
import datetime
import json
import os
from pathlib import Path

from src.pipeline.config import RestaurantConfig
from src.pipeline.sync_raw_data import _build_drive_service, _download_file, _list_files_in_folder
from src.pipeline.sync_state import load_last_signature, save_signature

GEMINI_MODEL = "gemini-2.5-flash"

EXTRACTION_PROMPT = """Questo e' un DDT (documento di trasporto) o fattura accompagnatoria \
di un fornitore per un ristorante. Estrai le seguenti informazioni e rispondi SOLO con un \
oggetto JSON, nessun altro testo:

{
  "date": "YYYY-MM-DD",
  "supplier": "nome del fornitore",
  "total": 123.45,
  "needs_review": false,
  "notes": ""
}

Regole importanti:
- "total" e' l'IMPORTO FINALE da pagare, gia' comprensivo di IVA, sconti e eventuali resi \
gia' netti -- quello vicino a "Totale", "Totale documento", "Totale fattura" o "Totale EUR". \
MAI un subtotale come "Imponibile" o "Totale Merce" (quello e' prima dell'IVA).
- "supplier" e' chi EMETTE il documento (il logo/intestazione in alto, di solito con P.IVA \
e IBAN propri) -- MAI il destinatario/cliente del documento (spesso un ristorante), anche se \
il nome del destinatario compare in modo piu' prominente.
- "date" e' la data del documento (di solito vicino al numero documento/fattura).
- Se un campo non e' leggibile con certezza, o il documento e' ambiguo/illeggibile, imposta \
"needs_review": true e spiega brevemente il motivo in "notes". NON INVENTARE MAI un numero \
o un nome se non sei sicuro -- meglio segnalare il dubbio.
"""


def _ddt_state_id(restaurant_id: str) -> str:
    """Identificativo di stato separato da quello di sync_raw_data.py (che
    usa restaurant_id puro per il fatturato) -- evita di sovrascrivere
    state/{restaurant_id}.json esistente."""
    return f"{restaurant_id}_ddt"


def sync_ddt_photos(config: RestaurantConfig) -> list[Path]:
    """Scarica in config.raw_ddt_dir le foto DDT non ancora presenti
    localmente. Il nome file (gia' univoco, include timestamp) e' il
    confronto usato per capire cosa e' gia' stato scaricato."""
    folder_id = config.food_cost.ddt_folder_id
    if not folder_id:
        return []

    service = _build_drive_service()
    files = _list_files_in_folder(service, folder_id, "*")

    config.raw_ddt_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in config.raw_ddt_dir.glob("*")}

    downloaded = []
    for file in files:
        if file["name"] in existing:
            continue
        destination = config.raw_ddt_dir / file["name"]
        _download_file(service, file, destination)
        downloaded.append(destination)

    return downloaded


def extract_ddt_data(image_path: Path) -> dict:
    """Chiama Gemini per estrarre date/supplier/total dalla foto. Non
    solleva mai un'eccezione verso il chiamante: qualunque problema (rete,
    quota, risposta non valida) produce un record needs_review=True invece
    di bloccare l'elaborazione delle altre foto."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY non impostata")

        client = genai.Client(api_key=api_key)
        image_bytes = image_path.read_bytes()
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                EXTRACTION_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        parsed = json.loads(response.text)
        return {
            "date": parsed.get("date"),
            "supplier": parsed.get("supplier"),
            "total": parsed.get("total"),
            "needs_review": bool(parsed.get("needs_review", False)),
            "notes": parsed.get("notes", ""),
        }
    except Exception as e:
        return {
            "date": None,
            "supplier": None,
            "total": None,
            "needs_review": True,
            "notes": f"Estrazione fallita: {e}",
        }


def _load_ddt_state(restaurant_id: str) -> dict:
    return load_last_signature(_ddt_state_id(restaurant_id)) or {"processed": {}}


def _save_ddt_record(restaurant_id: str, filename: str, record: dict) -> None:
    state = _load_ddt_state(restaurant_id)
    state["processed"][filename] = record
    save_signature(_ddt_state_id(restaurant_id), state)


def _append_records_csv(config: RestaurantConfig, records: dict[str, dict]) -> None:
    """Riscrive ddt_records.csv da zero a partire da tutti i record noti in
    stato (processed) -- piu' semplice e robusto di un append incrementale,
    dato il volume basso (decine di DDT al mese, non migliaia)."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    path = config.processed_path("ddt_records.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "date", "supplier", "total", "needs_review", "notes", "extracted_at"]
        )
        writer.writeheader()
        for filename, record in sorted(records.items()):
            writer.writerow({"filename": filename, **record})


def _aggregate_monthly(config: RestaurantConfig, records: dict[str, dict]) -> None:
    monthly: dict[str, dict] = {}
    for record in records.values():
        date_str = record.get("date")
        if not date_str:
            continue
        year_month = date_str[:7]  # "YYYY-MM"
        bucket = monthly.setdefault(year_month, {"total": 0.0, "n_ddt": 0, "n_needs_review": 0})
        if record.get("needs_review"):
            bucket["n_needs_review"] += 1
        total = record.get("total")
        if total is not None:
            bucket["total"] += float(total)
            bucket["n_ddt"] += 1

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    path = config.processed_path("food_cost_monthly.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["year_month", "total", "n_ddt", "n_needs_review"])
        writer.writeheader()
        for year_month, bucket in sorted(monthly.items()):
            writer.writerow({
                "year_month": year_month,
                "total": round(bucket["total"], 2),
                "n_ddt": bucket["n_ddt"],
                "n_needs_review": bucket["n_needs_review"],
            })


def run_food_cost_pipeline(config: RestaurantConfig) -> dict:
    """Orchestratore: sincronizza foto nuove, estrae quelle non ancora
    elaborate (salvando lo stato dopo ognuna, non solo a fine batch), poi
    riscrive ddt_records.csv e food_cost_monthly.csv da tutto lo stato noto.
    Skip silenzioso se ddt_folder_id non e' configurato."""
    if not config.food_cost.ddt_folder_id:
        return {"skipped": True}

    sync_ddt_photos(config)

    state = _load_ddt_state(config.restaurant_id)
    processed = state["processed"]

    new_files = [p for p in config.raw_ddt_dir.glob("*") if p.name not in processed]
    for path in new_files:
        record = extract_ddt_data(path)
        record["extracted_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        _save_ddt_record(config.restaurant_id, path.name, record)
        processed[path.name] = record

    _append_records_csv(config, processed)
    _aggregate_monthly(config, processed)

    return {
        "skipped": False,
        "n_new_photos": len(new_files),
        "n_total_records": len(processed),
    }


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    result = run_food_cost_pipeline(config)
    print(result)


if __name__ == "__main__":
    main()
