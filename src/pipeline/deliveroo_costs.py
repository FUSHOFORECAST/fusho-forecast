"""Costi della piattaforma Deliveroo (commissioni, IVA sulla commissione,
sconti marketing, rimborsi) estratti dai CSV settimanali ("statement") che
Deliveroo carica in un'unica cartella Drive condivisa da tutti i ristoranti.

A differenza di Glovo (una cartella per ristorante, fatture quindicinali in
PDF), Deliveroo organizza i file per periodo: una cartella radice condivisa
(config/deliveroo.yaml) contiene sottocartelle di periodo, ognuna con una
sottocartella per ristorante identificata da un codice breve (es. "JL",
"KANJI_CENTRALE"), non dal nome -- il codice va configurato per ristorante
(delivery_costs.deliveroo_code). Dentro ogni cartella ristorante ci sono file
settimanali *_statement.csv, un CSV machine-readable (a differenza della
fattura PDF Glovo) con un dettaglio ordine-per-ordine.

Se config.delivery_costs.deliveroo_code non e' impostato (o la root non e'
configurata), ogni funzione qui fa skip silenzioso -- stesso pattern di
delivery_costs.py.

La cartella Deliveroo va condivisa (permesso Visualizzatore) con l'email del
service account, come le cartelle Glovo.
"""

import csv
import datetime
import re
from pathlib import Path

import yaml

from src.pipeline.config import RestaurantConfig
from src.pipeline.sync_raw_data import _build_drive_service, _download_file
from src.pipeline.sync_state import load_last_signature, save_signature

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"

# Tipi di riga confermati sui CSV reali visti finora (file FUSHO e La Gyozeria
# Viale Monza): "Consegna" (ordine), "Pagamento in contanti" (il cliente paga
# cash, Deliveroo lo detrae dal bonifico -- non e' un costo, e' cassa gia'
# incassata dal ristorante) e "Rimborso al cliente" (storno per un problema
# sull'ordine, un costo vero e proprio). Qualunque altra "Attività" mai vista
# finisce comunque nel bucket generico "altro" invece di essere persa: stesso
# principio di LABEL_KEY_MAP/unmapped_fields in delivery_costs.py, non si
# guessa una struttura mai verificata sui byte reali del file.
ORDER_ACTIVITY = "Consegna"
CASH_ACTIVITY = "Pagamento in contanti"
REFUND_ACTIVITY = "Rimborso al cliente"
KNOWN_ACTIVITIES = {ORDER_ACTIVITY, CASH_ACTIVITY, REFUND_ACTIVITY}

# "Nota" sulle righe Consegna puo' contenere "Sconto offerta Marketer: 2,00"
# (visto su file reali, La Gyozeria Viale Monza, novembre 2026) -- importo del
# prezzo ordine gia' scontato grazie a una promozione. Non e' detto sia un
# costo fatturato al ristorante (il confronto Valore ordine * commissione%
# torna esatto sul valore gia' scontato, segno che la commissione si applica
# dopo lo sconto): tracciato come informativo, non sommato nei costi.
ADS_NOTE_RE = re.compile(r"Sconto offerta Marketer:\s*([\d.,]+)", re.IGNORECASE)

DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")


def _it_number_to_float(value: str) -> float:
    """Converte un numero in formato italiano ('3.312,08', '-525,34') in float.
    Stessa logica di delivery_costs._it_number_to_float, duplicata qui per
    tenere i due moduli piattaforma indipendenti."""
    value = (value or "").strip()
    if not value:
        return 0.0
    negative = value.startswith("-")
    value = value.lstrip("-").replace(".", "").replace(",", ".")
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    return -parsed if negative else parsed


def _load_deliveroo_root_folder_id(path: str | Path = "config/deliveroo.yaml") -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("root_folder_id")


def _list_children(service, folder_id: str) -> list[dict]:
    query = f"'{folder_id}' in parents and trashed = false"
    children: list[dict] = []
    page_token = None
    while True:
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()
        children.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return children


def _find_restaurant_statement_csvs(service, folder_id: str, code: str) -> list[dict]:
    """Attraversa ricorsivamente folder_id (struttura tipica periodo/codice/*.csv,
    ma generica: non assume profondita' fissa) cercando sottocartelle il cui
    nome coincide (case-insensitive) col codice ristorante, e dentro ognuna
    raccoglie i file *_statement.csv (ignora invoice.pdf/statement.pdf e la
    sottocartella ordini/, non ancora usati)."""
    children = _list_children(service, folder_id)
    subfolders = [c for c in children if c["mimeType"] == GOOGLE_FOLDER_MIME]
    matches = [c for c in subfolders if c["name"].strip().upper() == code.strip().upper()]

    results: list[dict] = []
    for match in matches:
        match_children = _list_children(service, match["id"])
        results.extend(c for c in match_children if c["name"].lower().endswith("_statement.csv"))

    match_ids = {m["id"] for m in matches}
    for sub in subfolders:
        if sub["id"] not in match_ids:
            results.extend(_find_restaurant_statement_csvs(service, sub["id"], code))

    return results


def sync_deliveroo_statements(config: RestaurantConfig) -> list[Path]:
    """Scarica in config.raw_deliveroo_dir i CSV settimanali non ancora
    presenti localmente (confronto per nome file, gia' univoco lato Deliveroo:
    <codice>_SRL_<data>_statement.csv)."""
    code = config.delivery_costs.deliveroo_code
    if not code:
        return []

    root_folder_id = _load_deliveroo_root_folder_id()
    if not root_folder_id:
        return []

    service = _build_drive_service()
    files = _find_restaurant_statement_csvs(service, root_folder_id, code)

    config.raw_deliveroo_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in config.raw_deliveroo_dir.glob("*.csv")}

    downloaded = []
    for file in files:
        if file["name"] in existing:
            continue
        destination = config.raw_deliveroo_dir / file["name"]
        _download_file(service, file, destination)
        downloaded.append(destination)

    return downloaded


def _parse_statement_sections(path: Path) -> list[dict]:
    """Il CSV Deliveroo ha piu' sezioni (titolo su una riga a se', poi
    l'intestazione, poi le righe dati fino alla riga vuota successiva):
    "Orders and related adjustments", "Payments for contested customer
    refunds (from a previous invoice)", "Other payments and fees". Le ultime
    due sono viste finora sempre vuote sui file reali, ma la struttura e'
    generica: qualunque sezione futura viene letta allo stesso modo.

    Le righe vengono lette usando l'intestazione COME SCRITTA nel file (non
    una lista di colonne fissata qui), cosi' se Deliveroo aggiunge/riordina
    colonne il parsing si adatta da solo invece di rompersi silenziosamente."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    records: list[dict] = []
    section = "Orders and related adjustments"
    header: list[str] | None = None

    for row in rows:
        non_empty = [c for c in row if c.strip()]
        if row and row[0].strip() == "Nome del ristorante":
            header = row
        elif header is not None and non_empty:
            record = {h.strip(): v for h, v in zip(header, row)}
            record["_section"] = section
            records.append(record)
        elif not non_empty:
            header = None
        elif len(non_empty) == 1 and header is None:
            section = non_empty[0].strip()

    return records


def _date_from_filename(path: Path) -> str | None:
    match = DATE_RE.search(path.stem)
    if not match:
        return None
    y, m, d = match.groups()
    return f"{y}-{m}-{d}"


def parse_statement_csv(path: Path) -> dict:
    """Estrae da un CSV settimanale Deliveroo i totali aggregati (commissione,
    IVA sulla commissione, netto da incassare, ordini, sconti marketing
    informativi), piu' tre bucket per le righe non-ordine confermate sui file
    reali: "contanti" (pagamenti in contanti gia' incassati dal ristorante,
    dedotti dal bonifico -- non un costo), "rimborsi" (storni al cliente, un
    costo vero) e un residuo "altro" generico per qualunque "Attività" mai
    vista -- stesso principio delle voci non mappate in delivery_costs.py: mai
    perse silenziosamente, finiscono in un totale visibile invece che dentro
    una delle due voci confermate."""
    records = _parse_statement_sections(path)

    orders = [r for r in records if r.get("Attività", "").strip() == ORDER_ACTIVITY]
    unknown_activities = sorted({
        r.get("Attività", "").strip() for r in records
        if r.get("Attività", "").strip() and r.get("Attività", "").strip() not in KNOWN_ACTIVITIES
    })

    cash_rows = [r for r in records if r.get("Attività", "").strip() == CASH_ACTIVITY]
    refund_rows = [r for r in records if r.get("Attività", "").strip() == REFUND_ACTIVITY]
    other_rows = [
        r for r in records
        if r.get("Attività", "").strip() not in (ORDER_ACTIVITY, CASH_ACTIVITY, REFUND_ACTIVITY)
    ]

    # Commissione/IVA calcolate solo sulle righe Consegna (il costo e' per
    # definizione legato all'ordine): qualunque altra riga (pagamento in
    # contanti, rimborso, o un tipo mai visto) va invece per intero in uno dei
    # tre bucket qui sotto usando il suo "Totale da pagare" gia' netto, cosi'
    # la somma totale_da_pagare = valore_ordini - commissione - commissione_iva
    # - contanti - rimborsi - altro torna sempre, senza contare due volte la
    # stessa commissione.
    valore_ordini = round(sum(_it_number_to_float(r.get("Valore dell'ordine (€)")) for r in orders), 2)
    commissione = round(sum(_it_number_to_float(r.get("Commissione Deliveroo (€)")) for r in orders), 2)
    commissione_iva = round(sum(_it_number_to_float(r.get("Commissione / rettifica IVA (€)")) for r in orders), 2)
    contanti = round(sum(_it_number_to_float(r.get("Totale da pagare")) for r in cash_rows), 2)
    rimborsi = round(sum(_it_number_to_float(r.get("Totale da pagare")) for r in refund_rows), 2)
    altro = round(sum(_it_number_to_float(r.get("Totale da pagare")) for r in other_rows), 2)
    totale_da_pagare = round(sum(_it_number_to_float(r.get("Totale da pagare")) for r in records), 2)

    sconti_marketing = 0.0
    for r in orders:
        match = ADS_NOTE_RE.search(r.get("Nota") or "")
        if match:
            sconti_marketing += _it_number_to_float(match.group(1))
    sconti_marketing = round(sconti_marketing, 2)

    dates = sorted({
        (r.get("Data e ora della consegna (UTC)") or "")[:10]
        for r in records
        if (r.get("Data e ora della consegna (UTC)") or "").strip()
    })
    filename_date = _date_from_filename(path)
    period_end = dates[-1] if dates else filename_date
    period_start = dates[0] if dates else filename_date

    restaurant_name = next(
        (r.get("Nome del ristorante", "").strip() for r in records if r.get("Nome del ristorante", "").strip()),
        None,
    )

    return {
        "filename": path.name,
        "restaurant_name": restaurant_name,
        "period_start": period_start,
        "period_end": period_end,
        "n_orders": len(orders),
        "valore_ordini": valore_ordini,
        "commissione": abs(commissione),
        "commissione_iva": abs(commissione_iva),
        "sconti_marketing": sconti_marketing,
        "contanti": abs(contanti),
        "rimborsi": abs(rimborsi),
        "altro": abs(altro),
        "totale_da_pagare": totale_da_pagare,
        "unknown_activities": unknown_activities,
    }


def _deliveroo_state_id(restaurant_id: str) -> str:
    """v2: separa dal bucket "altro" i due tipi di riga confermati su dati
    reali -- "contanti" (pagamenti in contanti, non un costo) e "rimborsi"
    (storni al cliente, un costo vero) -- cosi' il campo "altro" resta un
    residuo generico invece di mischiare le due cose. Se cambiano ancora i
    campi salvati andra' incrementato di nuovo (stesso schema di
    versionamento di delivery_costs.py), cosi' il riparsing scatta su tutti
    gli statement gia' noti invece di tenere per sempre i record col formato
    vecchio."""
    return f"{restaurant_id}_deliveroo_v2"


def _load_deliveroo_state(restaurant_id: str) -> dict:
    return load_last_signature(_deliveroo_state_id(restaurant_id)) or {"processed": {}}


def _save_deliveroo_record(restaurant_id: str, filename: str, record: dict) -> None:
    state = _load_deliveroo_state(restaurant_id)
    state["processed"][filename] = record
    save_signature(_deliveroo_state_id(restaurant_id), state)


_STATEMENT_FIELDS = [
    "filename", "restaurant_name", "period_start", "period_end", "n_orders",
    "valore_ordini", "commissione", "commissione_iva", "sconti_marketing",
    "contanti", "rimborsi", "altro", "totale_da_pagare",
]


def _append_statements_csv(config: RestaurantConfig, records: dict[str, dict]) -> None:
    """Riscrive delivery_costs_deliveroo_statements.csv da zero a partire da
    tutti i record noti in stato -- stesso approccio di delivery_costs.py
    (volume basso, non serve append incrementale)."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    path = config.processed_path("delivery_costs_deliveroo_statements.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_STATEMENT_FIELDS)
        writer.writeheader()
        for filename, record in sorted(records.items()):
            writer.writerow({k: record.get(k) for k in _STATEMENT_FIELDS} | {"filename": filename})


def _aggregate_monthly(config: RestaurantConfig, records: dict[str, dict]) -> None:
    monthly: dict[str, dict] = {}
    money_fields = [
        "valore_ordini", "commissione", "commissione_iva", "sconti_marketing",
        "contanti", "rimborsi", "altro", "totale_da_pagare",
    ]

    for record in records.values():
        period_start = record.get("period_start")
        if not period_start:
            continue
        year_month = period_start[:7]
        bucket = monthly.setdefault(year_month, {f: 0.0 for f in money_fields} | {"n_orders": 0, "n_statements": 0})
        for field in money_fields:
            value = record.get(field)
            if value is not None:
                bucket[field] += value
        bucket["n_orders"] += record.get("n_orders", 0)
        bucket["n_statements"] += 1

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    path = config.processed_path("delivery_costs_deliveroo_monthly.csv")
    fieldnames = ["year_month"] + money_fields + ["n_orders", "n_statements"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for year_month, bucket in sorted(monthly.items()):
            row = {field: round(bucket[field], 2) for field in money_fields}
            row["year_month"] = year_month
            row["n_orders"] = bucket["n_orders"]
            row["n_statements"] = bucket["n_statements"]
            writer.writerow(row)


def run_deliveroo_costs_pipeline(config: RestaurantConfig) -> dict:
    """Orchestratore: sincronizza i CSV settimanali nuovi, estrae quelli non
    ancora elaborati (salvando lo stato dopo ognuno), poi riscrive
    statements.csv e monthly.csv da tutto lo stato noto. Skip silenzioso se
    deliveroo_code (o la root condivisa) non e' configurato."""
    if not config.delivery_costs.deliveroo_code:
        return {"skipped": True}

    sync_deliveroo_statements(config)

    state = _load_deliveroo_state(config.restaurant_id)
    processed = state["processed"]

    new_files = [p for p in config.raw_deliveroo_dir.glob("*_statement.csv") if p.name not in processed]
    for path in new_files:
        record = parse_statement_csv(path)
        record["parsed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        _save_deliveroo_record(config.restaurant_id, path.name, record)
        processed[path.name] = record

    _append_statements_csv(config, processed)
    _aggregate_monthly(config, processed)

    return {
        "skipped": False,
        "n_new_statements": len(new_files),
        "n_total_statements": len(processed),
    }


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    result = run_deliveroo_costs_pipeline(config)
    print(result)


if __name__ == "__main__":
    main()
