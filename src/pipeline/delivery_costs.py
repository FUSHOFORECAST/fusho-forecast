"""Costi della piattaforma Glovo (commissioni, promozioni/ads a carico del
partner, costi incidenti) estratti dalle fatture PDF che Glovo carica in una
cartella Drive dedicata per ristorante (struttura AAAA/MM/*.pdf, quindicinale).

Non e' un input al forecast: e' un flusso dati indipendente, stesso pattern
di food_cost.py -- sincronizza da Drive, estrae con un parser deterministico
(non serve un LLM: il layout della fattura e' fisso), aggrega mensilmente.

Se config.delivery_costs.glovo_folder_id non e' impostato, ogni funzione qui
fa skip silenzioso -- zero impatto sui ristoranti che non hanno ancora
attivato questa fase.

La cartella Glovo associata al ristorante va condivisa (permesso
Visualizzatore) con l'email del service account, come per food_cost.
"""

import csv
import datetime
import re
from pathlib import Path

from src.pipeline.config import RestaurantConfig
from src.pipeline.sync_raw_data import _build_drive_service, _download_file
from src.pipeline.sync_state import load_last_signature, save_signature

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"

# Etichette esatte (minuscolo) come compaiono nella fattura PDF Glovo -> chiave
# interna stabile. Qualunque etichetta non in questa mappa viene comunque
# catturata (slug automatico) e segnalata in "unmapped_fields", invece di
# essere persa silenziosamente se Glovo cambia/aggiunge una voce.
LABEL_KEY_MAP = {
    "commissioni": "commissioni",
    "promozione sulla consegna a carico del partner": "promo_consegna",
    "costo di manutenzione": "costo_manutenzione",
    "supplemento per ordine con glovo prime": "supplemento_prime",
    "marketing-visibilità": "marketing_visibilita",
    "subtotale": "subtotale",
    "totale fattura": "totale_fattura",
    "prodotti": "prodotti",
    "servizio di consegna": "servizio_consegna",
    "promozione sui prodotti a carico del partner": "promo_prodotti",
    "costo degli incidenti relativi ai prodotti": "costo_incidenti_prodotti",
    "tariffa per tempo di attesa": "tariffa_attesa",
    "rimborsi al partner senza costo commissione glovo": "rimborsi_partner",
    "costo degli annullamenti e degli incidenti relativi al servizio": "costo_annullamenti",
    "consegna gratuita in seguito a incidente": "consegna_gratuita_incidente",
    "consegna gratuita in seguito a un incidente": "consegna_gratuita_incidente",
    "glovo già pagati": "glovo_gia_pagati",
    "debito accumulato": "debito_accumulato",
    "importo del bonifico": "importo_bonifico",
    # Viste solo su alcune fatture (es. Ramen Shifu durante le campagne
    # "offerta lampo" Glovo) -- assenti su FUSHO. "Costi per offerta lampo"
    # e' nel blocco di costi in cima (sibling di commissioni/ads/prime);
    # "Promozione lampo a carico del Partner" e' nel blocco di
    # riconciliazione, stessa natura di "promo_prodotti" (sconto sul
    # prodotto finanziato dal ristorante, non un costo sul fatturato).
    "costi per offerta lampo": "offerta_lampo",
    "promozione lampo a carico del partner": "promo_lampo",
}

# I bucket che l'utente vuole vedere, piu' "altro" per tutto cio' che non
# rientra chiaramente li' (incluse eventuali voci nuove non ancora mappate
# sopra -- mai perse, finiscono qui). "Sconti" (promozioni sul prezzo a
# carico del ristorante) e' tenuto separato da "ads" (vera spesa di
# marketing/visibilita'): sono entrambi "leve commerciali" ma di natura
# diversa, e l'utente li vuole confrontare separatamente col fatturato netto.
# "Prime" (supplemento per ordini Glovo Prime) e' a sua volta separato da
# "ads": non e' spesa di marketing, e' una fee sugli ordini dei clienti
# abbonati a Glovo Prime -- l'utente vuole vederle distinte.
# Le voci "meta" sono valori di riepilogo/bonifico della fattura, non costi
# autonomi, e vanno escluse dalla somma di "altro" per non contare due volte.
COMMISSIONE_FIELDS = {"commissioni"}
ADS_FIELDS = {"marketing_visibilita"}
PRIME_FIELDS = {"supplemento_prime"}
OFFERTA_LAMPO_FIELDS = {"offerta_lampo"}
SCONTI_FIELDS = {"promo_consegna", "promo_prodotti", "promo_lampo"}
ERRORI_FIELDS = {"costo_incidenti_prodotti", "tariffa_attesa", "costo_annullamenti", "consegna_gratuita_incidente"}
META_FIELDS = {
    "subtotale", "iva", "totale_fattura", "prodotti", "servizio_consegna",
    "rimborsi_partner", "glovo_gia_pagati", "debito_accumulato", "importo_bonifico",
}

# Nella sezione di riconciliazione della fattura (in basso nella pagina 1)
# queste voci sono stampate come sottrazioni dal valore "Prodotti" (segno
# negativo), ma rappresentano comunque un costo reale per il ristorante --
# vanno normalizzate in positivo prima di sommarle nei bucket, altrimenti un
# "costo ads/errori" finirebbe per ridurre il totale invece di aumentarlo.
RECONCILIATION_COST_FIELDS = {
    "promo_prodotti", "promo_lampo", "costo_incidenti_prodotti", "tariffa_attesa",
    "costo_annullamenti", "consegna_gratuita_incidente",
}

LINE_RE = re.compile(r"^[+\-]?\s*(.+?)\s+(-?[\d.,]+)\s*€\s*$")
INVOICE_NUM_RE = re.compile(r"N\.\s*fattura:\s*(\S+)")
PERIOD_RE = re.compile(r"Servizio fornito da\s*(\d{4}-\d{2}-\d{2})\s*a\s*(\d{4}-\d{2}-\d{2})")

_ACCENT_MAP = str.maketrans("àèéìòù", "aeeiou")


def _slugify(label: str) -> str:
    slug = label.strip().lower().translate(_ACCENT_MAP)
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _resolve_key(label: str) -> tuple[str, bool]:
    """Ritorna (chiave_interna, trovata_in_mappa)."""
    low = label.lower().strip()
    if low.startswith("iva"):
        return "iva", True
    if low in LABEL_KEY_MAP:
        return LABEL_KEY_MAP[low], True
    return _slugify(label), False


def _it_number_to_float(value: str) -> float:
    """Converte un numero in formato italiano ('3.312,08', '-525,34') in float."""
    value = value.strip()
    negative = value.startswith("-")
    value = value.lstrip("-").replace(".", "").replace(",", ".")
    parsed = float(value)
    return -parsed if negative else parsed


def _delivery_costs_state_id(restaurant_id: str) -> str:
    """Identificativo di stato separato da sync_raw_data.py e food_cost.py --
    evita di sovrascrivere lo stato di altri flussi dati. Il suffisso
    numerico forza un riparsing completo di tutte le fatture quando cambia lo
    schema dei campi salvati (v2: separazione sconti/ads + fatturato netto;
    v3: separazione ads/prime; v4: costo_piattaforma_totale; v5: importi al
    lordo IVA; v6: lettura multi-pagina, corregge l'importo del bonifico
    mancante sulle fatture con voci di costo extra; v7: "offerta lampo"
    mappata come voce propria invece di restare nascosta in "altro"/sconti
    generici) -- altrimenti la cache di stato su GitHub Actions terrebbe per
    sempre i record vecchi, dato che il riparsing scatta solo sui filename
    nuovi."""
    return f"{restaurant_id}_glovo_v7"


def _find_invoice_pdfs(service, folder_id: str) -> list[dict]:
    """Attraversa ricorsivamente folder_id (struttura tipica AAAA/MM/*.pdf,
    ma generica: non assume profondita' o nomi fissi) e ritorna tutti i file
    PDF trovati."""
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

    pdfs = [c for c in children if c["name"].lower().endswith(".pdf")]
    subfolders = [c for c in children if c["mimeType"] == GOOGLE_FOLDER_MIME]
    for sub in subfolders:
        pdfs.extend(_find_invoice_pdfs(service, sub["id"]))
    return pdfs


def sync_glovo_invoices(config: RestaurantConfig) -> list[Path]:
    """Scarica in config.raw_glovo_dir le fatture PDF non ancora presenti
    localmente (confronto per nome file, gia' univoco lato Glovo)."""
    folder_id = config.delivery_costs.glovo_folder_id
    if not folder_id:
        return []

    service = _build_drive_service()
    files = _find_invoice_pdfs(service, folder_id)

    config.raw_glovo_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in config.raw_glovo_dir.glob("*.pdf")}

    downloaded = []
    for file in files:
        if file["name"] in existing:
            continue
        destination = config.raw_glovo_dir / file["name"]
        _download_file(service, file, destination)
        downloaded.append(destination)

    return downloaded


def parse_invoice_pdf(path: Path) -> dict:
    """Estrae dalle pagine iniziali (il riepilogo) di una fattura Glovo tutte
    le voci 'etichetta valore €', e le aggrega nei bucket richiesti
    (commissione/ads/prime/errori/altro), ciascuno anche al lordo IVA dove la
    fattura fornisce la scomposizione. Parser deterministico su testo
    (pdfplumber), non un LLM: il layout del riepilogo e' fisso.

    Il riepilogo di solito sta tutto in pagina 1, ma se la fattura ha voci di
    costo in piu' (es. "Promozione lampo", vista solo su alcuni locali) la
    riga finale "Importo del bonifico" puo' slittare in pagina 2 -- quindi si
    continua a leggere pagina per pagina finche' non la si trova (o fino a un
    tetto di sicurezza, per non finire a leggere le pagine di dettaglio
    ordini in fondo alla fattura). Nota: "Debito accumulato" NON va bene come
    segnale di stop -- e' una riga fissa della riconciliazione, stampata
    sempre (anche a 0,00 €) gia' in pagina 1, non l'ultima riga della
    fattura.

    Le voci di costo extra viste solo su alcune fatture (es. "offerta
    lampo") non vengono mai perse: finiscono comunque nel totale tramite
    "altro"/RECONCILIATION_COST_FIELDS anche se non sono in LABEL_KEY_MAP --
    ma se sono mappate esplicitamente (vedi LABEL_KEY_MAP) diventano visibili
    come voce propria nel report invece di restare nascoste dentro "altro"."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text() or ""
        for page in pdf.pages[1:5]:
            if re.search(r"importo del bonifico", text, re.IGNORECASE):
                break
            text += "\n" + (page.extract_text() or "")

    invoice_match = INVOICE_NUM_RE.search(text)
    period_match = PERIOD_RE.search(text)

    raw: dict[str, float] = {}
    unmapped: dict[str, str] = {}
    for line in text.splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        label, value_str = match.group(1).strip(), match.group(2)
        key, known = _resolve_key(label)
        if not known:
            unmapped[key] = label
        value = _it_number_to_float(value_str)
        if key in RECONCILIATION_COST_FIELDS:
            value = abs(value)
        # Prima occorrenza vince: "Totale fattura" compare due volte (il
        # totale a inizio pagina, poi di nuovo -- stesso valore negato --
        # nella sezione di riconciliazione sotto); teniamo la prima.
        raw.setdefault(key, value)

    def sum_fields(fields: set[str]) -> float:
        return round(sum(raw.get(f, 0.0) for f in fields), 2)

    known_fields = (
        COMMISSIONE_FIELDS | ADS_FIELDS | PRIME_FIELDS | OFFERTA_LAMPO_FIELDS
        | SCONTI_FIELDS | ERRORI_FIELDS | META_FIELDS
    )
    altro = round(sum(v for k, v in raw.items() if k not in known_fields), 2)

    prodotti = raw.get("prodotti", 0.0)
    sconti = sum_fields(SCONTI_FIELDS)
    # "Prodotti" e' il valore degli ordini venduti tramite Glovo, IVA food
    # gia' inclusa (e' quello che il ristorante batte in cassa): confermato
    # col ristorante, non e' "prima di IVA". Il fatturato netto sottrae solo
    # gli sconti/promozioni che il ristorante ha finanziato -- non
    # commissioni/ads/errori, che sono costi sul fatturato netto, non
    # riduzioni del fatturato stesso.
    fatturato_netto = round(prodotti - sconti, 2)
    commissione = sum_fields(COMMISSIONE_FIELDS)
    ads = sum_fields(ADS_FIELDS)
    prime = sum_fields(PRIME_FIELDS)
    offerta_lampo = sum_fields(OFFERTA_LAMPO_FIELDS)
    errori = sum_fields(ERRORI_FIELDS)

    # IVA: commissione/ads/prime/altro vengono dal blocco in cima alla
    # fattura "Subtotale -> IVA -> Totale fattura", quindi sono gia' importi
    # al NETTO IVA per costruzione (e' la fattura stessa a sommarli in
    # "Subtotale" prima di aggiungere l'IVA). Il rapporto
    # totale_fattura/subtotale e' l'aliquota applicata *letta dalla fattura
    # stessa* (mai fissata qui: se Glovo cambia aliquota si adatta da sola) e
    # si riapplica alle singole voci per avere anche l'importo al LORDO.
    # "Errori" viene invece dal blocco di riconciliazione "Prodotti" in
    # fondo alla fattura, che non ha una riga IVA propria: resta un unico
    # importo, identico in netto e lordo, perche' la fattura non fornisce
    # quella scomposizione.
    subtotale = raw.get("subtotale", 0.0)
    totale_fattura_servizi = raw.get("totale_fattura", 0.0)
    iva_ratio = (totale_fattura_servizi / subtotale) if subtotale else 1.0

    commissione_lordo = round(commissione * iva_ratio, 2)
    ads_lordo = round(ads * iva_ratio, 2)
    prime_lordo = round(prime * iva_ratio, 2)
    offerta_lampo_lordo = round(offerta_lampo * iva_ratio, 2)
    altro_lordo = round(altro * iva_ratio, 2)

    # Costo totale che il ristorante paga a Glovo per l'uso della
    # piattaforma (commissioni + ads + prime + offerta lampo + errori +
    # altro): tutto cio' che non e' "sconti" (che riduce il fatturato
    # stesso, non e' un costo sul fatturato) ne' una voce "meta" di
    # riepilogo/bonifico.
    costo_piattaforma_totale = round(commissione + ads + prime + offerta_lampo + errori + altro, 2)
    costo_piattaforma_totale_lordo = round(
        commissione_lordo + ads_lordo + prime_lordo + offerta_lampo_lordo + errori + altro_lordo, 2
    )

    return {
        "filename": path.name,
        "invoice_number": invoice_match.group(1) if invoice_match else None,
        "period_start": period_match.group(1) if period_match else None,
        "period_end": period_match.group(2) if period_match else None,
        "prodotti": prodotti,
        "sconti": sconti,
        "fatturato_netto": fatturato_netto,
        "commissione": commissione,
        "commissione_lordo": commissione_lordo,
        "ads": ads,
        "ads_lordo": ads_lordo,
        "prime": prime,
        "prime_lordo": prime_lordo,
        "offerta_lampo": offerta_lampo,
        "offerta_lampo_lordo": offerta_lampo_lordo,
        "errori": errori,
        "altro": altro,
        "altro_lordo": altro_lordo,
        "costo_piattaforma_totale": costo_piattaforma_totale,
        "costo_piattaforma_totale_lordo": costo_piattaforma_totale_lordo,
        "iva_servizi": raw.get("iva"),
        "totale_fattura": raw.get("totale_fattura"),
        "importo_bonifico": raw.get("importo_bonifico"),
        "raw": raw,
        "unmapped_fields": unmapped,
    }


def _load_glovo_state(restaurant_id: str) -> dict:
    return load_last_signature(_delivery_costs_state_id(restaurant_id)) or {"processed": {}}


def _save_glovo_record(restaurant_id: str, filename: str, record: dict) -> None:
    state = _load_glovo_state(restaurant_id)
    state["processed"][filename] = record
    save_signature(_delivery_costs_state_id(restaurant_id), state)


_INVOICE_FIELDS = [
    "filename", "invoice_number", "period_start", "period_end",
    "prodotti", "sconti", "fatturato_netto",
    "commissione", "commissione_lordo", "ads", "ads_lordo", "prime", "prime_lordo",
    "offerta_lampo", "offerta_lampo_lordo",
    "errori", "altro", "altro_lordo",
    "costo_piattaforma_totale", "costo_piattaforma_totale_lordo", "iva_servizi",
    "totale_fattura", "importo_bonifico",
]


def _append_invoices_csv(config: RestaurantConfig, records: dict[str, dict]) -> None:
    """Riscrive delivery_costs_glovo_invoices.csv da zero a partire da tutti i
    record noti in stato -- stesso approccio di food_cost.py (volume basso,
    poche decine di fatture l'anno, non serve append incrementale)."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    path = config.processed_path("delivery_costs_glovo_invoices.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_INVOICE_FIELDS)
        writer.writeheader()
        for filename, record in sorted(records.items()):
            writer.writerow({k: record.get(k) for k in _INVOICE_FIELDS} | {"filename": filename})


def _aggregate_monthly(config: RestaurantConfig, records: dict[str, dict]) -> None:
    monthly: dict[str, dict] = {}
    money_fields = [
        "prodotti", "sconti", "fatturato_netto",
        "commissione", "commissione_lordo", "ads", "ads_lordo", "prime", "prime_lordo",
        "offerta_lampo", "offerta_lampo_lordo",
        "errori", "altro", "altro_lordo",
        "costo_piattaforma_totale", "costo_piattaforma_totale_lordo", "iva_servizi",
        "totale_fattura", "importo_bonifico",
    ]

    for record in records.values():
        period_start = record.get("period_start")
        if not period_start:
            continue
        year_month = period_start[:7]
        bucket = monthly.setdefault(year_month, {f: 0.0 for f in money_fields} | {"n_invoices": 0})
        for field in money_fields:
            value = record.get(field)
            if value is not None:
                bucket[field] += value
        bucket["n_invoices"] += 1

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    path = config.processed_path("delivery_costs_glovo_monthly.csv")
    fieldnames = ["year_month"] + money_fields + ["n_invoices"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for year_month, bucket in sorted(monthly.items()):
            row = {field: round(bucket[field], 2) for field in money_fields}
            row["year_month"] = year_month
            row["n_invoices"] = bucket["n_invoices"]
            writer.writerow(row)


def run_delivery_costs_pipeline(config: RestaurantConfig) -> dict:
    """Orchestratore: sincronizza fatture nuove, estrae quelle non ancora
    elaborate (salvando lo stato dopo ognuna, non solo a fine batch), poi
    riscrive invoices.csv e monthly.csv da tutto lo stato noto. Skip
    silenzioso se glovo_folder_id non e' configurato."""
    if not config.delivery_costs.glovo_folder_id:
        return {"skipped": True}

    sync_glovo_invoices(config)

    state = _load_glovo_state(config.restaurant_id)
    processed = state["processed"]

    new_files = [p for p in config.raw_glovo_dir.glob("*.pdf") if p.name not in processed]
    for path in new_files:
        record = parse_invoice_pdf(path)
        record["parsed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        _save_glovo_record(config.restaurant_id, path.name, record)
        processed[path.name] = record

    _append_invoices_csv(config, processed)
    _aggregate_monthly(config, processed)

    return {
        "skipped": False,
        "n_new_invoices": len(new_files),
        "n_total_invoices": len(processed),
    }


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    result = run_delivery_costs_pipeline(config)
    print(result)


if __name__ == "__main__":
    main()
