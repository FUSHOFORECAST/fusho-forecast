"""Costruisce il report HTML (tabella mensile, voci su riga / mesi su colonna,
netto e lordo IVA) dei margini delivery Glovo, e lo converte in PDF per la
pubblicazione su Google Drive."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.pipeline.config import RestaurantConfig

MONTHS_IT = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
    "agosto", "settembre", "ottobre", "novembre", "dicembre",
]

SUM_FIELDS = [
    "prodotti", "sconti", "fatturato_netto",
    "commissione", "commissione_lordo", "ads", "ads_lordo", "prime", "prime_lordo",
    "offerta_lampo", "offerta_lampo_lordo",
    "errori", "costo_piattaforma_totale", "costo_piattaforma_totale_lordo",
    "rettifiche_fatturazione", "contanti", "importo_bonifico",
]

# (etichetta, campo, formattatore, riga secondaria, separatore sopra)
TABLE_ROWS = [
    ("Fatturato lordo", "prodotti", "eur", False, False),
    ("Sconti €", "sconti", "eur", False, False),
    ("% sconti", "pct_sconti", "pct", True, False),
    ("Fatturato netto", "fatturato_netto", "eur", False, True),
    ("Costo tot. (netto IVA)", "costo_piattaforma_totale", "eur", False, False),
    ("Costo tot. (lordo IVA)", "costo_piattaforma_totale_lordo", "eur", False, False),
    ("% costo tot. (netto IVA)", "pct_costo_piattaforma_totale", "pct", True, False),
    ("% costo tot. (lordo IVA)", "pct_costo_piattaforma_totale_lordo", "pct", True, False),
    ("Commissioni (netto)", "commissione", "eur", False, True),
    ("Commissioni (lordo)", "commissione_lordo", "eur", False, False),
    ("% commissioni (netto IVA)", "pct_commissione", "pct", True, False),
    ("% commissioni (lordo IVA)", "pct_commissione_lordo", "pct", True, False),
    ("Ads (netto)", "ads", "eur", False, True),
    ("Ads (lordo)", "ads_lordo", "eur", False, False),
    ("% ads (netto IVA)", "pct_ads", "pct", True, False),
    ("% ads (lordo IVA)", "pct_ads_lordo", "pct", True, False),
    ("Prime (netto)", "prime", "eur", False, True),
    ("Prime (lordo)", "prime_lordo", "eur", False, False),
    ("% prime (netto IVA)", "pct_prime", "pct", True, False),
    ("% prime (lordo IVA)", "pct_prime_lordo", "pct", True, False),
    ("Offerta lampo (netto)", "offerta_lampo", "eur", False, True),
    ("Offerta lampo (lordo)", "offerta_lampo_lordo", "eur", False, False),
    ("% offerta lampo (netto IVA)", "pct_offerta_lampo", "pct", True, False),
    ("% offerta lampo (lordo IVA)", "pct_offerta_lampo_lordo", "pct", True, False),
    ("Errori €", "errori", "eur", False, True),
    ("% errori", "pct_errori", "pct", True, False),
    ("Rettifiche fatturazione (rimborsi/debito)", "rettifiche_fatturazione", "eur", False, True),
    ("Contanti (incassati dal ristorante, anticipati da Glovo)", "contanti", "eur", False, True),
    ("Netto incassato via bonifico (Glovo)", "importo_bonifico", "eur", False, True),
    ("Incasso totale (bonifico + contanti già anticipati)", "incasso_totale", "eur", False, True),
    ("% incasso totale / fatturato netto", "pct_incasso_totale", "pct", True, False),
]


def _fmt_eur(v: float) -> str:
    return "€ " + f"{v:,.0f}".replace(",", ".")


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f}%".replace(".", ",")


def _month_label(ym: str, n_invoices: float | None = None) -> str:
    """Glovo fattura a quindicine (2 fatture/mese, H1+H2): se per un mese ne
    e' arrivata solo una, il mese compare comunque nella tabella (non si
    aspetta che sia completo) ma va segnalato come parziale, altrimenti i
    suoi numeri sembrano un mese intero mentre sono solo meta'."""
    y, mo = ym.split("-")
    label = f"{MONTHS_IT[int(mo) - 1]} {y}"
    if n_invoices is not None and n_invoices < 2:
        label += " (parziale)"
    return label


def _num(v) -> float:
    """Il bonifico (o le altre voci meta) possono mancare del tutto su una
    fattura quando quel periodo il ristorante va in debito con Glovo invece
    di ricevere un pagamento -- non e' un errore di parsing, e' lo stato
    della fattura stessa. Trattato come 0 invece di far fallire il report."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _aggregate_by_month(invoices: list[dict]) -> dict[str, dict]:
    by_month: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for row in invoices:
        month = row["period_start"][:7]
        m = by_month[month]
        m["n_invoices"] += 1
        for field in SUM_FIELDS:
            m[field] += _num(row[field])

    for m in by_month.values():
        m["pct_sconti"] = (m["sconti"] / m["prodotti"] * 100) if m["prodotti"] else None
        # "Netto incassato via bonifico" e' solo la parte che Glovo versa in
        # banca. Ma i contanti anticipati ("glovo gia' pagati") sono soldi che
        # il ristorante ha gia' incassato alla consegna -- vanno sommati per
        # avere il vero incasso totale (stesso principio di Deliveroo: il
        # bonifico da solo sottostima quanto resta davvero al ristorante).
        m["incasso_totale"] = round(m["importo_bonifico"] + m["contanti"], 2)
        # Quanto resta davvero al ristorante (bonifico + contanti, gia' al
        # netto di tutti i costi) per ogni euro di fatturato netto (prodotti -
        # sconti) -- stessa base di tutte le altre percentuali della tabella
        # (costo tot., commissioni, ads, prime, offerta lampo, errori), per
        # restare confrontabili fra loro. Un unico valore (non netto/lordo
        # IVA): l'incasso totale e' gia' l'importo reale finale, non una voce
        # scomponibile per IVA.
        m["pct_incasso_totale"] = (m["incasso_totale"] / m["fatturato_netto"] * 100) if m["fatturato_netto"] else None
        for field in ["costo_piattaforma_totale", "commissione", "ads", "prime", "offerta_lampo", "errori"]:
            m[f"pct_{field}"] = (m[field] / m["fatturato_netto"] * 100) if m["fatturato_netto"] else None
        # Stesso fatturato (quello che batti in cassa, gia' IVA inclusa) a
        # denominatore: qui uso il costo lordo IVA al numeratore, cosi' chi
        # legge vede sia l'incidenza "sulla carta" (netto IVA) sia quella
        # sui soldi che escono davvero dal conto (lordo IVA). Niente aliquota
        # IVA sul cibo da scorporare: e' una scelta esplicita, non un calcolo
        # fiscalmente "giusto" in senso stretto (le due IVA, 10% food e 22%
        # servizi, restano diverse) ma e' quello che l'utente ha chiesto.
        for field in ["costo_piattaforma_totale", "commissione", "ads", "prime", "offerta_lampo"]:
            m[f"pct_{field}_lordo"] = (m[f"{field}_lordo"] / m["fatturato_netto"] * 100) if m["fatturato_netto"] else None

    return by_month


def build_delivery_costs_report(config: RestaurantConfig, year: int | None = None) -> tuple[str, str] | None:
    """Ritorna (titolo, html) con la tabella mensile dei margini delivery, o
    None se non ci sono ancora fatture elaborate per questo ristorante (o per
    l'anno richiesto). Se year e' dato, limita la tabella ai mesi di quell'anno
    -- usato per generare un report separato per ogni cartella Drive/anno."""
    csv_path = config.processed_path("delivery_costs_glovo_invoices.csv")
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        invoices = list(csv.DictReader(f))
    if year is not None:
        invoices = [r for r in invoices if r["period_start"][:4] == str(year)]
    if not invoices:
        return None

    by_month = _aggregate_by_month(invoices)
    months = sorted(by_month)

    th_style = "padding:6px 10px;text-align:right;border-bottom:1px solid #ddd;background:#f5f5f5;font-weight:600;color:#333;white-space:nowrap;"
    th_label_style = th_style + "text-align:left;"
    td_style = "padding:6px 10px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap;"
    td_label_style = "padding:6px 10px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap;font-weight:600;"
    sub_style = "color:#777;font-size:11px;font-weight:400;"
    divider_style = "border-top:2px solid #999;"

    header_cells = f'<th style="{th_label_style}">Voce</th>' + "".join(
        f'<th style="{th_style}">{_month_label(ym, by_month[ym]["n_invoices"])}</th>' for ym in months
    )

    body_rows = []
    for label, field, kind, is_sub, is_divider in TABLE_ROWS:
        row_td = td_style + (sub_style if is_sub else "") + (divider_style if is_divider else "")
        row_th = td_label_style + (sub_style if is_sub else "") + (divider_style if is_divider else "")
        cells = "".join(
            f'<td style="{row_td}">'
            + (_fmt_eur(by_month[ym][field]) if kind == "eur" else _fmt_pct(by_month[ym][field]))
            + "</td>"
            for ym in months
        )
        body_rows.append(f'<tr><th style="{row_th}">{label}</th>{cells}</tr>')

    last = by_month[months[-1]]
    lead = (
        f"{_month_label(months[-1], last['n_invoices'])}: fatturato netto {_fmt_eur(last['fatturato_netto'])}, "
        f"costo totale piattaforma {_fmt_eur(last['costo_piattaforma_totale'])} netto IVA / "
        f"{_fmt_eur(last['costo_piattaforma_totale_lordo'])} lordo IVA "
        f"({_fmt_pct(last['pct_costo_piattaforma_totale'])} netto IVA — "
        f"{_fmt_pct(last['pct_costo_piattaforma_totale_lordo'])} lordo IVA) — "
        f"incasso totale: {_fmt_eur(last['incasso_totale'])} "
        f"({_fmt_pct(last['pct_incasso_totale'])} del fatturato netto), "
        f"di cui {_fmt_eur(last['importo_bonifico'])} via bonifico Glovo e {_fmt_eur(last['contanti'])} in contanti."
    )

    html = f"""\
<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;color:#1a1a1a;max-width:900px;">
  <h2 style="font-size:18px;margin:0 0 4px 0;">Glovo Margini Delivery ({config.display_name})</h2>
  <p style="color:#555;font-size:13px;margin:0 0 16px 0;">{lead}</p>
  <table style="border-collapse:collapse;width:100%;font-size:12px;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
  <p style="color:#777;font-size:11px;margin-top:16px;">
    Netto/lordo IVA: commissioni, ads, prime e offerta lampo hanno sia l'importo netto IVA (come in
    fattura) sia il lordo IVA (netto + IVA, aliquota letta dalla fattura) -- "Offerta lampo" (costo per le
    campagne promozionali Glovo, vista solo su alcuni ristoranti/periodi) compare come riga propria invece
    di restare nascosta dentro "altro". Errori resta un unico importo (blocco di riconciliazione, senza
    IVA propria). <b>Fatturato netto</b> (prodotti − sconti) e' gia' IVA inclusa:
    e' l'incasso di cassa, al netto solo degli sconti finanziati al ristorante. Ogni voce di costo ha
    quindi due percentuali sullo stesso fatturato di cassa: "netto IVA" (il costo come scritto in
    fattura, IVA servizi esclusa) e "lordo IVA" (con l'IVA servizi inclusa, cioe' i soldi che escono
    davvero dal conto) -- le due aliquote (IVA food e IVA servizi) restano diverse, quindi sono due letture
    dello stesso costo, non un confronto "a pari IVA". % sconti resta su prodotti/fatturato lordo (prima
    degli sconti), non ha componente IVA. <b>Rettifiche fatturazione</b> raccoglie 2 voci della fattura che
    non sono un costo di questo periodo ma aggiustano comunque il bonifico finale: rimborsi al partner senza
    commissione Glovo (positivo), debito accumulato pregresso (negativo) -- spesso e' zero. <b>Contanti</b>
    NON e' un costo: e' "Glovo gia' pagati", un anticipo in contanti che il ristorante ha gia' incassato alla
    consegna -- Glovo lo detrae dal bonifico per non pagarlo due volte. <b>Netto incassato via bonifico</b> e'
    solo l'importo che Glovo versa in banca (gia' al netto di tutti i costi, lordo IVA, e delle rettifiche).
    <b>Incasso totale</b> e' il vero incasso complessivo del ristorante: bonifico + contanti gia' anticipati
    (i contanti non passano dal conto corrente, ma sono comunque soldi incassati) -- e' questa la cifra da
    confrontare col fatturato netto per capire quanto resta davvero al ristorante, non il solo bonifico:
    incasso totale = fatturato netto − costo tot. (lordo IVA) + rettifiche fatturazione + contanti.
    <b>% incasso totale / fatturato netto</b> e' quella stessa cifra rapportata al fatturato netto
    (prodotti - sconti, la stessa base di tutte le altre percentuali della tabella): quanto resta davvero al
    ristorante per ogni euro di incasso di cassa al netto degli sconti.
  </p>
</div>
"""

    title = f"Glovo Margini delivery {config.display_name} — {_month_label(months[-1])}"
    return title, html


def render_html_to_pdf(html: str, output_path: Path, landscape: bool = True) -> None:
    """Converte un frammento HTML (come quello di build_delivery_costs_report)
    in PDF usando Chromium headless (playwright), per la pubblicazione su
    Google Drive."""
    from playwright.sync_api import sync_playwright

    full_page = f"""\
<!doctype html>
<html lang="it">
<head><meta charset="utf-8"></head>
<body style="margin:24px;">{html}</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(full_page, wait_until="networkidle")
        page.pdf(path=str(output_path), format="A4", landscape=landscape, print_background=True)
        browser.close()


def publish_delivery_costs_report(config: RestaurantConfig) -> str | None:
    """Genera il PDF dei margini delivery dell'anno corrente e lo carica nella
    cartella Drive dedicata (config.delivery_costs.report_folder_id), in una
    sottocartella per anno creata automaticamente se non esiste gia'. Ritorna
    l'id del file su Drive, o None se il report non e' configurato o non ci
    sono ancora dati per l'anno corrente."""
    import datetime

    if not config.delivery_costs.report_folder_id:
        return None

    year = datetime.date.today().year
    result = build_delivery_costs_report(config, year=year)
    if result is None:
        return None
    _, html = result

    pdf_path = config.processed_path("margini_delivery_glovo.pdf")
    render_html_to_pdf(html, pdf_path)

    from src.pipeline.sync_raw_data import _build_drive_oauth_service, _find_or_create_subfolder, upload_file_to_drive

    # Scrittura con le credenziali OAuth del proprietario del Drive (non il
    # service account: su Drive personale non ha quota per possedere file).
    service = _build_drive_oauth_service()
    year_folder_id = _find_or_create_subfolder(service, config.delivery_costs.report_folder_id, str(year))

    # Migrazione una tantum: il report si chiamava "Margini Delivery Glovo.pdf"
    # prima di invertire l'ordine del nome -- rinomina il file gia' pubblicato
    # invece di lasciarlo orfano in Drive mentre se ne crea uno nuovo col nome
    # aggiornato. No-op una volta completata (il vecchio nome non c'e' piu').
    old_name_query = (
        f"'{year_folder_id}' in parents and trashed = false "
        "and name = 'Margini Delivery Glovo.pdf'"
    )
    old_files = service.files().list(q=old_name_query, fields="files(id)").execute().get("files", [])
    for old_file in old_files:
        service.files().update(fileId=old_file["id"], body={"name": "Glovo Margini Delivery.pdf"}).execute()

    file_id = upload_file_to_drive(service, year_folder_id, pdf_path, "Glovo Margini Delivery.pdf")
    return file_id
