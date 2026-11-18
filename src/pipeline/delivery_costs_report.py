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
    "errori", "costo_piattaforma_totale", "costo_piattaforma_totale_lordo", "importo_bonifico",
]

# (etichetta, campo, formattatore, riga secondaria, separatore sopra)
TABLE_ROWS = [
    ("Fatturato lordo", "prodotti", "eur", False, False),
    ("Sconti €", "sconti", "eur", False, False),
    ("% sconti", "pct_sconti", "pct", True, False),
    ("Fatturato netto", "fatturato_netto", "eur", False, True),
    ("Costo tot. (netto IVA)", "costo_piattaforma_totale", "eur", False, False),
    ("Costo tot. (lordo IVA)", "costo_piattaforma_totale_lordo", "eur", False, False),
    ("% costo tot.", "pct_costo_piattaforma_totale", "pct", True, False),
    ("Commissioni (netto)", "commissione", "eur", False, True),
    ("Commissioni (lordo)", "commissione_lordo", "eur", False, False),
    ("% commissioni", "pct_commissione", "pct", True, False),
    ("Ads (netto)", "ads", "eur", False, True),
    ("Ads (lordo)", "ads_lordo", "eur", False, False),
    ("% ads", "pct_ads", "pct", True, False),
    ("Prime (netto)", "prime", "eur", False, True),
    ("Prime (lordo)", "prime_lordo", "eur", False, False),
    ("% prime", "pct_prime", "pct", True, False),
    ("Errori €", "errori", "eur", False, True),
    ("% errori", "pct_errori", "pct", True, False),
    ("Netto incassato (bonifico Glovo)", "importo_bonifico", "eur", False, True),
]


def _fmt_eur(v: float) -> str:
    return "€ " + f"{v:,.0f}".replace(",", ".")


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f}%".replace(".", ",")


def _month_label(ym: str) -> str:
    y, mo = ym.split("-")
    return f"{MONTHS_IT[int(mo) - 1]} {y}"


def _aggregate_by_month(invoices: list[dict]) -> dict[str, dict]:
    by_month: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for row in invoices:
        month = row["period_start"][:7]
        m = by_month[month]
        for field in SUM_FIELDS:
            m[field] += float(row[field])

    for m in by_month.values():
        m["pct_sconti"] = (m["sconti"] / m["prodotti"] * 100) if m["prodotti"] else None
        for field in ["costo_piattaforma_totale", "commissione", "ads", "prime", "errori"]:
            m[f"pct_{field}"] = (m[field] / m["fatturato_netto"] * 100) if m["fatturato_netto"] else None

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
        f'<th style="{th_style}">{_month_label(ym)}</th>' for ym in months
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
        f"{_month_label(months[-1])}: fatturato netto {_fmt_eur(last['fatturato_netto'])}, "
        f"costo totale piattaforma {_fmt_eur(last['costo_piattaforma_totale'])} netto IVA / "
        f"{_fmt_eur(last['costo_piattaforma_totale_lordo'])} lordo IVA "
        f"({_fmt_pct(last['pct_costo_piattaforma_totale'])}) — "
        f"netto incassato (bonifico Glovo): {_fmt_eur(last['importo_bonifico'])}."
    )

    html = f"""\
<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;color:#1a1a1a;max-width:900px;">
  <h2 style="font-size:18px;margin:0 0 4px 0;">Margini Delivery — Glovo ({config.display_name})</h2>
  <p style="color:#555;font-size:13px;margin:0 0 16px 0;">{lead}</p>
  <table style="border-collapse:collapse;width:100%;font-size:12px;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
  <p style="color:#777;font-size:11px;margin-top:16px;">
    Netto/lordo IVA: commissioni, ads e prime hanno sia l'importo netto IVA (come in fattura) sia il
    lordo IVA (netto + IVA, aliquota letta dalla fattura). Errori resta un unico importo (blocco di
    riconciliazione, senza IVA propria). Le percentuali sono calcolate sul fatturato netto (importi
    netto IVA), tranne % sconti che e' su prodotti/fatturato lordo. <b>Netto incassato</b> e' l'importo
    del bonifico che Glovo versa davvero in banca, gia' al netto di tutti i costi (commissioni, ads,
    prime, errori, lordo IVA) -- il numero di fondo di tutta la tabella.
  </p>
</div>
"""

    title = f"Margini delivery Glovo {config.display_name} — {_month_label(months[-1])}"
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
    file_id = upload_file_to_drive(service, year_folder_id, pdf_path, "Margini Delivery Glovo.pdf")
    return file_id
