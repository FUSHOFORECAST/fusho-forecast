"""Costruisce il report HTML (tabella mensile, voci su riga / mesi su colonna)
dei margini delivery Deliveroo, e lo converte in PDF per la pubblicazione su
Google Drive -- stessa impostazione del report Glovo (delivery_costs_report.py),
cosi' i due PDF ("Glovo Margini Delivery", "Deliveroo Margini Delivery") sono
immediatamente confrontabili nella stessa cartella anno per ristorante.

Differenza principale rispetto a Glovo: Deliveroo fattura a settimane (non a
quindicine), quindi un mese puo' avere un numero variabile di settimane gia'
note -- "parziale" qui significa "l'ultima settimana nota finisce prima della
fine del mese di calendario", non "meno di N fatture"."""

from __future__ import annotations

import calendar
import csv
import datetime
from collections import defaultdict
from pathlib import Path

from src.pipeline.config import RestaurantConfig

MONTHS_IT = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
    "agosto", "settembre", "ottobre", "novembre", "dicembre",
]

SUM_FIELDS = [
    "valore_ordini", "commissione", "commissione_iva", "sconti_marketing",
    "contanti", "rimborsi", "pubblicita", "altro", "totale_da_pagare",
]

TABLE_ROWS = [
    ("Fatturato ordini", "valore_ordini", "eur", False, False),
    ("Commissione Deliveroo", "commissione", "eur", False, True),
    ("IVA su commissione", "commissione_iva", "eur", False, False),
    ("Commissione totale (lordo IVA)", "commissione_totale", "eur", False, False),
    ("% commissione totale / fatturato ordini", "pct_commissione_totale", "pct", True, False),
    ("Rimborsi al cliente", "rimborsi", "eur", False, True),
    ("% rimborsi / fatturato ordini", "pct_rimborsi", "pct", True, False),
    ("Contanti (incassati dal ristorante, dedotti dal bonifico)", "contanti", "eur", False, True),
    ("Pubblicità (canone Deliveroo \"Annunci Marketer\")", "pubblicita", "eur", False, True),
    ("% pubblicità / fatturato ordini", "pct_pubblicita", "pct", True, False),
    ("Altro (non ancora classificato)", "altro", "eur", False, True),
    ("Sconti marketing (informativo, gia' nel prezzo ordine)", "sconti_marketing", "eur", False, True),
    ("% sconti marketing / fatturato ordini", "pct_sconti_marketing", "pct", True, False),
    ("Netto incassato via bonifico (da Deliveroo)", "totale_da_pagare", "eur", False, True),
    ("Incasso totale (bonifico + contanti gia' in cassa)", "incasso_totale", "eur", False, True),
    ("% incasso totale / fatturato ordini", "pct_incasso_totale", "pct", True, False),
    ("N. ordini", "n_orders", "int", True, True),
    ("N. settimane rendicontate", "n_statements", "int", True, False),
]


def _fmt_eur(v: float) -> str:
    return "€ " + f"{v:,.0f}".replace(",", ".")


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1f}%".replace(".", ",")


def _fmt_int(v) -> str:
    return str(int(v)) if v is not None else "—"


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _month_is_partial(year_month: str, last_period_end: str | None) -> bool:
    """Deliveroo fattura a settimane: un mese e' "completo" solo se l'ultima
    settimana nota arriva fino alla fine del mese di calendario -- non si puo'
    contare un numero fisso di settimane attese (puo' variare 4/5 a seconda
    del mese) senza doverlo aggiornare a mano ogni volta."""
    if not last_period_end:
        return True
    year, month = (int(x) for x in year_month.split("-"))
    last_day = calendar.monthrange(year, month)[1]
    month_end = datetime.date(year, month, last_day)
    try:
        known_end = datetime.date.fromisoformat(last_period_end)
    except ValueError:
        return True
    return known_end < month_end


def _month_label(ym: str, is_partial: bool) -> str:
    y, mo = ym.split("-")
    label = f"{MONTHS_IT[int(mo) - 1]} {y}"
    if is_partial:
        label += " (parziale)"
    return label


def _aggregate_by_month(statements: list[dict]) -> dict[str, dict]:
    by_month: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    last_period_end: dict[str, str] = {}

    for row in statements:
        period_start = row.get("period_start")
        if not period_start:
            continue
        month = period_start[:7]
        m = by_month[month]
        m["n_statements"] += 1
        m["n_orders"] += _num(row.get("n_orders"))
        for field in SUM_FIELDS:
            m[field] += _num(row.get(field))
        period_end = row.get("period_end") or period_start
        if period_end > last_period_end.get(month, ""):
            last_period_end[month] = period_end

    for month, m in by_month.items():
        m["commissione_totale"] = round(m["commissione"] + m["commissione_iva"], 2)
        m["incasso_totale"] = round(m["totale_da_pagare"] + m["contanti"], 2)
        m["pct_commissione_totale"] = (m["commissione_totale"] / m["valore_ordini"] * 100) if m["valore_ordini"] else None
        m["pct_rimborsi"] = (m["rimborsi"] / m["valore_ordini"] * 100) if m["valore_ordini"] else None
        m["pct_pubblicita"] = (m["pubblicita"] / m["valore_ordini"] * 100) if m["valore_ordini"] else None
        m["pct_sconti_marketing"] = (m["sconti_marketing"] / m["valore_ordini"] * 100) if m["valore_ordini"] else None
        m["pct_incasso_totale"] = (m["incasso_totale"] / m["valore_ordini"] * 100) if m["valore_ordini"] else None
        m["_is_partial"] = _month_is_partial(month, last_period_end.get(month))

    return by_month


def build_deliveroo_costs_report(config: RestaurantConfig, year: int | None = None) -> tuple[str, str] | None:
    """Ritorna (titolo, html) con la tabella mensile dei margini Deliveroo, o
    None se non ci sono ancora statement elaborati per questo ristorante (o
    per l'anno richiesto)."""
    csv_path = config.processed_path("delivery_costs_deliveroo_statements.csv")
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        statements = list(csv.DictReader(f))
    if year is not None:
        statements = [r for r in statements if r.get("period_start", "")[:4] == str(year)]
    if not statements:
        return None

    by_month = _aggregate_by_month(statements)
    months = sorted(by_month)
    if not months:
        return None

    th_style = "padding:6px 10px;text-align:right;border-bottom:1px solid #ddd;background:#f5f5f5;font-weight:600;color:#333;white-space:nowrap;"
    th_label_style = th_style + "text-align:left;"
    td_style = "padding:6px 10px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap;"
    td_label_style = "padding:6px 10px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap;font-weight:600;"
    sub_style = "color:#777;font-size:11px;font-weight:400;"
    divider_style = "border-top:2px solid #999;"

    header_cells = f'<th style="{th_label_style}">Voce</th>' + "".join(
        f'<th style="{th_style}">{_month_label(ym, by_month[ym]["_is_partial"])}</th>' for ym in months
    )

    formatters = {"eur": _fmt_eur, "pct": _fmt_pct, "int": _fmt_int}

    body_rows = []
    for label, field, kind, is_sub, is_divider in TABLE_ROWS:
        row_td = td_style + (sub_style if is_sub else "") + (divider_style if is_divider else "")
        row_th = td_label_style + (sub_style if is_sub else "") + (divider_style if is_divider else "")
        cells = "".join(
            f'<td style="{row_td}">{formatters[kind](by_month[ym][field])}</td>' for ym in months
        )
        body_rows.append(f'<tr><th style="{row_th}">{label}</th>{cells}</tr>')

    last = by_month[months[-1]]
    lead = (
        f"{_month_label(months[-1], last['_is_partial'])}: fatturato ordini {_fmt_eur(last['valore_ordini'])}, "
        f"commissione totale {_fmt_eur(last['commissione_totale'])} "
        f"({_fmt_pct(last['pct_commissione_totale'])} del fatturato ordini) — "
        f"incasso totale: {_fmt_eur(last['incasso_totale'])} "
        f"({_fmt_pct(last['pct_incasso_totale'])} del fatturato ordini), "
        f"di cui {_fmt_eur(last['totale_da_pagare'])} via bonifico Deliveroo e {_fmt_eur(last['contanti'])} in contanti."
    )

    html = f"""\
<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;color:#1a1a1a;max-width:900px;">
  <h2 style="font-size:18px;margin:0 0 4px 0;">Deliveroo Margini Delivery ({config.display_name})</h2>
  <p style="color:#555;font-size:13px;margin:0 0 16px 0;">{lead}</p>
  <table style="border-collapse:collapse;width:100%;font-size:12px;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
  <p style="color:#777;font-size:11px;margin-top:16px;">
    Deliveroo fattura a settimane (non a quindicine come Glovo): un mese resta "(parziale)" finche' l'ultima
    settimana nota non arriva fino alla fine del mese di calendario. <b>Commissione totale</b> e' commissione +
    IVA sulla commissione (unica aliquota disponibile sul CSV, non c'e' la scomposizione netto/lordo per voce
    vista su Glovo). <b>Rimborsi al cliente</b> e' un costo vero: storni per problemi sull'ordine (es. articolo
    sbagliato o mancante) detratti dal bonifico -- equivalente agli "Errori" del report Glovo. <b>Contanti</b>
    NON e' un costo: sono ordini pagati in contanti direttamente al corriere, gia' incassati dal ristorante, che
    Deliveroo detrae dal bonifico per non pagarli due volte (altrimenti li pagherebbe anche lui). <b>Pubblicità</b>
    e' un costo vero: il canone del programma pubblicitario Deliveroo "Annunci Marketer" (visibile sul CSV come
    "Ad Fee", addebitato a giornata, non legato a un singolo ordine) -- presente solo per i ristoranti che aderiscono
    a questo programma, non va confuso con lo "sconto offerta Marketer" (voce informativa sulle righe ordine, gia'
    incluso nel prezzo). <b>Altro</b> e' un residuo generico per qualunque tipo di riga diversa da
    ordine/rimborso/contanti/pubblicità mai ancora vista sui CSV reali (es. le rare rettifiche quando un ordine va
    riconsegnato/rifatto): resta a zero finche' Deliveroo non introduce un nuovo tipo di rendiconto non ancora
    osservato, cosi' il totale resta corretto anche in quel caso invece di nascondere l'importo dentro una delle
    altre voci. A differenza delle altre voci (mostrate come importo assoluto, perche' la direzione e' gia'
    confermata sui dati reali), <b>Altro</b> mantiene il segno originale: un valore negativo ha ridotto l'incasso
    (e' un costo), uno positivo lo ha aumentato (e' un credito a favore del ristorante -- es. nella riconsegna qui
    sopra il credito ricevuto per il rifacimento supera la commissione addebitata, netto positivo). <b>Sconti marketing</b>
    e' il totale degli sconti "offerta Marketer" segnalati sugli ordini: e' gia' incluso nel prezzo ordine (la
    commissione si calcola sul valore gia' scontato) -- riportato come informazione, non e' un costo aggiuntivo
    confermato a carico del ristorante. <b>Netto incassato via bonifico</b> e' solo la parte che Deliveroo versa
    in banca (gia' al netto di commissione, IVA, rimborsi e contanti). <b>Incasso totale</b> e' il vero incasso
    complessivo del ristorante su questi ordini: bonifico + contanti gia' incassati in cassa (i contanti non
    passano dal conto, ma sono comunque soldi incassati) -- e' questa la cifra da confrontare col fatturato
    ordini per capire quanto resta davvero al ristorante, non il solo bonifico.
  </p>
</div>
"""

    title = f"Deliveroo Margini delivery {config.display_name} — {_month_label(months[-1], last['_is_partial'])}"
    return title, html


def publish_deliveroo_costs_report(config: RestaurantConfig) -> str | None:
    """Genera il PDF dei margini Deliveroo dell'anno corrente e lo carica
    nella stessa cartella Drive/anno del report Glovo (config.delivery_costs.
    report_folder_id), con un nome file distinto cosi' i due PDF convivono
    senza sovrascriversi. Ritorna l'id del file su Drive, o None se il report
    non e' configurato o non ci sono ancora dati per l'anno corrente."""
    from src.pipeline.delivery_costs_report import render_html_to_pdf

    if not config.delivery_costs.report_folder_id:
        return None

    year = datetime.date.today().year
    result = build_deliveroo_costs_report(config, year=year)
    if result is None:
        return None
    _, html = result

    pdf_path = config.processed_path("margini_delivery_deliveroo.pdf")
    render_html_to_pdf(html, pdf_path)

    from src.pipeline.sync_raw_data import _build_drive_oauth_service, _find_or_create_subfolder, upload_file_to_drive

    service = _build_drive_oauth_service()
    year_folder_id = _find_or_create_subfolder(service, config.delivery_costs.report_folder_id, str(year))
    file_id = upload_file_to_drive(service, year_folder_id, pdf_path, "Deliveroo Margini Delivery.pdf")
    return file_id
