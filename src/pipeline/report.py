import json

import pandas as pd

from src.pipeline.config import RestaurantConfig

WEEKDAY_IT = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]

CHANNEL_COLORS = {
    "delivery": "#2563eb",
    "digital": "#16a34a",
    "cash": "#d97706",
}
DEFAULT_COLOR = "#6b7280"


def _channel_color(channel: str) -> str:
    return CHANNEL_COLORS.get(channel, DEFAULT_COLOR)


def _format_date_it(date: pd.Timestamp) -> str:
    return f"{WEEKDAY_IT[date.dayofweek]} {date.strftime('%d/%m')}"


def _format_euro(value: float) -> str:
    return f"€{value:,.0f}".replace(",", ".")


def build_html_report(config: RestaurantConfig) -> tuple[str, str]:
    with open(config.reports_path("pipeline_summary.json")) as f:
        summary = json.load(f)

    forecast = pd.read_csv(summary["forecast_file"])
    forecast["date"] = pd.to_datetime(forecast["date"])

    channels = config.channels
    horizon = len(forecast)

    start_str = forecast["date"].min().strftime("%d/%m")
    end_str = forecast["date"].max().strftime("%d/%m")
    week_total = forecast["total_pred"].sum()

    subject = f"Previsione {config.display_name} — {start_str}-{end_str}: {_format_euro(week_total)}"

    max_total = forecast["total_pred"].max()

    rows_html = []
    for _, row in forecast.iterrows():
        bar_width = round(row["total_pred"] / max_total * 240) if max_total else 0
        channel_cells = "".join(
            f'<td style="padding:6px 10px;text-align:right;color:#374151;font-size:13px;">'
            f'{_format_euro(row[f"{channel}_pred"])}</td>'
            for channel in channels
        )
        rows_html.append(f"""
        <tr>
          <td style="padding:6px 10px;font-size:13px;color:#111827;">{_format_date_it(row['date'])}</td>
          <td style="padding:6px 10px;">
            <div style="background:#2563eb;height:10px;width:{bar_width}px;border-radius:4px;"></div>
          </td>
          <td style="padding:6px 10px;text-align:right;font-size:13px;font-weight:600;color:#111827;">
            {_format_euro(row['total_pred'])}
          </td>
          {channel_cells}
        </tr>
        """)

    channel_headers = "".join(
        f'<th style="padding:6px 10px;text-align:right;font-size:12px;color:#6b7280;">{channel}</th>'
        for channel in channels
    )

    backtest = summary["backtest"]
    accuracy_parts = [f"totale {backtest['total']['mape']:.0f}%"]
    for channel in channels:
        winner = backtest["winners"].get(channel)
        if winner and channel in backtest["per_channel_model"]:
            mape = backtest["per_channel_model"][channel][winner]["mape"]
            accuracy_parts.append(f"{channel} {mape:.0f}%")
    accuracy_line = ", ".join(accuracy_parts)

    html = f"""
    <div style="font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:600px;margin:0 auto;">
      <h2 style="font-size:18px;color:#111827;margin-bottom:4px;">{config.display_name}</h2>
      <p style="font-size:14px;color:#6b7280;margin-top:0;">
        Previsione {horizon} giorni: {start_str} - {end_str}
      </p>
      <p style="font-size:20px;font-weight:700;color:#111827;margin:16px 0;">
        {_format_euro(week_total)} <span style="font-size:13px;font-weight:400;color:#6b7280;">totale settimana</span>
      </p>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr>
            <th style="padding:6px 10px;text-align:left;font-size:12px;color:#6b7280;">giorno</th>
            <th></th>
            <th style="padding:6px 10px;text-align:right;font-size:12px;color:#6b7280;">totale</th>
            {channel_headers}
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
      <p style="font-size:12px;color:#9ca3af;margin-top:20px;border-top:1px solid #e5e7eb;padding-top:12px;">
        Errore medio stimato sugli ultimi {backtest['backtest_days']} giorni: {accuracy_line}.
      </p>
    </div>
    """

    return subject, html


def main():
    import argparse

    from src.pipeline.config import load_restaurant_config
    from src.pipeline.notify import send_email

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()

    config = load_restaurant_config(args.config)
    subject, html = build_html_report(config)

    if args.send:
        if not config.notify.recipient_email:
            raise ValueError(f"notify.recipient_email non impostato per {config.restaurant_id}")
        send_email(config.notify.recipient_email, subject, html)
        print(f"Email inviata a {config.notify.recipient_email}")
    else:
        print(subject)
        print(html)


if __name__ == "__main__":
    main()
