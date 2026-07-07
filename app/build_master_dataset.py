import pandas as pd
import os
import re

DATA_DIR = "data/raw"

MONTHS = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12
}

CHANNEL_MAP = {
    "POS": "pos",
    "UBER": "delivery",
    "JUST EAT": "delivery",
    "GLOVO": "delivery",
    "DELIVEROO": "delivery",
    "TICKET": "digital",
    "SATISPAY": "digital",
    "CONTANTI": "cash"
}


def extract_year(path):
    return int(re.findall(r"20\d{2}", path)[0])


def detect_month(sheet):
    s = sheet.lower()
    for k, v in MONTHS.items():
        if k in s:
            return v
    return None


def clean(v):
    if pd.isna(v):
        return 0
    if isinstance(v, str):
        v = v.replace("€", "").replace(".", "").replace(",", ".")
    try:
        return float(v)
    except:
        return 0


def parse_file(path):
    xls = pd.ExcelFile(path)
    year = extract_year(path)

    all_rows = []

    for sheet in xls.sheet_names:

        month = detect_month(sheet)
        if month is None:
            continue

        df = pd.read_excel(path, sheet_name=sheet, header=None)

        # riga 1 = giorni (come hai detto tu)
        days_row = df.iloc[1]

        for i in range(2, len(df)):  # saltiamo header

            raw_channel = str(df.iloc[i, 1]).strip().upper()

            if raw_channel not in CHANNEL_MAP:
                continue

            channel = CHANNEL_MAP[raw_channel]

            for col in range(2, df.shape[1]):

                value = clean(df.iloc[i, col])
                if value == 0:
                    continue

                try:
                    day = int(days_row[col])
                except:
                    continue

                date = pd.Timestamp(year=year, month=month, day=day)

                all_rows.append([date, channel, value])

    return pd.DataFrame(all_rows, columns=["date", "channel", "revenue"])


# =========================
# LOAD FILES
# =========================

files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")]

df_all = pd.concat([parse_file(f) for f in files], ignore_index=True)

print("\nCHECK CHANNELS:")
print(df_all["channel"].value_counts())

# =========================
# TOTAL
# =========================

df_total = df_all.groupby("date")["revenue"].sum().reset_index()
df_total.columns = ["date", "total"]

pivot = df_all.pivot_table(
    index="date",
    columns="channel",
    values="revenue",
    aggfunc="sum"
).fillna(0)

final = pivot.merge(df_total, on="date", how="left")

if "delivery" not in final.columns:
    final["delivery"] = 0

final["delivery_share"] = final["delivery"] / final["total"]

final = final.sort_values("date")

print("\n=== DATASET READY ===")
print(final.head())

print("\nROWS:", len(final))
print("\nAVG DELIVERY SHARE:", final["delivery_share"].mean())

final.to_csv("data/processed/master_dataset.csv", index=False)
