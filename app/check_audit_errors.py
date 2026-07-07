import pandas as pd

audit = pd.read_csv("data/processed/audit_totals.csv")
audit["date"] = pd.to_datetime(audit["date"])

audit["abs_difference"] = audit["difference"].abs()

errors = audit[audit["abs_difference"] > 1].copy()
errors = errors.sort_values("abs_difference", ascending=False)

print("\n=== ERRORI PRINCIPALI ===")
print(errors.head(50)[[
    "date",
    "excel_total",
    "computed_total",
    "difference",
    "source_file",
    "sheet"
]])

print("\nNUMERO ERRORI > 1 EURO:", len(errors))
