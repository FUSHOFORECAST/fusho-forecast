import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from src.pipeline.config import list_restaurant_configs, load_restaurant_config
from src.pipeline.delivery_costs import run_delivery_costs_pipeline
from src.pipeline.delivery_costs_report import publish_delivery_costs_report
from src.pipeline.deliveroo_costs import run_deliveroo_costs_pipeline
from src.pipeline.deliveroo_costs_report import publish_deliveroo_costs_report
from src.pipeline.food_cost import run_food_cost_pipeline

LOGS_DIR = Path("logs")
SUMMARY_LOG = LOGS_DIR / "run_all_summary.log"


def run_one(config_path: Path):
    config = load_restaurant_config(config_path)
    restaurant_id = config.restaurant_id

    log_dir = LOGS_DIR / restaurant_id
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.log"

    buffer = io.StringIO()
    start = time.time()
    success = True
    error_message = ""

    # Food cost e delivery costs sono due flussi dati indipendenti (foto DDT,
    # fatture Glovo): ciascuno isolato nel proprio try/except, cosi' un
    # fallimento nell'uno non impedisce all'altro di andare a buon fine.
    if config.food_cost.ddt_folder_id:
        try:
            with redirect_stdout(buffer):
                result = run_food_cost_pipeline(config)
                print(f"Food cost: {result}")
        except Exception:
            success = False
            error_message = traceback.format_exc()
            buffer.write("\nERRORE: pipeline food cost fallita:\n" + error_message)

    if config.delivery_costs.glovo_folder_id:
        try:
            with redirect_stdout(buffer):
                result = run_delivery_costs_pipeline(config)
                print(f"Delivery costs: {result}")
        except Exception:
            success = False
            error_message = traceback.format_exc()
            buffer.write("\nERRORE: pipeline delivery costs fallita:\n" + error_message)
        else:
            # Pubblica il PDF aggiornato su Drive solo se sono arrivate fatture
            # nuove (o se richiesto a mano via FORCE_REPORT_PUBLISH, es. dopo
            # una modifica al formato del report, da workflow_dispatch) --
            # un fallimento qui non invalida il run (i dati sono gia' salvi),
            # e' solo il report "bello" a non essersi aggiornato.
            if result.get("n_new_invoices") or os.environ.get("FORCE_REPORT_PUBLISH") == "true":
                try:
                    with redirect_stdout(buffer):
                        file_id = publish_delivery_costs_report(config)
                        print(f"Report delivery pubblicato su Drive: {file_id}")
                except Exception:
                    buffer.write("\nAVVISO: pubblicazione report delivery su Drive fallita:\n")
                    buffer.write(traceback.format_exc())

    if config.delivery_costs.deliveroo_code:
        try:
            with redirect_stdout(buffer):
                result = run_deliveroo_costs_pipeline(config)
                print(f"Deliveroo costs: {result}")
        except Exception:
            success = False
            error_message = traceback.format_exc()
            buffer.write("\nERRORE: pipeline deliveroo costs fallita:\n" + error_message)
        else:
            if result.get("n_new_statements") or os.environ.get("FORCE_REPORT_PUBLISH") == "true":
                try:
                    with redirect_stdout(buffer):
                        file_id = publish_deliveroo_costs_report(config)
                        print(f"Report Deliveroo pubblicato su Drive: {file_id}")
                except Exception:
                    buffer.write("\nAVVISO: pubblicazione report deliveroo su Drive fallita:\n")
                    buffer.write(traceback.format_exc())

    duration = time.time() - start
    log_path.write_text(buffer.getvalue())

    return restaurant_id, success, duration, error_message, log_path


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config_paths = list_restaurant_configs()

    if not config_paths:
        print("Nessun config ristorante trovato in config/restaurants/")
        return 0

    failures = []

    for config_path in config_paths:
        print(f"--- {config_path} ---")
        restaurant_id, success, duration, error_message, log_path = run_one(config_path)

        status = "OK" if success else "FALLITO"

        summary_line = (
            f"{datetime.now().isoformat(timespec='seconds')}\t{restaurant_id}\t{status}\t"
            f"{duration:.1f}s\t{log_path}"
        )
        if not success:
            failures.append(restaurant_id)
            last_error_line = error_message.strip().splitlines()[-1] if error_message else ""
            summary_line += f"\t{last_error_line}"

        with open(SUMMARY_LOG, "a") as f:
            f.write(summary_line + "\n")

        print(f"{restaurant_id}: {status} ({duration:.1f}s) -> {log_path}")

    print()
    print(f"Totale: {len(config_paths)} ristoranti, {len(failures)} falliti")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
