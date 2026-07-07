import io
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from src.pipeline.config import list_restaurant_configs, load_restaurant_config
from src.pipeline.run import run_pipeline
from src.pipeline.sync_raw_data import sync_from_drive

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
    error_message = ""
    success = True

    try:
        with redirect_stdout(buffer):
            n_synced = sync_from_drive(config)
            if n_synced:
                print(f"Sincronizzati {n_synced} file da Google Drive in {config.raw_dir}")
            run_pipeline(config)
    except Exception:
        success = False
        error_message = traceback.format_exc()
        buffer.write("\n" + error_message)

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
