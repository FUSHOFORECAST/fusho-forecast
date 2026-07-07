import subprocess
import sys

STEPS = [
    ("01 Extract Excel", "app/01_extract_excel.py"),
    ("02 Clean Dataset", "app/02_clean_dataset.py"),
    ("08 Fetch Weather", "app/08_fetch_weather.py"),
    ("13 Calendar Features", "app/13_add_calendar_features.py"),
    ("17 Events Calendar", "app/17_create_events_calendar.py"),
    ("18 Full External Dataset", "app/18_build_full_external_dataset.py"),
    ("Restaurant Profile", "src/evaluation/restaurant_profiler.py"),
    ("Restaurant DNA", "src/evaluation/restaurant_dna.py"),
    ("Profile Features", "src/features/profile_features.py"),
    ("Build Features", "src/features/build_features.py"),
    ("Hierarchical Forecast", "app/27_forecast_hierarchical.py"),
]

def run_step(name, script):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script],
        env={**dict(), "PYTHONPATH": "."},
    )

    if result.returncode != 0:
        print("\nERRORE NELLO STEP:", name)
        sys.exit(result.returncode)

def main():
    print("\nAVVIO PIPELINE FUSHO FORECAST")

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETATA")
    print("=" * 80)
    print("Output principali:")
    print("- data/processed/master_dataset_clean.csv")
    print("- data/processed/master_dataset_full.csv")
    print("- data/processed/model_dataset_profile.csv")
    print("- reports/restaurant_profile.json")
    print("- reports/restaurant_dna.json")
    print("- data/processed/forecast_hierarchical_7_days.csv")

if __name__ == "__main__":
    main()
