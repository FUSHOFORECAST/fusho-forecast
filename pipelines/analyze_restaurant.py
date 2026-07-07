import os
import subprocess
import sys


STEPS = [
    ("Adaptive Regime Engine", "src/evaluation/adaptive_regime_engine.py"),
    ("Business Calendar Analyzer", "src/evaluation/business_calendar_analyzer.py"),
    ("Business Model Profiler", "src/evaluation/business_model_profiler.py"),
    ("Universal Restaurant DNA", "src/dna/restaurant_dna_schema.py"),
    ("Similarity Engine", "src/similarity/similarity_engine.py"),
    ("Model Selection Engine", "src/model_selection/model_selection_engine.py"),
]


def run_step(name, script):
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    result = subprocess.run(
        [sys.executable, script],
        env={**os.environ, "PYTHONPATH": "."},
    )

    if result.returncode != 0:
        print("\nERRORE NELLO STEP:", name)
        sys.exit(result.returncode)


def main():
    print("\nAVVIO ANALISI RISTORANTE")

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "=" * 90)
    print("ANALISI RISTORANTE COMPLETATA")
    print("=" * 90)

    print("\nOutput principali:")
    print("- reports/adaptive_regime_profile.json")
    print("- reports/business_calendar_patterns.csv")
    print("- reports/business_season_patterns.csv")
    print("- reports/business_model_profile.json")
    print("- reports/restaurant_dna_universal.json")
    print("- reports/restaurant_similarity_vector.json")
    print("- reports/model_selection.json")


if __name__ == "__main__":
    main()
