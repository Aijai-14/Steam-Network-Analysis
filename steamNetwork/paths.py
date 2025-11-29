from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Repository root
DATA = ROOT / "data"                       # All data assets
RAW = DATA / "raw"                         # Unmodified inputs
INTERIM = DATA / "interim"                 # Partially processed artifacts
PROCESSED = DATA / "processed"             # Final model-ready datasets
REPORTS = ROOT / "reports"                 # Generated analyses
FIG = REPORTS / "figures"                  # Visualization outputs

# Ensure the entire directory tree exists before writing artifacts.
for p in [DATA, RAW, INTERIM, PROCESSED, REPORTS, FIG]:
    p.mkdir(parents=True, exist_ok=True)