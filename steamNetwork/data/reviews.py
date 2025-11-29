from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..utils import to_datetime

COLS = [
    "user_id","app_id","review_id","timestamp","recommended","votes_up","playtime_hours"
]


def load_reviews_csv(path: Path) -> pd.DataFrame:
    """Read reviews CSV, validate required columns, and enforce dtypes."""
    df = pd.read_csv(path)
    missing = [c for c in COLS if c not in df.columns] # Check if the columns are present
    if missing:
        raise ValueError(f"reviews.csv missing columns: {missing}")
    df = to_datetime(df, "timestamp") # Convert the timestamp column to datetime
    return df[COLS].astype({"app_id": int}) # Convert the app_id column to int