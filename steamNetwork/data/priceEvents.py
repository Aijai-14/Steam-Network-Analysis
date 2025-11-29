from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..utils import to_datetime

PE_COLS = ["timestamp","app_id","discount_pct","price_old","price_new"]


def load_price_events(path: Path) -> pd.DataFrame:
    """Read price events CSV, enforce schema, and normalize timestamp dtype."""
    df = pd.read_csv(path)
    missing = [c for c in PE_COLS if c not in df.columns] # Check if the columns are present
    if missing:
        raise ValueError(f"price_events.csv missing columns: {missing}")
    
    df = to_datetime(df, "timestamp") # Convert the timestamp column to datetime
    df = df.astype({"app_id": int, "discount_pct": float}) # Convert the app_id and discount_pct columns to int and float respectively
    return df