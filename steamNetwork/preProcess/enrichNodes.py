from __future__ import annotations
import pandas as pd
from pathlib import Path


def build_node_attributes(games_csv: Path, price_events_csv: Path | None = None) -> pd.DataFrame:
    """Combine app metadata with latest known discount info to create node attributes.

    Args:
        games_csv: Path to enriched Steam app metadata.
        price_events_csv: Optional path to price-event timelines for latest discounts.

    Returns:
        DataFrame ready for node-level joins (app_id, prices, genres, temporal info).
    """
    # Read the games CSV
    meta = pd.read_csv(games_csv)
    # Convert the release date to year
    meta["release_year"] = pd.to_datetime(meta["release_date"], errors="coerce").dt.year

    if price_events_csv and price_events_csv.exists():
        # Read the price events CSV
        ev = pd.read_csv(price_events_csv, parse_dates=["timestamp"])
        last = ev.sort_values("timestamp").groupby("app_id").tail(1) # Get the last price event for each app
        last = last[["app_id","discount_pct","timestamp"]].rename(columns={
            "discount_pct":"last_discount_pct","timestamp":"last_discount_ts"
        }) # Rename the columns
        out = meta.merge(last, left_on="app_id", right_on="app_id", how="left") # Merge the metadata with the last price event
    else:
        out = meta.rename(columns={"discount_pct":"last_discount_pct"}) # Rename the columns
        out["last_discount_ts"] = pd.NaT # Set the last discount timestamp to NaN
    
    # columns: app_id, title, genres, base_price, current_price, last_discount_pct, last_discount_ts, release_year
    return out # Return the node attributes
