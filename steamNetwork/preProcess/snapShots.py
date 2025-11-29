from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import pandas as pd
from ..utils import month_floor
from .affinityProjection import compute_affinity


@dataclass
class SnapshotConfig:
    """Configuration for aggregating reviews into monthly affinity graphs."""
    window_days: int = 30 # window size in days
    min_coreviews: int = 3 # minimum number of coreviews to include an edge
    weight_scheme: str = "wjaccard" # weight scheme
    engagement_weights: Mapping[str, float] | None = None


def build_monthly_windows(reviews: pd.DataFrame) -> pd.DataFrame:
    """Annotate each review with its calendar month (month-end timestamp)."""
    reviews = reviews.copy()
    reviews["month"] = month_floor(reviews["timestamp"])  # month end marker
    return reviews


def build_snapshots(reviews: pd.DataFrame, cfg: SnapshotConfig) -> pd.DataFrame:
    """Aggregate reviews month-by-month into weighted co-view edges.

    Args:
        reviews: Raw review table containing ``timestamp`` and ``app_id`` columns.
        cfg: SnapshotConfig controlling weight scheme and minimum coreviews.

    Returns:
        DataFrame with columns ``[month, i, j, w, co, deg_i, deg_j]``.
    """
    reviews = build_monthly_windows(reviews) # Annotate each review with its calendar month (month-end timestamp)
    rows = []
    # Iterate through the months
    for month, block in reviews.groupby("month"):
        # Compute the affinity
        edges = compute_affinity(
            block,
            scheme=cfg.weight_scheme,
            engagement_weights=cfg.engagement_weights,
        )
        # Filter the edges by the minimum number of coreviews
        edges = edges[edges["co"] >= cfg.min_coreviews].copy() 
        edges.insert(0, "month", month) # Insert the month column
        rows.append(edges) # Append the edges to the rows list
    if rows:
        return pd.concat(rows, axis=0, ignore_index=True) # Concatenate the rows into a DataFrame
    return pd.DataFrame(columns=["month","i","j","w","co","deg_i","deg_j"]) # Return an empty DataFrame