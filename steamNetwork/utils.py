from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable

from pandas.core.arrays import DatetimeArray

RNG = np.random.default_rng(42)


def to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse ``col`` into UTC timestamps (coercing invalid values) in-place.

    Args:
        df: DataFrame that will be mutated.
        col: Name of the column to parse.

    Returns:
        pd.DataFrame: Same object, provided for fluent chaining.
    """

    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def month_floor(ts: pd.Series) -> DatetimeArray:
    """Convert timestamps to month-end (floor) UTC datetimes without tz info.

    Args:
        ts: Series or array-like timestamp values.

    Returns:
        DatetimeArray: Month-floored timestamps normalized to UTC.
    """

    return pd.to_datetime(ts, utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")


def ensure_parquet(path: Path, df: pd.DataFrame):
    """Guarantee the parent directory exists and persist ``df`` as parquet.

    Args:
        path: Destination file path.
        df: DataFrame to serialize.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def chunked(iterable: Iterable, n: int):
    """Yield successive lists of size ``n`` from ``iterable`` (last may be short).

    Args:
        iterable: Source sequence or generator.
        n: Target batch size (>0).

    Yields:
        list: Up to ``n`` items from ``iterable`` in order.
    """

    bucket = []
    for x in iterable:
        bucket.append(x)
        if len(bucket) == n:
            yield bucket
            bucket = []
    if bucket:
        yield bucket