from __future__ import annotations
import math
from typing import Literal, Tuple, Mapping, Any
import networkx as nx
import numpy as np
import pandas as pd

# Weight schemes
WeightScheme = Literal["count", "jaccard", "wjaccard", "engagement"] 

# Default engagement weights
DEFAULT_ENGAGEMENT_WEIGHTS: Mapping[str, float] = {
    "base": 1.0, # base weight
    "rec_bonus": 0.75, # bonus for recommended reviews
    "votes_scale": 0.15, # scale for votes
    "playtime_scale": 0.1, # scale for playtime
    "vote_cap": 500.0, # cap for votes
    "playtime_cap": 200.0, # cap for playtime
}


def _pair_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count co-reviews (coreviews) for every unordered app pair within a window."""
    # within a window: group by user, list of games they reviewed, then make pairs
    out = []
    # Iterate through the users
    for uid, block in df.groupby("user_id"):
        games = block["app_id"].unique() # Get the unique games for the user
        games = np.sort(games) # Sort the games
        # Iterate through the games
        games = np.sort(games)
        for i in range(len(games)):
            # Iterate through the games
            for j in range(i+1, len(games)):
                out.append((games[i], games[j], 1))
    if not out:
        return pd.DataFrame(columns=["i","j","co"]) # Return an empty DataFrame
    co = pd.DataFrame(out, columns=["i","j","co"]).groupby(["i","j"], as_index=False)["co"].sum() # Group by the games and sum the co-reviews
    return co


def _node_degrees(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-app reviewer counts (used as degree proxies)."""
    # degree = number of distinct users per game in window
    deg = df.groupby("app_id")["user_id"].nunique().reset_index(name="deg") # Group by the games and count the unique users
    return deg.rename(columns={"app_id": "id"})  


def _engagement_score(row: pd.Series, weights: Mapping[str, float]) -> float:
    """Map a single review row to a capped engagement weight."""
    # Get the votes and playtime
    votes = min(float(row.get("votes_up", 0.0) or 0.0), weights["vote_cap"])
    play = min(max(float(row.get("playtime_hours", 0.0) or 0.0), 0.0), weights["playtime_cap"])
    # Calculate the engagement weight
    return (
        weights["base"] # base weight
        + weights["rec_bonus"] * float(row.get("recommended", 0)) # bonus for recommended reviews
        + weights["votes_scale"] * math.log1p(votes) # scale for votes
        + weights["playtime_scale"] * math.log1p(play) # scale for playtime
    )


def _engagement_tables(df: pd.DataFrame, weights: Mapping[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise engagement overlaps and per-node strengths."""
    if df.empty:
        empty_pairs = pd.DataFrame(columns=["i", "j", "score", "co"])
        empty_strength = pd.DataFrame(columns=["id", "strength"])
        return empty_pairs, empty_strength

    df = df.copy()
    # Calculate the engagement weight for each review
    df["engagement"] = df.apply(lambda row: _engagement_score(row, weights), axis=1)

    # Group by the games and sum the engagement weights
    node_strength = (
        df.groupby("app_id")["engagement"]
        .sum()
        .reset_index()
        .rename(columns={"engagement": "strength", "app_id": "id"})
    )

    # Iterate through the users
    rows: list[tuple[int, int, float, int]] = []
    for _, block in df.groupby("user_id"):
        block = block.drop_duplicates("app_id", keep="last") # Drop the duplicates
        if len(block) < 2:
            continue
        # Sort the games by the engagement weight
        block = block.sort_values("app_id")
        apps = block[["app_id", "engagement"]].to_numpy()
        # Iterate through the games
        for i in range(len(apps)):
            # Iterate through the games
            for j in range(i + 1, len(apps)):
                # Get the games and the engagement weights
                gi, score_i = apps[i]
                gj, score_j = apps[j]
                rows.append((int(gi), int(gj), float(min(score_i, score_j)), 1))

    if not rows:
        empty_pairs = pd.DataFrame(columns=["i", "j", "score", "co"])
        return empty_pairs, node_strength

    # Create a DataFrame of the pairs and the engagement weights
    pairs = pd.DataFrame(rows, columns=["i", "j", "score", "co"])
    # Group by the pairs and sum the engagement weights
    pairs = pairs.groupby(["i", "j"], as_index=False).sum()
    # Return the pairs and the node strengths
    return pairs, node_strength


def compute_affinity(
    df: pd.DataFrame,
    scheme: WeightScheme = "wjaccard",
    engagement_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return edge list with weights w_ij according to desired scheme.

    The optional ``engagement_weights`` argument is only used when
    ``scheme='engagement'``. It controls how review metadata (recommendation,
    votes, playtime) is folded into asymmetric overlap scores.
    """
    if scheme == "engagement":
        weights = dict(DEFAULT_ENGAGEMENT_WEIGHTS) # Default engagement weights
        if engagement_weights:
            weights.update(engagement_weights) # Update the engagement weights
        pair_scores, strengths = _engagement_tables(df, weights) # Compute the engagement tables
        if pair_scores.empty:
            return pair_scores.assign(
                w=[],
                deg_i=[],
                deg_j=[],
                strength_i=[],
                strength_j=[],
            )
        
        # Compute the node degrees
        deg = _node_degrees(df)
        # Merge the pairs and the node degrees
        merged = (
            pair_scores.merge(deg.rename(columns={"id": "i", "deg": "deg_i"}), on="i")
            .merge(deg.rename(columns={"id": "j", "deg": "deg_j"}), on="j")
            .merge(strengths.rename(columns={"id": "i", "strength": "strength_i"}), on="i")
            .merge(strengths.rename(columns={"id": "j", "strength": "strength_j"}), on="j")
        )
        # Compute the weight
        denom = np.minimum(merged["strength_i"], merged["strength_j"]) + 1e-9
        merged["w"] = merged["score"] / denom
        # Return the merged DataFrame
        columns = ["i", "j", "w", "co", "deg_i", "deg_j", "strength_i", "strength_j", "score"]
        return merged[columns]

    # Compute the co-reviews
    co = _pair_counts(df)
    if co.empty:
        return co.assign(w=[])  # empty

    # Compute the node degrees
    deg = _node_degrees(df)
    # Merge the co-reviews and the node degrees
    co = co.merge(deg.rename(columns={"id":"i","deg":"deg_i"}), on="i")
    # Merge the co-reviews and the node degrees
    co = co.merge(deg.rename(columns={"id":"j","deg":"deg_j"}), on="j")

    # Compute the weight
    if scheme == "count":
        co["w"] = co["co"].astype(float)
    elif scheme == "jaccard":
        co["w"] = co["co"] / (co["deg_i"] + co["deg_j"] - co["co"])  # classic Jaccard
    elif scheme == "wjaccard":
        # Weighted Jaccard: bias towards pairs with balanced degrees
        co["w"] = co["co"] / (np.minimum(co["deg_i"], co["deg_j"]) + 1e-9)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")

    return co[["i","j","w","co","deg_i","deg_j"]]


def to_graph(edges: pd.DataFrame, threshold: float = 0.0) -> nx.Graph:
    """Project an edge table into a NetworkX graph, preserving extra attributes."""
    if edges.empty:
        return nx.Graph()

    columns = edges.columns.tolist() # Get the columns
    G = nx.Graph()
    # Iterate through the edges
    for row in edges.itertuples(index=False):
        row_dict = {col: getattr(row, col) for col in columns} # Get the row dictionary
        weight = float(row_dict.get("w", 0.0))
        if weight < threshold: # If the weight is less than the threshold, continue
            continue
        # Create the attributes
        attrs: dict[str, Any] = {}
        for key, value in row_dict.items():
            # If the key is "i" or "j", continue
            if key in {"i", "j"}:
                continue
            # If the key is "w", set the weight
            if key == "w":
                attrs["weight"] = float(value)
            else:
                attrs[key] = value # Set the attribute
        # Add the edge
        G.add_edge(int(row_dict["i"]), int(row_dict["j"]), **attrs)
    return G # Return the graph