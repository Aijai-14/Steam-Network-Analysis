from __future__ import annotations
import pandas as pd
import networkx as nx
from pathlib import Path
from steamNetwork.preProcess.enrichNodes import build_node_attributes

USER_PREFIX = "u_"


def build_bipartite_from_reviews(df: pd.DataFrame) -> nx.Graph:
    """Create a bipartite Users–Games graph from reviews.
    Nodes: users prefixed with 'u_' and integer app_id for games.
    Edge attrs: timestamp, recommended, votes_up, playtime_hours.
    """
    B = nx.Graph()
    for row in df.itertuples(index=False): # Iterate through the rows
        u = f"{USER_PREFIX}{row.user_id}"
        g = int(row.app_id)
        B.add_node(u, bipartite=0) # Add the user node
        B.add_node(g, bipartite=1) # Add the game node
        # Add the edge
        B.add_edge(u, g, timestamp=row.timestamp, recommended=int(row.recommended),
                   votes_up=int(row.votes_up), playtime=float(row.playtime_hours))
    return B


def write_node_attributes(games_csv, price_events_csv, out_csv):
    """Build node attributes from CSV inputs and persist them to disk."""
    # Build the node attributes
    df = build_node_attributes(games_csv, price_events_csv)
    # Create the parent directory if it doesn't exist
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # Write the node attributes to a CSV file
    df.to_csv(out_csv, index=False)
