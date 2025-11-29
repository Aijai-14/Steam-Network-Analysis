from __future__ import annotations
import pandas as pd
import networkx as nx
import ruptures as rpt
from ..preProcess.affinityProjection import to_graph


def network_series(edges_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Compute network-level statistics per month as a time series."""
    rows = []
    # for each month, compute the network-level statistics
    for month, edges in edges_snapshots.groupby("month"):
        G = to_graph(edges[["i","j","w","co"]], threshold=0.0)
        if G.number_of_nodes() == 0:
            rows.append({"month": month, "n": 0, "m": 0, "avg_w": 0, "avg_deg": 0, "cc": 0})
            continue
        n = G.number_of_nodes(); m = G.number_of_edges() # get the number of nodes and edges
        avg_w = sum(d.get("weight",1.0) for _,_,d in G.edges(data=True)) / max(m,1) # get the average weight
        avg_deg = sum(d for _, d in G.degree()) / max(n, 1) # get the average degree
        cc = nx.average_clustering(G, weight="weight") if n > 1 else 0 # get the average clustering coefficient
        rows.append({"month": month, "n": n, "m": m, "avg_w": avg_w, "avg_deg": avg_deg, "cc": cc})
    return pd.DataFrame(rows).sort_values("month") # return the network-level statistics


def detect_changepoints(ts: pd.DataFrame, column: str = "cc", pen: float = 1.0) -> pd.DataFrame:
    """Use ruptures to find change points on a selected column.
    Returns DataFrame with detected break indices and timestamps.
    """
    # reset the index
    ts = ts.reset_index(drop=True)
    if ts.empty or ts.shape[0] < 4:
        return pd.DataFrame(columns=["idx","month","column"])
    y = ts[[column]].to_numpy() # get the signal
    # Normalize the signal to improve sensitivity
    y_mean = y.mean() # get the mean of the signal
    y_std = y.std() # get the standard deviation of the signal
    if y_std > 0:
        y = (y - y_mean) / y_std
    algo = rpt.Pelt(model="rbf").fit(y)
    idxs = algo.predict(pen=pen) # predict the change points
    out = [] # initialize the output
    for idx in idxs: # for each change point
        if idx < len(ts): # if the change point is less than the length of the time series
            out.append({"idx": idx, "month": ts.loc[idx, "month"], "column": column}) # add the change point to the output
    return pd.DataFrame(out) # return the change points