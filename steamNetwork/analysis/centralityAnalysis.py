from __future__ import annotations
import pandas as pd
import networkx as nx
from steamNetwork.preProcess.affinityProjection import to_graph


def compute_centralities(edges_month: pd.DataFrame) -> pd.DataFrame:
    """Return centrality table for one month: degree, strength, betweenness (approx), eigenvector."""
    G = to_graph(edges_month[["i","j","w", "co"]], threshold=0.0)
    deg = dict(G.degree()) # get the degree of each node
    strength = {n: sum(G[n][nbr].get("weight", 1.0) for nbr in G.neighbors(n)) for n in G.nodes()} # get the strength of each node
    # Approx betweenness for speed
    try:
        btw = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()), weight="weight", seed=42) # get the betweenness centrality of each node
    except Exception:
        btw = {n: 0.0 for n in G.nodes()} # if the betweenness centrality is not available, set it to 0
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight") # get the eigenvector centrality of each node
    except Exception:
        eig = {n: 0.0 for n in G.nodes()} # if the eigenvector centrality is not available, set it to 0
    
    # create the dataframe
    df = pd.DataFrame({
        "node": list(G.nodes()), # get the nodes
        "deg": [deg[n] for n in G.nodes()], # get the degree of each node
        "strength": [strength[n] for n in G.nodes()], # get the strength of each node
        "betweenness": [btw[n] for n in G.nodes()], # get the betweenness centrality of each node
        "eigenvector": [eig[n] for n in G.nodes()], # get the eigenvector centrality of each node
    })
    return df # return the dataframe

