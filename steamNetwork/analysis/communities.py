from __future__ import annotations
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
from community import community_louvain  
from ..preProcess.affinityProjection import to_graph


def _nx_to_igraph(G: nx.Graph) -> ig.Graph:
    """Convert a NetworkX graph to igraph while preserving weights."""
    # create a mapping of the nodes to the indices
    mapping = {n: i for i, n in enumerate(G.nodes())}
    # create the edges
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    # create the graph
    g = ig.Graph(n=len(mapping), edges=edges)
    # add the names to the vertices
    g.vs["name"] = list(mapping.keys())
    # add the weights to the edges
    g.es["weight"] = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    return g


def find_communities(edges_month: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    """Detect communities for one month using Leiden, Louvain, or components."""
    
    # create the graph
    G = to_graph(edges_month[["i","j","w", "co"]], threshold=0.0)
    if method == "leiden":
        print("Using Leiden community detection") 
        g = _nx_to_igraph(G)
        # find the partition
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights=g.es["weight"])
        # create the communities
        comms = {g.vs[idx]["name"]: int(cid) for idx, cid in enumerate(part.membership)}
    
    elif method == "louvain":
        print("Using Louvain community detection")
        part = community_louvain.best_partition(G, weight="weight") # find the partition
        # create the communities
        comms = {int(n): int(c) for n, c in part.items()}
    
    # return the communities
    return pd.DataFrame({"node": list(comms.keys()), "community": list(comms.values())})