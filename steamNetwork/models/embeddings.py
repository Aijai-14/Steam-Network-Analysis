from __future__ import annotations
import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec

from ..preProcess.affinityProjection import to_graph


def node2vec_embeddings(edges_month: pd.DataFrame, dimensions: int = 128, walk_length: int = 40,
                         num_walks: int = 10, p: float = 1.0, q: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Train Node2Vec embeddings on the weighted monthly network snapshot.

    Args:
        edges_month: DataFrame with at least ``i``, ``j``, ``w``, ``co`` columns.
        dimensions: Size of the embedding vectors.
        walk_length: Length of each random walk.
        num_walks: Number of walks per node.
        p: Return parameter controlling BFS bias.
        q: In-out parameter controlling DFS bias.
        seed: RNG seed for reproducibility.

    Returns:
        pd.DataFrame: Rows contain ``node`` plus ``f0..f{dimensions-1}``.
    """

    G = to_graph(edges_month[["i","j","w", "co"]], threshold=0.0) # convert to graph
    
    # Train Node2Vec model on the graph. Uses random walk sampling to generate embeddings.
    n2v = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
                   p=p, q=q, workers=1, seed=seed, quiet=True, weight_key="weight")
    model = n2v.fit(window=10, min_count=1, batch_words=4) # fit the model
    
    
    rows = [] # store the embeddings in a list.
    for n in G.nodes():
        # get the embedding for the node.
        vec = model.wv[str(n)] if str(n) in model.wv else model.wv[n]
        rows.append([n] + vec.tolist())

    # create a dataframe with the embeddings.
    cols = ["node"] + [f"f{i}" for i in range(dimensions)]
    return pd.DataFrame(rows, columns=cols) # return the dataframe.