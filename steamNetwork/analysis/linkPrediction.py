"""Heuristic-based link prediction datasets and baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ..preProcess.affinityProjection import to_graph

# Using NetworkX heuristic utilities
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    adamic_adar_index,
    resource_allocation_index,
    preferential_attachment,
)


@dataclass
class Split:
    train_month: object
    test_month: object


@dataclass
class LinkPredDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    train_pos_pairs: list[tuple[int, int]] # positive training pairs
    train_neg_pairs: list[tuple[int, int]] # negative training pairs
    test_pos_pairs: list[tuple[int, int]] # positive test pairs
    test_neg_pairs: list[tuple[int, int]] # negative test pairs


def _pairs_from_graph(G: nx.Graph) -> set[tuple[int,int]]:
    """Return the set of undirected edges as sorted integer tuples."""
    return set((int(u), int(v)) for u, v in G.edges()) # return the set of undirected edges as sorted integer tuples


def _candidate_non_edges(G: nx.Graph, k: int) -> list[tuple[int,int]]:
    """Uniformly sample k non-edges without replacement."""
    nodes = list(G.nodes()) # get the nodes
    non = set()
    attempts = 0
    # while the number of non-edges is less than k and the attempts are less than 25 * k
    while len(non) < k and attempts < 25 * k: 
        u = random.choice(nodes); v = random.choice(nodes) # sample a random node
        if u == v: # if the nodes are the same, increment the attempts
            attempts += 1
            continue
        a, b = (u, v) if u < v else (v, u) # sort the nodes
        if not G.has_edge(a, b): # if the edge does not exist, add the edge to the non-edges
            non.add((a, b))
        attempts += 1
    return list(non)


def _heuristics(G: nx.Graph, pairs: list[tuple[int,int]]) -> pd.DataFrame:
    """Compute structural heuristics for candidate node pairs."""
    if not pairs:
        return pd.DataFrame(columns=["i", "j"])

    def to_df(gen, name):
        return pd.DataFrame([(int(u), int(v), float(p)) for u, v, p in gen], columns=["i","j",name])

    ja = to_df(jaccard_coefficient(G, pairs), "jaccard") # compute the jaccard coefficient
    aa = to_df(adamic_adar_index(G, pairs), "adamic_adar") # compute the adamic adar index
    ra = to_df(resource_allocation_index(G, pairs), "res_alloc") # compute the resource allocation index
    pa = to_df(preferential_attachment(G, pairs), "pref_attach") # compute the preferential attachment

    # Common neighbors count for the pairs
    cn_vals = []
    for u, v in pairs:
        cn = len(list(nx.common_neighbors(G, u, v))) # compute the common neighbors
        cn_vals.append((int(u), int(v), float(cn))) # add the common neighbors to the values
    cn = pd.DataFrame(cn_vals, columns=["i","j","common_neighbors"]) # create the common neighbors dataframe

    out = ja.merge(aa, on=["i","j"]).merge(ra, on=["i","j"]).merge(pa, on=["i","j"]).merge(cn, on=["i","j"])
    return out


def build_train_test(
    edges_snapshots: pd.DataFrame,
    split: Split,
    train_frac: float = 0.7,
    random_state: int = 42,
) -> Optional[LinkPredDataset]:
    """Construct train/test heuristic feature tables for a temporal split."""
    E_train = edges_snapshots.query("month == @split.train_month") # get the training edges
    E_test = edges_snapshots.query("month == @split.test_month") # get the test edges

    Gtr = to_graph(E_train[["i","j","w", "co"]], threshold=0.0) # create the training graph
    Gte = to_graph(E_test[["i","j","w", "co"]], threshold=0.0) # create the test graph

    # New edges only, but filter to edges where both nodes exist in the training graph
    train_nodes = set(Gtr.nodes())
    # get the positive pairs (edges that exist in the test graph but not in the training graph)
    pos_pairs = [
        (u, v) for u, v in (_pairs_from_graph(Gte) - _pairs_from_graph(Gtr))
        if u in train_nodes and v in train_nodes
    ]
    if not pos_pairs:
        return None
    
    # get the negative pairs (non-edges that exist in the training graph)
    neg_pairs = _candidate_non_edges(Gtr, k=max(len(pos_pairs), 1))
    if not neg_pairs:
        return None

    # shuffle the positive and negative pairs
    rng = random.Random(random_state)
    rng.shuffle(pos_pairs)
    rng.shuffle(neg_pairs)

    pos_split = max(1, int(train_frac * len(pos_pairs))) # get the split for the positive pairs
    neg_split = max(1, int(train_frac * len(neg_pairs))) # get the split for the negative pairs

    train_pos = pos_pairs[:pos_split] # get the training positive pairs
    test_pos = pos_pairs[pos_split:] or train_pos[-1:] # get the test positive pairs
    train_pos = train_pos if len(train_pos) > 1 else pos_pairs # get the training positive pairs if the length is greater than 1

    train_neg = neg_pairs[:neg_split] # get the training negative pairs
    test_neg = neg_pairs[neg_split:] or train_neg[-1:] # get the test negative pairs
    train_neg = train_neg if len(train_neg) > 1 else neg_pairs # get the training negative pairs if the length is greater than 1

    # concatenate the training and test positive and negative pairs with the heuristics
    train_df = pd.concat(
        [
            _heuristics(Gtr, train_pos).assign(label=1),
            _heuristics(Gtr, train_neg).assign(label=0),
        ],
        ignore_index=True,
    )

    # concatenate the training and test positive and negative pairs with the heuristics
    test_df = pd.concat(
        [
            _heuristics(Gtr, test_pos).assign(label=1),
            _heuristics(Gtr, test_neg).assign(label=0),
        ],
        ignore_index=True,
    )

    return LinkPredDataset(
        train=train_df,
        test=test_df,
        train_pos_pairs=train_pos,
        train_neg_pairs=train_neg,
        test_pos_pairs=test_pos,
        test_neg_pairs=test_neg,
    )


def run_link_prediction(edges_snapshots: pd.DataFrame) -> dict:
    """Train heuristic link prediction models across successive month splits."""
    months = sorted(edges_snapshots["month"].unique())
    if len(months) < 2:
        return {"splits": [], "results": []}
    results = []
    splits = []

    # for each month, train the heuristic link prediction model
    for i in range(len(months)-1):
        split = Split(train_month=months[i], test_month=months[i+1]) # create the split
        dataset = build_train_test(edges_snapshots, split) # build the train/test dataset
        if dataset is None or dataset.train.empty or dataset.test.empty:
            continue

        splits.append(split) # add the split to the splits
        train = dataset.train.copy() # copy the training data
        test = dataset.test.copy() # copy the test data
        y_train = train.pop("label").values # get the training labels
        y_test = test.pop("label").values # get the test labels
        X_train = train.drop(columns=["i","j"]).fillna(0.0).values # get the training features
        X_test = test.drop(columns=["i","j"]).fillna(0.0).values # get the test features
        clf = LogisticRegression(max_iter=200) # create the logistic regression model
        clf.fit(X_train, y_train) # train the logistic regression model
        preds = clf.predict_proba(X_test)[:,1] # predict the probabilities
        auc = roc_auc_score(y_test, preds) # compute the area under the ROC curve
        ap = average_precision_score(y_test, preds) # compute the average precision
        results.append(
            {
                "train": split.train_month,
                "test": split.test_month,
                "AUROC": auc,
                "AP": ap,
            }
        )
    return {"splits": splits, "results": results}