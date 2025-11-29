"""Embedding-Based Link Prediction"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from ..config import SEED, LinkPredParams
from ..models.embeddings import node2vec_embeddings
from .linkPrediction import LinkPredDataset
from .linkPrediction import build_train_test, Split

# Set random seed
RNG = np.random.default_rng(SEED)


def edge_features_from_embeddings(
    pairs: list[tuple[int, int]],
    embeddings: pd.DataFrame,
    operators: list[str] = ['hadamard', 'average', 'l1', 'l2']
) -> pd.DataFrame:
    """
    Generate edge features from node embeddings using various operators.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        List of node pairs to compute features for
    embeddings : pd.DataFrame
        Node embeddings with columns: node, f0, f1, ..., f{d-1}
    operators : list[str]
        List of operators to apply

    Returns
    -------
    pd.DataFrame
        Edge features with one row per pair, columns for each dimension and operator
    """
    # Create embedding lookup dictionary
    emb_dict = {}
    # get the feature columns
    feature_cols = [c for c in embeddings.columns if c.startswith('f')]
    n_dims = len(feature_cols)
    # for each row in the embeddings, add the embedding to the embedding dictionary
    for _, row in embeddings.iterrows():
        node_id = int(row['node'])
        emb_dict[node_id] = row[feature_cols].values

    # Compute features for each pair
    features_list = []
    valid_pairs = []

    for i, j in pairs:
        if i not in emb_dict or j not in emb_dict:
            # Skip pairs where embeddings are missing
            continue

        emb_i = emb_dict[i] # get the embedding for the first node
        emb_j = emb_dict[j] # get the embedding for the second node

        feat = {}

        # hadamard is the element-wise product of the two embeddings
        if 'hadamard' in operators:
            hadamard = emb_i * emb_j
            for d in range(n_dims):
                feat[f'had_{d}'] = hadamard[d]

        # average is the element-wise average of the two embeddings
        if 'average' in operators:
            avg = (emb_i + emb_j) / 2
            for d in range(n_dims):
                feat[f'avg_{d}'] = avg[d]

        # l1 is the element-wise absolute difference of the two embeddings
        if 'l1' in operators:
            l1 = np.abs(emb_i - emb_j)
            for d in range(n_dims):
                feat[f'l1_{d}'] = l1[d]

        # l2 is the element-wise squared difference of the two embeddings
        if 'l2' in operators:
            l2 = (emb_i - emb_j) ** 2
            for d in range(n_dims):
                feat[f'l2_{d}'] = l2[d]

        features_list.append(feat) 
        valid_pairs.append((i, j)) # add the valid pair to the valid pairs list

    df = pd.DataFrame(features_list)
    df['i'] = [p[0] for p in valid_pairs] # add the first node to the dataframe
    df['j'] = [p[1] for p in valid_pairs] # add the second node to the dataframe

    return df


def link_prediction_with_embeddings(
    train_edges: pd.DataFrame,
    test_positive: list[tuple[int, int]],
    test_negative: list[tuple[int, int]],
    embedding_params: Optional[LinkPredParams] = None,
    operators: list[str] = ['hadamard'],
    train_positive: Optional[list[tuple[int, int]]] = None,
    train_negative: Optional[list[tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Perform link prediction using only node embeddings.

    Parameters
    ----------
    train_edges : pd.DataFrame
        Training edges with columns: i, j, w
    test_positive : pd.DataFrame
        Positive test pairs (new edges)
    test_negative : pd.DataFrame
        Negative test pairs (non-edges)
    embedding_params : LinkPredParams, optional
        Parameters for Node2Vec
    operators : list[str]
        Embedding operators to use

    Returns
    -------
    Dict[str, float]
        Performance metrics: AUROC, AP
    """
    if embedding_params is None:
        embedding_params = LinkPredParams()

    # Generate embeddings on training graph
    embeddings = node2vec_embeddings(
        train_edges,
        dimensions=embedding_params.embed_dim,
        walk_length=embedding_params.walk_length,
        num_walks=embedding_params.num_walks,
        p=embedding_params.p,
        q=embedding_params.q,
        seed=SEED
    )

    # Generate features for test pairs
    X_pos = edge_features_from_embeddings(test_positive, embeddings, operators)
    X_neg = edge_features_from_embeddings(test_negative, embeddings, operators)

    if len(X_pos) == 0 or len(X_neg) == 0:
        return {'AUROC': np.nan, 'AP': np.nan, 'error': 'insufficient valid pairs'}

    # Combine and create labels
    X_test = pd.concat([X_pos, X_neg], ignore_index=True)
    y_test = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    # Generate training features (sample from training edges + random non-edges)
    if train_positive is None:
        sample = train_edges.head(min(1000, len(train_edges))) # get a sample of the training edges
        train_positive = [(int(r.i), int(r.j)) for r in sample.itertuples()]
    if train_negative is None:
        nodes = pd.concat([train_edges['i'], train_edges['j']]).unique() # get the unique nodes
        train_negative = []

        # while the number of training negative pairs is less than the number of training positive pairs, sample a random node
        while len(train_negative) < len(train_positive):
            i, j = RNG.choice(nodes, size=2, replace=False) # sample a random node
            train_negative.append((int(i), int(j)))

    # generate the training features for the positive and negative pairs
    X_train_pos = edge_features_from_embeddings(train_positive, embeddings, operators)
    X_train_neg = edge_features_from_embeddings(train_negative, embeddings, operators)

    if len(X_train_pos) == 0 or len(X_train_neg) == 0:
        return {'AUROC': np.nan, 'AP': np.nan, 'error': 'insufficient training pairs'}

    # concatenate the training features for the positive and negative pairs
    X_train = pd.concat([X_train_pos, X_train_neg], ignore_index=True)
    y_train = np.concatenate([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])

    # train the classifier
    feature_cols = [c for c in X_train.columns if c not in ['i', 'j']]
    clf = LogisticRegression(max_iter=1000, random_state=SEED) # create the logistic regression model
    clf.fit(X_train[feature_cols], y_train) # train the logistic regression model

    # predict the probabilities
    y_pred = clf.predict_proba(X_test[feature_cols])[:, 1]

    # compute the area under the ROC curve and the average precision
    auroc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    
    return {'AUROC': float(auroc), 'AP': float(ap)}


def hybrid_link_prediction(
    train_edges: pd.DataFrame,
    dataset: LinkPredDataset,
    embedding_params: Optional[LinkPredParams] = None,
    operators: list[str] = ['hadamard']
) -> Dict[str, float]:
    """
    Hybrid link prediction combining embeddings and heuristic features.

    Combines structural local features (Jaccard, Adamic-Adar, etc.) with learned global
    embeddings for potentially better performance.

    Parameters
    ----------
    train_edges : pd.DataFrame
        Training edges
    test_positive : pd.DataFrame
        Positive test pairs
    test_negative : pd.DataFrame
        Negative test pairs
    heuristic_features : pd.DataFrame
        Pre-computed heuristic features (from linkPrediction.py)
    embedding_params : LinkPredParams, optional
        Node2Vec parameters
    operators : list[str]
        Embedding operators

    Returns
    -------
    Dict[str, float]
        Performance metrics: AUROC, AP
    """
    if embedding_params is None:
        embedding_params = LinkPredParams()

    # Generate embeddings
    embeddings = node2vec_embeddings(
        train_edges,
        dimensions=embedding_params.embed_dim,
        walk_length=embedding_params.walk_length,
        num_walks=embedding_params.num_walks,
        p=embedding_params.p,
        q=embedding_params.q,
        seed=SEED
    )

    def _merge_embeddings(df: pd.DataFrame) -> pd.DataFrame:
        """Merge the embeddings with the training and test data"""
        pairs = list(df[["i", "j"]].itertuples(index=False, name=None)) # get the pairs
        emb = edge_features_from_embeddings(pairs, embeddings, operators) # generate the embeddings for the pairs
        if emb.empty:
            return pd.DataFrame() # return an empty dataframe if the embeddings are empty
        return df.merge(emb, on=['i', 'j'], how='inner') # merge the embeddings with the training and test data

    train_df = _merge_embeddings(dataset.train) # merge the embeddings with the training data
    test_df = _merge_embeddings(dataset.test) # merge the embeddings with the test data

    if train_df.empty or test_df.empty:
        return {'AUROC': np.nan, 'AP': np.nan, 'error': 'no embedding coverage'}

    # get the feature columns
    feature_cols = [c for c in train_df.columns if c not in ['i', 'j', 'label']]
    clf = LogisticRegression(max_iter=1000, random_state=SEED) # create the logistic regression model
    clf.fit(train_df[feature_cols].fillna(0), train_df['label']) # train the logistic regression model

    y_pred = clf.predict_proba(test_df[feature_cols].fillna(0))[:, 1] # predict the probabilities
    auroc = roc_auc_score(test_df['label'], y_pred) # compute the area under the ROC curve
    ap = average_precision_score(test_df['label'], y_pred) # compute the average precision

    return {'AUROC': float(auroc), 'AP': float(ap)}


def compare_link_prediction_methods(
    snapshots: pd.DataFrame,
    embedding_params: Optional[LinkPredParams] = None
) -> pd.DataFrame:
    """
    Compare link prediction performance across methods:
    1. Heuristics only 
    2. Embeddings only
    3. Hybrid (heuristics + embeddings)

    Parameters
    ----------
    snapshots : pd.DataFrame
        Monthly network snapshots
    embedding_params : LinkPredParams, optional
        Node2Vec parameters

    Returns
    -------
    pd.DataFrame
        Comparison results with columns:
        - split_id: Temporal split identifier
        - method: 'heuristic', 'embedding', or 'hybrid'
        - AUROC: Area under ROC curve
        - AP: Average precision
    """

    months = sorted(snapshots['month'].unique()) # get the unique months
    results = []

    # for each month, compare the link prediction performance
    for idx in range(len(months) - 1):
        train_month = months[idx] # get the training month
        test_month = months[idx + 1] # get the test month

        split = Split(train_month=train_month, test_month=test_month) # create the split

        try:
            # Get heuristic-based features and labels
            dataset = build_train_test(snapshots, split)
            if dataset is None or dataset.train.empty or dataset.test.empty:
                continue

            # Get train edges (select only columns needed for embeddings)
            train_edges = snapshots[snapshots['month'] == train_month][['i', 'j', 'w', "co"]].copy()

            # Method 1: Heuristics only (baseline)
            heur_features = [c for c in dataset.train.columns if c not in ['i', 'j', 'label']]
            if len(heur_features) > 0:
                # train the logistic regression model
                clf_h = LogisticRegression(max_iter=1000, random_state=SEED) # create the logistic regression model
                clf_h.fit(dataset.train[heur_features].fillna(0), dataset.train['label']) # train the logistic regression model
                y_pred_h = clf_h.predict_proba(dataset.test[heur_features].fillna(0))[:, 1] # predict the probabilities

                results.append({
                    'split_id': idx,
                    'train_month': str(train_month)[:7],
                    'test_month': str(test_month)[:7],
                    'method': 'heuristic',
                    'AUROC': roc_auc_score(dataset.test['label'], y_pred_h),
                    'AP': average_precision_score(dataset.test['label'], y_pred_h)
                })

            # Method 2: Embeddings only
            emb_result = link_prediction_with_embeddings(
                train_edges,
                dataset.test_pos_pairs,
                dataset.test_neg_pairs,
                embedding_params,
                train_positive=dataset.train_pos_pairs,
                train_negative=dataset.train_neg_pairs,
            )
            if 'error' not in emb_result:
                results.append({
                    'split_id': idx,
                    'train_month': str(train_month)[:7],
                    'test_month': str(test_month)[:7],
                    'method': 'embedding',
                    'AUROC': emb_result['AUROC'],
                    'AP': emb_result['AP']
                })

            # Method 3: Hybrid (heuristics + embeddings)
            hybrid_result = hybrid_link_prediction(
                train_edges, dataset, embedding_params
            )
            if 'error' not in hybrid_result:
                results.append({
                    'split_id': idx,
                    'train_month': str(train_month)[:7],
                    'test_month': str(test_month)[:7],
                    'method': 'hybrid',
                    'AUROC': hybrid_result['AUROC'],
                    'AP': hybrid_result['AP']
                })

        except Exception as e:
            print(f"  Warning: Error in split {idx}: {e}")
            continue

    return pd.DataFrame(results)
