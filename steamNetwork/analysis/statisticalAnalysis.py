from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
from ..config import SEED
from sklearn.preprocessing import LabelEncoder
from ..preProcess.affinityProjection import to_graph

RNG = np.random.default_rng(SEED)


def community_homophily_analysis(
    communities: pd.DataFrame,
    node_attributes: pd.DataFrame,
    attribute: str = 'genres'
) -> Dict[str, float]:
    """
    Analyze community homophily: Do nodes in same community share attributes?

    Measures whether games in the same community tend to share genres,
    indicating topical clustering.

    Parameters
    ----------
    communities : pd.DataFrame
        Community assignments
    node_attributes : pd.DataFrame
        Node metadata including attribute column
    attribute : str
        Attribute to analyze (default: 'genres')

    Returns
    -------
    Dict[str, float]
        - within_comm_similarity: Avg similarity within communities
        - between_comm_similarity: Avg similarity between communities
        - homophily_ratio: Ratio of within/between
        - nmi: Normalized Mutual Information
    """
    # Merge communities with attributes
    merged = communities.merge(node_attributes, left_on='node', right_on='app_id', how='inner')

    if attribute not in merged.columns:
        return {'error': f'Attribute {attribute} not found'}

    # Process genres (split by '|')
    merged['genres_set'] = merged[attribute].fillna('').apply(lambda x: set(x.split('|')) if x else set())

    # Compute pairwise Jaccard similarities within and between communities
    within_similarities = []
    between_similarities = []

    communities_list = merged['community'].unique()

    for comm in communities_list:
        comm_nodes = merged[merged['community'] == comm]
        genre_sets = comm_nodes['genres_set'].tolist()

        # Within community similarities
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                if genre_sets[i] or genre_sets[j]:
                    jaccard = len(genre_sets[i] & genre_sets[j]) / len(genre_sets[i] | genre_sets[j]) if (genre_sets[i] | genre_sets[j]) else 0
                    within_similarities.append(jaccard)

    # Between community similarities (sampling for efficiency)
    n_samples = min(1000, len(communities_list) * 10)
    for _ in range(n_samples):
        comm1, comm2 = RNG.choice(communities_list, size=2, replace=False)
        nodes1 = merged[merged['community'] == comm1]['genres_set'].tolist()
        nodes2 = merged[merged['community'] == comm2]['genres_set'].tolist()

        if nodes1 and nodes2:
            node1 = RNG.choice(nodes1)
            node2 = RNG.choice(nodes2)
            if node1 or node2:
                jaccard = len(node1 & node2) / len(node1 | node2) if (node1 | node2) else 0
                between_similarities.append(jaccard)

    # Compute averages
    within_avg = np.mean(within_similarities) if within_similarities else 0
    between_avg = np.mean(between_similarities) if between_similarities else 0
    homophily_ratio = within_avg / between_avg if between_avg > 0 else np.inf

    # Compute Normalized Mutual Information
    # Create genre-based labels (the most frequent genre)
    merged['primary_genre'] = merged['genres_set'].apply(lambda x: sorted(x)[0] if x else None)

    # Encode categorical variables (community and genre)
    le_comm = LabelEncoder()
    le_genre = LabelEncoder()

    comm_labels = le_comm.fit_transform(merged['community'])
    genre_labels = le_genre.fit_transform(merged['primary_genre'])

    nmi = normalized_mutual_info_score(comm_labels, genre_labels) # compute the normalized mutual information

    return {
        'within_comm_similarity': float(within_avg),
        'between_comm_similarity': float(between_avg),
        'homophily_ratio': float(homophily_ratio),
        'nmi': float(nmi),
        'interpretation': f"Homophily ratio of {homophily_ratio:.2f} indicates {'strong' if homophily_ratio > 1.5 else 'moderate' if homophily_ratio > 1.0 else 'weak'} topical clustering."
    }


def temporal_stability_analysis(
    net_stats: pd.DataFrame
) -> Dict[str, any]:
    """
    Analyze temporal stability of network properties.

    Computes autocorrelation, trend analysis, and variability metrics.

    Parameters
    ----------
    net_stats : pd.DataFrame
        Network time series statistics

    Returns
    -------
    Dict[str, any]
        - autocorrelations: Lag-1 autocorrelation for each metric
        - trends: Linear trend slopes
        - variability: Coefficient of variation for each metric
    """
    metrics = ['n', 'm', 'avg_w', 'avg_deg', 'cc']
    available_metrics = [m for m in metrics if m in net_stats.columns]

    results = {
        'autocorrelations': {},
        'trends': {},
        'variability': {},
        'stationarity': {}
    }

    # for each metric, compute the autocorrelation, trend, variability, and stationarity
    for metric in available_metrics:
        series = net_stats[metric].dropna() # get the series

        if len(series) < 3:
            continue

        # Lag-1 autocorrelation
        if len(series) > 1:
            autocorr = series.autocorr(lag=1)
            results['autocorrelations'][metric] = float(autocorr) if not np.isnan(autocorr) else 0.0

        # Linear trend
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        results['trends'][metric] = {
            'slope': float(slope),
            'r_squared': float(r_value**2),
            'p_value': float(p_value)
        }

        # Coefficient of variation
        cv = series.std() / series.mean() if series.mean() != 0 else 0
        results['variability'][metric] = float(cv)

        # Augmented Dickey-Fuller test for stationarity 
        is_stationary = abs(autocorr) < 0.7 if not np.isnan(autocorr) else True
        results['stationarity'][metric] = bool(is_stationary)

    return results


def degree_distribution_analysis(
    edges_month: pd.DataFrame
) -> Dict[str, any]:
    """
    Analyze degree distribution and test for power-law fit.

    Tests whether the network follows a scale-free (power-law) distribution.

    Parameters
    ----------
    edges_month : pd.DataFrame
        Network edges

    Returns
    -------
    Dict[str, any]
        - power_law_alpha: Estimated power-law exponent
        - ks_statistic: KS test statistic
        - is_power_law: Boolean flag (heuristic)
        - mean_degree: Mean degree
        - median_degree: Median degree
        - max_degree: Maximum degree
        - degree_variance: Variance of degree distribution
    """

    G = to_graph(edges_month[['i', 'j', 'w', "co"]], threshold=0.0)
    degrees = [d for _, d in G.degree()]

    # Basic statistics
    mean_deg = np.mean(degrees)
    median_deg = np.median(degrees)
    max_deg = np.max(degrees)
    var_deg = np.var(degrees)

    # Power-law fit (simplified - use log-log linear regression)
    # Filter out zeros
    degrees_nonzero = [d for d in degrees if d > 0]

    if len(degrees_nonzero) < 10:
        return {
            'mean_degree': float(mean_deg),
            'median_degree': float(median_deg),
            'max_degree': float(max_deg),
            'degree_variance': float(var_deg),
            'error': 'Insufficient data for power-law fit'
        }

    # Degree distribution
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    x = degree_counts.index.values
    y = degree_counts.values

    # Log-log regression
    log_x = np.log10(x[x > 0])
    log_y = np.log10(y[x > 0])

    if len(log_x) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        alpha = -slope  # Power-law exponent (negative slope in log-log)

        # Heuristic: power-law if alpha in reasonable range and good fit
        is_power_law = (2.0 <= alpha <= 3.5) and (r_value**2 > 0.7)

        return {
            'power_law_alpha': float(alpha),
            'r_squared': float(r_value**2),
            'is_power_law': bool(is_power_law),
            'mean_degree': float(mean_deg),
            'median_degree': float(median_deg),
            'max_degree': float(max_deg),
            'degree_variance': float(var_deg),
            'interpretation': f"Alpha={alpha:.2f}, R²={r_value**2:.3f}. " +
                            ("Consistent with power-law." if is_power_law else "Does not follow power-law.")
        }
    else:
        return {
            'mean_degree': float(mean_deg),
            'median_degree': float(median_deg),
            'max_degree': float(max_deg),
            'degree_variance': float(var_deg),
            'error': 'Insufficient unique degrees for power-law fit'
        }


def cascade_seed_centrality_test(
    seeds: list[int],
    centrality_df: pd.DataFrame,
    cascade_results: pd.DataFrame,
    n_permutations: int = 1000
) -> Dict[str, any]:
    """
    Permutation test: Are observed cascades significantly related to seed centrality?

    Tests null hypothesis by randomly permuting seed-cascade assignments.

    Parameters
    ----------
    seeds : list[int]
        Seed node IDs
    centrality_df : pd.DataFrame
        Node centralities
    cascade_results : pd.DataFrame
        Cascade results
    n_permutations : int
        Number of permutations for test

    Returns
    -------
    Dict[str, any]
        - observed_correlation: Actual correlation
        - null_distribution: Array of permuted correlations
        - p_value: Permutation test p-value
        - is_significant: Boolean flag
    """
    # Get observed correlation
    seed_cents = centrality_df[centrality_df['node'].isin(seeds)].copy()

    if 'deg' not in seed_cents.columns:
        return {'error': 'Degree centrality not available'}

    # Map seeds to cascade sizes
    cascade_sizes = cascade_results.groupby(cascade_results.index // (len(cascade_results) // len(seeds)))['size'].mean().values

    if len(cascade_sizes) != len(seeds):
        cascade_sizes = cascade_results['size'].values[:len(seeds)]

    seed_cents = seed_cents.iloc[:len(cascade_sizes)] # get the seed centralities
    seed_cents['cascade_size'] = cascade_sizes # add the cascade sizes to the seed centralities

    observed_corr = seed_cents[['deg', 'cascade_size']].corr().iloc[0, 1] # compute the observed correlation

    # Permutation test
    null_corrs = []
    # for each permutation, compute the null correlation
    for _ in range(n_permutations):
        permuted_sizes = RNG.permutation(cascade_sizes) # permute the cascade sizes
        seed_cents['cascade_size_perm'] = permuted_sizes # add the permuted cascade sizes to the seed centralities
        null_corr = seed_cents[['deg', 'cascade_size_perm']].corr().iloc[0, 1] # compute the null correlation
        null_corrs.append(null_corr)

    null_corrs = np.array(null_corrs) # convert the null correlations to an array

    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))

    return {
        'observed_correlation': float(observed_corr),
        'null_mean': float(np.mean(null_corrs)),
        'null_std': float(np.std(null_corrs)),
        'p_value': float(p_value),
        'is_significant': bool(p_value < 0.05),
        'interpretation': f"Observed correlation (r={observed_corr:.3f}) is {'significant' if p_value < 0.05 else 'not significant'} (p={p_value:.4f})"
    }


def comprehensive_statistical_report(
    snapshots: pd.DataFrame,
    communities: pd.DataFrame,
    centralities: pd.DataFrame,
    node_attributes: pd.DataFrame,
    net_stats: pd.DataFrame,
    month: pd.Timestamp
) -> Dict[str, any]:
    """
    Generate comprehensive statistical report for a single time point.

    Parameters
    ----------
    snapshots : pd.DataFrame
        Network snapshots
    communities : pd.DataFrame
        Community assignments
    centralities : pd.DataFrame
        Node centralities
    node_attributes : pd.DataFrame
        Node metadata
    net_stats : pd.DataFrame
        Network time series
    month : pd.Timestamp
        Target month

    Returns
    -------
    Dict[str, any]
        Comprehensive report with all statistical analyses
    """
    edges_month = snapshots[snapshots['month'] == month]

    report = {
        'month': str(month)[:7],
        'degree_distribution': degree_distribution_analysis(edges_month),
        'community_homophily': community_homophily_analysis(communities, node_attributes),
        'temporal_stability': temporal_stability_analysis(net_stats)
    }

    return report
