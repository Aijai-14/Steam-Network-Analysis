from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from ..config import SEED

# Set random seed for reproducibility
RNG = np.random.default_rng(SEED)


def statistical_comparison(
    ic_sizes: pd.Series,
    null_sizes: pd.Series,
    observed_deltas: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Perform statistical tests comparing IC model cascades with null model and observations.

    Tests performed:
    1. Mann-Whitney U test: Non-parametric test for difference in distributions
    2. Kolmogorov-Smirnov test: Test if samples come from same distribution
    3. Effect size (Cohen's d): Standardized difference between means

    Parameters
    ----------
    ic_sizes : pd.Series
        Cascade sizes from IC model simulations
    null_sizes : pd.Series
        Cascade sizes from null model (random seeds)
    observed_deltas : pd.Series, optional
        Observed review deltas around events

    Returns
    -------
    Dict[str, float]
        Dictionary containing test statistics and p-values:
        - mw_statistic: Mann-Whitney U statistic
        - mw_pvalue: P-value for MW test
        - ks_statistic: KS test statistic
        - ks_pvalue: P-value for KS test
        - cohens_d: Effect size
        - ic_mean: Mean IC cascade size
        - null_mean: Mean null cascade size
        - observed_mean: Mean observed delta (if provided)
    """
    results = {}

    # Mann-Whitney U test (tests if distributions differ)
    mw_stat, mw_p = stats.mannwhitneyu(ic_sizes, null_sizes, alternative='two-sided')
    results['mw_statistic'] = float(mw_stat)
    results['mw_pvalue'] = float(mw_p)

    # Kolmogorov-Smirnov test (tests if samples from same distribution)
    ks_stat, ks_p = stats.ks_2samp(ic_sizes, null_sizes)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_p)

    # Cohen's d effect size
    pooled_std = np.sqrt((ic_sizes.std()**2 + null_sizes.std()**2) / 2)
    cohens_d = (ic_sizes.mean() - null_sizes.mean()) / pooled_std if pooled_std > 0 else 0
    results['cohens_d'] = float(cohens_d)

    # Means
    results['ic_mean'] = float(ic_sizes.mean())
    results['ic_std'] = float(ic_sizes.std())
    results['null_mean'] = float(null_sizes.mean())
    results['null_std'] = float(null_sizes.std())

    if observed_deltas is not None:
        results['observed_mean'] = float(observed_deltas.mean())
        results['observed_std'] = float(observed_deltas.std())

        # Compare IC with observed
        if len(observed_deltas) > 0:
            # Normalize observed to similar scale
            obs_positive = observed_deltas[observed_deltas > 0]
            if len(obs_positive) > 0:
                results['observed_positive_mean'] = float(obs_positive.mean())

    return results


def analyze_cascade_depth(cascade_results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the relationship between cascade depth (steps) and final size.

    Parameters
    ----------
    cascade_results : pd.DataFrame
        DataFrame with columns: sim_id, size, depth

    Returns
    -------
    pd.DataFrame
        Summary statistics by depth level:
        - depth: Number of propagation steps
        - count: Number of cascades reaching this depth
        - mean_size: Average cascade size at this depth
        - std_size: Standard deviation of cascade size
        - max_size: Maximum cascade size at this depth
    """
    if 'depth' not in cascade_results.columns:
        return pd.DataFrame(columns=['depth', 'count', 'mean_size', 'std_size', 'max_size'])

    # group the cascade results by depth and compute the count, mean size, standard deviation, and maximum size
    depth_analysis = cascade_results.groupby('depth').agg(
        count=('size', 'count'),
        mean_size=('size', 'mean'),
        std_size=('size', 'std'),
        max_size=('size', 'max')
    ).reset_index()

    return depth_analysis


def centrality_cascade_correlation(
    seeds: list[int],
    centrality_df: pd.DataFrame,
    cascade_results: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Analyze correlation between seed node centrality and cascade performance.

    Parameters
    ----------
    seeds : list[int]
        List of seed node IDs used in cascades
    centrality_df : pd.DataFrame
        Centrality measures for nodes (columns: node, deg, strength, betweenness, eigenvector)
    cascade_results : pd.DataFrame
        Cascade simulation results (columns: sim_id, size, depth)

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        - DataFrame: Aggregated cascade performance per seed with centrality measures
        - Dict: Pearson correlation coefficients between centrality and cascade size
    """
    
    # Aggregate cascade sizes per seed 
    if len(seeds) != len(cascade_results):
        # If multiple sims per seed, aggregate
        n_sims_per_seed = len(cascade_results) // len(seeds)
        if n_sims_per_seed > 0:
            # Truncate to ensure even division
            n_to_use = n_sims_per_seed * len(seeds)
            
            # subset the cascade results to the number of simulations per seed
            cascade_subset = cascade_results.iloc[:n_to_use]
            cascade_agg = cascade_subset.groupby(cascade_subset.index // n_sims_per_seed).agg({
                'size': ['mean', 'std', 'max'],
                'depth': 'mean'
            }).reset_index(drop=True)
            cascade_agg.columns = ['_'.join(col).strip('_') for col in cascade_agg.columns]
        else:
            # More seeds than results, just use what we have
            cascade_agg = cascade_results.copy()
            cascade_agg.columns = ['size_mean', 'depth_mean']
    else:
        cascade_agg = cascade_results.copy()
        cascade_agg.columns = ['size_mean', 'depth_mean']

    # Add seed node IDs with matching length
    cascade_agg['node'] = seeds[:len(cascade_agg)]

    # Merge with centrality data
    merged = cascade_agg.merge(centrality_df, on='node', how='left') # merge the cascade results with the centrality data

    # calculate the correlations between the centrality and cascade size
    correlations = {}
    for cent_col in ['deg', 'strength', 'betweenness', 'eigenvector']:
        if cent_col in merged.columns:
            size_col = 'size_mean' if 'size_mean' in merged.columns else 'size'
            valid = merged[[cent_col, size_col]].dropna() # drop the rows with missing values
            if len(valid) > 1:
                # check if the centrality and cascade size are not constant
                if valid[cent_col].nunique() > 1 and valid[size_col].nunique() > 1:
                    corr, p_value = stats.pearsonr(valid[cent_col], valid[size_col]) # calculate the Pearson correlation coefficient and the p-value
                    correlations[f'{cent_col}_corr'] = float(corr)
                    correlations[f'{cent_col}_pvalue'] = float(p_value)
                else:
                    # One or both arrays are constant
                    correlations[f'{cent_col}_corr'] = np.nan
                    correlations[f'{cent_col}_pvalue'] = np.nan

    return merged, correlations


def compare_cascade_distributions(
    ic_results: pd.DataFrame,
    null_results: pd.DataFrame,
    observed_deltas: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """
    Comprehensive comparison of cascade distributions across IC, null, and observed.

    Provides percentiles, quantiles, and distribution characteristics.

    Parameters
    ----------
    ic_results : pd.DataFrame
        IC model cascade results
    null_results : pd.DataFrame
        Null model cascade results
    observed_deltas : pd.DataFrame, optional
        Observed adoption deltas

    Returns
    -------
    Dict[str, any]
        Comprehensive distribution statistics:
        - percentiles: 25th, 50th, 75th, 90th, 95th percentiles for each model
        - statistical_tests: Results from statistical comparisons
        - distribution_params: Mean, median, std, skewness, kurtosis
    """
    comparison = {
        'percentiles': {},
        'distribution_params': {},
        'statistical_tests': {}
    }

    # Percentiles
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:

        # compute the percentiles for the IC and null results
        comparison['percentiles'][f'ic_p{p}'] = float(np.percentile(ic_results['size'], p))
        comparison['percentiles'][f'null_p{p}'] = float(np.percentile(null_results['size'], p))
        if observed_deltas is not None and 'delta_reviews' in observed_deltas.columns:
            obs_valid = observed_deltas['delta_reviews'].dropna() # drop the rows with missing values
            if len(obs_valid) > 0:
                comparison['percentiles'][f'observed_p{p}'] = float(np.percentile(obs_valid, p)) # compute the percentiles for the observed results

    # compute the distribution parameters for the IC and null results
    for name, data in [('ic', ic_results['size']), ('null', null_results['size'])]:
        comparison['distribution_params'][f'{name}_mean'] = float(data.mean())
        comparison['distribution_params'][f'{name}_median'] = float(data.median())
        comparison['distribution_params'][f'{name}_std'] = float(data.std())
        comparison['distribution_params'][f'{name}_skewness'] = float(stats.skew(data))
        comparison['distribution_params'][f'{name}_kurtosis'] = float(stats.kurtosis(data))

    if observed_deltas is not None and 'delta_reviews' in observed_deltas.columns:
        obs_valid = observed_deltas['delta_reviews'].dropna() # drop the rows with missing values
        if len(obs_valid) > 0:
            comparison['distribution_params']['observed_mean'] = float(obs_valid.mean()) # compute the mean for the observed results
            comparison['distribution_params']['observed_median'] = float(obs_valid.median()) # compute the median for the observed results
            comparison['distribution_params']['observed_std'] = float(obs_valid.std()) # compute the standard deviation for the observed results

    # compute the statistical tests for the IC and null results
    comparison['statistical_tests'] = statistical_comparison(
        ic_results['size'],
        null_results['size'],
        observed_deltas['delta_reviews'] if observed_deltas is not None else None
    )

    return comparison


def seed_selection_analysis(
    edges_month: pd.DataFrame,
    price_events: pd.DataFrame,
    centrality_df: pd.DataFrame,
    month: pd.Timestamp
) -> pd.DataFrame:
    """
    Analyze characteristics of games selected as seeds (discounted games).

    Compares centrality measures of discounted games vs non-discounted games.

    Parameters
    ----------
    edges_month : pd.DataFrame
        Network edges for the target month
    price_events : pd.DataFrame
        Price events with columns: timestamp, app_id, discount_pct
    centrality_df : pd.DataFrame
        Node centrality measures
    month : pd.Timestamp
        Target month for analysis

    Returns
    -------
    pd.DataFrame
        Comparison of centrality measures between seeds and non-seeds:
        - group: 'seed' or 'non_seed'
        - count: Number of nodes
        - deg_mean, strength_mean, betweenness_mean, eigenvector_mean
    """
    # Get nodes in the network
    nodes_in_network = pd.concat([edges_month['i'], edges_month['j']]).unique()

    # Get seeds (discounted games in this month)
    month_start = pd.Timestamp(month.year, month.month, 1)
    month_end = month_start + pd.DateOffset(months=1)
    seeds = price_events[
        (price_events['timestamp'] >= month_start) &
        (price_events['timestamp'] < month_end)
    ]['app_id'].unique()

    # Filter to nodes in network
    seeds_in_network = [s for s in seeds if s in nodes_in_network]

    # Label nodes as seeds or non-seeds
    cent_copy = centrality_df[centrality_df['node'].isin(nodes_in_network)].copy()
    cent_copy['group'] = cent_copy['node'].apply(
        lambda x: 'seed' if x in seeds_in_network else 'non_seed'
    )

    # Aggregate by group
    comparison = cent_copy.groupby('group').agg({
        'node': 'count',
        'deg': 'mean',
        'strength': 'mean',
        'betweenness': 'mean',
        'eigenvector': 'mean'
    }).reset_index()

    comparison.rename(columns={'node': 'count'}, inplace=True)

    return comparison
