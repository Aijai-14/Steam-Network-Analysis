from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Dict, List, Iterable, Tuple
import networkx as nx
from ..paths import FIG
from ..config import FIG_DPI


def plot_community_network(
    edges_month: pd.DataFrame,
    communities: pd.DataFrame,
    month,
    outdir: Path = FIG,
    top_n_communities: int = 5,
    layout: str = 'spring'
):
    """
    Visualize network with nodes colored by community membership.

    Shows the largest N communities in different colors, with force-directed
    layout revealing community structure.

    Parameters
    ----------
    edges_month : pd.DataFrame
        Network edges for a specific month
    communities : pd.DataFrame
        Community assignments (columns: node, community)
    month : pd.Timestamp
        Month identifier
    outdir : Path
        Output directory for figures
    top_n_communities : int
        Number of largest communities to highlight
    layout : str
        NetworkX layout algorithm ('spring', 'kamada_kawai', 'circular')
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Build graph
    G = nx.Graph()
    for r in edges_month.itertuples():
        G.add_edge(int(r.i), int(r.j), weight=float(r.w))

    # Get top N communities by size
    comm_sizes = communities['community'].value_counts()
    top_comms = comm_sizes.head(top_n_communities).index.tolist()

    # Create color map
    node_to_comm = communities.set_index('node')['community'].to_dict()
    color_map = []
    for node in G.nodes():
        comm = node_to_comm.get(node, -1)
        if comm in top_comms:
            color_map.append(top_comms.index(comm))
        else:
            color_map.append(top_n_communities)  # "Other" category

    # Select layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=color_map,
        node_size=30,
        cmap=plt.cm.Set3,
        vmin=0,
        vmax=top_n_communities,
        ax=ax
    )

    # Create legend
    legend_elements = [
        mpatches.Patch(color=plt.cm.Set3(i / top_n_communities), label=f'Community {top_comms[i]} (n={comm_sizes[top_comms[i]]})')
        for i in range(top_n_communities)
    ]
    legend_elements.append(
        mpatches.Patch(color=plt.cm.Set3(1.0), label=f'Other (n={len(G) - sum(comm_sizes[top_comms])})')
    )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    ax.set_title(f'Community Structure — {str(month)[:7]}', fontsize=14)
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(outdir / f"community_network_{str(month)[:7]}.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_evolution_heatmap(
    stability_df: pd.DataFrame,
    outdir: Path = FIG
):
    """
    Visualize community evolution stability metrics as a heatmap over time.

    Parameters
    ----------
    stability_df : pd.DataFrame
        Stability metrics from community evolution analysis
    outdir : Path
        Output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if stability_df.empty:
        print("  Warning: No stability data to plot")
        return

    # Prepare data
    stability_df = stability_df.copy()
    stability_df['transition'] = stability_df['month_t1'].astype(str).str[:7] + ' → ' + stability_df['month_t2'].astype(str).str[:7]

    metrics = ['ari', 'avg_jaccard', 'retention_rate', 'persistent_communities']
    data_to_plot = stability_df[metrics].values

    fig, ax = plt.subplots(figsize=(10, len(stability_df) * 0.5 + 1))

    im = ax.imshow(data_to_plot, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(['ARI', 'Avg Jaccard', 'Retention', 'Persistent'], rotation=45, ha='right')
    ax.set_yticks(range(len(stability_df)))
    ax.set_yticklabels(stability_df['transition'])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Stability Score', rotation=270, labelpad=20)

    # Add values as text
    for i in range(len(stability_df)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data_to_plot[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Community Evolution Stability Over Time')
    fig.tight_layout()
    fig.savefig(outdir / "evolution_heatmap.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_evolution_events(
    events_summary: pd.DataFrame,
    outdir: Path = FIG
):
    """
    Plot community evolution events over time as a stacked bar chart.

    Parameters
    ----------
    events_summary : pd.DataFrame
        Summary of events from summarize_evolution_statistics
    outdir : Path
        Output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if events_summary.empty:
        print("  Warning: No events data to plot")
        return

    # Create transition labels
    events_summary = events_summary.copy()
    events_summary['transition'] = events_summary['month_t1'].astype(str).str[:7]

    # Get event columns (excluding month columns)
    event_cols = [c for c in events_summary.columns if c not in ['month_t1', 'month_t2', 'transition']]

    if len(event_cols) == 0:
        return

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    events_summary.set_index('transition')[event_cols].plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#3498db', '#1abc9c']
    )

    ax.set_xlabel('Time Transition')
    ax.set_ylabel('Event Count')
    ax.set_title('Community Evolution Events Over Time')
    ax.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(outdir / "evolution_events.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_diffusion_comparison(
    comparison_results: Dict,
    outdir: Path = FIG
):
    """
    Visualize statistical comparison of IC, null, and observed cascades.

    Creates a comprehensive multi-panel figure showing distributions,
    box plots, and statistical test results.

    Parameters
    ----------
    comparison_results : Dict
        Results from compare_cascade_distributions in advancedDiffusion.py
    outdir : Path
        Output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Information Cascade Model Comparison', fontsize=16)

    # Extract data
    stats_tests = comparison_results.get('statistical_tests', {})
    dist_params = comparison_results.get('distribution_params', {})
    percentiles = comparison_results.get('percentiles', {})

    # Plot 1: Distribution parameters
    ax = axes[0, 0]
    metrics = ['mean', 'median', 'std']
    models = ['ic', 'null']
    x = np.arange(len(metrics))
    width = 0.35

    ic_vals = [dist_params.get(f'ic_{m}', 0) for m in metrics]
    null_vals = [dist_params.get(f'null_{m}', 0) for m in metrics]

    ax.bar(x - width/2, ic_vals, width, label='IC Model', color='#3498db')
    ax.bar(x + width/2, null_vals, width, label='Null Model', color='#e74c3c')

    ax.set_ylabel('Value')
    ax.set_title('Distribution Parameters')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean', 'Median', 'Std Dev'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Percentiles
    ax = axes[0, 1]
    perc_values = [25, 50, 75, 90, 95]
    ic_percs = [percentiles.get(f'ic_p{p}', 0) for p in perc_values]
    null_percs = [percentiles.get(f'null_p{p}', 0) for p in perc_values]

    ax.plot(perc_values, ic_percs, marker='o', label='IC Model', color='#3498db', linewidth=2)
    ax.plot(perc_values, null_percs, marker='s', label='Null Model', color='#e74c3c', linewidth=2)

    ax.set_xlabel('Percentile')
    ax.set_ylabel('Cascade Size')
    ax.set_title('Cascade Size Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Statistical test results
    ax = axes[1, 0]
    ax.axis('off')

    # Create text summary
    text_content = "Statistical Tests\n" + "="*40 + "\n\n"
    text_content += f"Mann-Whitney U Test:\n"
    text_content += f"  Statistic: {stats_tests.get('mw_statistic', 0):.2f}\n"
    text_content += f"  P-value: {stats_tests.get('mw_pvalue', 0):.4f}\n\n"

    text_content += f"Kolmogorov-Smirnov Test:\n"
    text_content += f"  Statistic: {stats_tests.get('ks_statistic', 0):.3f}\n"
    text_content += f"  P-value: {stats_tests.get('ks_pvalue', 0):.4f}\n\n"

    text_content += f"Effect Size (Cohen's d):\n"
    text_content += f"  {stats_tests.get('cohens_d', 0):.3f}\n\n"

    text_content += f"Means:\n"
    text_content += f"  IC: {stats_tests.get('ic_mean', 0):.2f} (±{stats_tests.get('ic_std', 0):.2f})\n"
    text_content += f"  Null: {stats_tests.get('null_mean', 0):.2f} (±{stats_tests.get('null_std', 0):.2f})\n"

    # Add observed mean if it exists.
    if 'observed_mean' in stats_tests:
        text_content += f"  Observed: {stats_tests['observed_mean']:.2f}\n"

    # Add text to the plot.
    ax.text(0.1, 0.9, text_content, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Plot 4: Interpretation
    ax = axes[1, 1]
    ax.axis('off')

    # Interpret results
    interpretation = "Interpretation\n" + "="*40 + "\n\n"

    mw_p = stats_tests.get('mw_pvalue', 1.0)
    cohens_d = stats_tests.get('cohens_d', 0.0)

    if mw_p < 0.001:
        sig_level = "highly significant"
    elif mw_p < 0.01:
        sig_level = "very significant"
    elif mw_p < 0.05:
        sig_level = "significant"
    else:
        sig_level = "not significant"

    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    interpretation += f"The difference between IC and null\n"
    interpretation += f"models is {sig_level} (p={mw_p:.4f})\n"
    interpretation += f"with a {effect} effect size (d={cohens_d:.3f}).\n\n"

    # Interpret the results.
    if cohens_d > 0:
        interpretation += f"IC model produces larger cascades\n"
        interpretation += f"than the null model, suggesting that\n"
        interpretation += f"network structure matters for\n"
        interpretation += f"information diffusion.\n"
    else:
        interpretation += f"Null model produces larger cascades,\n"
        interpretation += f"suggesting random spread may be\n"
        interpretation += f"more effective.\n"

    ax.text(0.1, 0.9, interpretation, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.tight_layout()
    fig.savefig(outdir / "diffusion_statistical_comparison.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_method_comparison(
    comparison_df: pd.DataFrame,
    outdir: Path = FIG
):
    """
    Compare link prediction methods (heuristic, embedding, hybrid) over time.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Results from compare_link_prediction_methods
    outdir : Path
        Output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if comparison_df.empty:
        print("  Warning: No comparison data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: AUROC over splits
    ax = axes[0]
    for method in comparison_df['method'].unique():
        data = comparison_df[comparison_df['method'] == method]
        ax.plot(data['split_id'], data['AUROC'], marker='o', label=method.capitalize(), linewidth=2)

    ax.set_xlabel('Temporal Split')
    ax.set_ylabel('AUROC')
    ax.set_title('Link Prediction Performance (AUROC)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: AP over splits
    ax = axes[1]
    for method in comparison_df['method'].unique():
        data = comparison_df[comparison_df['method'] == method]
        ax.plot(data['split_id'], data['AP'], marker='s', label=method.capitalize(), linewidth=2)

    ax.set_xlabel('Temporal Split')
    ax.set_ylabel('Average Precision')
    ax.set_title('Link Prediction Performance (AP)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "linkpred_method_comparison.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_centrality_scatter_matrix(
    centralities: pd.DataFrame,
    month,
    outdir: Path = FIG
):
    """
    Create scatter plot matrix showing relationships between centrality measures.

    Parameters
    ----------
    centralities : pd.DataFrame
        Node centralities
    month : pd.Timestamp
        Month identifier
    outdir : Path
        Output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = ['deg', 'strength', 'betweenness', 'eigenvector']
    available_metrics = [m for m in metrics if m in centralities.columns]

    if len(available_metrics) < 2:
        return

    n = len(available_metrics)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    fig.suptitle(f'Centrality Scatter Matrix — {str(month)[:7]}', fontsize=14)

    for i, metric_i in enumerate(available_metrics):
        for j, metric_j in enumerate(available_metrics):
            ax = axes[i, j] if n > 1 else axes

            if i == j:
                # Diagonal: histogram
                ax.hist(centralities[metric_i].dropna(), bins=30, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency')
            else:
                # Off-diagonal: scatter
                valid = centralities[[metric_i, metric_j]].dropna()
                ax.scatter(valid[metric_j], valid[metric_i], alpha=0.5, s=10)

                # Compute correlation
                if len(valid) > 1:
                    corr = valid[metric_j].corr(valid[metric_i])
                    ax.text(0.05, 0.95, f'r={corr:.2f}', transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Labels
            if i == n - 1:
                ax.set_xlabel(metric_j.capitalize())
            if j == 0:
                ax.set_ylabel(metric_i.capitalize())

            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"centrality_scatter_matrix_{str(month)[:7]}.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_cascade_depth_analysis(
    depth_analysis: pd.DataFrame,
    outdir: Path = FIG
):
    """
    Visualize relationship between cascade depth and size.

    Parameters
    ----------
    depth_analysis : pd.DataFrame
        Results from analyze_cascade_depth
    outdir : Path
        Output directory
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if depth_analysis.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean size by depth
    ax = axes[0]
    ax.errorbar(depth_analysis['depth'], depth_analysis['mean_size'],
                yerr=depth_analysis['std_size'], marker='o', capsize=5,
                linewidth=2, markersize=8)
    ax.set_xlabel('Cascade Depth (steps)')
    ax.set_ylabel('Mean Cascade Size')
    ax.set_title('Cascade Size vs Depth')
    ax.grid(True, alpha=0.3)

    # Plot 2: Count by depth
    ax = axes[1]
    ax.bar(depth_analysis['depth'], depth_analysis['count'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cascade Depth (steps)')
    ax.set_ylabel('Number of Cascades')
    ax.set_title('Cascade Depth Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(outdir / "cascade_depth_analysis.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_diffusion_hypergrid(
    summary_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    outdir: Path = FIG,
    fname: str = "diffusion_hypergrid.png"
):
    """
    Render a heatmap of a diffusion metric (e.g., mean cascade size) over a 2D hyperparameter grid.

    Args:
        summary_df: Aggregated simulation statistics with parameter sweeps.
        x_param: Column name to map to the x-axis (e.g., base probability).
        y_param: Column name to map to the y-axis (e.g., alpha).
        metric: Column with the scalar metric to visualize.
        outdir: Directory where the figure is saved.
        fname: Output filename.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty or x_param not in summary_df or y_param not in summary_df:
        return

    # Pivot the summary data to create a grid of the metric across the parameter grid.
    pivot = summary_df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
    if pivot.empty:
        return

    # Create heatmap of the metric across the parameter grid.
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" if isinstance(v, (int, float)) else v for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" if isinstance(v, (int, float)) else v for v in pivot.index])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{metric} across parameter grid")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cascade_timelines(
    timeline_map: Dict[str, List[Dict[int, Iterable[int]]]],
    outdir: Path = FIG,
    max_panels: int = 4,
    fname: str = "cascade_timelines.png"
):
    """
    Plot representative cascade timelines to visualize propagation depth by strategy.

    Args:
        timeline_map: Mapping of strategy name to a list of cascades (each cascade
            is a dict of step → iterable of activated nodes).
        outdir: Directory where the figure is saved.
        max_panels: Maximum number of cascades to render to avoid clutter.
        fname: Output filename.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if not timeline_map:
        return

    samples: List[Tuple[str, Dict[int, Iterable[int]]]] = []
    strategy_list = list(timeline_map.items())
    if not strategy_list:
        return

    idx = 0
    # Cycle through cascades per strategy so each method is represented fairly.
    while len(samples) < max_panels:
        strategy, timelines = strategy_list[idx % len(strategy_list)] 
        if timelines:
            sample_idx = (len(samples) // len(strategy_list)) % len(timelines)
            samples.append((strategy, timelines[sample_idx]))
        idx += 1
        if idx > max_panels * len(strategy_list):
            break

    samples = [s for s in samples if s[1]] # filter out empty timelines
    if not samples:
        return

    # Create figure with one panel per sample.
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 3 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    # Plot each sample.
    for ax, (strategy, timeline) in zip(axes, samples):
        steps = sorted(timeline.keys())
        sizes = [len(set(timeline[step])) for step in steps]
        ax.step(steps, sizes, where="post", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("New activations")
        ax.set_title(f"Strategy {strategy}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
