from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from ..paths import FIG
from ..config import FIG_DPI
from ..preProcess.affinityProjection import to_graph


def plot_cascade_sizes(df: pd.DataFrame, month, outdir: Path = FIG):
    """Visualize the distribution of IC cascade sizes for a given month.

    Args:
        df: DataFrame with a ``size`` column.
        month: Label for the time period.
        outdir: Directory where the figure is written.
    """
    fig = plt.figure(figsize=(6,4))
    plt.title(f"IC cascade sizes (n={len(df)}) — {month}")
    plt.hist(df["size"], bins=30)
    plt.xlabel("cascade size"); plt.ylabel("count")
    fig.tight_layout();
    fig.savefig(outdir / f"ic_cascade_sizes_{str(month)[:7]}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_deg_hist(edges_month: pd.DataFrame, month, outdir: Path = FIG):
    """Alternate degree histogram that rebuilds a NetworkX graph manually.

    Args:
        edges_month: Edge list with ``i``, ``j``, ``w`` columns.
        month: Label for the time period (used in title/filename).
        outdir: Directory where the figure is written.
    """
    from networkx import Graph
    G = Graph()
    for r in edges_month[["i","j","w"]].itertuples(index=False):
        G.add_edge(int(r.i), int(r.j), weight=float(r.w))
    degs = [d for _, d in G.degree()]
    fig = plt.figure(figsize=(6,4))
    plt.title(f"Degree distribution — {str(month)[:7]}")
    plt.hist(degs, bins=30)
    plt.xlabel("degree"); plt.ylabel("count")
    fig.tight_layout()
    fig.savefig(outdir / f"deg_hist_{str(month)[:7]}.png", dpi=140)
    plt.close(fig)


def plot_lp_scores(scores: pd.DataFrame, outdir: Path = FIG, tag: str = "heuristics"):
    """Plot AUROC and AP curves for heuristic link prediction scores.

    Args:
        scores: DataFrame sorted by split with ``AUROC`` and ``AP`` columns.
        outdir: Directory where the figure is written.
        tag: Suffix for the output filename to distinguish methods.
    """
    if scores.empty:
        return
    xs = list(range(len(scores)))
    fig = plt.figure(figsize=(6,4))
    plt.title("Link prediction — AUROC/AP per split")
    plt.plot(xs, scores["AUROC"], marker="o", label="AUROC")
    plt.plot(xs, scores["AP"], marker="s", label="AP")
    plt.xlabel("split (train_t → test_t)"); plt.ylabel("score")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"linkpred_scores_{tag}.png", dpi=140)
    plt.close(fig)


def plot_network_timeseries(net_stats: pd.DataFrame, outdir: Path = FIG, changepoints=None):
    """Plot the evolution of key network statistics (n, m, weight, cc).

    Args:
        net_stats: DataFrame containing per-month metrics (n, m, avg_w, avg_deg, cc).
        outdir: Directory where the figure is written.
        changepoints: Optional DataFrame with ``month`` column marking structural shifts.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Network Evolution Over Time", fontsize=14)

    # Convert month to datetime for plotting
    months = pd.to_datetime(net_stats['month'])

    # Plot 1: Number of nodes and edges
    ax = axes[0, 0]
    ax.plot(months, net_stats['n'], marker='o', label='Nodes', color='blue')
    ax.set_ylabel('Nodes', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2 = ax.twinx()
    ax2.plot(months, net_stats['m'], marker='s', label='Edges', color='red')
    ax2.set_ylabel('Edges', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Network Size')
    ax.grid(True, alpha=0.3)

    # Plot 2: Average weight
    ax = axes[0, 1]
    ax.plot(months, net_stats['avg_w'], marker='o', color='green')
    ax.set_ylabel('Average Weight')
    ax.set_title('Average Edge Weight')
    ax.grid(True, alpha=0.3)

    # Plot 3: Average degree
    ax = axes[1, 0]
    ax.plot(months, net_stats['avg_deg'], marker='o', color='purple')
    ax.set_ylabel('Average Degree')
    ax.set_title('Average Node Degree')
    ax.grid(True, alpha=0.3)

    # Plot 4: Clustering coefficient with changepoints
    ax = axes[1, 1]
    ax.plot(months, net_stats['cc'], marker='o', color='orange', label='Clustering Coeff.')
    if changepoints is not None and len(changepoints) > 0:
        cp_months = pd.to_datetime(changepoints['month'])
        for cp in cp_months:
            ax.axvline(cp, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.plot([], [], 'r--', label='Change Points')
    ax.set_ylabel('Clustering Coefficient')
    ax.set_title('Clustering Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for all subplots
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    fig.savefig(outdir / "network_timeseries.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_community_sizes(communities: pd.DataFrame, month, outdir: Path = FIG):
    """Plot both ranked community sizes and their distribution.

    Args:
        communities: DataFrame with node → community assignments.
        month: Label for the time period.
        outdir: Directory where the figure is written.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    comm_sizes = communities['community'].value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Community Structure — {str(month)[:7]}", fontsize=14)

    # Plot 1: Top 20 communities by size
    ax = axes[0]
    top_n = min(20, len(comm_sizes))
    ax.bar(range(top_n), comm_sizes.head(top_n).values)
    ax.set_xlabel('Community Rank')
    ax.set_ylabel('Size (# nodes)')
    ax.set_title(f'Top {top_n} Communities')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Size distribution histogram
    ax = axes[1]
    ax.hist(comm_sizes.values, bins=30, edgecolor='black')
    ax.set_xlabel('Community Size')
    ax.set_ylabel('Frequency')
    ax.set_title('Community Size Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"community_sizes_{str(month)[:7]}.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_adoption_deltas(adoption_df: pd.DataFrame, outdir: Path = FIG):
    """Plot review adoption changes around discount events.

    Args:
        adoption_df: DataFrame with ``delta_reviews`` and ``discount_pct`` columns.
        outdir: Directory where the figure is written.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Review Adoption Around Discount Events", fontsize=14)

    # Plot 1: Scatter of delta vs discount percentage
    ax = axes[0]
    valid = adoption_df.dropna(subset=['delta_reviews', 'discount_pct'])
    ax.scatter(valid['discount_pct'], valid['delta_reviews'], alpha=0.5, s=20)
    ax.set_xlabel('Discount %')
    ax.set_ylabel('Δ Reviews (post - pre)')
    ax.set_title('Adoption Change vs Discount Size')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of review deltas
    ax = axes[1]
    ax.hist(adoption_df['delta_reviews'].dropna(), bins=50, edgecolor='black')
    ax.set_xlabel('Δ Reviews')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Review Changes')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='No change')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "adoption_deltas.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_centrality_distributions(centralities: pd.DataFrame, month, outdir: Path = FIG):
    """Plot distributions of different centrality measures.

    Args:
        centralities: DataFrame containing deg/strength/betweenness/eigenvector.
        month: Label for the time period.
        outdir: Directory where the figure is written.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Centrality Distributions — {str(month)[:7]}", fontsize=14)

    # Plot 1: Degree
    ax = axes[0, 0]
    ax.hist(centralities['deg'], bins=30, edgecolor='black')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 2: Strength
    ax = axes[0, 1]
    ax.hist(centralities['strength'], bins=30, edgecolor='black')
    ax.set_xlabel('Strength (weighted degree)')
    ax.set_ylabel('Frequency')
    ax.set_title('Strength Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Betweenness
    ax = axes[1, 0]
    valid_bw = centralities['betweenness'].dropna()
    if len(valid_bw) > 0:
        ax.hist(valid_bw, bins=30, edgecolor='black')
        ax.set_xlabel('Betweenness Centrality')
        ax.set_ylabel('Frequency')
        ax.set_title('Betweenness Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    # Plot 4: Eigenvector
    ax = axes[1, 1]
    valid_ev = centralities['eigenvector'].dropna()
    if len(valid_ev) > 0:
        ax.hist(valid_ev, bins=30, edgecolor='black')
        ax.set_xlabel('Eigenvector Centrality')
        ax.set_ylabel('Frequency')
        ax.set_title('Eigenvector Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"centrality_distributions_{str(month)[:7]}.png", dpi=FIG_DPI)
    plt.close(fig)

