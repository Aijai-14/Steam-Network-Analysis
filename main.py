"""full Steam affinity analysis + visualization pipeline."""

from __future__ import annotations
import sys
import pandas as pd
import json

# Import steamNetwork modules
from steamNetwork.paths import RAW, INTERIM, PROCESSED, FIG
from steamNetwork.config import DiffusionParams, LinkPredParams, SEED
from steamNetwork.utils import month_floor

# Data loading
from steamNetwork.data.reviews import load_reviews_csv
from steamNetwork.data.priceEvents import load_price_events

# Preprocessing
from steamNetwork.preProcess.enrichNodes import build_node_attributes
from steamNetwork.preProcess.snapShots import build_snapshots, SnapshotConfig

# Analysis - Core
from steamNetwork.analysis.communities import find_communities
from steamNetwork.analysis.changePoints import network_series, detect_changepoints
from steamNetwork.analysis.diffusion import (
    simulate_discount_cascades, adoption_delta_around_events,
    review_counts_by_day, simulate_null_cascades
)
from steamNetwork.analysis.linkPrediction import run_link_prediction
from steamNetwork.analysis.centralityAnalysis import compute_centralities

# Analysis - Advanced
from steamNetwork.analysis.diffusionStats import (
    analyze_cascade_depth, centrality_cascade_correlation,
    compare_cascade_distributions
)
from steamNetwork.analysis.embeddingLinkPred import compare_link_prediction_methods
from steamNetwork.analysis.communityEvolution import (
    track_community_evolution,
    summarize_evolution_statistics
)
from steamNetwork.analysis.statisticalAnalysis import (
    community_homophily_analysis,
    temporal_stability_analysis, degree_distribution_analysis,
    comprehensive_statistical_report
)

# Models
from steamNetwork.models.embeddings import node2vec_embeddings

# Visualization - Core
from steamNetwork.visualizations.plotting import (
    plot_cascade_sizes,
    plot_deg_hist, plot_lp_scores, plot_network_timeseries,
    plot_community_sizes, plot_adoption_deltas, plot_centrality_distributions,
)

# Visualization - Advanced
from steamNetwork.visualizations.advancedPlotting import (
    plot_community_network, plot_evolution_heatmap, plot_evolution_events,
    plot_diffusion_comparison, plot_method_comparison, plot_centrality_scatter_matrix,
    plot_cascade_depth_analysis
)

RUN_TAG = "engage" # tag for the run


def main():
    """Execute the end-to-end pipeline from raw data to analysis artifacts."""
    # Ensure output directories exist
    FIG.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    def tagged(base):
        """Return a path within the run-specific namespace (if defined)."""
        return base / RUN_TAG if RUN_TAG else base

    # get the figure, interim, and processed directories
    fig_dir = tagged(FIG)
    interim_dir = tagged(INTERIM)
    proc_dir = tagged(PROCESSED)

    # ensure the figure, interim, and processed directories exist
    for path in (fig_dir, interim_dir, proc_dir):
        path.mkdir(parents=True, exist_ok=True)

    def proc_file(name: str):
        """Ensure the processed-data file path exists before returning it."""
        path = proc_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # DATA LOADING & PREPROCESSING
    # Load reviews
    reviews = load_reviews_csv(RAW / "reviews.csv")

    # Load game metadata
    games_df = pd.read_csv(RAW / "games.csv")

    # Load price events
    price_events = load_price_events(RAW / "price_events_itad.csv")

    # Build enriched node attributes
    node_attrs = build_node_attributes(
        games_csv=RAW / "games.csv",
        price_events_csv=RAW / "price_events_itad.csv" if price_events is not None else None
    )
    node_attrs.to_csv(interim_dir / "node_attributes.csv", index=False)

    # NETWORK CONSTRUCTION 
    # Configure affinity network parameters
    cfg = SnapshotConfig(
        window_days=30,
        min_coreviews=3,
        weight_scheme="engagement", # for final but wjaccard used for milestone 
        engagement_weights={
            "base": 1.0,
            "rec_bonus": 0.8,
            "votes_scale": 0.2,
            "playtime_scale": 0.12,
            "vote_cap": 500.0,
            "playtime_cap": 200.0,
        },
    )

    # Build monthly snapshots
    snapshots = build_snapshots(reviews, cfg)
    snapshots_path = proc_file("snapshots_monthly.parquet")
    snapshots.to_parquet(snapshots_path)

    # Compute network statistics time series
    net_stats = network_series(snapshots)
    net_stats.to_csv(proc_file("network_timeseries.csv"), index=False)

    # Plot network evolution 
    plot_network_timeseries(net_stats, outdir=fig_dir)

    # COMMUNITY DETECTION
    # Select a representative month for community analysis
    representative_month = snapshots['month'].value_counts().idxmax() # get the month with the most edges
    edges_month = snapshots[snapshots['month'] == representative_month]

    # Detect communities
    communities = find_communities(edges_month, method="auto")
    communities.to_csv(proc_file(f"communities_{representative_month.strftime('%Y%m')}.csv"), index=False)

    # Plot degree distribution for this month
    plot_deg_hist(edges_month, representative_month, outdir=fig_dir)

    # Plot community sizes
    plot_community_sizes(communities, representative_month, outdir=fig_dir)

    # STRUCTURAL CHANGE POINT DETECTION
    # Detect change points in clustering coefficient
    changepoints = detect_changepoints(net_stats, column="cc", pen=3.0)
    changepoints.to_csv(proc_file("changepoints.csv"), index=False)

    # Re-plot network timeseries with changepoints
    plot_network_timeseries(net_stats, outdir=fig_dir, changepoints=changepoints)

    # DIFFUSION MODELING 
    # if there are price events, run the diffusion modeling
    if price_events is not None and len(price_events) > 0:
        # Configure diffusion parameters
        diff_params = DiffusionParams()

        # Select a month with price events
        price_events_monthly = price_events.copy()
        price_events_monthly['month'] = month_floor(price_events_monthly['timestamp']) # convert the price events to a month-end timestamp

        target_month = price_events_monthly['month'].value_counts().idxmax() # get the month with the most price events

        # Run cascade simulations
        cascade_results = simulate_discount_cascades(
            edges_monthly=snapshots,
            price_events=price_events,
            month=target_month,
            params=diff_params,
            n_sims=150
        )
        # save the cascade results to a csv file
        cascade_path = proc_file(f"cascades_{target_month.strftime('%Y%m')}.csv")
        cascade_results.to_csv(cascade_path, index=False)

        # Plot cascade size distribution
        plot_cascade_sizes(cascade_results, target_month, outdir=fig_dir)

        # Analyze adoption deltas (post - pre review counts) around discount events
        reviews_daily = review_counts_by_day(reviews)
        adoption_deltas = adoption_delta_around_events(reviews_daily, price_events, window_days=14)
        adoption_path = proc_file("adoption_deltas.csv")
        adoption_deltas.to_csv(adoption_path, index=False)

        # Plot adoption deltas
        plot_adoption_deltas(adoption_deltas, outdir=fig_dir)

        # Compute centralities for cascade analysis
        edges_target = snapshots[snapshots['month'] == target_month] # get the edges for the target month
        centralities = compute_centralities(edges_target)
        centralities_path = proc_file(f"centralities_{target_month.strftime('%Y%m')}.csv")
        centralities.to_csv(centralities_path, index=False)

        # Plot centrality distributions
        plot_centrality_distributions(centralities, target_month, outdir=fig_dir)

    # LINK PREDICTION
    # Get sorted months
    months_sorted = sorted(snapshots['month'].unique())

    if len(months_sorted) >= 2: # if there are at least 2 months, run the link prediction
        # Run link prediction across consecutive months
        lp_results = run_link_prediction(snapshots)

        # Save results
        results_df = pd.DataFrame(lp_results['results'])
        results_df.to_csv(proc_file("linkpred_scores.csv"), index=False)

        # Plot link prediction scores
        plot_lp_scores(results_df, outdir=fig_dir, tag=f"heuristics_{RUN_TAG}")

        # perform node2vec embeddings link prediction
        emb_params = LinkPredParams()

        # Use first month for embeddings
        edges_first = snapshots[snapshots['month'] == months_sorted[0]]
        embeddings = node2vec_embeddings( # generate the node2vec embeddings
            edges_first,
            dimensions=emb_params.embed_dim,
            walk_length=emb_params.walk_length,
            num_walks=emb_params.num_walks,
            p=emb_params.p,
            q=emb_params.q,
            seed=SEED
        )
        embeddings.to_csv(proc_file(f"embeddings_{months_sorted[0].strftime('%Y%m')}.csv"), index=False)

    # Diffusion Statistical Analysis
    if price_events is not None and len(price_events) > 0:
        price_events_monthly = price_events.copy() # copy the price events
        price_events_monthly['month'] = month_floor(price_events_monthly['timestamp']) # convert the price events to a month-end timestamp
        target_month = price_events_monthly['month'].value_counts().idxmax() # get the month with the most price events

        # Make target_month timezone-aware to match price_events['timestamp']
        target_month = pd.Timestamp(target_month).tz_localize('UTC') # convert the target month to a timezone-aware timestamp

        # get the seeds for the target month
        seeds_for_month = price_events[
            (price_events['timestamp'] >= target_month) &
            (price_events['timestamp'] < target_month + pd.DateOffset(months=1))
        ]['app_id'].unique().tolist()

        # Load cascade results
        cascade_csv = proc_file(f"cascades_{target_month.strftime('%Y%m')}.csv")
        cascade_results = pd.read_csv(cascade_csv)

        # Simulate null model
        null_results = simulate_null_cascades(
            edges_monthly=snapshots,
            true_seeds=seeds_for_month,
            params=DiffusionParams(),
            n_sims=100,
            mode='degree_matched'
        )
        null_results.to_csv(proc_file(f"null_cascades_{target_month.strftime('%Y%m')}.csv"), index=False)

        # Statistical comparison
        comparison_results = compare_cascade_distributions( # compare the cascade distributions
            cascade_results,
            null_results,
            pd.read_csv(proc_file("adoption_deltas.csv")) if proc_file("adoption_deltas.csv").exists() else None # get the adoption deltas
        )
        pd.DataFrame([comparison_results['statistical_tests']]).to_csv( # save the statistical tests to a csv file
            proc_file("diffusion_statistical_tests.csv"), index=False
        )
        # Visualization
        plot_diffusion_comparison(comparison_results, outdir=fig_dir) # plot the diffusion comparison

        # Cascade depth analysis
        if 'depth' in cascade_results.columns:
            depth_analysis = analyze_cascade_depth(cascade_results) # analyze the cascade depth
            depth_analysis.to_csv(proc_file("cascade_depth_analysis.csv"), index=False) # save the cascade depth analysis to a csv file
            plot_cascade_depth_analysis(depth_analysis, outdir=fig_dir) # plot the cascade depth analysis

        # Centrality-cascade correlation
        centralities = pd.read_csv(proc_file(f"centralities_{target_month.strftime('%Y%m')}.csv")) # get the centralities for the target month
        corr_analysis = centrality_cascade_correlation(seeds_for_month, centralities, cascade_results) # analyze the centrality-cascade correlations
        pd.DataFrame([corr_analysis[1]]).to_csv(proc_file("centrality_cascade_correlations.csv"), index=False) # save the centrality-cascade correlations to a csv file

    # COMPARE LINK PREDICTION METHODS
    if len(months_sorted) >= 2: # if there are at least 2 months, run the embedding-based link prediction
        comparison_results = compare_link_prediction_methods(snapshots, LinkPredParams()) # compare the link prediction methods

        if not comparison_results.empty:
            comparison_results.to_csv(proc_file("linkpred_method_comparison.csv"), index=False)
            plot_method_comparison(comparison_results, outdir=fig_dir) # plot the link prediction methods

    # COMMUNITY EVOLUTION TRACKING
    if len(months_sorted) >= 2:
        communities_by_month = {}

        # for each month, detect the communities
        for month_ts in months_sorted:
            edges_m = snapshots[snapshots['month'] == month_ts] # get the edges for the month
            if len(edges_m) > 0:
                comms = find_communities(edges_m, method="auto") # detect the communities
                communities_by_month[month_ts] = comms

        # track the community evolution
        events_df, stability_df = track_community_evolution(snapshots, communities_by_month)

        if not events_df.empty:
            events_df.to_csv(proc_file("community_evolution_events.csv"), index=False) # save the community evolution events to a csv file

            # summarize the community evolution events
            events_summary = summarize_evolution_statistics(events_df) # summarize the community evolution events
            if not events_summary.empty:
                events_summary.to_csv(proc_file("community_evolution_summary.csv"), index=False) # save the community evolution summary to a csv file

        if not stability_df.empty:
            stability_df.to_csv(proc_file("community_stability.csv"), index=False) # save the community stability to a csv file

        # Visualizations
        plot_evolution_heatmap(stability_df, outdir=fig_dir) # plot the community evolution heatmap
        if not events_df.empty:
            plot_evolution_events(summarize_evolution_statistics(events_df), outdir=fig_dir) # plot the community evolution events

        # Network visualization with communities
        plot_community_network(edges_month, communities, representative_month, outdir=fig_dir, top_n_communities=5)


    # EXTRA STATISTICAL ANALYSIS
    # Degree distribution analysis
    degree_dist = degree_distribution_analysis(edges_month) # analyze the degree distribution
    pd.DataFrame([degree_dist]).to_csv(proc_file("degree_distribution_analysis.csv"), index=False)

    # Community homophily
    homophily = community_homophily_analysis(communities, node_attrs, attribute='genres') # analyze the community homophily
    pd.DataFrame([homophily]).to_csv(proc_file("community_homophily.csv"), index=False)

    # Temporal stability
    temporal_stability = temporal_stability_analysis(net_stats) # analyze the temporal stability
    pd.DataFrame(temporal_stability).to_csv(proc_file("temporal_stability.csv"), index=False)

    # Centrality scatter matrix
    plot_centrality_scatter_matrix(centralities, representative_month, outdir=fig_dir) # plot the centrality scatter matrix

    # Comprehensive report of all statistical analyses
    comprehensive_report = comprehensive_statistical_report(
        snapshots, communities, centralities, node_attrs, net_stats, representative_month
    )

    # save the comprehensive statistical report to a json file
    with open(proc_file("comprehensive_statistical_report.json"), 'w') as f:
        json.dump(comprehensive_report, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
