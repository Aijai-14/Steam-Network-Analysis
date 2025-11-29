#!/usr/bin/env python
"""Legacy milestone runner for wJaccard snapshots and link prediction."""

from __future__ import annotations

import pandas as pd

from steamNetwork.analysis.embeddingLinkPred import compare_link_prediction_methods
from steamNetwork.analysis.linkPrediction import run_link_prediction
from steamNetwork.config import LinkPredParams
from steamNetwork.data.reviews import load_reviews_csv
from steamNetwork.paths import FIG, PROCESSED, RAW
from steamNetwork.preProcess.snapShots import SnapshotConfig, build_snapshots
from steamNetwork.visualizations.advancedPlotting import plot_method_comparison
from steamNetwork.visualizations.plotting import plot_lp_scores


def main():
    """Build wJaccard snapshots, run heuristic LP, and compare methods."""
    reviews = load_reviews_csv(RAW / "reviews.csv") # load the reviews data

    cfg = SnapshotConfig(window_days=30, min_coreviews=3, weight_scheme="wjaccard") # configure the snapshot config
    snapshots = build_snapshots(reviews, cfg) # build the snapshots

    out_dir = PROCESSED / "legacy_linkpred" # create the output directory
    fig_dir = FIG / "legacy_linkpred" # create the figure directory
    out_dir.mkdir(parents=True, exist_ok=True) # create the output directory
    fig_dir.mkdir(parents=True, exist_ok=True) # create the figure directory

    snapshots_path = out_dir / "snapshots_monthly_wjaccard.parquet"
    snapshots.to_parquet(snapshots_path) # save the snapshots to a parquet file

    lp_results = run_link_prediction(snapshots) # run the link prediction
    scores_df = pd.DataFrame(lp_results["results"]) # convert the results to a dataframe
    scores_path = out_dir / "linkpred_scores_wjaccard.csv"
    scores_df.to_csv(scores_path, index=False) # save the scores to a csv file

    if not scores_df.empty:
        plot_lp_scores(scores_df, outdir=fig_dir, tag="heuristics_wjaccard") # plot the scores

    comparison_results = compare_link_prediction_methods(snapshots, LinkPredParams()) # compare the link prediction methods

    if not comparison_results.empty:
        cmp_path = out_dir / "linkpred_method_comparison_wjaccard.csv"
        comparison_results.to_csv(cmp_path, index=False) # save the comparison results to a csv file
        plot_method_comparison(comparison_results, outdir=fig_dir) # plot the comparison results


if __name__ == "__main__":
    main()

