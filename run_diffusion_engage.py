#!/usr/bin/env python
"""Command-line workflow for monthly diffusion hyperparameter sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from steamNetwork.analysis.diffusion_engage import (
    DiffusionStrategy,
    SeedSelectionConfig,
    monthly_hyperparam_sweep,
)
from steamNetwork.analysis.diffusion import (
    adoption_delta_around_events,
    review_counts_by_day,
    simulate_null_cascades,
    ICParams,
)

from steamNetwork.analysis.diffusionStats import (
    analyze_cascade_depth,
    compare_cascade_distributions,
)

from steamNetwork.data.priceEvents import load_price_events
from steamNetwork.data.reviews import load_reviews_csv
from steamNetwork.paths import FIG, PROCESSED, RAW
from steamNetwork.visualizations.advancedPlotting import (
    plot_cascade_timelines,
    plot_diffusion_hypergrid,
    plot_cascade_depth_analysis,
    plot_diffusion_comparison,
)
from steamNetwork.visualizations.plotting import (
    plot_adoption_deltas,
    plot_cascade_sizes,
)


def _parse_month(month_str: str) -> pd.Timestamp:
    """Convert a YYYY-MM string into a month-end timestamp."""
    period = pd.Period(month_str, freq="M")
    return period.to_timestamp("M")


def build_default_strategies() -> list[DiffusionStrategy]:
    """Return the handful of baseline strategies used across experiments."""
    return [
        DiffusionStrategy(name="ic_balanced", base_p=0.12, weight_scale=9, max_p=0.75), # balanced IC strategy
        DiffusionStrategy(name="ic_aggressive", base_p=0.18, weight_scale=14, max_p=0.9), # aggressive IC strategy
        DiffusionStrategy(name="lt_responsive", model="lt", lt_threshold=0.3, max_steps=10), # responsive LT strategy
    ]


def parse_args():
    """Parse CLI arguments for diffusion batch execution."""
    parser = argparse.ArgumentParser(description="Run enhanced diffusion experiments.")
    parser.add_argument( # optional path to snapshots parquet
        "--snapshots",
        type=Path,
        help="Optional path to snapshots parquet. Defaults to tagged engagement run if present.",
    )
    parser.add_argument(
        "--months", # optional list of YYYY-MM values to restrict analysis to
        type=str,
        nargs="+",
        help="Optional list of YYYY-MM values to restrict analysis to.",
    )
    parser.add_argument(
        "--min-discount", # minimum discount percentage to treat as seed
        type=float,
        default=10.0,
        help="Minimum discount percentage to treat as seed.",
    )
    parser.add_argument(
        "--top-k", # top k discounted games to seed
        type=int,
        default=15,
        help="Top-K discounted games to seed.",
    )
    parser.add_argument(
        "--n-sims", # number of simulations per strategy
        type=int,
        default=150,
        help="Simulations per strategy.",
    )
    parser.add_argument(
        "--highlight-months", # months to generate detailed adoption/cascade plots for
        type=str, 
        nargs="*",
        default=["2025-07", "2025-08", "2025-09", "2025-10"], # busiest months
        help="Months to generate detailed adoption/cascade plots for.",
    )
    return parser.parse_args()


def main():
    """Run diffusion sweeps, plots, and summaries across chosen months."""
    args = parse_args()
    snapshots_path = args.snapshots
    if snapshots_path is None: # if no snapshots path is provided, use the default paths
        candidates = [
            PROCESSED / "engage" / "snapshots_monthly.parquet",
            PROCESSED / "snapshots_monthly.parquet",
        ]
        snapshots_path = next((c for c in candidates if c.exists()), None)
    
    if snapshots_path is None or not snapshots_path.exists(): # if the snapshots path does not exist, raise an error
        raise SystemExit("Could not locate snapshots parquet. Provide --snapshots to override.")

    snapshots = pd.read_parquet(snapshots_path) # read the snapshots from the parquet file
    reviews = load_reviews_csv(RAW / "reviews.csv") # load the reviews data
    price_events = load_price_events(RAW / "price_events_itad.csv") # load the price events data
    price_events["month"] = price_events["timestamp"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M") # convert the price events to a month-end timestamp

    available_months = sorted(set(price_events["month"]).intersection(set(snapshots["month"]))) # get the available months
    if not available_months:
        raise SystemExit("No overlapping months between price events and snapshots.")

    if args.months: # if the months are provided, parse the months
        requested = {_parse_month(m) for m in args.months}
        target_months = [m for m in available_months if m in requested]
        if not target_months:
            raise SystemExit("None of the requested months overlap with processed data.")
    else:
        target_months = available_months

    highlight_set = { # get the highlight months
        pd.Period(m, freq="M").to_timestamp("M")
        for m in (args.highlight_months or [])
    }

    strategies = build_default_strategies() # build the default strategies
    seed_cfg = SeedSelectionConfig(min_discount=args.min_discount, top_k=args.top_k) # configure the seed selection

    out_dir = snapshots_path.parent / "diffusion_engage" # create the output directory
    fig_dir = FIG / "diffusion_engage" # create the figure directory
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True) # create the figure directory

    all_summaries = []
    all_results = []
    reviews_daily = review_counts_by_day(reviews) # get the reviews daily
    adoption_global = adoption_delta_around_events(reviews_daily, price_events, window_days=14) # get the adoption deltas
    if not adoption_global.empty:
        adoption_global["month_tag"] = pd.to_datetime(adoption_global["timestamp"], utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")
    ic_params = ICParams(base_p=0.12, alpha=4.0, max_steps=8) # configure the IC parameters
 
    # for each month, run the diffusion sweep
    for month in target_months:
        month_tag = str(month)[:7] # get the month tag
        month_out = out_dir / month_tag # create the output directory
        fig_month = fig_dir / month_tag # create the figure directory
        month_out.mkdir(parents=True, exist_ok=True) # create the output directory
        fig_month.mkdir(parents=True, exist_ok=True) # create the figure directory

        result = monthly_hyperparam_sweep( # run the diffusion sweep
            snapshots=snapshots,
            price_events=price_events,
            reviews=reviews,
            month=month,
            strategies=strategies,
            seed_cfg=seed_cfg,
            n_sims=args.n_sims,
        )

        # get the summary, results, and seeds
        summary = result["summary"].copy()
        summary["month"] = month
        sims = result["results"].copy()
        sims["month"] = month
        seeds = result["seeds"].copy()
        seeds["month"] = month

        # save the summary, results, and seeds
        summary.to_csv(month_out / "summary.csv", index=False)
        sims.to_csv(month_out / "simulations.csv", index=False)
        seeds.to_csv(month_out / "seeds.csv", index=False)

       
        # append the summary, results, and seeds to the lists
        all_summaries.append(summary)
        all_results.append(sims)

        # plot the diffusion hypergrid
        plot_diffusion_hypergrid(
            summary,
            x_param="base_p",
            y_param="weight_scale",
            metric="mean_size",
            outdir=fig_month,
            fname="diffusion_hypergrid.png",
        )

        # plot the cascade timelines
        plot_cascade_timelines(
            result["timelines"],
            outdir=fig_month,
            fname="cascade_timelines.png",
        )

        # if the month is in the highlight set, plot the adoption deltas, IC balanced simulations, cascade sizes, cascade depth analysis, and null cascades
        if month in highlight_set:
            if not adoption_global.empty:
                adoption_month = adoption_global[adoption_global["month_tag"] == month].copy() # get the adoption deltas for the month
            else:
                adoption_month = pd.DataFrame() # if there are no adoption deltas, create an empty dataframe
            adoption_month.to_csv(month_out / "adoption_deltas.csv", index=False) # save the adoption deltas to a csv file
            plot_adoption_deltas(adoption_month if not adoption_month.empty else adoption_global, outdir=fig_month) # plot the adoption deltas

            ic_balanced_df = result["results"][result["results"]["strategy"] == "ic_balanced"][["sim_id", "size", "depth"]].copy() # get the IC balanced simulations
            ic_balanced_df.to_csv(month_out / "ic_balanced_simulations.csv", index=False) # save the IC balanced simulations to a csv file

            if not ic_balanced_df.empty:
                plot_cascade_sizes(ic_balanced_df[["size"]], month, outdir=fig_month) # plot the cascade sizes

                depth_analysis = analyze_cascade_depth(ic_balanced_df) # analyze the cascade depth
                depth_analysis.to_csv(month_out / "cascade_depth_analysis.csv", index=False) # save the cascade depth analysis to a csv file
                plot_cascade_depth_analysis(depth_analysis, outdir=fig_month) # plot the cascade depth analysis

                edges_month = snapshots[snapshots["month"] == month][["i", "j", "w", "co"]]
                seed_ids = seeds["app_id"].astype(int).tolist() # get the seed ids

                if seed_ids:
                    # simulate the null cascades
                    null_results = simulate_null_cascades(
                        edges_month,
                        seed_ids,
                        params=ic_params,
                        n_sims=args.n_sims,
                        mode="degree_matched",
                    )
                    null_results.to_csv(month_out / "null_cascades.csv", index=False)
                else:
                    null_results = pd.DataFrame(columns=["sim_id", "size", "depth"])

                # compare the cascade distributions
                comparison = compare_cascade_distributions(
                    ic_balanced_df[["sim_id", "size", "depth"]],
                    null_results,
                    adoption_month if not adoption_month.empty else None,
                )
                stats = comparison.get("statistical_tests")
                if stats:
                    pd.DataFrame([stats]).to_csv(month_out / "diffusion_statistical_tests.csv", index=False)
                plot_diffusion_comparison(comparison, outdir=fig_month) # plot the diffusion comparison

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(out_dir / "diffusion_summary_all_months.csv", index=False) # save the summary to a csv file
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(out_dir / "diffusion_sims_all_months.csv", index=False)


if __name__ == "__main__":
    main()

