# Steam Network Analysis

Comprehensive end-to-end tooling for studying how games spread through the Steam ecosystem. The project ingests raw review and pricing data, builds monthly weighted affinity networks, simulates price-driven diffusion cascades, evaluates a range of link-prediction baselines, and exports publication-ready reports plus figures.

## Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Data Lifecycle](#data-lifecycle)
- [Data Inputs](#data-inputs)
- [Getting Started](#getting-started)
- [Running the Pipelines](#running-the-pipelines)
- [Outputs & Reports](#outputs--reports)
- [Package Tour](#package-tour)
- [Customization & Extension](#customization--extension)
- [Troubleshooting](#troubleshooting)

## Overview

This repo models player co-interest on Steam by projecting user reviews into weighted game-to-game networks. The default **engagement-weighted** affinity reflects co-reviews while considering review helpfulness, playtime, and recommendation flags. With these networks the project:

1. Detects structural shifts (centrality, clustering, community splits/merges) over time.
2. Simulates Independent Cascade vs. Linear Threshold diffusion for discount-driven adoption.
3. Benchmarks heuristic, embedding-only, and hybrid link prediction.
4. Summarizes statistical relationships (homophily, degree distributions, temporal stability).
5. Exports dashboards of PNGs and CSVs for downstream analysis.

## Highlights
- Single command (`python main.py`) to go from raw CSVs to ready-to-plot artifacts.
- Configurable snapshot builder supporting Weighted Jaccard or engagement-driven weights.
- Price-event–aware diffusion experiments with null models and statistical tests.
- Monthly hyper-parameter grid search for diffusion strategies (`run_diffusion_engage.py`).
- Combined heuristic + Node2Vec link-prediction comparisons (AUROC/AP scores).
- Community evolution tracking (births, splits, merges, stability metrics).
- Deterministic paths (`steamNetwork/paths.py`) keep data/figures organized per run tag.

## Repository Layout

| Path | Description |
| --- | --- |
| `main.py` | Master pipeline orchestrating ingest, snapshotting, diffusion, link prediction, and visualization for the default engagement run. |
| `run_diffusion_engage.py` | CLI for monthly diffusion hyper-parameter sweeps with richer plots/statistics. |
| `run_linkpred_milestone.py` | Legacy workflow that rebuilds Weighted-Jaccard snapshots and reproduces heuristic vs. embedding LP benchmarks. |
| `steamNetwork/` | Python package housing preprocessing, analysis, modeling, and plotting modules (see [Package Tour](#package-tour)). |
| `data/` | Standardized data lake (`raw`, `interim`, `processed`) populated by the pipelines. |
| `reports/figures/` | All PNG outputs grouped by experiment (e.g., `engage`, `diffusion_engage`, `legacy_linkpred`). |
| `requirements.txt` | Python dependencies (igraph/leidenalg, NetworkX, pandas, sklearn, etc.). |

## Data Lifecycle

1. **Raw ingestion** (`data/raw/`): curated Steam reviews, game metadata, and price events.
2. **Intermediate enrichment** (`data/interim/<tag>/`): node attributes, cached metadata.
3. **Processed artifacts** (`data/processed/<tag>/`): monthly snapshots, cascades, link-pred scores, statistical summaries.
4. **Reports** (`reports/figures/<tag>/`): visuals for each analytic stage.

The `RUN_TAG` constant near the top of `main.py` (default `engage`) keeps outputs partitioned so multiple experiments can coexist.

## Data Inputs

Place the following CSVs in `data/raw/`:

| File | Purpose | Required Columns |
| --- | --- | --- |
| `reviews.csv` | Steam review history used for affinity edges and adoption signals. | `user_id`, `app_id`, `review_id`, `timestamp`, `recommended`, `votes_up`, `playtime_hours` |
| `games.csv` | Game metadata merged into node attributes (genres, prices, release year). | `app_id`, `title`, `genres`, `base_price`, `current_price`, `release_date`, `discount_pct` |
| `price_events_itad.csv` | Historical discount events that seed diffusion simulations. | `timestamp`, `app_id`, `discount_pct`, `price_old`, `price_new` |

Optional supporting files (`steam_appids_curated.csv`, `ownership.csv`, etc.) can also live under `data/raw/` for future extensions, but the core pipeline depends on the three assets above.

## Getting Started

1. **Python version**: 3.10+ is recommended (matches igraph/leidenalg wheels).
2. **System packages** (macOS): `brew install igraph` ensures native libs for `python-igraph`/`leidenalg`.
3. **Create a virtual environment**:
   ```bash
   cd /Users/aijay/Desktop/Info\ Networks/SteamNetworkAnalysis
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Verify data placement** under `data/raw/`.
5. **(Optional)** export `PYTHONPATH` if running scripts from outside the repo root: `export PYTHONPATH=$PYTHONPATH:/Users/aijay/Desktop/Info\ Networks/SteamNetworkAnalysis`.

## Running the Pipelines

### 1. End-to-End Engagement Run (`main.py`)

```bash
python main.py
```

The script performs:

1. **Directory bootstrap** using `steamNetwork.paths`.
2. **Data loading** (`steamNetwork.data.reviews`, `priceEvents`, `games`).
3. **Node enrichment** via `preProcess.enrichNodes.build_node_attributes`.
4. **Monthly snapshots** using `preProcess.snapShots.build_snapshots` with engagement weights (`SnapshotConfig`).
5. **Network analytics**: community detection (Leiden/Louvain), degree histograms, change-point detection, temporal statistics.
6. **Diffusion modeling** (if price events exist): cascade simulations, adoption deltas, centrality correlations, null-model comparisons, depth analyses.
7. **Link prediction**: heuristic logistic regression, Node2Vec embeddings, hybrid comparisons.
8. **Community evolution**: matching, stability scoring, event summaries.
9. **Extra stats**: degree distributions, homophily, comprehensive JSON report.
10. **Visualization**: figures saved under `reports/figures/engage/` plus run-tag-specific folders.

### 2. Diffusion Hyper-Parameter Sweeps (`run_diffusion_engage.py`)

Use after `main.py` generates `data/processed/engage/snapshots_monthly.parquet`:

```bash
python run_diffusion_engage.py \
  --snapshots data/processed/engage/snapshots_monthly.parquet \
  --months 2025-07 2025-08 2025-09 \
  --min-discount 15 \
  --top-k 20 \
  --n-sims 250 \
  --highlight-months 2025-07 2025-09
```

Key flags:

- `--snapshots`: Override path to the parquet snapshots (defaults to engage tag if omitted).
- `--months`: Restrict experiments to specific `YYYY-MM` periods.
- `--min-discount` / `--top-k`: Control seed selection (`SeedSelectionConfig`).
- `--n-sims`: Simulation count per strategy.
- `--highlight-months`: Months that receive deeper plots (adoption deltas, null comparisons).

Artifacts land in `data/processed/diffusion_engage/<YYYY-MM>/` with matching plots under `reports/figures/diffusion_engage/<YYYY-MM>/`.

### 3. Legacy Weighted-Jaccard Link Prediction (`run_linkpred_milestone.py`)

Rebuilds historical milestones with the simpler weighted-Jaccard scheme:

```bash
python run_linkpred_milestone.py
```

Outputs:
- `data/processed/legacy_linkpred/snapshots_monthly_wjaccard.parquet`
- `data/processed/legacy_linkpred/linkpred_scores_wjaccard.csv`
- `data/processed/legacy_linkpred/linkpred_method_comparison_wjaccard.csv`
- Figures under `reports/figures/milestoneLinkPred/`

## Outputs & Reports

| Artifact | Location | Notes |
| --- | --- | --- |
| Node attributes | `data/interim/engage/node_attributes.csv` | Enriched metadata (genres, prices, last discount). |
| Snapshots | `data/processed/engage/snapshots_monthly.parquet` | Monthly edge list with weights (`month`, `i`, `j`, `w`, `co`, `deg_i`, `deg_j`). |
| Network KPIs | `data/processed/engage/network_timeseries.csv` | Node/edge counts, avg weight, avg degree, clustering coefficient. |
| Community results | `data/processed/engage/communities_*.csv` | Month-specific community assignments plus plots (`community_sizes`, `community_network`). |
| Cascades & adoption | `data/processed/engage/cascades_*.csv`, `adoption_deltas.csv`, `null_cascades_*.csv` | Simulation outputs plus observed deltas. |
| Centralities | `data/processed/engage/centralities_*.csv` + scatter matrix figure | Degree, strength, betweenness, eigenvector per node. |
| Link prediction | `data/processed/engage/linkpred_scores.csv`, `linkpred_method_comparison.csv`, `embeddings_*.csv` | Heuristic AUROC/AP plus embedding comparisons. |
| Community evolution | `community_evolution_events.csv`, `community_evolution_summary.csv`, `community_stability.csv` | Event timelines and stability measures. |
| Statistics | `cascade_depth_analysis.csv`, `centrality_cascade_correlations.csv`, `degree_distribution_analysis.csv`, `community_homophily.csv`, `temporal_stability.csv`, `comprehensive_statistical_report.json` | Aggregated stats for reporting. |
| Figures | `reports/figures/engage/*.png`, `reports/figures/diffusion_engage/<YYYY-MM>/*.png`, `reports/figures/legacy_linkpred/*.png` | Histograms, timelines, heatmaps, diffusion grids, LP bar charts, etc. |

## Package Tour

| Module | Responsibility |
| --- | --- |
| `steamNetwork/paths.py` | Centralized directory constants (`RAW`, `INTERIM`, `PROCESSED`, `FIG`, etc.). |
| `steamNetwork/config.py` | Dataclasses for affinity, diffusion, and link-prediction parameters + global seed. |
| `steamNetwork/utils.py` | Shared helpers (month flooring, RNG, parquet persistence). |
| `steamNetwork/data/` | Raw data loaders (`reviews`, `priceEvents`, SteamSpy stubs). |
| `steamNetwork/preProcess/` | Affinity projection (`compute_affinity`, `to_graph`), node enrichment, snapshot builders. |
| `steamNetwork/analysis/` | Core analytics: communities, change points, diffusion models/statistics, link prediction, embeddings, community evolution, advanced reporting. |
| `steamNetwork/models/embeddings.py` | Node2Vec wrapper that respects weighted edges. |
| `steamNetwork/visualizations/` | Matplotlib utilities for both basic (`plotting.py`) and advanced (`advancedPlotting.py`) figures. |

Each module can be imported independently for notebooks or ad-hoc exploration; the orchestration scripts simply wire them together.

## Customization & Extension

- **Run names**: Set `RUN_TAG` in `main.py` before execution to isolate outputs (`engage`, `wjaccard`, etc.).
- **Snapshot config**: Adjust `SnapshotConfig` (window size, min co-reviews, `weight_scheme`, engagement weights) to explore alternative affinity definitions.
- **Diffusion tuning**: Instantiate new `DiffusionStrategy` objects (IC vs LT, probability maps, caps) and pass them into `run_diffusion_engage.py` or bespoke notebooks.
- **Seed selection**: Tweak `SeedSelectionConfig` (discount thresholds, review windows, weighting) to test different promotional targeting rules.
- **Link prediction**: Modify `LinkPredParams` (embedding dimension, walk lengths) or extend `compare_link_prediction_methods` with additional classifiers.
- **Visual themes**: `steamNetwork/visualizations` exposes plotting primitives; add new charts without touching analytics code.
- **Additional data**: Drop future CSVs under `data/raw/` and adjust `build_node_attributes` or custom loaders to include them in node features.

## Troubleshooting

- **igraph/leidenalg build errors**: Install system libs first (`brew install igraph pkg-config`) and ensure you are using Python 3.10+. Reinstall via `pip install --no-binary :all: python-igraph`.
- **`snapshots_monthly.parquet` not found**: Run `python main.py` or supply `--snapshots` pointing to an existing parquet when invoking `run_diffusion_engage.py`.
- **High memory usage**: Reduce `SnapshotConfig.window_days`, process fewer months via `run_diffusion_engage.py --months`, or down-sample price events.
- **Empty diffusion seeds**: Lower `--min-discount`, increase `--top-k`, or verify `price_events_itad.csv` timestamps align with snapshot months (UTC month floors).
- **Sklearn convergence warnings**: Increase `max_iter` in the logistic regressions (both link prediction and hybrid models accept the change) or standardize features before fitting.
- **Figure overlaps**: Delete outdated PNGs under `reports/figures/<tag>/` before re-running if you want a clean slate; paths are deterministic so stale plots may persist otherwise.

With the steps above you can rerun the entire analytical stack, dig into individual modules, or extend the project for new research questions without touching unrelated components. Enjoy exploring the Steam network!

