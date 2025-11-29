from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Literal, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ..preProcess.affinityProjection import to_graph
from ..utils import month_floor
from .diffusion import adoption_delta_around_events, review_counts_by_day


@dataclass
class SeedSelectionConfig:
    """Controls how discounted games are selected as diffusion seeds."""
    min_discount: float = 10.0 # minimum discount percentage to consider a game as a seed
    top_k: int | None = 15 # top k games to consider as seeds
    weight_by: str = "discount_pct" # weight by discount percentage
    include_negative_deltas: bool = False # include negative deltas
    review_window: int = 14 # review window in days


@dataclass
class DiffusionStrategy:
    """Diffusion hyper-parameters for either Independent Cascade or Linear Threshold."""
    name: str # name of the strategy
    model: Literal["ic", "lt"] = "ic" # model to use for the strategy
    base_p: float = 0.1 # base probability
    min_p: float = 0.01 # minimum probability
    max_p: float = 0.85 # maximum probability
    alpha: float = 4.0 # alpha parameter for the sigmoid function
    weight_scale: float = 10.0 # weight scale
    weight_shift: float = 0.5 # weight shift
    max_steps: int = 8 # maximum number of steps
    lt_threshold: float = 0.35 # linear threshold
    prob_map: Literal["sigmoid", "linear", "sqrt"] = "sigmoid" # probability map
    edge_weight_key: str = "weight" # edge weight key
    metadata: Dict[str, float] = field(default_factory=dict) # metadata


def _month_bounds(month: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the inclusive start and exclusive end timestamps for ``month``."""
    # Convert the month to a period and get the start and end timestamps
    period = pd.Timestamp(month).to_period("M")
    start = period.to_timestamp()
    end = (period + 1).to_timestamp()
    return start, end


def select_seeds(
    price_events: pd.DataFrame,
    month: pd.Timestamp,
    cfg: SeedSelectionConfig,
    adoption_deltas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a ranked seed table for the requested month."""
    if price_events is None or price_events.empty:
        return pd.DataFrame(columns=["app_id", "seed_score", "discount_pct"])
    
    # Convert the month to a period and get the start and end timestamps
    month = pd.Timestamp(month)
    target_month = month.to_period("M").to_timestamp("M")
    start, end = _month_bounds(month)
    pe = price_events.copy()

    # Convert the timestamp to a datetime and filter the price events for the target month
    pe["timestamp"] = pd.to_datetime(pe["timestamp"], utc=True)
    mask = (pe["timestamp"] >= start.tz_localize("UTC")) & (pe["timestamp"] < end.tz_localize("UTC"))
    month_events = pe.loc[mask].copy()
    if month_events.empty:
        return pd.DataFrame(columns=["app_id", "seed_score", "discount_pct"])

    # Convert the discount percentage to a float and filter the price events for the target month
    month_events["discount_pct"] = month_events["discount_pct"].astype(float)
    month_events = month_events[month_events["discount_pct"] >= cfg.min_discount]
    if month_events.empty:
        return pd.DataFrame(columns=["app_id", "seed_score", "discount_pct"])

    # Group by the app_id and get the maximum discount percentage
    grouped = month_events.groupby("app_id").agg({"discount_pct": "max"}).reset_index()

    if adoption_deltas is not None and not adoption_deltas.empty:
        # Convert the timestamp to a datetime and filter the adoption deltas for the target month
        adoption = adoption_deltas.copy()
        adoption["timestamp"] = pd.to_datetime(adoption["timestamp"], utc=True)
        adoption["month"] = month_floor(adoption["timestamp"])
        # Filter the adoption deltas for the target month
        adoption = adoption[adoption["month"] == target_month]
        # Group by the app_id and get the mean of the delta reviews
        adoption = adoption.groupby("app_id")["delta_reviews"].mean().reset_index()
        # Merge the grouped data with the adoption deltas
        grouped = grouped.merge(adoption, on="app_id", how="left")

    # Get the score column
    score_col = cfg.weight_by if cfg.weight_by in grouped.columns else "discount_pct"
    grouped["seed_score"] = grouped[score_col].fillna(0.0)
    # If negative deltas are not allowed, filter the grouped data
    if not cfg.include_negative_deltas:
        grouped = grouped[grouped["seed_score"] >= 0]

    # Sort the grouped data by the seed score
    grouped = grouped.sort_values("seed_score", ascending=False)
    if cfg.top_k:
        grouped = grouped.head(cfg.top_k)
    return grouped.reset_index(drop=True)


def _edge_value(G: nx.Graph, u: int, v: int, strategy: DiffusionStrategy) -> float:
    """Fetch the configured edge weight between ``u`` and ``v``."""
    attrs = G[u][v] 
    # Get the edge weight
    return float(attrs.get(strategy.edge_weight_key, attrs.get("weight", 0.0)))


def _map_probability(value: float, strategy: DiffusionStrategy) -> float:
    """Map a raw edge weight to an activation probability under the strategy."""
    
    # If the probability map is sigmoid, map the edge weight to an activation probability
    if strategy.prob_map == "sigmoid":
        scaled = strategy.weight_scale * value - strategy.weight_shift
        sig = 1.0 / (1.0 + math.exp(-strategy.alpha * scaled)) 
        prob = strategy.min_p + (strategy.max_p - strategy.min_p) * sig
    elif strategy.prob_map == "linear": # If the probability map is linear, map the edge weight to an activation probability
        prob = strategy.base_p + strategy.weight_scale * value
    elif strategy.prob_map == "sqrt": # If the probability map is sqrt, map the edge weight to an activation probability
        prob = strategy.base_p + strategy.weight_scale * math.sqrt(max(value, 0.0))
    else:
        prob = strategy.base_p
    return float(np.clip(prob, strategy.min_p, strategy.max_p))


def _ic_once(G: nx.Graph, seeds: Iterable[int], strategy: DiffusionStrategy, rng: np.random.Generator):
    """Run a single Independent Cascade rollout."""
    active = set(int(s) for s in seeds if s in G) # Get the active nodes
    if not active:
        return {"size": 0, "depth": 0, "timeline": {}}
    timeline: Dict[int, set[int]] = {0: set(active)} # Initialize the timeline
    frontier = set(active)
    step = 0 # Initialize the step
    # While the frontier is not empty and the step is less than the maximum number of steps
    while frontier and step < strategy.max_steps:
        step += 1
        newly_active: set[int] = set() # Initialize the newly active nodes
        # For each node in the frontier
        for u in frontier:
            for v in G.neighbors(u): # For each neighbor of the node
                if v in active:
                    continue
                p = _map_probability(_edge_value(G, u, v, strategy), strategy) # Map the edge weight to an activation probability
                if rng.random() < p: # If the random number is less than the activation probability, add the node to the newly active nodes
                    newly_active.add(v)
        if not newly_active:
            break
        timeline[step] = newly_active # Add the newly active nodes to the timeline
        active |= newly_active # Add the newly active nodes to the active nodes
        frontier = newly_active # Update the frontier
    return {
        "size": len(active),
        "depth": max(timeline.keys()),
        "timeline": {k: sorted(v) for k, v in timeline.items()},
    }


def _lt_once(G: nx.Graph, seeds: Iterable[int], strategy: DiffusionStrategy, rng: np.random.Generator):
    """Run a single Linear Threshold rollout."""
    active = set(int(s) for s in seeds if s in G) # Get the active nodes
    if not active:
        return {"size": 0, "depth": 0, "timeline": {}}

    # Get the thresholds
    thresholds = {
        node: float(np.clip(rng.normal(strategy.lt_threshold, 0.05), 0.05, 0.95))
        for node in G.nodes()
    }

    timeline: Dict[int, set[int]] = {0: set(active)} # Initialize the timeline
    step = 0
    # While the step is less than the maximum number of steps
    while step < strategy.max_steps:
        step += 1
        newly_active: set[int] = set() # Initialize the newly active nodes
        # For each node in the graph
        for node in G.nodes():
            if node in active:
                continue
            total_weight = sum(_edge_value(G, node, nbr, strategy) for nbr in G.neighbors(node)) # Get the total weight of the node
            if total_weight == 0:
                continue
            # Get the influence of the node
            influence = sum(
                _edge_value(G, node, nbr, strategy)
                for nbr in G.neighbors(node)
                if nbr in active
            )
            # Get the fraction of the total weight that is active
            frac = influence / total_weight
            # If the fraction is greater than or equal to the threshold, add the node to the newly active nodes
            if frac >= thresholds[node]:
                newly_active.add(node)
        newly_active -= active
        # If the newly active nodes are empty, break
        if not newly_active:
            break
        timeline[step] = newly_active # Add the newly active nodes to the timeline
        active |= newly_active # Add the newly active nodes to the active nodes
        active |= newly_active # Update the active nodes
    return {
        "size": len(active),
        "depth": max(timeline.keys()),
        "timeline": {k: sorted(v) for k, v in timeline.items()},
    }


def run_diffusion_batch(
    G: nx.Graph,
    seeds: Sequence[int],
    strategy: DiffusionStrategy,
    n_sims: int = 100,
    rng_seed: int = 42,
    timeline_samples: int = 5,
) -> tuple[pd.DataFrame, List[Dict[int, List[int]]]]:
    """Run multiple simulations for a single strategy."""

    # Initialize random number generator, records, and samples
    rng = np.random.default_rng(rng_seed)
    records = []
    samples: List[Dict[int, List[int]]] = []
    strategy_payload = asdict(strategy)
    strategy_payload.pop("name", None)

    # For each simulation
    for sim_id in range(n_sims):
        # based on the strategy model, run the appropriate diffusion model
        if strategy.model == "ic":
            out = _ic_once(G, seeds, strategy, rng)
        else:
            out = _lt_once(G, seeds, strategy, rng)

        # Create a record of the simulation
        record = {
            "strategy": strategy.name,
            "model": strategy.model,
            "sim_id": sim_id,
            "size": out["size"],
            "depth": out["depth"],
            **strategy_payload,
        }
        records.append(record)
        # If the number of samples is less than the number of timeline samples, add the timeline to the samples
        if len(samples) < timeline_samples:
            # Add the timeline to the samples
            samples.append(out["timeline"])
    return pd.DataFrame(records), samples


def summarize_diffusion(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation outcomes by strategy with basic stats."""
    if results.empty:
        return pd.DataFrame()
    
    # Group by the strategy and calculate the basic stats
    summary = (
        results.groupby("strategy")
        .agg(
            mean_size=("size", "mean"),
            std_size=("size", "std"),
            mean_depth=("depth", "mean"),
            propagation_rate=("depth", lambda d: (d > 0).mean()),
            sims=("size", "count"),
        )
        .reset_index()
    )
    
    # Merge the summary with the results
    summary = summary.merge(
        results.drop(columns=["size", "depth", "sim_id"])
        .drop_duplicates(subset=["strategy"]),
        on="strategy",
        how="left",
    )
    return summary


def make_strategy_grid(
    base: DiffusionStrategy,
    grid: Dict[str, Sequence[float]],
) -> List[DiffusionStrategy]:
    """Expand a parameter grid into concrete strategy objects."""
    
    keys = list(grid.keys()) # Get the keys
    combos = list(product(*(grid[k] for k in keys))) # Get the combinations
    strategies = []
    for combo in combos: # For each combination
        kwargs = {k: v for k, v in zip(keys, combo)}
        name_suffix = "_".join(f"{k}{v}" for k, v in zip(keys, combo)) # Get the name suffix
        
        strat_kwargs = asdict(base) # Get the base strategy kwargs
        strat_kwargs.update(kwargs) # Update the base strategy kwargs with the combination
        strat_kwargs["name"] = f"{base.name}_{name_suffix}" # Get the name
        strategies.append(DiffusionStrategy(**strat_kwargs))
    return strategies # Return the strategies


def monthly_hyperparam_sweep(
    snapshots: pd.DataFrame,
    price_events: pd.DataFrame,
    reviews: pd.DataFrame,
    month: pd.Timestamp,
    strategies: List[DiffusionStrategy],
    seed_cfg: SeedSelectionConfig | None = None,
    n_sims: int = 100,
) -> dict:
    """High-level helper: build graph, pick seeds, run strategies, and summarize."""
    if seed_cfg is None:
        seed_cfg = SeedSelectionConfig() # Get the seed configuration
    if price_events is None or price_events.empty:
        raise ValueError("Price events are required for diffusion experiments.")

    edges = snapshots.query("month == @month") # Get the edges
    if edges.empty:
        raise ValueError(f"No snapshot edges found for month {month}")
    G = to_graph(edges, threshold=0.0)

    adoption = None
    # If the reviews are not empty, calculate the adoption deltas
    if reviews is not None and not reviews.empty:
        reviews_daily = review_counts_by_day(reviews) # Get the reviews daily
        # Calculate the adoption deltas
        adoption = adoption_delta_around_events(
            reviews_daily,
            price_events,
            window_days=seed_cfg.review_window,
        )

    # Select the seeds
    seeds_df = select_seeds(price_events, month, seed_cfg, adoption)
    seed_ids = seeds_df["app_id"].astype(int).tolist()
    if not seed_ids:
        raise ValueError(f"No seeds selected for month {month}")

    all_results = []
    all_timelines: Dict[str, List[Dict[int, List[int]]]] = {}
    # For each strategy, run the diffusion batch
    for strategy in strategies:
        sims, timelines = run_diffusion_batch(
            G,
            seed_ids,
            strategy,
            n_sims=n_sims,
            rng_seed=42 + len(all_results),
        )
        all_results.append(sims)
        all_timelines[strategy.name] = timelines

    # Concatenate the results
    result_df = pd.concat(all_results, ignore_index=True)
    # Summarize the results
    summary_df = summarize_diffusion(result_df)
    # Rename the seed score column to weight
    seeds_df = seeds_df.rename(columns={"seed_score": "weight"})

    return {
        "results": result_df,
        "summary": summary_df,
        "seeds": seeds_df,
        "timelines": all_timelines,
        "graph": G,
    }

