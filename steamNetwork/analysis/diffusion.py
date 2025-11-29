"""Milestone Report diffusion code for non-engagement weighted affinity graphs."""

from __future__ import annotations
import math
import random
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Set, Iterable, Tuple
from ..preProcess.affinityProjection import to_graph
import numpy as np


@dataclass
class ICParams:
    base_p: float = 0.15 # base probability
    alpha: float = 4.0 # alpha parameter for the sigmoid function
    max_steps: int = 8 # maximum number of steps

random.seed(42)


def _edge_prob(weight: float, base_p: float, alpha: float) -> float:
    """Map an edge weight to an activation probability via scaled sigmoid."""
    scaled_weight = weight * 10 # scale the weight by 10
    return max(0.0, min(1.0, base_p + (1.0 / (1.0 + math.exp(-alpha * (scaled_weight - 0.5))) - 0.5)))


def independent_cascade(G: nx.Graph, seeds: Iterable[int], params: ICParams) -> Dict[str, object]:
    """Run a single IC simulation and return cascade info."""
    active: Set[int] = set(int(s) for s in seeds if s in G)
    t = 0
    timeline: Dict[int, Set[int]] = {0: set(active)}
    newly_active = set(active)
    while newly_active and t < params.max_steps: # while the newly active nodes are not empty and the step is less than the maximum number of steps
        t += 1
        next_active: Set[int] = set()
        for u in newly_active: # for each node in the newly active nodes
            for v in G.neighbors(u): # for each neighbor of the node
                if v in active:
                    continue
                w = float(G[u][v].get("weight", 0.0)) 
                p = _edge_prob(w, params.base_p, params.alpha) # map the edge weight to an activation probability
                if random.random() < p: # if the random number is less than the activation probability, add the node to the next active nodes
                    next_active.add(v)
        newly_active = next_active
        active |= newly_active # add the newly active nodes to the active nodes
        if newly_active:
            timeline[t] = set(newly_active)
    return {
        "size": len(active),
        "depth": max(timeline.keys()),
        "timeline": {k: list(v) for k, v in timeline.items()},
    }


def simulate_discount_cascades(
    edges_monthly: pd.DataFrame,
    price_events: pd.DataFrame,
    month,
    params: ICParams,
    n_sims: int = 100,
) -> pd.DataFrame:
    """Run IC simulations for all discounted games in ``month``."""

    edges = edges_monthly.query("month == @month")[["i","j","w", "co"]] # get the edges for the month
    G = to_graph(edges, threshold=0.0)

    # get the seeds for the month that have a discount
    seeds = price_events.query("timestamp.dt.tz_localize(None).dt.to_period('M').dt.to_timestamp('M') == @month and discount_pct > 0")[
        "app_id"
    ].unique().tolist()

    if not seeds:
        return pd.DataFrame(columns=["sim_id","size","depth"])  # nothing discounted this month

    rows = []
    # for each simulation, run the independent cascade simulation
    for s in range(n_sims): 
        out = independent_cascade(G, seeds, params) # run the independent cascade simulation
        rows.append({"sim_id": s, "size": out["size"], "depth": out["depth"]})
    return pd.DataFrame(rows)


def review_counts_by_day(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews per app per day (proxy for adoption)."""
    r = reviews_df.copy()
    # convert the timestamp to a day
    r["day"] = pd.to_datetime(r["timestamp"], utc=True).dt.floor("D") 
    return r.groupby(["app_id","day"], as_index=False).size().rename(columns={"size":"reviews"}) # group by the app_id and day and count the number of reviews


def adoption_delta_around_events(reviews_daily: pd.DataFrame,
                                 events_df: pd.DataFrame,
                                 window_days: int = 14) -> pd.DataFrame:
    """For each discount event, compute pre/post review counts in ±window."""
    rows = []
    # for each event, compute the pre/post review counts in ±window
    for ev in events_df.itertuples(index=False):
        t0 = pd.to_datetime(ev.timestamp, utc=True) # convert the timestamp to a datetime
        app = int(ev.app_id) # convert the app_id to an integer
        block = reviews_daily.query("app_id == @app") # get the reviews for the app
        if block.empty: # if the block is empty, continue
            continue
        # get the pre/post review counts in ±window
        pre = block[(block["day"] >= t0 - pd.Timedelta(days=window_days)) &
                    (block["day"] <  t0)]["reviews"].sum() # get the pre review counts
        post = block[(block["day"] >  t0) &
                     (block["day"] <= t0 + pd.Timedelta(days=window_days))]["reviews"].sum() # get the post review counts
        rows.append(dict(app_id=app, timestamp=t0, discount_pct=ev.discount_pct, # add the row to the rows
                         pre_reviews=int(pre), post_reviews=int(post), # add the pre/post review counts
                         delta_reviews=int(post - pre))) # add the delta review counts
    return pd.DataFrame(rows)


def _degree_matched_random_seeds(G: nx.Graph, seeds: list[int], rng: np.random.Generator) -> list[int]:
    """Sample random seeds matching the degree quantiles of the real seeds (null)."""
    deg = pd.DataFrame(G.degree(), columns=["node","deg"]) # get the degree of the nodes
    q = deg["deg"].rank(pct=True) # get the rank of the degree
    qmap = dict(zip(deg["node"], q)) # get the map of the degree to the rank
    seed_qs = np.array([qmap.get(s, 0.5) for s in seeds])
    bins = np.clip(np.floor(seed_qs * 10), 0, 9).astype(int) # get the bins
    sampled = []
    for b in bins: # for each bin
        pool = deg[(np.floor(q*10).astype(int) == b)]["node"].tolist() # get the pool of nodes in the bin
        if pool:
            sampled.append(int(rng.choice(pool))) # sample a random node from the pool
    return sampled


def simulate_null_cascades(edges_monthly: pd.DataFrame,
                           true_seeds: list[int],
                           params: ICParams,
                           n_sims: int = 200,
                           mode: str = "degree_matched",
                           seed: int = 42) -> pd.DataFrame:
    """Null IC: random seeds matched on degree quantiles in the same graph."""
    from steamNetwork.preProcess.affinityProjection import to_graph  # adjust import if named differently
    rng = np.random.default_rng(seed) # initialize the random number generator
    G = to_graph(edges_monthly[["i","j","w", "co"]], threshold=0.0) # create the graph
    sims = [] # initialize the simulations
    for s in range(n_sims): # for each simulation
        if mode == "degree_matched": # if the mode is degree matched
            seeds = _degree_matched_random_seeds(G, true_seeds, rng) # sample the seeds
        else: # if the mode is not degree matched
            seeds = rng.choice(list(G.nodes()), size=len(true_seeds), replace=False).tolist() # sample the seeds
        out = independent_cascade(G, seeds, params) # run the independent cascade simulation
        sims.append(dict(sim_id=s, size=out["size"], depth=out["depth"])) # add the simulation to the simulations
    return pd.DataFrame(sims) # return the simulations
