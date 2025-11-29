from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class AffinityParams:
    """Hyperparameters for turning raw co-view data into weighted edges."""
    window_days: int = 30 # window size in days
    min_coreviews: int = 3 # minimum number of coreviews to include an edge
    weight_scheme: str = "wjaccard"  # ["count", "jaccard", "wjaccard", "pmi", "engagement"]
    normalize: bool = True # whether to normalize the weights
    engagement_weights: Optional[Mapping[str, float]] = None # weights for the engagement scheme


@dataclass
class DiffusionParams:
    """Controls the simulated diffusion process on the network."""
    base_p: float = 0.05 # base transmission probability
    alpha: float = 4.0 # strength of the edge weight -> probability mapping
    max_steps: int = 8 # maximum number of steps in the diffusion process
    n_sims: int = 100 # number of simulations


@dataclass
class LinkPredParams:
    """Configuration for random-walk embeddings and downstream classifiers."""
    train_ratio: float = 0.7 # ratio of training data
    neg_pos_ratio: float = 1.0 # ratio of negative to positive samples
    embed_dim: int = 128 # dimension of the embedding
    walk_length: int = 40 # length of each walk
    num_walks: int = 20 # number of walks per node
    p: float = 1.0 # explore parameter
    q: float = 1.0 # return parameter

# drawing
FIG_DPI = 140
SEED = 42