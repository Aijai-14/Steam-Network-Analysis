from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score


def match_communities_across_time(
    comm_t1: pd.DataFrame,
    comm_t2: pd.DataFrame,
    min_overlap: float = 0.3
) -> Dict[int, List[int]]:
    """
    Match communities between two time steps based on node overlap.

    Uses Jaccard similarity to find the best matches between communities
    at consecutive time points.

    Parameters
    ----------
    comm_t1 : pd.DataFrame
        Community assignments at time t1 (columns: node, community)
    comm_t2 : pd.DataFrame
        Community assignments at time t2 (columns: node, community)
    min_overlap : float
        Minimum Jaccard similarity to consider a match (default: 0.3)

    Returns
    -------
    Dict[int, List[int]]
        Mapping from communities at t1 to matched communities at t2
        Format: {comm_t1_id: [comm_t2_id, comm_t2_id, ...]}
    """
    # Group nodes by community
    comm_t1_groups = comm_t1.groupby('community')['node'].apply(set).to_dict()
    comm_t2_groups = comm_t2.groupby('community')['node'].apply(set).to_dict()

    matches = defaultdict(list)

    # For each community at t1, find overlapping communities at t2
    for c1, nodes1 in comm_t1_groups.items():
        best_matches = []

        for c2, nodes2 in comm_t2_groups.items():
            # Compute Jaccard similarity
            intersection = len(nodes1 & nodes2)
            union = len(nodes1 | nodes2)

            if union > 0:
                jaccard = intersection / union
                if jaccard >= min_overlap:
                    best_matches.append((c2, jaccard))

        # Sort by Jaccard and keep all above threshold
        best_matches.sort(key=lambda x: x[1], reverse=True)
        matches[c1] = [c for c, _ in best_matches]

    return dict(matches)


def detect_community_events(
    comm_t1: pd.DataFrame,
    comm_t2: pd.DataFrame,
    min_overlap: float = 0.3
) -> pd.DataFrame:
    """
    Detect community evolution events between two time steps.

    Events detected:
    - Birth: New community appears in t2
    - Death: Community from t1 disappears in t2
    - Merge: Multiple t1 communities map to single t2 community
    - Split: Single t1 community maps to multiple t2 communities
    - Growth: Community gains members
    - Shrink: Community loses members
    - Stable: Community maintains similar membership

    Parameters
    ----------
    comm_t1 : pd.DataFrame
        Communities at time t1
    comm_t2 : pd.DataFrame
        Communities at time t2
    min_overlap : float
        Minimum overlap threshold for matching

    Returns
    -------
    pd.DataFrame
        Events with columns:
        - event_type: Type of event
        - comm_t1: Community ID(s) at t1 (comma-separated if multiple)
        - comm_t2: Community ID(s) at t2 (comma-separated if multiple)
        - size_t1: Size at t1
        - size_t2: Size at t2
        - jaccard: Jaccard similarity (for stable/growth/shrink events)
    """
    # Match communities
    forward_matches = match_communities_across_time(comm_t1, comm_t2, min_overlap)
    backward_matches = match_communities_across_time(comm_t2, comm_t1, min_overlap)

    # Get community sizes
    size_t1 = comm_t1.groupby('community')['node'].count().to_dict()
    size_t2 = comm_t2.groupby('community')['node'].count().to_dict()

    # Get node sets
    nodes_t1 = comm_t1.groupby('community')['node'].apply(set).to_dict()
    nodes_t2 = comm_t2.groupby('community')['node'].apply(set).to_dict()

    events = []

    # Detect deaths (communities in t1 with no matches in t2)
    for c1 in nodes_t1:
        if c1 not in forward_matches or len(forward_matches[c1]) == 0:
            events.append({
                'event_type': 'death',
                'comm_t1': str(c1),
                'comm_t2': '',
                'size_t1': size_t1[c1],
                'size_t2': 0,
                'jaccard': 0.0
            })

    # Detect births (communities in t2 with no matches in t1)
    for c2 in nodes_t2:
        if c2 not in backward_matches or len(backward_matches[c2]) == 0:
            events.append({
                'event_type': 'birth',
                'comm_t1': '',
                'comm_t2': str(c2),
                'size_t1': 0,
                'size_t2': size_t2[c2],
                'jaccard': 0.0
            })

    # Detect splits, merges, growth, shrink, stable
    for c1, matched_c2s in forward_matches.items():
        if len(matched_c2s) == 0:
            continue  # Already handled as death

        elif len(matched_c2s) == 1:
            c2 = matched_c2s[0]

            # Check if it's also a one-to-one match backwards
            if c2 in backward_matches and len(backward_matches[c2]) == 1:
                # One-to-one match: stable, growth, or shrink
                jaccard = len(nodes_t1[c1] & nodes_t2[c2]) / len(nodes_t1[c1] | nodes_t2[c2])
                size_change = size_t2[c2] - size_t1[c1]

                if size_change > size_t1[c1] * 0.2:  # Grew by >20%
                    event_type = 'growth'
                elif size_change < -size_t1[c1] * 0.2:  # Shrunk by >20%
                    event_type = 'shrink'
                else:
                    event_type = 'stable'

                events.append({
                    'event_type': event_type,
                    'comm_t1': str(c1),
                    'comm_t2': str(c2),
                    'size_t1': size_t1[c1],
                    'size_t2': size_t2[c2],
                    'jaccard': jaccard
                })

            else:
                # One-to-one forward, but many-to-one backward: merge
                events.append({
                    'event_type': 'merge',
                    'comm_t1': ','.join(map(str, backward_matches[c2])),
                    'comm_t2': str(c2),
                    'size_t1': sum(size_t1.get(c, 0) for c in backward_matches[c2]),
                    'size_t2': size_t2[c2],
                    'jaccard': np.mean([
                        len(nodes_t1.get(c, set()) & nodes_t2[c2]) / len(nodes_t1.get(c, set()) | nodes_t2[c2])
                        for c in backward_matches[c2]
                    ])
                })

        else:
            # One-to-many: split
            events.append({
                'event_type': 'split',
                'comm_t1': str(c1),
                'comm_t2': ','.join(map(str, matched_c2s)),
                'size_t1': size_t1[c1],
                'size_t2': sum(size_t2.get(c, 0) for c in matched_c2s),
                'jaccard': np.mean([
                    len(nodes_t1[c1] & nodes_t2.get(c, set())) / len(nodes_t1[c1] | nodes_t2.get(c, set()))
                    for c in matched_c2s
                ])
            })

    return pd.DataFrame(events)


def community_stability_score(
    comm_t1: pd.DataFrame,
    comm_t2: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute stability metrics for community structure over time.

    Metrics:
    - Adjusted Rand Index: Measures similarity of partitions (-1 to 1, higher is more stable)
    - Average Jaccard: Average Jaccard similarity of matched communities
    - Node retention rate: Fraction of nodes that stay in same matched community

    Parameters
    ----------
    comm_t1 : pd.DataFrame
        Communities at time t1
    comm_t2 : pd.DataFrame
        Communities at time t2

    Returns
    -------
    Dict[str, float]
        Stability metrics:
        - ari: Adjusted Rand Index
        - avg_jaccard: Average Jaccard similarity
        - retention_rate: Node retention rate
        - persistent_communities: Fraction of communities that persist
    """
    # Get common nodes between time steps
    common_nodes = set(comm_t1['node']) & set(comm_t2['node'])

    if len(common_nodes) == 0:
        return {
            'ari': 0.0,
            'avg_jaccard': 0.0,
            'retention_rate': 0.0,
            'persistent_communities': 0.0
        }

    # Filter to common nodes
    comm_t1_filtered = comm_t1[comm_t1['node'].isin(common_nodes)].sort_values('node')
    comm_t2_filtered = comm_t2[comm_t2['node'].isin(common_nodes)].sort_values('node')

    # Compute Adjusted Rand Index
    ari = adjusted_rand_score(
        comm_t1_filtered['community'].values,
        comm_t2_filtered['community'].values
    )

    # Compute average Jaccard for matched communities
    matches = match_communities_across_time(comm_t1, comm_t2, min_overlap=0.1)
    nodes_t1 = comm_t1.groupby('community')['node'].apply(set).to_dict()
    nodes_t2 = comm_t2.groupby('community')['node'].apply(set).to_dict()

    jaccards = []
    # for each community at t1, compute the average Jaccard similarity for the matched communities at t2
    for c1, matched_c2s in matches.items():
        for c2 in matched_c2s:
            jaccard = len(nodes_t1[c1] & nodes_t2[c2]) / len(nodes_t1[c1] | nodes_t2[c2])
            jaccards.append(jaccard)

    avg_jaccard = np.mean(jaccards) if jaccards else 0.0 # compute the average Jaccard similarity

    # Compute persistence rate
    n_persistent = len([c1 for c1, matched in matches.items() if len(matched) > 0])
    persistent_rate = n_persistent / len(nodes_t1) if len(nodes_t1) > 0 else 0.0

    # Compute node retention (nodes that stay in the same matched community)
    node_to_comm_t1 = comm_t1.set_index('node')['community'].to_dict()
    node_to_comm_t2 = comm_t2.set_index('node')['community'].to_dict()

    retained = 0
    # for each node in the common nodes, check if it stays in the same matched community
    for node in common_nodes:
        c1 = node_to_comm_t1[node] # get the community at t1
        c2 = node_to_comm_t2[node] # get the community at t2
        # Check if c2 is a match for c1
        if c1 in matches and c2 in matches[c1]:
            retained += 1 # increment the retained count

    retention_rate = retained / len(common_nodes) if len(common_nodes) > 0 else 0.0

    return {
        'ari': float(ari),
        'avg_jaccard': float(avg_jaccard),
        'retention_rate': float(retention_rate),
        'persistent_communities': float(persistent_rate)
    }


def track_community_evolution(
    snapshots: pd.DataFrame,
    communities_by_month: Dict[pd.Timestamp, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Track community evolution across all temporal snapshots.

    Parameters
    ----------
    snapshots : pd.DataFrame
        Monthly network snapshots
    communities_by_month : Dict[pd.Timestamp, pd.DataFrame]
        Community assignments for each month

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - events_df: DataFrame of all community events
        - stability_df: DataFrame of stability metrics between consecutive months
    """
    months = sorted(communities_by_month.keys())

    all_events = []
    stability_scores = []

    # for each month, compute the community events and stability
    for i in range(len(months) - 1):
        t1 = months[i] # get the first month
        t2 = months[i + 1] # get the second month

        comm_t1 = communities_by_month[t1] # get the communities at t1
        comm_t2 = communities_by_month[t2] # get the communities at t2

        # Detect events
        events = detect_community_events(comm_t1, comm_t2)
        events['month_t1'] = t1
        events['month_t2'] = t2
        all_events.append(events)

        # Compute stability
        stability = community_stability_score(comm_t1, comm_t2)
        stability['month_t1'] = t1
        stability['month_t2'] = t2
        stability_scores.append(stability)

    # concatenate the events and stability scores
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    stability_df = pd.DataFrame(stability_scores) # convert the stability scores to a dataframe

    return events_df, stability_df


def summarize_evolution_statistics(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize community evolution events across all time steps.

    Parameters
    ----------
    events_df : pd.DataFrame
        All evolution events from track_community_evolution

    Returns
    -------
    pd.DataFrame
        Summary with counts of each event type per time step
    """
    if events_df.empty:
        return pd.DataFrame()

    # group the events by the month and event type and count the number of events
    summary = events_df.groupby(['month_t1', 'month_t2', 'event_type']).size().reset_index(name='count')

    # pivot the summary for easier reading
    pivot = summary.pivot_table(
        index=['month_t1', 'month_t2'],
        columns='event_type',
        values='count',
        fill_value=0
    ).reset_index()

    return pivot
