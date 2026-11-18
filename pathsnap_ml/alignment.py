"""OFFLINE reference alignment only. Never imported by inference.py.

This produces provisional labels from supplied ordered routes, not verified
per-observation ground truth. Member 1 must review maps/ambiguous cases or
replace these labels with an organiser-approved observation mapping.
"""
import numpy as np
import pandas as pd
from .features import motion
from .geometry import point_segments


def align_reference(trip, geometry, route, cfg):
    raw = np.asarray(route)
    if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError('Reference must be a nonempty integer sequence')
    route = raw.astype(int)
    if route.min() < 0 or route.max() >= len(trip.arcs):
        raise ValueError('Reference arc ID out of range')
    n = len(trip.track)
    k = min(cfg['nearest_occurrences'], len(route))
    starts, ends = geometry.starts[route], geometry.ends[route]
    length = geometry.lengths[route]
    offset = np.r_[0, np.cumsum(length)[:-1]]
    route_edges = trip.arcs[route]
    breaks = route_edges[:-1, 1] != route_edges[1:, 0]
    chunks = np.r_[0, np.cumsum(breaks)]
    break_points = np.concatenate((ends[:-1][breaks], starts[1:][breaks]))
    dt, speed, headings, hv, step = motion(trip.track, geometry.gps_xy)
    road_headings = np.arctan2((ends - starts)[:, 1], (ends - starts)[:, 0])
    occ = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)
    fractions = np.empty((n, k), dtype=np.float32)
    emission = np.empty((n, k), dtype=np.float32)
    margin = np.zeros(n)
    near_break = np.zeros(n, dtype=bool)
    for lo in range(0, n, 128):
        hi = min(n, lo + 128)
        dist, fraction, _ = point_segments(geometry.gps_xy[lo:hi, None, :], starts[None], ends[None])
        diff = np.abs(np.arctan2(np.sin(headings[lo:hi, None] - road_headings), np.cos(headings[lo:hi, None] - road_headings)))
        cost = np.minimum((dist / 15) ** 2, 400) + 2 * diff / np.pi * hv[lo:hi, None]
        order = np.argsort(cost, axis=1, kind='stable')[:, :k]
        occ[lo:hi] = order
        distances[lo:hi] = np.take_along_axis(dist, order, axis=1)
        fractions[lo:hi] = np.take_along_axis(fraction, order, axis=1)
        emission[lo:hi] = np.take_along_axis(cost, order, axis=1)
        ids = route[order]
        selected_cost = emission[lo:hi]
        other = np.where(ids != ids[:, :1], selected_cost, np.inf)
        margin[lo:hi] = np.min(other, axis=1) - selected_cost[:, 0]
        if len(break_points):
            near_break[lo:hi] = np.linalg.norm(geometry.gps_xy[lo:hi, None] - break_points[None], axis=2).min(axis=1) < cfg['break_exclusion_m']

    progress = offset[occ] + fractions * length[occ]
    cost_history = np.empty((n, k), dtype=np.float32)
    back = np.full((n, k), -1, dtype=np.int16)
    resets = np.zeros(n, dtype=bool)
    cost_history[0] = emission[0]
    resets[0] = True
    for t in range(1, n):
        prev = cost_history[t - 1]
        change = progress[t][None, :] - progress[t - 1][:, None]
        same_chunk = chunks[occ[t - 1]][:, None] == chunks[occ[t]][None, :]
        # Allow small GPS jitter, but not reverse traversal of an entire route.
        feasible = (change >= -15) & (change <= max(150, dt[t] * 60 + 50)) & same_chunk
        transition = np.abs(change - min(step[t], dt[t] * 60)) / 30
        values = np.where(feasible, prev[:, None] + transition, np.inf)
        if dt[t] > cfg['gap_reset_s'] or not np.isfinite(values).any():
            cost_history[t] = emission[t]
            resets[t] = True
        else:
            back[t] = np.argmin(values, axis=0)
            cost_history[t] = emission[t] + np.min(values, axis=0)
            cost_history[t] -= np.min(cost_history[t])
    chosen = np.empty(n, dtype=int)
    state = int(np.argmin(cost_history[-1]))
    for t in range(n - 1, -1, -1):
        chosen[t] = state
        parent = int(back[t, state])
        if t:
            state = parent if parent >= 0 else int(np.argmin(cost_history[t - 1]))
    idx = np.arange(n)
    selected_distance = distances[idx, chosen]
    locally_consistent = emission[idx, chosen] <= emission[:, 0] + 1.0
    eligible = (selected_distance <= cfg['max_label_distance_m']) & locally_consistent & (margin >= cfg['min_emission_margin']) & ~near_break
    # Exclude unverified alignment resets (first observation has a valid initial prior).
    eligible[resets & (idx != 0)] = False
    result = pd.DataFrame({
        'observation_id': idx, 'reference_arc_id': route[occ[idx, chosen]],
        'route_occurrence': occ[idx, chosen], 'reference_distance_m': selected_distance,
        'emission_margin': np.minimum(margin, 1e6), 'eligible': eligible,
        'alignment_reset': resets, 'near_reference_break': near_break,
        'local_consistency': locally_consistent,
    })
    return result
