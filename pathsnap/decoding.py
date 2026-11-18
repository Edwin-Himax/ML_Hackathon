"""Log-space hybrid HMM/Viterbi over candidate road positions.

Emissions come from the frozen classifier, transitions from the road graph.
No reference data is accepted. Gaps and impossible transitions restart chains.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from .graph import movement_limit


@dataclass
class Lattice:
    groups: list
    distances: list
    observations: np.ndarray
    time_steps: np.ndarray
    gps_steps: np.ndarray
    reverse_pairs: list


@dataclass
class Decoded:
    assignments: np.ndarray
    candidate_rows: np.ndarray
    break_before: np.ndarray
    resets: list


def prepare_lattice(trip, geometry, candidates, graph, config):
    groups = [np.empty(0, dtype=int) for _ in trip.track]
    observations = candidates.observation_id.to_numpy(dtype=int)
    if len(observations):
        ids, first, count = np.unique(observations, return_index=True, return_counts=True)
        for obs, start, size in zip(ids, first, count):
            groups[obs] = np.arange(start, start + size)
    dt = np.r_[0., np.diff(trip.track[:, 2])]
    gps_steps = np.r_[0., np.linalg.norm(np.diff(geometry.gps_xy, axis=0), axis=1)]
    arcs = candidates.arc_id.to_numpy(dtype=int)
    fractions = candidates.projection_fraction.to_numpy()
    states = np.column_stack((arcs, fractions))
    matrices, reverse = [None], [None]
    for t in range(1, len(groups)):
        a, b = groups[t - 1], groups[t]
        if not len(a) or not len(b) or dt[t] > config['gap_reset_s']:
            matrices.append(None); reverse.append(None)
            continue
        matrix = graph.distance_matrix(states[a], states[b], movement_limit(dt[t], config), config['backward_jitter_m'])
        matrices.append(matrix)
        pa, cb = trip.arcs[arcs[a]], trip.arcs[arcs[b]]
        reverse.append((pa[:, 0, None] == cb[None, :, 1]) & (pa[:, 1, None] == cb[None, :, 0]) & (arcs[a, None] != arcs[None, b]))
    return Lattice(groups, matrices, observations, dt, gps_steps, reverse)


def viterbi(trip, candidates, scores, lattice, config):
    n = len(trip.track)
    assignments = np.full(n, -1, dtype=np.int64)
    chosen_rows = np.full(n, -1, dtype=np.int64)
    breaks = np.zeros(n, dtype=bool)
    resets = []
    arcs = candidates.arc_id.to_numpy(dtype=int)
    log_emission = scores.log_emission_potential.to_numpy()
    active, parents, values = [], [], None

    def flush():
        if not active:
            return
        state = int(np.argmax(values))
        for index in range(len(active) - 1, -1, -1):
            obs, rows = active[index]
            row = int(rows[state])
            chosen_rows[obs] = row
            assignments[obs] = arcs[row]
            if index:
                state = int(parents[index][state])

    for obs, rows in enumerate(lattice.groups):
        if not len(rows):
            flush(); active, parents, values = [], [], None
            breaks[obs] = True
            resets.append({'observation_id': obs, 'reason': 'no_candidates'})
            continue
        emission = config['emission_weight'] * log_emission[rows]
        reason = None
        if not active:
            reason = 'initial' if obs == 0 else 'after_unmatched'
        elif lattice.time_steps[obs] > config['gap_reset_s']:
            reason = 'gps_gap'
        if reason is None:
            distances = lattice.distances[obs]
            transition = -np.abs(distances - lattice.gps_steps[obs]) / config['transition_beta_m']
            transition -= lattice.reverse_pairs[obs] * config['u_turn_penalty']
            normalizer = logsumexp(transition, axis=1, keepdims=True)
            transition = np.where(np.isfinite(normalizer), transition - np.where(np.isfinite(normalizer), normalizer, 0), -np.inf)
            proposed = values[:, None] + transition
            next_values = emission + proposed.max(axis=0)
            if not np.isfinite(next_values).any():
                reason = 'infeasible_transition'
            else:
                parents.append(proposed.argmax(axis=0))
                active.append((obs, rows))
                values = next_values - np.max(next_values)
                continue
        flush()
        active = [(obs, rows)]
        parents = [np.full(len(rows), -1, dtype=int)]
        values = emission - np.max(emission)
        breaks[obs] = True
        resets.append({'observation_id': obs, 'reason': reason})
    flush()
    return Decoded(assignments, chosen_rows, breaks, resets)


def classifier_decoded(trip, candidates, scores):
    n = len(trip.track)
    chosen = np.full(n, -1, dtype=int)
    assignments = np.full(n, -1, dtype=int)
    if len(scores):
        order = scores.sort_values(['observation_id', 'compatibility_probability', 'arc_id'], ascending=[True, False, True], kind='stable')
        selected = order.drop_duplicates('observation_id')
        obs = selected.observation_id.to_numpy(dtype=int)
        chosen[obs] = selected.index.to_numpy(dtype=int)
        assignments[obs] = selected.arc_id.to_numpy(dtype=int)
    return Decoded(assignments, chosen, np.zeros(n, dtype=bool), [])


def reconstruct(trip, geometry, candidates, graph, decoded, config):
    """Connected route pieces, original-arc sequence and label-free continuity."""
    coords = []
    pieces, sequences, current, sequence = [], [], [], []
    comparable = feasible = 0
    length_m = 0.
    connected_before = np.zeros(len(trip.track), dtype=bool)
    fractions = np.full(len(trip.track), np.nan)
    valid = decoded.candidate_rows >= 0
    fractions[valid] = candidates.projection_fraction.to_numpy()[decoded.candidate_rows[valid]]
    snapped = np.full((len(trip.track), 2), np.nan)
    for i, arc in enumerate(decoded.assignments):
        if arc >= 0:
            a, b = trip.nodes[trip.arcs[arc]]
            snapped[i] = a + fractions[i] * (b - a)
    def finish():
        if current:
            pieces.append(current.copy())
            sequences.append(sequence.copy())
    for t, arc in enumerate(decoded.assignments):
        dt = trip.track[t, 2] - trip.track[t - 1, 2] if t else 0.
        if t and dt <= config['gap_reset_s']:
            comparable += 1
        distance, path = np.inf, []
        if t and arc >= 0 and decoded.assignments[t - 1] >= 0 and dt <= config['gap_reset_s']:
            distance, path = graph.connection(decoded.assignments[t - 1], fractions[t - 1], arc, fractions[t], movement_limit(dt, config), config['backward_jitter_m'], with_path=True)
            if np.isfinite(distance):
                feasible += 1
        can_connect = np.isfinite(distance) and not decoded.break_before[t]
        if not can_connect:
            finish(); current, sequence = [], []
        if arc < 0:
            continue
        if not current:
            current = [snapped[t].tolist()]
            sequence = [int(arc)]
        else:
            connected_before[t] = True
            length_m += distance
            if len(path) > 1:
                for edge in path[:-1]:
                    current.append(trip.nodes[trip.arcs[edge, 1]].tolist())
                for edge in path:
                    if not sequence or sequence[-1] != edge:
                        sequence.append(int(edge))
            current.append(snapped[t].tolist())
    finish()
    return {
        'pieces': pieces, 'arc_sequences': sequences, 'snapped_coordinates': snapped,
        'connected_before': connected_before,
        'summary': {'matched_observations': int(valid.sum()), 'unmatched_observations': int((~valid).sum()),
                    'route_pieces': len(pieces), 'route_length_m': float(length_m),
                    'comparable_pairs': comparable, 'feasible_pairs': feasible,
                    'continuity': feasible / comparable if comparable else None,
                    'gap_pairs_excluded': int((np.diff(trip.track[:, 2]) > config['gap_reset_s']).sum())},
    }
