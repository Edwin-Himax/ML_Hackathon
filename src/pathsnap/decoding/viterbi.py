import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.pathsnap.schemas import CandidateObservation, DecodedPoint


@dataclass
class HMMConfig:
    beta_m: float = 50.0          # Scale parameter (meters) for |d_net - d_gps| penalty
    max_speed_mps: float = 45.0   # Feasibility speed ceiling (~162 km/h)
    max_gap_sec: float = 30.0     # Elapsed time threshold triggering trellis reset
    eps: float = 1e-9             # Floor for classifier probabilities before log


TransitionDistFn = Callable[[int, float, int, float], Optional[float]]
GpsDistFn = Callable[[float, float, float, float], float]


def default_haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two (lon, lat) coordinates."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_log_emissions(candidates: List[Dict], eps: float = 1e-9) -> List[float]:
    """Candidate-normalized log emission potentials from binary classifier scores."""
    raw = [max(float(c.get("score", eps)), eps) for c in candidates]
    total = sum(raw)
    return [math.log(p / total) for p in raw]


def compute_log_transition(
    cand_prev: Dict,
    cand_curr: Dict,
    dt_sec: float,
    d_gps_m: float,
    transition_dist_fn: TransitionDistFn,
    config: HMMConfig,
) -> float:
    """Log transition probability; returns -inf if disconnected or infeasible."""
    if dt_sec <= 0:
        dt_sec = 1e-3

    d_net_m = transition_dist_fn(
        int(cand_prev["arc_id"]),
        float(cand_prev.get("proj_fraction", 0.5)),
        int(cand_curr["arc_id"]),
        float(cand_curr.get("proj_fraction", 0.5)),
    )

    if d_net_m is None or math.isinf(d_net_m) or d_net_m < 0:
        return -math.inf

    if (d_net_m / dt_sec) > config.max_speed_mps:
        return -math.inf

    return -abs(d_net_m - d_gps_m) / config.beta_m


def decode_classifier_only(
    observations: List[CandidateObservation],
) -> List[DecodedPoint]:
    """Baseline classifier-only predictions (argmax score) for HMM ablation."""
    results: List[DecodedPoint] = []
    for obs in observations:
        if not obs.candidates:
            results.append(
                DecodedPoint(
                    observation_id=obs.observation_id,
                    timestamp=obs.timestamp,
                    gps_lon=obs.gps_lon,
                    gps_lat=obs.gps_lat,
                    arc_id=None,
                    proj_fraction=None,
                    status="unmatched",
                )
            )
            continue

        best = max(obs.candidates, key=lambda c: float(c.get("score", 0.0)))
        results.append(
            DecodedPoint(
                observation_id=obs.observation_id,
                timestamp=obs.timestamp,
                gps_lon=obs.gps_lon,
                gps_lat=obs.gps_lat,
                arc_id=int(best["arc_id"]),
                proj_fraction=float(best.get("proj_fraction", 0.5)),
                status="matched",
            )
        )
    return results


def viterbi_decode(
    observations: List[CandidateObservation],
    transition_dist_fn: TransitionDistFn,
    config: Optional[HMMConfig] = None,
    gps_dist_fn: GpsDistFn = default_haversine_m,
) -> Tuple[List[DecodedPoint], List[str]]:
    """Log-space Viterbi route reconstruction with gap and discontinuity resets."""
    cfg = config or HMMConfig()
    warnings: List[str] = []
    n = len(observations)
    if n == 0:
        return [], warnings

    v_table: List[List[float]] = [[] for _ in range(n)]
    backptr: List[List[Optional[int]]] = [[] for _ in range(n)]
    step_status: List[str] = ["matched"] * n
    prev_valid_idx: Optional[int] = None

    for t, obs in enumerate(observations):
        cands = obs.candidates
        if not cands:
            step_status[t] = "unmatched"
            warnings.append(
                f"Obs {obs.observation_id} (t={obs.timestamp}s): no candidate arcs; marked unmatched."
            )
            continue

        log_emis = compute_log_emissions(cands, eps=cfg.eps)
        k_curr = len(cands)

        if prev_valid_idx is None:
            v_table[t] = list(log_emis)
            backptr[t] = [None] * k_curr
            prev_valid_idx = t
            continue

        prev_obs = observations[prev_valid_idx]
        dt_sec = obs.timestamp - prev_obs.timestamp

        if dt_sec > cfg.max_gap_sec:
            v_table[t] = list(log_emis)
            backptr[t] = [None] * k_curr
            step_status[t] = "gap_reset"
            warnings.append(
                f"Obs {obs.observation_id}: GPS gap {dt_sec:.1f}s > {cfg.max_gap_sec}s; trellis reset."
            )
            prev_valid_idx = t
            continue

        d_gps_m = gps_dist_fn(
            prev_obs.gps_lon, prev_obs.gps_lat, obs.gps_lon, obs.gps_lat
        )
        prev_cands = prev_obs.candidates
        prev_v = v_table[prev_valid_idx]

        curr_v = [-math.inf] * k_curr
        curr_ptr: List[Optional[int]] = [None] * k_curr
        any_feasible = False

        for j, cand_j in enumerate(cands):
            best_score = -math.inf
            best_i: Optional[int] = None
            for i, cand_i in enumerate(prev_cands):
                if math.isinf(prev_v[i]):
                    continue
                log_tr = compute_log_transition(
                    cand_i, cand_j, dt_sec, d_gps_m, transition_dist_fn, cfg
                )
                if math.isinf(log_tr):
                    continue
                score = prev_v[i] + log_tr
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i is not None:
                curr_v[j] = log_emis[j] + best_score
                curr_ptr[j] = best_i
                any_feasible = True

        if not any_feasible:
            v_table[t] = list(log_emis)
            backptr[t] = [None] * k_curr
            step_status[t] = "gap_reset"
            warnings.append(
                f"Obs {obs.observation_id}: no feasible road connection from obs {prev_obs.observation_id}; route split."
            )
        else:
            v_table[t] = curr_v
            backptr[t] = curr_ptr

        prev_valid_idx = t

    # Backtrack across contiguous trellis segments
    chosen_cand_idx: List[Optional[int]] = [None] * n
    valid_indices = [t for t in range(n) if step_status[t] != "unmatched"]

    if valid_indices:
        t = valid_indices[-1]
        while t is not None:
            best_j = max(range(len(v_table[t])), key=lambda idx: v_table[t][idx])
            chosen_cand_idx[t] = best_j

            curr_t = t
            curr_j = best_j
            while curr_t is not None and backptr[curr_t][curr_j] is not None:
                prev_j = backptr[curr_t][curr_j]
                prev_t = max(idx for idx in valid_indices if idx < curr_t)
                chosen_cand_idx[prev_t] = prev_j
                curr_t = prev_t
                curr_j = prev_j

            earlier = [idx for idx in valid_indices if idx < curr_t]
            t = earlier[-1] if earlier else None

    decoded: List[DecodedPoint] = []
    for t, obs in enumerate(observations):
        idx = chosen_cand_idx[t]
        if idx is None:
            decoded.append(
                DecodedPoint(
                    observation_id=obs.observation_id,
                    timestamp=obs.timestamp,
                    gps_lon=obs.gps_lon,
                    gps_lat=obs.gps_lat,
                    arc_id=None,
                    proj_fraction=None,
                    status="unmatched",
                )
            )
        else:
            cand = obs.candidates[idx]
            decoded.append(
                DecodedPoint(
                    observation_id=obs.observation_id,
                    timestamp=obs.timestamp,
                    gps_lon=obs.gps_lon,
                    gps_lat=obs.gps_lat,
                    arc_id=int(cand["arc_id"]),
                    proj_fraction=float(cand.get("proj_fraction", 0.5)),
                    status=step_status[t],
                )
            )

    return decoded, warnings