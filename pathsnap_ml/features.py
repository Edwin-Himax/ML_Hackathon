"""Inference-only features. No function accepts reference labels or route files."""
import numpy as np

FEATURE_NAMES = [
    'distance_m', 'heading_difference_rad', 'heading_valid',
    'projection_fraction', 'road_length_m', 'from_in_degree',
    'from_out_degree', 'to_in_degree', 'to_out_degree',
    'speed_mps', 'time_gap_s', 'distance_rank', 'relative_distance_m',
    'candidate_count', 'search_radius_m', 'zero_length_arc',
]


def motion(track, gps_xy):
    n = len(track)
    dt = np.zeros(n)
    displacement = np.zeros((n, 2))
    if n > 1:
        dt[1:] = np.diff(track[:, 2])
        displacement[1:] = np.diff(gps_xy, axis=0)
    distance = np.linalg.norm(displacement, axis=1)
    speed = np.divide(distance, dt, out=np.zeros(n), where=dt > 0)
    heading = np.arctan2(displacement[:, 1], displacement[:, 0])
    valid = (dt > 0) & (dt <= 60) & (distance >= 3) & (speed >= 1) & (speed <= 60)
    return dt, speed, heading, valid, distance


def make_features(trip, geometry, candidates):
    """Member 1 adapter input -> fixed-order float32 features for Member 3.

    Required candidate columns: observation_id, arc_id, distance_m,
    projection_fraction, distance_rank, search_radius_m. IDs are metadata,
    never features. Candidate rows must be contiguous by observation.
    """
    required = {
        'observation_id', 'arc_id', 'distance_m',
        'projection_fraction', 'distance_rank', 'search_radius_m',
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f'Missing candidate columns: {sorted(missing)}')
    obs = candidates.observation_id.to_numpy(dtype=int)
    arc = candidates.arc_id.to_numpy(dtype=int)
    if len(obs) and (np.any(np.diff(obs) < 0) or obs.min() < 0 or obs.max() >= len(trip.track) or arc.min() < 0 or arc.max() >= len(trip.arcs)):
        raise ValueError('Invalid or unsorted candidate keys')
    dt, speed, heading, valid, _ = motion(trip.track, geometry.gps_xy)
    road_heading = np.arctan2(geometry.vectors[arc, 1], geometry.vectors[arc, 0])
    road_valid = geometry.lengths[arc] > 1e-6
    hv = valid[obs] & road_valid
    diff = np.abs(np.arctan2(np.sin(heading[obs] - road_heading), np.cos(heading[obs] - road_heading)))
    diff[~hv] = 0
    starts, ends = trip.arcs[arc].T
    counts = np.bincount(obs, minlength=len(trip.track))
    minimum = np.full(len(trip.track), np.inf)
    np.minimum.at(minimum, obs, candidates.distance_m)
    columns = [
        candidates.distance_m, diff, hv.astype(float), candidates.projection_fraction,
        geometry.lengths[arc], geometry.in_degree[starts], geometry.out_degree[starts],
        geometry.in_degree[ends], geometry.out_degree[ends],
        np.minimum(speed[obs], 150), np.minimum(dt[obs], 3600), candidates.distance_rank,
        candidates.distance_m.to_numpy() - minimum[obs], counts[obs],
        candidates.search_radius_m, (~road_valid).astype(float),
    ]
    x = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    return x
