"""Temporary Member 1 adapter; inference functions never open .route files.

Distances use a per-trip spherical azimuthal-equidistant projection, radius
6371008.8 m. It is a local spherical approximation, not an ellipsoidal CRS.
The segment index samples every edge at <= spacing: querying samples within
radius + spacing/2 guarantees inclusion of any segment within radius, before
exact planar point-to-segment filtering. Directed parallel edges remain distinct.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6371008.8


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def load_trip(folder):
    folder = Path(folder)
    tid = folder.name
    track = np.loadtxt(folder / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(folder / f"{tid}.nodes", ndmin=2)
    raw = np.loadtxt(folder / f"{tid}.arcs", ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or raw.shape[1] != 2:
        raise ValueError(f"Invalid columns in trip {tid}")
    if not all(np.isfinite(x).all() for x in (track, nodes, raw)):
        raise ValueError(f"Nonfinite inputs in trip {tid}; preserve IDs and resolve upstream")
    if np.any(np.abs(track[:, 0]) > 180) or np.any(np.abs(track[:, 1]) > 90) or np.any(np.abs(nodes[:, 0]) > 180) or np.any(np.abs(nodes[:, 1]) > 90):
        raise ValueError("Invalid longitude/latitude")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Arc node IDs must be integers")
    arcs = raw.astype(np.int64)
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError("Invalid node index")
    if len(track) > 1 and np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError("Timestamps must increase; repair upstream without losing row IDs")
    return Trip(tid, track, nodes, arcs)


def project(lonlat, origin):
    """Spherical AEQD, with an explicit trip origin (longitude, latitude)."""
    lon, lat = np.radians(np.asarray(lonlat)).T
    lon0, lat0 = np.radians(origin)
    dl = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    cosc = np.clip(np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dl), -1, 1)
    c = np.arccos(cosc)
    if np.any(c > np.radians(10)):
        raise ValueError("Trip extent exceeds 10 degrees from projection origin; Member 1 must choose a projection")
    k = np.divide(c, np.sin(c), out=np.ones_like(c), where=c > 1e-9)
    return EARTH_RADIUS_M * np.column_stack((k * np.cos(lat) * np.sin(dl), k * (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dl))))


def point_segments(point, starts, ends):
    v = ends - starts
    sq = (v * v).sum(axis=-1)
    numerator = ((point - starts) * v).sum(axis=-1)
    f = np.divide(numerator, sq, out=np.zeros_like(numerator), where=sq > 0).clip(0, 1)
    projected = starts + f[..., None] * v
    return np.linalg.norm(point - projected, axis=-1), f, projected


class Geometry:
    def __init__(self, trip, spacing_m=30.0):
        self.origin = tuple(np.median(trip.track[:, :2], axis=0))
        self.nodes_xy = project(trip.nodes, self.origin)
        self.gps_xy = project(trip.track[:, :2], self.origin)
        self.starts = self.nodes_xy[trip.arcs[:, 0]]
        self.ends = self.nodes_xy[trip.arcs[:, 1]]
        self.vectors = self.ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.in_degree = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
        self.out_degree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
        self.spacing_m = spacing_m
        pieces = np.maximum(1, np.ceil(self.lengths / spacing_m).astype(int))
        counts = pieces + 1
        self.sample_arcs = np.repeat(np.arange(len(pieces)), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        fractions = (np.arange(counts.sum()) - offsets) / pieces[self.sample_arcs]
        samples = self.starts[self.sample_arcs] + fractions[:, None] * self.vectors[self.sample_arcs]
        self.index = cKDTree(samples)

    def candidates(self, trip, cfg):
        """Return geometry rows with original observation and arc IDs intact."""
        radii = [cfg['radius_m'], *cfg['fallback_radii_m']]
        query_lists = self.index.query_ball_point(self.gps_xy, radii[0] + self.spacing_m / 2)
        rows = []
        for obs, (p, sample_ids) in enumerate(zip(self.gps_xy, query_lists)):
            for radius in radii:
                if radius != radii[0]:
                    sample_ids = self.index.query_ball_point(p, radius + self.spacing_m / 2)
                arc_ids = np.unique(self.sample_arcs[sample_ids])
                if len(arc_ids):
                    dist, frac, _ = point_segments(p, self.starts[arc_ids], self.ends[arc_ids])
                    order = np.lexsort((arc_ids, dist))
                    order = order[dist[order] <= radius][:cfg['max_candidates']]
                    if len(order):
                        for rank, j in enumerate(order):
                            rows.append((obs, int(arc_ids[j]), float(dist[j]), float(frac[j]), rank, radius))
                        break
        import pandas as pd
        return pd.DataFrame(rows, columns=['observation_id', 'arc_id', 'distance_m', 'projection_fraction', 'distance_rank', 'search_radius_m']).astype({'observation_id': 'int64', 'arc_id': 'int64', 'distance_rank': 'int64'})
