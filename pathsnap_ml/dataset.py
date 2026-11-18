"""Offline cache builder and persisted trip-grouped folds."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .geometry import load_trip, Geometry
from .features import make_features, FEATURE_NAMES
from .alignment import align_reference

CACHE_VERSION = 'member2-v1'


def fingerprint(folder, config):
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(json.dumps({'candidate': config['candidate'], 'alignment': config['alignment'], 'features': FEATURE_NAMES}, sort_keys=True).encode())
    # Include preprocessing implementation in cache identity, not just file timestamps.
    for name in ['geometry.py', 'features.py', 'alignment.py', 'dataset.py']:
        h.update((Path(__file__).parent / name).read_bytes())
    for ext in ['track', 'nodes', 'arcs', 'route']:
        h.update((folder / f'{folder.name}.{ext}').read_bytes())
    return h.hexdigest()


def prepare(data_root, result_root, config, limit=None):
    data_root, result_root = Path(data_root), Path(result_root)
    cache = result_root / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    partitions = {}
    inventory = []
    for partition in ['train', 'student_test']:
        trips = []
        folders = sorted(p for p in (data_root / partition).iterdir() if p.is_dir())
        if limit:
            folders = folders[:limit]
        for folder in folders:
            start = time.perf_counter()
            key = fingerprint(folder, config)
            dest = cache / f'{partition}_{folder.name}_{key[:16]}.npz'
            if not dest.exists():
                trip = load_trip(folder)
                geo = Geometry(trip, config['candidate']['index_spacing_m'])
                candidates = geo.candidates(trip, config['candidate'])
                x = make_features(trip, geo, candidates)
                raw_route = np.loadtxt(folder / f'{folder.name}.route', ndmin=1)
                alignment = align_reference(trip, geo, raw_route, config['alignment'])
                obs = candidates.observation_id.to_numpy(dtype=np.int32)
                arc = candidates.arc_id.to_numpy(dtype=np.int32)
                ref = alignment.reference_arc_id.to_numpy(dtype=np.int32)
                eligible = alignment.eligible.to_numpy(dtype=bool)
                y = (arc == ref[obs]).astype(np.int8)
                np.savez_compressed(dest, x=x, y=y, observation_id=obs, arc_id=arc,
                                    reference_arc_id=ref, eligible=eligible,
                                    n_observations=len(trip.track), fingerprint=key)
                audit = result_root / 'alignment_audits'
                audit.mkdir(exist_ok=True)
                alignment['longitude'] = trip.track[:, 0]
                alignment['latitude'] = trip.track[:, 1]
                alignment['timestamp_s'] = trip.track[:, 2]
                alignment.to_csv(audit / f'{partition}_{folder.name}.csv', index=False)
                del geo
            with np.load(dest) as d:
                hit = np.zeros(int(d['n_observations']), dtype=bool)
                hit[d['observation_id'][d['y'] == 1]] = True
                eligible = d['eligible']
                counts = np.bincount(d['observation_id'], minlength=len(hit))
                row = {'partition': partition, 'trip_id': folder.name,
                       'observations': len(hit), 'candidate_rows': len(d['x']),
                       'eligible_aligned_observations': int(eligible.sum()),
                       'alignment_excluded_observations': int((~eligible).sum()),
                       'candidate_misses_eligible': int((eligible & ~hit).sum()),
                       'observations_without_candidates': int((counts == 0).sum()),
                       'fingerprint': key, 'preparation_s': time.perf_counter() - start}
            inventory.append(row)
            trips.append({'trip_id': folder.name, 'partition': partition, 'path': str(dest.resolve()), 'fingerprint': key})
            print(f"Prepared {partition}/{folder.name}: eligible={row['eligible_aligned_observations']}/{row['observations']}, candidates={row['candidate_rows']}", flush=True)
        partitions[partition] = trips
    train_ids = {t['trip_id'] for t in partitions['train']}
    if train_ids & {t['trip_id'] for t in partitions['student_test']}:
        raise ValueError('Partition trip IDs overlap')
    pd.DataFrame(inventory).to_csv(result_root / 'preparation_summary.csv', index=False)
    (result_root / 'cache_manifest.json').write_text(json.dumps(partitions, indent=2), encoding='utf-8')
    return partitions


def make_folds(train, n_splits, seed):
    ids = np.array([t['trip_id'] for t in train])
    # Shuffling the trip list first provides compatibility with sklearn <1.6.
    shuffled = np.random.default_rng(seed).permutation(len(ids))
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for number, (a, b) in enumerate(splitter.split(shuffled, groups=ids[shuffled])):
        ti, vi = shuffled[a], shuffled[b]
        assert not set(ids[ti]) & set(ids[vi])
        folds.append({'fold': number, 'train_trip_ids': sorted(ids[ti].tolist()), 'validation_trip_ids': sorted(ids[vi].tolist())})
    return folds


def observation_sample(eligible, maximum):
    selected = np.flatnonzero(eligible)
    if maximum and len(selected) > maximum:
        selected = selected[np.linspace(0, len(selected) - 1, maximum).astype(int)]
    return selected


def training_matrix(trips, maximum):
    xx, yy, keys = [], [], []
    for trip in trips:
        with np.load(trip['path']) as d:
            eligible = d['eligible'].copy()
            hit = np.zeros(len(eligible), dtype=bool)
            hit[d['observation_id'][d['y'] == 1]] = True
            # No positive candidate -> no usable supervised observation group.
            obs = observation_sample(eligible & hit, maximum)
            keep = np.isin(d['observation_id'], obs)
            xx.append(d['x'][keep]); yy.append(d['y'][keep])
            keys.append(np.column_stack((np.full(keep.sum(), int(trip['trip_id'])), d['observation_id'][keep])))
    x, y = np.concatenate(xx), np.concatenate(yy)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Training requires positive and negative candidate examples')
    return x, y, np.concatenate(keys)
