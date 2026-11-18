"""Directed bounded shortest paths including partial candidate arcs.

Parallel arcs retain their identity; no sparse-matrix duplicate summation.
Truncated Dijkstra stores only visited nodes, with a bounded LRU source cache.
The graph has no access to references, labels, or stored predictions.
"""
from collections import OrderedDict
import heapq
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class RoadGraph:
    def __init__(self, trip, geometry, cache_entries=512):
        self.trip = trip
        self.geometry = geometry
        self.lengths = geometry.lengths
        self.order = np.argsort(trip.arcs[:, 0], kind='stable')
        self.offsets = np.r_[0, np.cumsum(np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes)))]
        sparse = coo_matrix((np.ones(len(trip.arcs)), tuple(trip.arcs.T)), shape=(len(trip.nodes), len(trip.nodes))).tocsr()
        self.n_components, self.components = connected_components(sparse, directed=True, connection='weak')
        self.cache = OrderedDict()
        self.cache_entries = cache_entries
        self.searches = 0
        self.cache_hits = 0

    def search(self, source, limit):
        source = int(source)
        # Round up for reuse; the final candidate transition is checked against its exact bound.
        bound = max(250., math.ceil(limit / 250) * 250.)
        previous = self.cache.get(source)
        if previous is not None and previous[0] >= bound:
            self.cache.move_to_end(source)
            self.cache_hits += 1
            return previous[1], previous[2]
        distances, parents = {source: 0.}, {}
        heap = [(0., source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance > distances[node]:
                continue
            for arc in self.order[self.offsets[node]:self.offsets[node + 1]]:
                destination = int(self.trip.arcs[arc, 1])
                alternative = distance + float(self.lengths[arc])
                if alternative <= bound and alternative < distances.get(destination, math.inf):
                    distances[destination] = alternative
                    parents[destination] = int(arc)
                    heapq.heappush(heap, (alternative, destination))
        self.cache[source] = (bound, distances, parents)
        self.cache.move_to_end(source)
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        self.searches += 1
        return distances, parents

    def connection(self, arc_a, fraction_a, arc_b, fraction_b, limit, jitter_m=8., with_path=False):
        """Return finite travel metres, optional original arc path; inf if infeasible.

        Small backward projection jitter on the same directed arc is assigned
        zero travel. Larger backward moves require a legal directed cycle.
        """
        arc_a, arc_b = int(arc_a), int(arc_b)
        if arc_a < 0 or arc_b < 0:
            return math.inf, []
        if arc_a == arc_b:
            direct = float((fraction_b - fraction_a) * self.lengths[arc_a])
            if direct >= -jitter_m and max(0., direct) <= limit:
                return max(0., direct), [arc_a] if with_path else []
        source = int(self.trip.arcs[arc_a, 1])
        target = int(self.trip.arcs[arc_b, 0])
        partial = float((1 - fraction_a) * self.lengths[arc_a] + fraction_b * self.lengths[arc_b])
        if partial > limit or self.components[source] != self.components[target]:
            return math.inf, []
        if source == target:
            return partial, [arc_a, arc_b] if with_path else []
        distances, parents = self.search(source, limit - partial)
        total = partial + distances.get(target, math.inf)
        if total > limit:
            return math.inf, []
        if not with_path:
            return total, []
        middle, node = [], target
        while node != source:
            edge = parents[node]
            middle.append(edge)
            node = int(self.trip.arcs[edge, 0])
        return total, [arc_a, *reversed(middle), arc_b]

    def distance_matrix(self, previous, current, limit, jitter_m):
        result = np.full((len(previous), len(current)), np.inf)
        for i, (arc_a, fraction_a) in enumerate(previous):
            for j, (arc_b, fraction_b) in enumerate(current):
                result[i, j] = self.connection(arc_a, fraction_a, arc_b, fraction_b, limit, jitter_m)[0]
        return result


def movement_limit(dt, config):
    return min(config['maximum_path_m'], config['max_speed_mps'] * dt + config['distance_slack_m'])
