"""Exact graph summaries for native scalability evaluations."""

import numpy as np


def _largest_component(neighbors, vertices):
    remaining = vertices
    largest = 0
    while remaining:
        frontier = remaining & -remaining
        remaining ^= frontier
        size = 0
        while frontier:
            vertex = frontier & -frontier
            size += 1
            new = neighbors[vertex.bit_length() - 1] & remaining
            remaining ^= new
            frontier = (frontier ^ vertex) | new
        largest = max(largest, size)
    return largest


def _maximum_clique(neighbors, vertices):
    """Bound the search with greedy coloring of each candidate subgraph."""
    best = 0

    def expand(candidates, size):
        nonlocal best
        unchecked = candidates
        colors = 0
        order = []
        while unchecked:
            available = unchecked
            colors += 1
            while available:
                vertex = available & -available
                order.append((vertex, colors))
                unchecked ^= vertex
                available &= ~(vertex | neighbors[vertex.bit_length() - 1])
        for vertex, bound in reversed(order):
            if size + bound <= best:
                break
            followers = candidates & neighbors[vertex.bit_length() - 1]
            if followers:
                expand(followers, size + 1)
            else:
                best = max(best, size + 1)
            candidates ^= vertex

    expand(vertices, 0)
    return best


class GraphStatistics:
    def __init__(self):
        self._decisions = 0
        self._aircraft_decisions = 0
        self._degree_sum = 0
        self._peak_aircraft = 0
        self._max_degree = 0
        self._max_clique = 0
        self._max_component = 0

    def add(self, adjacency, active):
        graph = (adjacency | adjacency.T) & active[:, None] & active[None, :]
        np.fill_diagonal(graph, False)
        neighbors = [sum(1 << int(neighbor) for neighbor in row.nonzero()[0]) for row in graph]
        vertices = sum(1 << int(index) for index in active.nonzero()[0])
        self._decisions += 1
        aircraft = vertices.bit_count()
        self._aircraft_decisions += aircraft
        self._peak_aircraft = max(self._peak_aircraft, aircraft)
        degrees = [links.bit_count() for links in neighbors]
        self._degree_sum += sum(degrees)
        self._max_degree = max(self._max_degree, max(degrees, default=0))
        self._max_clique = max(self._max_clique, _maximum_clique(neighbors, vertices))
        self._max_component = max(self._max_component, _largest_component(neighbors, vertices))

    def report(self):
        return dict(
            decisions=self._decisions, aircraft_decisions=self._aircraft_decisions,
            peak_aircraft=self._peak_aircraft, degree_sum=self._degree_sum,
            mean_degree=self._degree_sum / self._aircraft_decisions if self._aircraft_decisions else 0.0,
            max_degree=self._max_degree, max_clique_size=self._max_clique,
            max_component_size=self._max_component,
        )
