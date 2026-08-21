"""Initialization strategies for evolutionary Bayesian-network search.

The default strategy samples the initial population directly from the feasible
DAG space. This is not simply a convenience: it avoids letting the repair
operator determine the distribution of generation zero. A second, deliberately
less disciplined strategy generates raw ternary chromosomes and repairs them;
it is retained solely for the initialization ablation proposed in the paper.
"""
from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Mapping, Sequence

import networkx as nx

from ._encoding import code_length, encode_dag
from ._repair import RepairStats, repair_code

Edge = tuple[Hashable, Hashable]


def _validate_constraints(
    nodes: Sequence[Hashable],
    required_edges: Iterable[Edge],
    forbidden_edges: Iterable[Edge],
    search_space: Iterable[Edge],
    max_indegree: int | None,
) -> tuple[set[Edge], set[Edge], set[Edge]]:
    node_set = set(nodes)
    required = set(required_edges)
    forbidden = set(forbidden_edges)
    search = set(search_space)

    for edge in required | forbidden | search:
        u, v = edge
        if u not in node_set or v not in node_set:
            raise ValueError(f"constraint edge {edge!r} contains an unknown node")
        if u == v:
            raise ValueError("self-loops are not valid structural constraints")

    if required & forbidden:
        raise ValueError("an edge cannot be both required and forbidden")
    if search and not required.issubset(search):
        raise ValueError("all required edges must lie in the search space")

    required_graph = nx.DiGraph()
    required_graph.add_nodes_from(nodes)
    required_graph.add_edges_from(required)
    if not nx.is_directed_acyclic_graph(required_graph):
        raise ValueError("required_edges create a directed cycle")
    if max_indegree is not None:
        if max_indegree < 0:
            raise ValueError("max_indegree must be non-negative or None")
        if any(required_graph.in_degree(v) > max_indegree for v in nodes):
            raise ValueError("required_edges violate max_indegree")

    return required, forbidden, search


def randomized_topological_order(
    nodes: Sequence[Hashable],
    required_edges: Iterable[Edge],
    rng: random.Random,
) -> list[Hashable]:
    """Return a randomized topological order consistent with required edges.

    This is a randomized Kahn sort. At each step one currently admissible node
    is chosen uniformly from the available set. The resulting distribution is
    not claimed to be uniform over all DAGs or all topological orders; the only
    required property is feasibility with respect to the partial order induced
    by ``required_edges``.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(required_edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("required_edges create a directed cycle")

    indegree = dict(graph.in_degree())
    available = sorted(
        [node for node, degree in indegree.items() if degree == 0], key=repr
    )
    order: list[Hashable] = []

    while available:
        index = rng.randrange(len(available))
        node = available.pop(index)
        order.append(node)
        for child in sorted(graph.successors(node), key=repr):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(child)
                available.sort(key=repr)

    if len(order) != len(nodes):
        raise ValueError("required_edges create a directed cycle")
    return order


def sample_feasible_dag(
    nodes: Sequence[Hashable],
    *,
    edge_prob: float,
    required_edges: Iterable[Edge] = (),
    forbidden_edges: Iterable[Edge] = (),
    search_space: Iterable[Edge] = (),
    max_indegree: int | None = None,
    rng: random.Random,
) -> nx.DiGraph:
    """Sample a constraint-compatible DAG without post-hoc cycle repair.

    Procedure
    ---------
    1. Seed the graph with required edges.
    2. Draw a randomized topological order compatible with those edges.
    3. Consider only forward edges in that order.
    4. Add an admissible edge with probability ``edge_prob`` while respecting
       forbidden edges, a search-space whitelist, and ``max_indegree``.

    Because every added edge points forward in a topological order, acyclicity
    is guaranteed by construction.
    """
    if not 0 <= edge_prob <= 1:
        raise ValueError("edge_prob must lie in [0, 1]")

    nodes = list(nodes)
    required, forbidden, search = _validate_constraints(
        nodes,
        required_edges,
        forbidden_edges,
        search_space,
        max_indegree,
    )
    order = randomized_topological_order(nodes, required, rng)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(required)

    for i, parent in enumerate(order):
        for child in order[i + 1 :]:
            edge = (parent, child)
            if edge in required or edge in forbidden:
                continue
            if search and edge not in search:
                continue
            if max_indegree is not None and graph.in_degree(child) >= max_indegree:
                continue
            if rng.random() < edge_prob:
                graph.add_edge(parent, child)

    # Defensive postconditions double as useful executable documentation.
    assert nx.is_directed_acyclic_graph(graph)
    assert required.issubset(graph.edges())
    assert not (set(graph.edges()) & forbidden)
    if search:
        assert set(graph.edges()).issubset(search)
    if max_indegree is not None:
        assert all(graph.in_degree(v) <= max_indegree for v in graph.nodes())
    return graph


def sample_feasible_code(
    nodes: Sequence[Hashable],
    **kwargs,
) -> list[int]:
    """Sample a feasible DAG and return its ternary chromosome."""
    return encode_dag(sample_feasible_dag(nodes, **kwargs), nodes)


def sample_raw_repaired_code(
    nodes: Sequence[Hashable],
    *,
    edge_weights: Mapping[Edge, float],
    required_edges: Iterable[Edge] = (),
    forbidden_edges: Iterable[Edge] = (),
    search_space: Iterable[Edge] = (),
    max_indegree: int | None = None,
    edge_selection: str = "mutual_info",
    repair_operation: str = "reverse_then_delete",
    edge_prob: float = 0.2,
    rng: random.Random,
    repair_rng: random.Random,
) -> tuple[list[int], RepairStats]:
    """Generate an unconstrained chromosome, then repair it into feasibility.

    This function exists for the paper's initialization ablation. It should not
    be the production default because repair can bias the initial population.
    """
    if not 0 <= edge_prob <= 1:
        raise ValueError("edge_prob must lie in [0, 1]")

    # Density-matched control for the initialization ablation: an unordered
    # pair is adjacent with the same marginal probability ``edge_prob`` used by
    # feasible-DAG initialization. Conditional on adjacency, its orientation is
    # chosen uniformly. This prevents the control from accidentally comparing
    # a dense (2/3-adjacent) raw ternary population with a sparse feasible one.
    length = code_length(len(nodes))
    raw = []
    for _ in range(length):
        if rng.random() >= edge_prob:
            raw.append(0)
        else:
            raw.append(1 if rng.random() < 0.5 else 2)
    return repair_code(
        raw,
        nodes,
        edge_weights,
        required_edges=required_edges,
        forbidden_edges=forbidden_edges,
        search_space=search_space,
        max_indegree=max_indegree,
        edge_selection=edge_selection,
        repair_operation=repair_operation,
        rng=repair_rng,
    )
