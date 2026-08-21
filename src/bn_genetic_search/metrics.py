"""Recovery metrics for Bayesian-network structure experiments.

These helpers are intentionally small and graph-library agnostic: any object
with NetworkX-like ``nodes()``, ``has_edge()``, ``predecessors()``, and
``edges()`` methods can be used. That includes ``networkx.DiGraph`` and the
relevant pgmpy graph classes.

The paper should emphasize equivalence-class-aware recovery rather than exact
DAG orientation alone. Accordingly, this module provides skeleton and
v-structure metrics plus a pair-state distance for CPDAG/PDAG objects.

The pair-state distance is *defined here explicitly* rather than silently being
called ``SHD``. For each unordered pair, it counts one disagreement when the
two graphs differ among the states: absent, undirected, u->v, or v->u. This is
easy to audit and avoids depending on version-specific metric semantics.
"""
from __future__ import annotations

from collections.abc import Hashable
from itertools import combinations
from typing import NamedTuple

import networkx as nx

Node = Hashable
Edge = tuple[Node, Node]
UnorderedEdge = frozenset[Node]
VStructure = tuple[Node, Node, Node]


class PrecisionRecallF1(NamedTuple):
    """Precision, recall, and F1 with the corresponding set counts."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _same_nodes(true_graph, estimated_graph) -> list[Node]:
    """Validate node sets and return a deterministic pair-enumeration order."""
    true_nodes = set(true_graph.nodes())
    estimated_nodes = set(estimated_graph.nodes())
    if true_nodes != estimated_nodes:
        raise ValueError(
            "true_graph and estimated_graph must contain exactly the same nodes"
        )
    return sorted(true_nodes, key=repr)


def _prf(true_items: set, estimated_items: set) -> PrecisionRecallF1:
    tp = len(true_items & estimated_items)
    fp = len(estimated_items - true_items)
    fn = len(true_items - estimated_items)

    # The empty/empty case is a perfect match. If only the estimate is empty,
    # precision is conventionally set to 0; if only truth is empty, recall is 0.
    if not true_items and not estimated_items:
        return PrecisionRecallF1(1.0, 1.0, 1.0, 0, 0, 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return PrecisionRecallF1(precision, recall, f1, tp, fp, fn)


def skeleton_edges(graph) -> set[UnorderedEdge]:
    """Return adjacencies while ignoring edge orientation.

    A two-node ``frozenset`` is used so the result behaves naturally as a set
    even when a PDAG stores an undirected edge as both u->v and v->u.
    """
    result: set[UnorderedEdge] = set()
    for u, v in graph.edges():
        if u == v:
            raise ValueError("self-loops are not valid BN adjacencies")
        result.add(frozenset((u, v)))
    return result


def skeleton_precision_recall_f1(true_graph, estimated_graph) -> PrecisionRecallF1:
    """Compare the undirected skeletons of two graphs."""
    _same_nodes(true_graph, estimated_graph)
    return _prf(skeleton_edges(true_graph), skeleton_edges(estimated_graph))


def v_structures(graph) -> set[VStructure]:
    """Return the unshielded colliders ``X -> Z <- Y`` in a DAG.

    Parent order is canonicalized by ``repr`` so ``(X, Z, Y)`` and
    ``(Y, Z, X)`` represent the same v-structure.
    """
    nx_graph = nx.DiGraph(graph)
    if not nx.is_directed_acyclic_graph(nx_graph):
        raise ValueError("v_structures expects a DAG")

    result: set[VStructure] = set()
    for collider in nx_graph.nodes():
        parents = list(nx_graph.predecessors(collider))
        for left, right in combinations(parents, 2):
            # Shielded colliders are not v-structures.
            if nx_graph.has_edge(left, right) or nx_graph.has_edge(right, left):
                continue
            a, b = sorted((left, right), key=repr)
            result.add((a, collider, b))
    return result


def v_structure_precision_recall_f1(true_dag, estimated_dag) -> PrecisionRecallF1:
    """Compare the unshielded colliders of two DAGs."""
    _same_nodes(true_dag, estimated_dag)
    return _prf(v_structures(true_dag), v_structures(estimated_dag))


def _pair_state(graph, u: Node, v: Node) -> int:
    """Encode one unordered-pair state relative to ordered pair ``(u, v)``.

    Returns
    -------
    0 : no adjacency
    1 : u -> v
    2 : v -> u
    3 : undirected / represented in both directions
    """
    uv = graph.has_edge(u, v)
    vu = graph.has_edge(v, u)
    if uv and vu:
        return 3
    if uv:
        return 1
    if vu:
        return 2
    return 0


def cpdag_pairwise_distance(true_cpdag, estimated_cpdag) -> int:
    """Count unordered node pairs whose CPDAG/PDAG edge state disagrees.

    The function assumes the graph objects encode an undirected edge by having
    both directed representations, which is the convention used by pgmpy's PDAG
    graph representation. It also works for ordinary DAGs (which simply never
    have state 3).
    """
    nodes = _same_nodes(true_cpdag, estimated_cpdag)
    return sum(
        _pair_state(true_cpdag, u, v) != _pair_state(estimated_cpdag, u, v)
        for u, v in combinations(nodes, 2)
    )
