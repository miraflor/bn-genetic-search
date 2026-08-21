"""Cycle repair for evolutionary Bayesian-network structure search.

The central research contribution represented here is not generic "repair a
cycle" logic. It is the author's specific modernized variant of a 2022 design:

1. localize repair to an actually detected directed cycle;
2. select a removable cycle edge either by low pairwise mutual information or
   at random (the latter is useful as an ablation control);
3. optionally try a *cycle-safe reversal* first, thereby preserving the
   adjacency when possible;
4. delete the edge only when safe reversal is unavailable.

The two independent switches -- edge selection and repair operation -- are
exposed separately so a DAG-recovery paper can test the intended 2x2 factorial
comparison without changing any other part of the genetic search.
"""
from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import networkx as nx
from sklearn.metrics import mutual_info_score

from ._encoding import decode_code, encode_dag

Node = Hashable
Edge = tuple[Node, Node]
EdgeSelection = Literal["mutual_info", "random"]
RepairOperation = Literal["delete", "reverse_then_delete"]


@dataclass
class RepairStats:
    """Mutable counters describing what the repair operator actually did.

    These counters are deliberately part of the implementation because they are
    useful mechanism diagnostics in DAG-recovery experiments. They are not used
    to make search decisions.
    """

    calls: int = 0
    cycles_found: int = 0
    reversals: int = 0
    cycle_deletions: int = 0
    forbidden_deletions: int = 0
    indegree_deletions: int = 0

    def __iadd__(self, other: RepairStats) -> RepairStats:
        self.calls += other.calls
        self.cycles_found += other.cycles_found
        self.reversals += other.reversals
        self.cycle_deletions += other.cycle_deletions
        self.forbidden_deletions += other.forbidden_deletions
        self.indegree_deletions += other.indegree_deletions
        return self


def pairwise_mutual_information(data) -> dict[Edge, float]:
    """Compute symmetric pairwise mutual-information weights for data columns.

    ``mutual_info_score`` is exactly the categorical mutual-information measure
    used elsewhere in pgmpy's tree-search machinery. MI-guided repair is intended
    for discrete variables; :class:`GeneticSearch` rejects pgmpy scores that
    explicitly target continuous or mixed data when this heuristic is selected.
    """
    columns = list(data.columns)
    weights: dict[Edge, float] = {}
    for i, u in enumerate(columns):
        for v in columns[i + 1 :]:
            value = float(mutual_info_score(data[u], data[v]))
            weights[(u, v)] = value
            weights[(v, u)] = value
    return weights


def _edge_weight(edge: Edge, weights: Mapping[Edge, float]) -> float:
    # MI is symmetric, but accepting either orientation makes the helper robust
    # to user-supplied weight tables.
    return float(weights.get(edge, weights.get((edge[1], edge[0]), 0.0)))


def _pick_edge(
    candidates: Sequence[Edge],
    *,
    edge_selection: EdgeSelection,
    edge_weights: Mapping[Edge, float],
    rng: random.Random,
) -> Edge:
    """Choose one edge from a detected cycle under the requested ablation arm."""
    ordered = sorted(candidates, key=lambda e: (repr(e[0]), repr(e[1])))
    if edge_selection == "random":
        return rng.choice(ordered)
    if edge_selection == "mutual_info":
        return min(ordered, key=lambda e: (_edge_weight(e, edge_weights), repr(e)))
    raise ValueError("edge_selection must be 'mutual_info' or 'random'")


def _reverse_is_legal(
    graph: nx.DiGraph,
    u: Node,
    v: Node,
    *,
    forbidden_edges: set[Edge],
    search_space: set[Edge],
    max_indegree: int | None,
) -> bool:
    """Return whether replacing ``u -> v`` by ``v -> u`` is admissible.

    The caller has already removed ``u -> v``. Adding ``v -> u`` creates a
    directed cycle iff a path ``u => v`` still exists. This is the exact
    cycle-safety condition and avoids a full graph-wide DAG reconstruction.
    """
    reverse = (v, u)
    # If the reverse edge is already present (for example because it is a
    # required edge), removing u->v is a deletion of a conflicting orientation,
    # not a genuine reversal. Keeping the cases distinct also makes the repair
    # diagnostics interpretable.
    if graph.has_edge(v, u):
        return False
    if reverse in forbidden_edges:
        return False
    if search_space and reverse not in search_space:
        return False
    if max_indegree is not None and graph.in_degree(u) + 1 > max_indegree:
        return False
    return not nx.has_path(graph, u, v)


def repair_graph(
    graph: nx.DiGraph,
    edge_weights: Mapping[Edge, float],
    *,
    required_edges: Iterable[Edge] = (),
    forbidden_edges: Iterable[Edge] = (),
    search_space: Iterable[Edge] = (),
    max_indegree: int | None = None,
    edge_selection: EdgeSelection = "mutual_info",
    repair_operation: RepairOperation = "reverse_then_delete",
    rng: random.Random | None = None,
) -> tuple[nx.DiGraph, RepairStats]:
    """Repair a candidate directed graph into a constraint-compatible DAG.

    Notes
    -----
    The cycle loop always changes an edge that belongs to the detected cycle.
    Under safe reversal, the replacement orientation creates no new cycle. Under
    deletion, at least one edge of the cycle disappears. Therefore the loop must
    terminate; in particular, no more than the number of non-required candidate
    edges can be permanently processed as cycle-breaking edges.
    """
    if edge_selection not in {"mutual_info", "random"}:
        raise ValueError("edge_selection must be 'mutual_info' or 'random'")
    if repair_operation not in {"delete", "reverse_then_delete"}:
        raise ValueError("repair_operation must be 'delete' or 'reverse_then_delete'")
    if max_indegree is not None and max_indegree < 0:
        raise ValueError("max_indegree must be non-negative or None")

    rng = rng or random.Random()
    stats = RepairStats(calls=1)
    work = nx.DiGraph(graph)
    nodes = set(work.nodes())
    required = set(required_edges)
    forbidden = set(forbidden_edges)
    search = set(search_space)

    for edge in required | forbidden | search:
        if edge[0] not in nodes or edge[1] not in nodes:
            raise ValueError(f"expert edge {edge!r} contains a node outside the graph")
        if edge[0] == edge[1]:
            raise ValueError("expert knowledge cannot contain self-loops")

    if required & forbidden:
        raise ValueError("an edge cannot be both required and forbidden")
    if search and not required.issubset(search):
        raise ValueError("all required edges must belong to expert_knowledge.search_space")

    # First enforce hard exclusions. A search-space whitelist is equivalent to
    # forbidding every directed edge not listed in it.
    for u, v in list(work.edges()):
        if (u, v) in forbidden or (search and (u, v) not in search):
            work.remove_edge(u, v)
            stats.forbidden_deletions += 1

    # Then impose required edges. The cycle loop may remove the opposite edge,
    # but is never allowed to remove a required edge itself. Insert in sorted
    # order: set iteration order varies with interpreter hash randomization,
    # and edge insertion order steers ``nx.find_cycle`` traversal, so sorting
    # keeps repair reproducible across processes.
    work.add_edges_from(sorted(required, key=repr))

    required_graph = nx.DiGraph()
    required_graph.add_nodes_from(work.nodes())
    required_graph.add_edges_from(required)
    if not nx.is_directed_acyclic_graph(required_graph):
        raise ValueError("required_edges create a directed cycle")

    while True:
        try:
            cycle = nx.find_cycle(work, orientation="original")
        except nx.NetworkXNoCycle:
            break

        stats.cycles_found += 1
        cycle_edges = [(u, v) for u, v, *_ in cycle]
        candidates = [edge for edge in cycle_edges if edge not in required]
        if not candidates:
            raise ValueError("required_edges force a directed cycle")

        u, v = _pick_edge(
            candidates,
            edge_selection=edge_selection,
            edge_weights=edge_weights,
            rng=rng,
        )
        work.remove_edge(u, v)

        if repair_operation == "reverse_then_delete" and _reverse_is_legal(
            work,
            u,
            v,
            forbidden_edges=forbidden,
            search_space=search,
            max_indegree=max_indegree,
        ):
            work.add_edge(v, u)
            stats.reversals += 1
        else:
            stats.cycle_deletions += 1

    # max_indegree is a standard pgmpy search constraint. It is deliberately
    # applied after acyclicity repair: deleting an incoming edge cannot create a
    # cycle, so this stage cannot undo the proof above.
    if max_indegree is not None:
        for child in list(work.nodes()):
            while work.in_degree(child) > max_indegree:
                parents = [
                    parent
                    for parent in work.predecessors(child)
                    if (parent, child) not in required
                ]
                if not parents:
                    raise ValueError(
                        f"required_edges violate max_indegree={max_indegree} at {child!r}"
                    )
                if edge_selection == "random":
                    parent = rng.choice(sorted(parents, key=repr))
                else:
                    parent = min(
                        parents,
                        key=lambda p: (_edge_weight((p, child), edge_weights), repr(p)),
                    )
                work.remove_edge(parent, child)
                stats.indegree_deletions += 1

    if not nx.is_directed_acyclic_graph(work):  # defensive postcondition
        raise RuntimeError("repair_graph postcondition failed: result is cyclic")
    return work, stats


def repair_code(
    code,
    nodes,
    edge_weights,
    **kwargs,
):
    """Decode a chromosome, repair it, and encode the resulting DAG again."""
    graph, stats = repair_graph(decode_code(code, nodes), edge_weights, **kwargs)
    return encode_dag(graph, nodes), stats
