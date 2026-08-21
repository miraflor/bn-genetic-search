import random

import networkx as nx
import pandas as pd
import pytest

from bn_genetic_search._repair import pairwise_mutual_information, repair_graph


def _cycle():
    return nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A")])


def _weights():
    return {
        ("A", "B"): 0.1,
        ("B", "A"): 0.1,
        ("B", "C"): 0.8,
        ("C", "B"): 0.8,
        ("C", "A"): 0.9,
        ("A", "C"): 0.9,
    }


def test_mi_delete_removes_lowest_mi_cycle_edge():
    repaired, stats = repair_graph(
        _cycle(), _weights(), edge_selection="mutual_info", repair_operation="delete", rng=random.Random(1)
    )
    assert nx.is_directed_acyclic_graph(repaired)
    assert not repaired.has_edge("A", "B")
    assert repaired.number_of_edges() == 2
    assert stats.cycle_deletions == 1
    assert stats.reversals == 0


def test_mi_reverse_then_delete_preserves_adjacency_when_safe():
    repaired, stats = repair_graph(
        _cycle(),
        _weights(),
        edge_selection="mutual_info",
        repair_operation="reverse_then_delete",
        rng=random.Random(1),
    )
    assert nx.is_directed_acyclic_graph(repaired)
    assert repaired.has_edge("B", "A")
    assert repaired.number_of_edges() == 3
    assert stats.reversals == 1


def test_safe_reversal_falls_back_to_delete_when_reverse_creates_cycle():
    # A->B lies in A->B->C->A. After removing A->B, A still reaches B via
    # A->D->B, so adding B->A would create B->A->D->B and must be rejected.
    graph = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A"), ("A", "D"), ("D", "B")])
    weights = {edge: 1.0 for edge in graph.edges()}
    weights[("A", "B")] = 0.0
    repaired, stats = repair_graph(
        graph, weights, edge_selection="mutual_info", repair_operation="reverse_then_delete", rng=random.Random(1)
    )
    assert nx.is_directed_acyclic_graph(repaired)
    assert not repaired.has_edge("B", "A")
    assert stats.cycle_deletions >= 1


def test_required_edge_is_never_removed():
    repaired, _ = repair_graph(
        _cycle(), _weights(), required_edges=[("A", "B")], rng=random.Random(2)
    )
    assert repaired.has_edge("A", "B")
    assert nx.is_directed_acyclic_graph(repaired)


def test_required_cycle_rejected():
    with pytest.raises(ValueError, match="required_edges create a directed cycle"):
        repair_graph(
            _cycle(),
            _weights(),
            required_edges=[("A", "B"), ("B", "C"), ("C", "A")],
            rng=random.Random(2),
        )


def test_search_space_and_forbidden_edges_are_enforced():
    graph = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "C")])
    repaired, stats = repair_graph(
        graph,
        _weights(),
        search_space=[("A", "B"), ("B", "C")],
        forbidden_edges=[("B", "C")],
        rng=random.Random(3),
    )
    assert set(repaired.edges()) == {("A", "B")}
    assert stats.forbidden_deletions == 2


def test_max_indegree_respected_without_touching_required_parent():
    graph = nx.DiGraph([("A", "D"), ("B", "D"), ("C", "D")])
    weights = {("A", "D"): 0.1, ("B", "D"): 0.2, ("C", "D"): 0.3}
    repaired, _ = repair_graph(
        graph,
        weights,
        required_edges=[("A", "D")],
        max_indegree=1,
        rng=random.Random(4),
    )
    assert set(repaired.predecessors("D")) == {"A"}


def test_random_repair_is_reproducible_for_fixed_seed():
    a, _ = repair_graph(
        _cycle(), _weights(), edge_selection="random", repair_operation="delete", rng=random.Random(99)
    )
    b, _ = repair_graph(
        _cycle(), _weights(), edge_selection="random", repair_operation="delete", rng=random.Random(99)
    )
    assert set(a.edges()) == set(b.edges())


def test_mutual_information_table_is_symmetric():
    data = pd.DataFrame({"A": [0, 0, 1, 1], "B": [0, 0, 1, 1], "C": [0, 1, 0, 1]})
    weights = pairwise_mutual_information(data)
    assert weights[("A", "B")] == weights[("B", "A")]
    assert weights[("A", "B")] > weights[("A", "C")]


def test_repair_random_graphs_are_acyclic_and_iteration_bound_holds():
    # A deterministic stress test for the key repair postcondition and the
    # linear-in-initial-edges bound on the number of cycle-breaking iterations.
    for seed in range(100):
        rng = random.Random(seed)
        n = rng.randint(3, 8)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.30:
                    graph.add_edge(u, v)

        weights = {}
        for u in range(n):
            for v in range(u + 1, n):
                value = rng.random()
                weights[(u, v)] = value
                weights[(v, u)] = value

        repaired, stats = repair_graph(
            graph,
            weights,
            edge_selection="mutual_info",
            repair_operation="reverse_then_delete",
            rng=random.Random(seed + 10_000),
        )
        assert nx.is_directed_acyclic_graph(repaired)
        assert stats.cycles_found <= graph.number_of_edges()
