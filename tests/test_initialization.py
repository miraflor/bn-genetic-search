import random

import networkx as nx
import pytest

from bn_genetic_search._encoding import decode_code
from bn_genetic_search._initialization import (
    randomized_topological_order,
    sample_feasible_code,
    sample_feasible_dag,
    sample_raw_repaired_code,
)


def test_randomized_order_respects_required_partial_order():
    nodes = ["A", "B", "C", "D"]
    required = [("A", "C"), ("B", "C"), ("C", "D")]
    order = randomized_topological_order(nodes, required, random.Random(12))
    pos = {node: i for i, node in enumerate(order)}
    assert all(pos[u] < pos[v] for u, v in required)


def test_feasible_dag_satisfies_all_hard_constraints():
    nodes = ["A", "B", "C", "D", "E"]
    required = {("A", "C"), ("B", "D")}
    forbidden = {("A", "D"), ("C", "E")}
    search_space = {
        (u, v)
        for u in nodes
        for v in nodes
        if u != v and (u, v) not in {("E", "A"), ("E", "B")}
    }
    dag = sample_feasible_dag(
        nodes,
        edge_prob=0.8,
        required_edges=required,
        forbidden_edges=forbidden,
        search_space=search_space,
        max_indegree=2,
        rng=random.Random(7),
    )
    assert nx.is_directed_acyclic_graph(dag)
    assert required.issubset(dag.edges())
    assert not (set(dag.edges()) & forbidden)
    assert set(dag.edges()).issubset(search_space)
    assert max(dict(dag.in_degree()).values()) <= 2


def test_feasible_code_is_reproducible():
    kwargs = dict(
        edge_prob=0.4,
        required_edges=[("A", "B")],
        forbidden_edges=[("C", "A")],
        max_indegree=2,
    )
    a = sample_feasible_code(["A", "B", "C", "D"], rng=random.Random(99), **kwargs)
    b = sample_feasible_code(["A", "B", "C", "D"], rng=random.Random(99), **kwargs)
    assert a == b
    assert nx.is_directed_acyclic_graph(decode_code(a, ["A", "B", "C", "D"]))


def test_cyclic_required_edges_rejected_before_sampling():
    with pytest.raises(ValueError, match="required_edges create a directed cycle"):
        sample_feasible_dag(
            ["A", "B", "C"],
            edge_prob=0.2,
            required_edges=[("A", "B"), ("B", "C"), ("C", "A")],
            rng=random.Random(1),
        )


def test_raw_chromosome_arm_is_repaired_to_dag():
    nodes = ["A", "B", "C", "D"]
    code, stats = sample_raw_repaired_code(
        nodes,
        edge_weights={},
        required_edges=[("A", "B")],
        forbidden_edges=[("D", "A")],
        max_indegree=2,
        edge_selection="random",
        repair_operation="reverse_then_delete",
        edge_prob=0.2,
        rng=random.Random(4),
        repair_rng=random.Random(5),
    )
    dag = decode_code(code, nodes)
    assert nx.is_directed_acyclic_graph(dag)
    assert dag.has_edge("A", "B")
    assert not dag.has_edge("D", "A")
    assert max(dict(dag.in_degree()).values()) <= 2
    assert stats.calls == 1


def test_raw_control_matches_requested_marginal_edge_density():
    # With two nodes there can be no directed cycle, so repair cannot change
    # adjacency presence. Across deterministic seeds the raw control should
    # therefore reproduce the requested marginal edge probability rather than
    # the 2/3 density of a uniform ternary draw.
    adjacent = 0
    trials = 2000
    for seed in range(trials):
        code, _ = sample_raw_repaired_code(
            ["A", "B"],
            edge_weights={},
            edge_prob=0.2,
            edge_selection="random",
            repair_operation="delete",
            rng=random.Random(seed),
            repair_rng=random.Random(seed + 10000),
        )
        adjacent += int(code[0] != 0)
    observed = adjacent / trials
    assert 0.17 < observed < 0.23
