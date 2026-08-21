import random

import networkx as nx
import pytest

from bn_genetic_search._encoding import (
    code_length,
    decode_code,
    encode_dag,
    locus_index,
)


def test_code_length_and_indices_are_consistent():
    for n in range(2, 12):
        seen = []
        for i in range(n):
            for j in range(i + 1, n):
                seen.append(locus_index(i, j, n))
        assert sorted(seen) == list(range(code_length(n)))


def test_roundtrip_random_dags():
    rng = random.Random(42)
    for n in (2, 3, 5, 8):
        nodes = [f"X{i}" for i in range(n)]
        for _ in range(40):
            order = nodes.copy()
            rng.shuffle(order)
            dag = nx.DiGraph()
            dag.add_nodes_from(nodes)
            for i, u in enumerate(order):
                for v in order[i + 1 :]:
                    if rng.random() < 0.35:
                        dag.add_edge(u, v)
            code = encode_dag(dag, nodes)
            recovered = decode_code(code, nodes)
            assert set(recovered.nodes()) == set(dag.nodes())
            assert set(recovered.edges()) == set(dag.edges())


def test_two_cycle_cannot_be_encoded():
    graph = nx.DiGraph([("A", "B"), ("B", "A")])
    with pytest.raises(ValueError, match="both edge directions"):
        encode_dag(graph, ["A", "B"])


def test_invalid_code_rejected():
    with pytest.raises(ValueError):
        decode_code([3], ["A", "B"])
