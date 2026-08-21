"""Compact chromosome representation used by :class:`GeneticSearch`.

The representation descends directly from the author's 2022 implementation.
For each unordered pair of variables {X_i, X_j}, i < j, one ternary locus is
stored:

    0  -> no adjacency
    1  -> X_i -> X_j
    2  -> X_j -> X_i

This representation is not claimed as novel by itself. Its practical value is
that self-loops and directed 2-cycles are impossible by construction while a
candidate structure needs only n(n-1)/2 loci.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence

import networkx as nx

Node = Hashable


def code_length(n_nodes: int) -> int:
    """Return the number of ternary loci required for ``n_nodes`` variables."""
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    return n_nodes * (n_nodes - 1) // 2


def locus_index(i: int, j: int, n_nodes: int) -> int:
    """Return the chromosome position for unordered pair ``(i, j)``.

    ``i`` and ``j`` are positions in the fixed node ordering and must satisfy
    ``0 <= i < j < n_nodes``. Keeping this calculation in one helper makes the
    encoding auditable and avoids subtly different triangular-index formulas in
    mutation, constraint, and repair code.
    """
    if not 0 <= i < j < n_nodes:
        raise ValueError("expected 0 <= i < j < n_nodes")
    return i * n_nodes - i * (i + 1) // 2 + (j - i - 1)


def encode_dag(graph: nx.DiGraph, nodes: Sequence[Node]) -> list[int]:
    """Encode a directed graph using the triangular ternary representation.

    The function accepts any ``networkx``-compatible directed graph, but rejects
    self-loops, unknown nodes, and opposite edges between the same pair because
    those states cannot be represented by a single ternary locus.
    """
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")

    index = {node: i for i, node in enumerate(nodes)}
    code = [0] * code_length(len(nodes))

    for u, v in graph.edges():
        if u == v:
            raise ValueError("self-loops cannot be represented")
        if u not in index or v not in index:
            raise ValueError(f"edge ({u!r}, {v!r}) contains a node outside nodes")

        iu, iv = index[u], index[v]
        if iu < iv:
            i, j, value = iu, iv, 1
        else:
            i, j, value = iv, iu, 2

        pos = locus_index(i, j, len(nodes))
        if code[pos] != 0:
            raise ValueError(f"both edge directions are present between {u!r} and {v!r}")
        code[pos] = value

    return code


def decode_code(code: Sequence[int], nodes: Sequence[Node]) -> nx.DiGraph:
    """Decode a ternary chromosome into a ``networkx.DiGraph``."""
    expected = code_length(len(nodes))
    if len(code) != expected:
        raise ValueError(f"expected chromosome length {expected}; got {len(code)}")
    if any(value not in (0, 1, 2) for value in code):
        raise ValueError("chromosome values must be 0, 1, or 2")

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    pos = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            value = code[pos]
            if value == 1:
                graph.add_edge(nodes[i], nodes[j])
            elif value == 2:
                graph.add_edge(nodes[j], nodes[i])
            pos += 1
    return graph
