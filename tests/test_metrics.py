import networkx as nx

from bn_genetic_search.metrics import (
    cpdag_pairwise_distance,
    skeleton_precision_recall_f1,
    v_structure_precision_recall_f1,
    v_structures,
)


def test_skeleton_metrics_ignore_direction():
    true = nx.DiGraph([("A", "B"), ("B", "C")])
    est = nx.DiGraph([("B", "A"), ("B", "C")])
    report = skeleton_precision_recall_f1(true, est)
    assert report.f1 == 1.0
    assert report.false_positives == 0
    assert report.false_negatives == 0


def test_skeleton_metrics_count_missing_and_extra_edges():
    true = nx.DiGraph([("A", "B"), ("B", "C")])
    true.add_nodes_from(["A", "B", "C", "D"])
    est = nx.DiGraph([("A", "B"), ("C", "D")])
    est.add_nodes_from(["A", "B", "C", "D"])
    report = skeleton_precision_recall_f1(true, est)
    assert report.true_positives == 1
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.f1 == 0.5


def test_v_structures_exclude_shielded_colliders():
    unshielded = nx.DiGraph([("A", "C"), ("B", "C")])
    shielded = nx.DiGraph([("A", "C"), ("B", "C"), ("A", "B")])
    assert v_structures(unshielded) == {("A", "C", "B")}
    assert v_structures(shielded) == set()


def test_v_structure_f1_detects_orientation_error():
    true = nx.DiGraph([("A", "C"), ("B", "C")])
    est = nx.DiGraph([("C", "A"), ("B", "C")])
    report = v_structure_precision_recall_f1(true, est)
    assert report.true_positives == 0
    assert report.false_negatives == 1
    assert report.f1 == 0.0


def test_cpdag_pairwise_distance_handles_undirected_state():
    # Mimic pgmpy's convention: an undirected A--B appears in both directions.
    true = nx.DiGraph()
    true.add_nodes_from(["A", "B", "C"])
    true.add_edges_from([("A", "B"), ("B", "A"), ("B", "C")])

    same = true.copy()
    assert cpdag_pairwise_distance(true, same) == 0

    oriented = nx.DiGraph()
    oriented.add_nodes_from(["A", "B", "C"])
    oriented.add_edges_from([("A", "B"), ("B", "C")])
    assert cpdag_pairwise_distance(true, oriented) == 1
