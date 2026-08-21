import numpy as np
import pandas as pd
import pytest

pgmpy = pytest.importorskip("pgmpy")
deap = pytest.importorskip("deap")

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.structure_score import BDeu

from bn_genetic_search import GeneticSearch


def synthetic_chain(n=500, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=n)
    b = np.bitwise_xor(a, rng.binomial(1, 0.08, size=n))
    c = np.bitwise_xor(b, rng.binomial(1, 0.08, size=n))
    return pd.DataFrame({"A": a, "B": b, "C": c}).astype("category")


def test_estimator_exposes_pgmpy_style_attributes():
    X = synthetic_chain()
    search = GeneticSearch(
        scoring_method="bdeu",
        return_type="dag",
        population_size=24,
        max_iter=6,
        patience=3,
        random_state=7,
        show_progress=False,
    ).fit(X)
    assert isinstance(search.causal_graph_, DAG)
    assert set(search.causal_graph_.nodes()) == set(X.columns)
    assert list(search.adjacency_matrix_.index) == list(X.columns)
    assert search.n_features_in_ == 3
    assert list(search.feature_names_in_) == list(X.columns)
    assert search.best_score_ == search.best_score_
    assert search.n_score_evaluations_ > 0
    assert len(search.deap_hall_of_fame_) == 1
    assert len(search.deap_logbook_) >= 1


def test_pdag_return_type():
    X = synthetic_chain()
    search = GeneticSearch(
        scoring_method="bdeu",
        return_type="pdag",
        population_size=20,
        max_iter=4,
        patience=2,
        random_state=3,
        show_progress=False,
    ).fit(X)
    assert hasattr(search.causal_graph_, "to_dag")


def test_expert_knowledge_and_custom_bdeu_score():
    X = synthetic_chain()
    expert = ExpertKnowledge(required_edges=[("A", "B")], forbidden_edges=[("C", "A")])
    score = BDeu(X, equivalent_sample_size=5)
    search = GeneticSearch(
        scoring_method=score,
        expert_knowledge=expert,
        return_type="dag",
        population_size=20,
        max_iter=5,
        patience=2,
        random_state=5,
        show_progress=False,
    ).fit(X)
    assert search.causal_graph_.has_edge("A", "B")
    assert not search.causal_graph_.has_edge("C", "A")


def test_start_dag_can_seed_population():
    X = synthetic_chain()
    start = DAG([("A", "B")])
    start.add_nodes_from(X.columns)
    search = GeneticSearch(
        scoring_method="bdeu",
        start_dag=start,
        return_type="dag",
        population_size=16,
        max_iter=3,
        patience=2,
        random_state=1,
        show_progress=False,
    ).fit(X)
    assert set(search.dag_.nodes()) == set(X.columns)


def test_mutual_info_repair_rejects_explicit_continuous_score():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"X": rng.normal(size=80), "Y": rng.normal(size=80)})
    with pytest.raises(ValueError, match="intended for discrete data"):
        GeneticSearch(
            scoring_method="bic-g",
            repair_edge_selection="mutual_info",
            population_size=8,
            max_iter=2,
            patience=1,
            random_state=11,
            show_progress=False,
        ).fit(X)


def test_search_space_is_hard_constraint_and_input_knowledge_is_not_mutated():
    X = synthetic_chain()
    allowed = {("A", "B"), ("B", "C")}
    expert = ExpertKnowledge(search_space=allowed)

    # Record constructor-level knowledge before fitting. Current pgmpy-dev uses
    # fitted *_ attributes; pgmpy 1.1.2 mutates forbidden_edges internally.
    before_search_space = set(expert.search_space)
    before_forbidden = set(expert.forbidden_edges)

    search = GeneticSearch(
        scoring_method="bdeu",
        expert_knowledge=expert,
        return_type="dag",
        population_size=18,
        max_iter=4,
        patience=2,
        random_state=19,
        show_progress=False,
    ).fit(X)

    assert set(search.dag_.edges()).issubset(allowed)
    assert set(expert.search_space) == before_search_space
    assert set(expert.forbidden_edges) == before_forbidden


def test_same_seed_reproduces_best_dag():
    X = synthetic_chain(n=300, seed=13)
    kwargs = dict(
        scoring_method="bdeu",
        return_type="dag",
        population_size=18,
        max_iter=5,
        patience=3,
        random_state=23,
        show_progress=False,
    )
    left = GeneticSearch(**kwargs).fit(X)
    right = GeneticSearch(**kwargs).fit(X)
    assert set(left.dag_.edges()) == set(right.dag_.edges())
    assert left.best_score_ == right.best_score_


def test_estimator_is_sklearn_cloneable():
    from sklearn.base import clone

    original = GeneticSearch(
        scoring_method="bdeu",
        population_size=12,
        max_iter=3,
        random_state=31,
        show_progress=False,
    )
    copied = clone(original)
    assert copied.get_params(deep=False) == original.get_params(deep=False)
    assert copied is not original
