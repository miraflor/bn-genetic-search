import pytest

deap = pytest.importorskip("deap")

from bn_genetic_search._deap import get_individual_class, make_toolbox


def test_toolbox_uses_deap_fitness_and_operators():
    toolbox, individual_cls = make_toolbox(
        tournament_size=3,
        gene_mutation_prob=1.0,
    )
    ind = individual_cls([0, 0, 0, 0])
    assert not ind.fitness.valid
    ind.fitness.values = (1.5,)
    assert ind.fitness.valid
    toolbox.mutate(ind)
    assert all(value in (0, 1, 2) for value in ind)


def test_creator_types_are_idempotent():
    first = get_individual_class()
    second = get_individual_class()
    assert first is second
