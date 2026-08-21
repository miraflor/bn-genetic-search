import random

from bn_genetic_search._rng import isolated_global_random, make_random_streams


def test_child_streams_do_not_advance_one_another():
    a = make_random_streams(2026)
    b = make_random_streams(2026)

    for _ in range(100):
        a.repair.random()
        a.initialization_repair.random()

    assert [a.initialization.random() for _ in range(10)] == [
        b.initialization.random() for _ in range(10)
    ]


def test_deap_global_random_context_is_reproducible_and_restores_state():
    random.seed(12345)
    before = random.getstate()
    with isolated_global_random(777):
        first = [random.random() for _ in range(5)]
    after = random.getstate()
    assert before == after

    with isolated_global_random(777):
        second = [random.random() for _ in range(5)]
    assert first == second
