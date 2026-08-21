"""Random-stream helpers used to make ablation experiments reproducible.

The paper compares initialization and repair mechanisms. If every stochastic
subroutine shares the same pseudo-random stream, changing one mechanism can
shift the random draws used by later mechanisms. That makes a paired ablation
less clean: a treatment can accidentally change crossover and mutation merely
because it consumed a different number of random numbers during initialization
or repair.

This module therefore gives each stochastic role a deterministic child seed.
DEAP uses Python's module-level :mod:`random` generator internally, so the
``deap_random_seed`` value is installed only around the evolutionary loop and
then the caller's global random state is restored.
"""
from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

# Distinct odd 64-bit constants. They are not cryptographic; they simply make
# deterministic child streams far apart in the integer seed space.
_INIT_XOR = 0x9E3779B97F4A7C15
_INIT_REPAIR_XOR = 0xD1B54A32D192ED03
_DEAP_XOR = 0x94D049BB133111EB
_REPAIR_XOR = 0xBF58476D1CE4E5B9


@dataclass(frozen=True)
class RandomStreams:
    """Independent pseudo-random streams used by one ``GeneticSearch`` fit."""

    initialization: random.Random
    initialization_repair: random.Random
    repair: random.Random
    deap_random_seed: int | None


def _child_seed(seed: int | None, salt: int) -> int | None:
    return None if seed is None else int(seed) ^ salt


def make_random_streams(random_state: int | None) -> RandomStreams:
    """Create independent deterministic streams from one public seed."""
    return RandomStreams(
        initialization=random.Random(_child_seed(random_state, _INIT_XOR)),
        initialization_repair=random.Random(
            _child_seed(random_state, _INIT_REPAIR_XOR)
        ),
        repair=random.Random(_child_seed(random_state, _REPAIR_XOR)),
        deap_random_seed=_child_seed(random_state, _DEAP_XOR),
    )


@contextlib.contextmanager
def isolated_global_random(seed: int | None):
    """Temporarily seed Python's global RNG and restore it afterwards.

    DEAP's standard operators such as ``selTournament``, ``cxTwoPoint`` and
    ``mutUniformInt`` intentionally use :mod:`random`. This context manager lets
    us use those operators unmodified while keeping ``GeneticSearch`` from
    perturbing the application's global random state.
    """
    state = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(state)
