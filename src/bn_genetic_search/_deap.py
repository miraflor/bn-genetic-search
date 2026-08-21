"""Thin DEAP integration layer.

Generic evolutionary computation belongs to DEAP. This project deliberately
*does not* reimplement tournament selection, crossover, mutation, Hall-of-Fame
tracking, or DEAP fitness containers. The only custom pieces are those that are
specific to Bayesian-network structure learning: initialization, repair,
constraint handling, pgmpy scoring, and evaluation.
"""
from __future__ import annotations

from deap import base, creator, tools

_FITNESS_NAME = "BNStructureFitnessMax"
_INDIVIDUAL_NAME = "BNStructureIndividual"


def get_individual_class():
    """Create (once) and return the DEAP individual class used by the package."""
    if not hasattr(creator, _FITNESS_NAME):
        creator.create(_FITNESS_NAME, base.Fitness, weights=(1.0,))
    fitness_cls = getattr(creator, _FITNESS_NAME)

    if not hasattr(creator, _INDIVIDUAL_NAME):
        creator.create(_INDIVIDUAL_NAME, list, fitness=fitness_cls)
    return getattr(creator, _INDIVIDUAL_NAME)


def make_toolbox(*, tournament_size: int, gene_mutation_prob: float):
    """Return a DEAP toolbox containing only generic evolutionary operators."""
    individual_cls = get_individual_class()
    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=0,
        up=2,
        indpb=gene_mutation_prob,
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    return toolbox, individual_cls
