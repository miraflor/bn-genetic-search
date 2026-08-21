"""Evolutionary Bayesian-network structure learning for pgmpy.

The public estimator is imported lazily. This keeps dependency-light helpers
such as encoding, repair, initialization, and recovery metrics testable even in
minimal environments where pgmpy or DEAP is not installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["GeneticSearch"]
__version__ = "0.4.0"

if TYPE_CHECKING:  # pragma: no cover
    from .GeneticSearch import GeneticSearch


def __getattr__(name):
    if name == "GeneticSearch":
        from .GeneticSearch import GeneticSearch

        # Importing the submodule above also binds the *module* named
        # ``GeneticSearch`` onto this package, shadowing this lazy hook for
        # every later lookup. Rebind the class explicitly so that
        # ``from bn_genetic_search import GeneticSearch`` resolves to the
        # estimator class rather than its defining module.
        globals()["GeneticSearch"] = GeneticSearch
        return GeneticSearch
    raise AttributeError(name)
