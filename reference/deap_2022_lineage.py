"""Historical mapping from the original 2022 research code to this package.

This file is documentation only; it is not imported by the package. The modern
implementation deliberately retains DEAP as the generic evolutionary engine.
The important modernization is therefore not replacement of DEAP, but a clean
division of responsibility: DEAP handles generic evolutionary operators while
Bayesian-network-specific feasibility, repair, constraints, and pgmpy scoring
remain in this project.
"""

# Historical -> current conceptual mapping:
#
#   DEAP tools.selTournament  -> _deap.make_toolbox(...).select
#   DEAP tools.cxTwoPoint     -> _deap.make_toolbox(...).mate
#   DEAP mutUniformInt        -> _deap.make_toolbox(...).mutate
#   DEAP HallOfFame/Logbook   -> GeneticSearch fitted diagnostics
#   2022 ternary encoding     -> _encoding.py
#   2022 MI repair idea       -> _repair.py (cycle-localized and cycle-safe)
#   random valid DAG idea     -> _initialization.py (constraint-aware form)
