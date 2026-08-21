# BN Genetic Search

> **Status:** independent research implementation accompanying a work-in-progress paper; not an official pgmpy package.

An experimental Bayesian-network structure learner built with **pgmpy** for graphical-model
semantics and scoring and **DEAP** for generic evolutionary computation. The code modernizes a
2022 pgmpy+DEAP research implementation while keeping the public estimator deliberately close to
pgmpy's current causal-discovery nomenclature.

The research question is not whether genetic algorithms can learn Bayesian networks; that has a
long literature. The package isolates two feasibility mechanisms that can be tested directly:

1. **constraint-aware feasible-DAG initialization**: generation zero is sampled directly from the
   admissible DAG space instead of generating arbitrary chromosomes and repairing them; and
2. **cycle-localized information-guided safe repair**: when variation creates a directed cycle,
   choose a cycle edge (lowest pairwise mutual information or random), try a cycle-safe legal
   reversal, and fall back to deletion.

Required and forbidden edges, search-space restrictions, and maximum indegree are hard constraints
throughout initialization and repair.


## Author and affiliation

**James Matthew Miraflor**  
Scientific Computing Laboratory (SCL)  
Department of Computer Science (DCS)  
University of the Philippines Diliman  
Email: `jbmiraflor@up.edu.ph`

The affiliation identifies the author and does not imply that this repository is an official release
of SCL, DCS, the University of the Philippines, pgmpy, or DEAP.

## Division of labour

- **pgmpy:** `DAG`/`PDAG`, structure scores, `ExpertKnowledge`, optional hill-climb refinement,
  example Bayesian networks, and compatible estimator conventions.
- **DEAP:** fitness containers, tournament selection, two-point crossover, uniform-integer
  mutation, cloning, elitist helpers, Hall of Fame, and logging primitives.
- **This package:** BN chromosome interpretation, feasible initialization, constraint handling,
  cycle repair, score caching, experimental diagnostics, and DAG-recovery benchmarks.

Generic evolutionary operators are intentionally *not* reimplemented.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Basic use

```python
from bn_genetic_search import GeneticSearch

search = GeneticSearch(
    scoring_method="bdeu",
    return_type="pdag",
    initialization="feasible_dag",
    population_size=100,
    max_iter=200,
    repair_edge_selection="mutual_info",
    repair_operation="reverse_then_delete",
    random_state=42,
)
search.fit(data)

print(search.causal_graph_)
print(search.best_score_)
print(search.repair_stats_)
```

For a non-default BDeu equivalent sample size, use pgmpy's score object:

```python
from pgmpy.structure_score import BDeu

score = BDeu(data, equivalent_sample_size=5)
search = GeneticSearch(scoring_method=score, random_state=42).fit(data)
```


## Data scope

The paper and the default `repair_edge_selection="mutual_info"` configuration target **discrete**
Bayesian networks. Pairwise mutual information is used only as a repair heuristic; it treats
observed values as categorical states. For genuinely continuous or mixed data, use a non-MI repair
rule or a future data-type-appropriate edge-weight heuristic.

## Expert knowledge

```python
from pgmpy.causal_discovery import ExpertKnowledge

expert = ExpertKnowledge(
    required_edges=[("A", "B")],
    forbidden_edges=[("C", "A")],
)

search = GeneticSearch(
    scoring_method="bdeu",
    expert_knowledge=expert,
    max_indegree=4,
    random_state=42,
).fit(data)
```

The feasible initializer begins with all required edges, draws a randomized topological order
consistent with them, and adds only forward admissible edges. During repair, required edges are
never candidates for deletion; a reversal is rejected if it would be cyclic, forbidden, outside
the search-space whitelist, or inconsistent with `max_indegree`.

## Paper experiments

### Initialization ablation

The raw-chromosome control uses the same marginal `edge_prob` as feasible-DAG
initialization and assigns either orientation with equal probability. This avoids
confounding feasibility with starting graph density.

```python
for initialization in ["random_chromosome_repair", "feasible_dag"]:
    search = GeneticSearch(
        scoring_method="bdeu",
        initialization=initialization,
        repair_edge_selection="mutual_info",
        repair_operation="reverse_then_delete",
        random_state=42,
    ).fit(data)
```

### Repair 2 x 2 ablation

```python
for edge_selection in ["random", "mutual_info"]:
    for operation in ["delete", "reverse_then_delete"]:
        search = GeneticSearch(
            scoring_method="bdeu",
            initialization="feasible_dag",
            repair_edge_selection=edge_selection,
            repair_operation=operation,
            random_state=42,
        ).fit(data)
```

`benchmarks/dag_recovery.py` runs both experiments resumably on pgmpy's standard benchmark
networks. The proposed feasible+MI+safe-reversal configuration is shared by both contrasts and is
executed only once per model/sample/replicate block when `--experiment all` is used. When analyzing
results, filter rows whose `analysis_roles` value **contains** the experiment name — the shared arm
is stored once with `analysis_roles=initialization+repair`, so an equality filter on `repair` would
silently drop the proposed MI+reversal arm. The runner records equivalence-class-aware recovery metrics, skeleton/v-structure recovery,
runtime, score evaluations, generation-zero score/edge summaries, and repair diagnostics separated into
initialization and evolutionary phases.

## Reproducible ablations

The master `random_state` is split into independent streams for:

- initial population sampling,
- repair used only during the raw-chromosome initialization arm,
- DEAP selection/crossover/mutation, and
- repair after evolutionary variation.

This prevents one treatment from changing another treatment's later random draws simply because it
consumes a different number of random numbers.

## Validation status

See `docs/VALIDATION.md`. In the present build environment the dependency-light test suite passes,
and stress tests cover repair termination/acyclicity and constraint-aware initialization. The
full pgmpy+DEAP estimator integration suite is included in CI and runs when those external packages
are installed.

## Scope

This repository intentionally excludes the earlier DBN extension. Dynamic Bayesian networks,
large hyperparameter sweeps, and additional evolutionary frameworks are better treated as later
extensions after the static DAG-recovery study is complete.
