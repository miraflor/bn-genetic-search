# Algorithm design and contribution boundary

## 1. Search object

For `n` variables, a chromosome contains `n(n-1)/2` ternary loci, one per unordered pair. For the
canonical ordered pair `(Xi, Xj)` with `i < j`:

- `0`: no adjacency;
- `1`: `Xi -> Xj`;
- `2`: `Xj -> Xi`.

This representation makes self-loops and two-cycles impossible at the locus level, but longer
cycles can still arise after variation. Ternary pair representations have prior art, including
modern BN-GOMEA work, so the representation is infrastructure rather than the primary novelty.

## 2. Generic evolution is DEAP

The implementation delegates generic evolutionary mechanics to DEAP:

- `tools.selTournament` for selection;
- `tools.cxTwoPoint` for crossover;
- `tools.mutUniformInt(low=0, up=2)` for mutation;
- DEAP fitness classes and cloning;
- `tools.selBest` for elitism;
- `tools.HallOfFame` and `tools.Logbook` for diagnostics.

The evolutionary loop remains explicit only because Bayesian-network repair must occur *after*
variation and *before* pgmpy score evaluation.

## 3. Constraint-aware feasible-DAG initialization

Production initialization does not generate arbitrary ternary strings. It:

1. validates that required edges are mutually consistent and acyclic;
2. constructs a randomized topological order extending the partial order implied by required edges;
3. inserts all required edges;
4. considers only forward edges in that order;
5. skips forbidden edges and directions outside an optional search-space whitelist;
6. respects `max_indegree`; and
7. samples each remaining admissible edge with probability `edge_prob`.

Every resulting individual is therefore a DAG satisfying the declared hard constraints by
construction. The paper does **not** claim that random-DAG initialization in general is new. It
tests the narrower design hypothesis that beginning directly in the feasible space is preferable
to generating unconstrained chromosomes and allowing the repair operator to shape generation zero.

## 4. Cycle-localized repair

After DEAP crossover or mutation, a candidate is decoded as a directed graph. Hard exclusions are
applied, required edges are inserted, and directed cycles are repaired one at a time.

For a detected cycle `C`, only non-required edges in `C` are eligible. The edge can be selected by:

- `random`: experimental control; or
- `mutual_info`: choose the lowest pairwise mutual-information edge in the detected cycle.

With `reverse_then_delete`, an eligible edge `u -> v` is removed and `v -> u` is inserted only if:

- the reverse direction is not forbidden;
- it lies inside the search-space whitelist, if one exists;
- it respects `max_indegree`; and
- after removing `u -> v`, there is no path `u => v`.

The final condition is exact: if a path `u => v` remained, adding `v -> u` would create a directed
cycle. If reversal is not legal, the selected edge is deleted.

Mutual-information-guided deletion and reversal-based cycle handling both have prior art. The
candidate contribution is therefore the integrated, cycle-localized, constraint-preserving
reverse-then-delete variant and its controlled evaluation, not the isolated primitives.

## 5. Termination

Each repair iteration selects an edge on a currently existing directed cycle. Deletion removes that
edge. A legal reversal destroys the selected cycle and creates no new directed cycle. The reversed
edge cannot later enter a cycle because subsequent repair operations also create no new cycles.
Consequently each iteration permanently removes one currently cyclic edge from future cycle
participation. The number of cycle-breaking iterations is therefore at most the number of edges in
the graph entering the cycle-repair stage.

With a graph traversal/path check bounded by `O(|V|+|E|)` per iteration, the repair stage is bounded
by `O(|E|(|V|+|E|))`, excluding the one-time mutual-information preprocessing.

## 6. Structural knowledge

Required/forbidden edges are established ideas in Bayesian-network learning and are **not** claimed
as novel. They are nevertheless first-class algorithm capabilities because feasibility should be
preserved throughout the search, not patched onto the final answer. The code follows pgmpy's
`ExpertKnowledge` terminology to make an eventual upstream contribution easier to review.

## 7. Paper ablations

The WCTP paper is designed around two focused experiments rather than a large factorial sweep.

### A. Initialization

Hold repair fixed at `mutual_info + reverse_then_delete` and compare:

- raw ternary chromosome + repair;
- feasible-DAG initialization.

### B. Repair

Hold feasible-DAG initialization fixed and cross:

| edge selection | delete | safe reverse -> delete |
|---|---:|---:|
| random | R-D | R-R |
| mutual information | MI-D | MI-R |

The proposed method is MI-R. This design estimates the effect of information guidance, the effect
of adjacency-preserving safe reversal, and their interaction without changing the rest of the GA.

## Initialization control used in the paper

The `random_chromosome_repair` arm is **density-matched** to the feasible-DAG
initializer: each unordered pair is adjacent with marginal probability `edge_prob`,
and conditional on adjacency its orientation is chosen with probability 1/2. The
raw chromosome is then repaired. A uniform draw over {0,1,2} would imply 2/3 edge
density and would confound the initialization comparison.
