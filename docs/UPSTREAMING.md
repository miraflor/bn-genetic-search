# Eventual pgmpy contribution path

The standalone package intentionally uses pgmpy nomenclature:

- estimator method: `fit(X)`;
- fitted graph: `causal_graph_`;
- fitted adjacency: `adjacency_matrix_`;
- feature metadata: `n_features_in_`, `feature_names_in_`;
- arguments: `scoring_method`, `start_dag`, `max_indegree`, `expert_knowledge`, `return_type`,
  `epsilon`, `max_iter`, `show_progress`.

The remaining parameters are specific to evolutionary search.

The main policy question for an upstream proposal is DEAP. This research package should use DEAP
because generic evolutionary machinery is not the research contribution. If pgmpy maintainers are
interested in the estimator but do not want a core DEAP dependency, reasonable later options are:

1. make DEAP an optional pgmpy extra for evolutionary search;
2. keep `bn-genetic-search` as a companion package with a pgmpy-compatible API; or
3. only after maintainer discussion, replace the thin DEAP layer for upstream inclusion.

The research implementation should not prematurely reimplement DEAP merely to anticipate that
future packaging decision.
