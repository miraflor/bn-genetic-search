# Changelog

## 0.4.0 - 2026-08-21

- Restored **DEAP as a required dependency** for generic evolutionary computation.
- Added constraint-aware **feasible-DAG initialization** as the production default.
- Added `initialization="random_chromosome_repair"` as a controlled ablation arm.
- Preserved cycle-localized mutual-information repair and safe reverse-then-delete logic.
- Enforced required/forbidden/search-space/max-indegree constraints during initialization and repair.
- Split random streams for initialization, initialization repair, DEAP variation, and evolutionary repair.
- Added DEAP Hall-of-Fame and Logbook diagnostics.
- Expanded the DAG-recovery benchmark to separate initialization and repair experiments.
- Made benchmark seeds independent of CLI model ordering, deterministically shuffled arm order, and deduplicated the shared proposed arm across the two paper contrasts.
- Added benchmark protocol/dependency metadata, explicit BDeu equivalent sample size, normalized CPDAG distance, and CSV schema protection.
- Rejected MI-guided repair when pgmpy explicitly identifies a continuous or mixed structure score.
- Added SCL/DCS affiliation metadata and public-repository release hygiene.
- Kept pgmpy-compatible estimator naming and current-release/dev compatibility shims.
- Local validation: 33 tests passed; two external integration modules skipped because pgmpy and DEAP
  are unavailable in the build runtime; 5,000 repair and 1,000 initialization stress cases passed.

## 0.3.0

- Introduced pgmpy-shaped causal-discovery API and DAG-recovery metrics.
- Temporarily replaced DEAP with local evolutionary primitives; reversed in 0.4.0 after design review.
