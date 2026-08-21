# Validation record

This file records what has actually been executed for v0.4.0.

## Executed in the build environment

- `PYTHONPATH=src pytest -q`: **33 passed, 2 skipped**.
- The two skipped modules are the DEAP-specific integration test and the true pgmpy+DEAP estimator
  integration suite because those external packages are not installed in the build runtime.
- Repair stress test: **5,000** random directed graphs; every result was acyclic and the observed
  number of cycle-breaking iterations never exceeded the input edge count.
- Feasible-initialization stress test: **1,000** constrained random DAGs; every result was acyclic,
  retained all required edges, excluded all forbidden edges, and respected `max_indegree`.
- All Python sources compile with `compileall` as part of the release checks.
- The project builds successfully as a wheel with the locally available build toolchain.
- Benchmark protocol now uses model-name-derived stable seeds, density-matched initialization arms,
  deterministic within-block arm shuffling, fresh per-arm BDeu score objects, and CSV schema guards.

## CI validation after push

GitHub Actions installs the declared dependencies and runs the complete suite on supported Python
versions. A separate job replaces the released pgmpy package with pgmpy's current `dev` branch to
identify API drift relevant to eventual upstreaming.

## Interpretation

A green local dependency-light suite validates the algorithmic helpers and invariants, but it is
not a substitute for the external integration suite. The repository should not claim full pgmpy
integration until the CI job is green.
