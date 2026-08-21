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

GitHub Actions completed successfully on 21 August 2026. The full test suite passed
against the released pgmpy dependency on Python 3.10, 3.11, 3.12, 3.13, and 3.14.
The separate compatibility job against pgmpy's current `dev` branch also passed.

## Interpretation

The dependency-light local suite validates the algorithmic helpers and invariants,
while the successful GitHub Actions matrix validates the pgmpy+DEAP estimator
integration across the supported Python versions and against pgmpy's development
branch at the time of testing. These software checks do not establish superiority
in DAG recovery; that remains the purpose of the pre-specified benchmark study.