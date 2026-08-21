# Contributing

Contributions are welcome, especially reproducibility checks, tests, pgmpy API-compatibility fixes,
and carefully scoped improvements to the evolutionary search or DAG-recovery benchmarks.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Before opening a pull request, please also run:

```bash
python -m compileall -q src benchmarks tests
python -m build
python -m pip check
```

## Design boundaries

- Keep generic evolutionary operators in **DEAP** rather than reimplementing them locally.
- Keep graph semantics, structure scores, and expert-knowledge conventions aligned with **pgmpy**.
- Keep Bayesian-network-specific initialization and repair logic explicit and testable.
- Do not change benchmark protocol defaults silently. If a change affects comparability of saved
  results, increment `BENCHMARK_PROTOCOL` and document it in `CHANGELOG.md`.
- New stochastic mechanisms should use an isolated RNG stream when needed for paired ablations.

## Tests

Bug fixes should include a regression test when practical. Algorithmic changes should add invariant
tests (acyclicity, hard-constraint preservation, reproducibility) in addition to outcome tests.

## Research claims

This repository contains work-in-progress research software. Please distinguish implementation
features from novelty claims and avoid strengthening claims beyond what is supported by the cited
literature and completed experiments.
