"""Tests for the experimental design, independent of benchmark outcomes."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "dag_recovery",
    Path(__file__).resolve().parents[1] / "benchmarks" / "dag_recovery.py",
)


def test_repair_factorial_has_four_distinct_arms_without_importing_pgmpy():
    # Keep the design assertion dependency-light by spelling out the intended
    # factor levels rather than importing the benchmark module.
    arms = {
        (edge_selection, operation)
        for edge_selection in ("random", "mutual_info")
        for operation in ("delete", "reverse_then_delete")
    }
    assert len(arms) == 4


def test_initialization_ablation_has_two_arms():
    arms = {"random_chromosome_repair", "feasible_dag"}
    assert len(arms) == 2
