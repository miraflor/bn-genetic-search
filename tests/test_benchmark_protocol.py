"""Dependency-light tests for benchmark reproducibility and resume safety."""
import csv
import importlib.util
from pathlib import Path

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "_protocol.py"
SPEC = importlib.util.spec_from_file_location("benchmark_protocol", PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(protocol)


def _condition(**updates):
    row = {
        "benchmark_protocol": protocol.BENCHMARK_PROTOCOL,
        "software_version": "0.4.0",
        "pgmpy_version": "1.1.2",
        "deap_version": "1.4.4",
        "analysis_roles": "repair",
        "model": "child",
        "n_samples": 500,
        "replicate": 0,
        "initialization": "feasible_dag",
        "edge_selection": "mutual_info",
        "operation": "reverse_then_delete",
        "knowledge_fraction": 0.0,
        "edge_prob": 0.2,
        "population_size": 100,
        "max_iter": 200,
        "patience": 20,
        "equivalent_sample_size": 10.0,
    }
    row.update(updates)
    return row


def test_block_seed_depends_on_block_identity_not_list_position():
    assert protocol.block_seed("child", 500, 0) == protocol.block_seed("child", 500, 0)
    assert protocol.block_seed("child", 500, 0) != protocol.block_seed("asia", 500, 0)
    assert protocol.block_seed("child", 500, 0) != protocol.block_seed("child", 2000, 0)
    assert protocol.block_seed("child", 500, 0) != protocol.block_seed("child", 500, 1)


def test_row_key_normalizes_csv_strings():
    native = _condition()
    csv_like = {key: str(value) for key, value in native.items()}
    assert protocol.row_key(native) == protocol.row_key(csv_like)


def test_append_row_refuses_schema_mismatch(tmp_path):
    path = tmp_path / "results.csv"
    first = {**_condition(), "metric": 1.0}
    protocol.append_row(path, first)

    # A changed column set must not be appended under the old CSV header.
    incompatible = {**_condition(), "different_metric": 2.0}
    try:
        protocol.append_row(path, incompatible)
    except ValueError as exc:
        assert "different schema" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("schema mismatch should have been rejected")

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1


def test_read_done_rejects_legacy_schema(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_text("experiment,model,n_samples\nrepair,asia,500\n", encoding="utf-8")
    try:
        protocol.read_done(path)
    except ValueError as exc:
        assert "older/incompatible schema" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("legacy schema should have been rejected")
