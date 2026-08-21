"""Dependency-light reproducibility helpers for the DAG-recovery benchmark."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

BENCHMARK_PROTOCOL = "wctp2026-v1"


def block_seed(model_name: str, n_samples: int, replicate: int) -> int:
    """Return a stable 32-bit seed independent of CLI model ordering.

    Python's built-in ``hash`` is randomized between interpreter processes, so
    it is unsuitable for a resumable benchmark. Deriving the seed from the
    semantic block identity makes ``child`` generate the same dataset whether it
    is run alone or inside a longer ``--models`` list.
    """
    key = f"{BENCHMARK_PROTOCOL}|{model_name}|{n_samples}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def row_key(row):
    """Normalize the fields that uniquely identify one benchmark condition."""
    return (
        str(row.get("benchmark_protocol", "unknown")),
        str(row.get("software_version", "unknown")),
        str(row.get("pgmpy_version", "unknown")),
        str(row.get("deap_version", "unknown")),
        str(row["analysis_roles"]),
        str(row["model"]),
        int(row["n_samples"]),
        int(row["replicate"]),
        str(row["initialization"]),
        str(row["edge_selection"]),
        str(row["operation"]),
        str(row.get("knowledge_fraction", "0.0")),
        str(row.get("edge_prob", "0.2")),
        str(row.get("population_size", "100")),
        str(row.get("max_iter", "200")),
        str(row.get("patience", "20")),
        str(row.get("equivalent_sample_size", "10.0")),
    )


def read_done(path: Path):
    """Read completed condition keys, rejecting incompatible legacy schemas."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "analysis_roles" not in reader.fieldnames:
            raise ValueError(
                "Existing benchmark CSV uses an older/incompatible schema. "
                "Use a new --output path for this benchmark protocol."
            )
        return {row_key(row) for row in reader}


def append_row(path: Path, row: dict):
    """Append one result row without silently corrupting an older CSV schema."""
    fieldnames = list(row)
    first = not path.exists() or path.stat().st_size == 0

    if not first:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            existing_header = next(reader, [])
        if existing_header != fieldnames:
            raise ValueError(
                "Existing benchmark CSV has a different schema. "
                "Use a new --output path (recommended after protocol/code changes) "
                "instead of mixing incompatible result rows."
            )

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if first:
            writer.writeheader()
        writer.writerow(row)
