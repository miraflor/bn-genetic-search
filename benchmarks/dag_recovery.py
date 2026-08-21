"""Resumable DAG-recovery experiments for the WCTP paper.

The runner intentionally separates two scientific questions:

A. Initialization: does feasible-DAG initialization outperform generating raw
   ternary chromosomes and repairing generation zero?
B. Repair: with feasible initialization fixed, what are the main and interaction
   effects of information-guided edge selection and safe reversal?

Examples
--------
Run a small smoke benchmark:

    python benchmarks/dag_recovery.py --experiment all --models asia --samples 500 --replicates 2

Run the proposed WCTP design:

    python benchmarks/dag_recovery.py --experiment all \
        --models asia child insurance alarm \
        --samples 500 2000 10000 --replicates 20

Rows are appended as soon as a condition finishes. Re-running the same command
skips completed conditions, making long experiments resumable.
"""
from __future__ import annotations

import argparse
import random
import time
from importlib.metadata import PackageNotFoundError, version
from itertools import permutations
from pathlib import Path

from _protocol import (  # direct-script sibling import
    BENCHMARK_PROTOCOL,
    append_row,
    block_seed,
    read_done,
    row_key,
)
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.example_models import load_model
from pgmpy.metrics import SHD
from pgmpy.structure_score import BDeu

from bn_genetic_search import GeneticSearch
from bn_genetic_search import __version__ as SOFTWARE_VERSION
from bn_genetic_search.metrics import (
    cpdag_pairwise_distance,
    skeleton_precision_recall_f1,
    v_structure_precision_recall_f1,
)

# Five unique algorithm configurations support two overlapping contrasts.
# The proposed feasible+MI+safe-reversal arm belongs to *both* experiments,
# so ``--experiment all`` runs it once rather than duplicating an identical fit.
ARMS = [
    {
        "analysis_roles": {"initialization"},
        "initialization": "random_chromosome_repair",
        "edge_selection": "mutual_info",
        "operation": "reverse_then_delete",
    },
    {
        "analysis_roles": {"initialization", "repair"},
        "initialization": "feasible_dag",
        "edge_selection": "mutual_info",
        "operation": "reverse_then_delete",
    },
    *[
        {
            "analysis_roles": {"repair"},
            "initialization": "feasible_dag",
            "edge_selection": edge_selection,
            "operation": operation,
        }
        for edge_selection in ("random", "mutual_info")
        for operation in ("delete", "reverse_then_delete")
        if (edge_selection, operation) != ("mutual_info", "reverse_then_delete")
    ],
]


def experiment_arms(experiment: str):
    if experiment == "all":
        return list(ARMS)
    if experiment in {"initialization", "repair"}:
        return [arm for arm in ARMS if experiment in arm["analysis_roles"]]
    raise ValueError(experiment)


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "source-tree"


def constraint_sample(true_dag, fraction: float, seed: int):
    """Create correct required/forbidden knowledge for mechanism validation.

    This is not a primary WCTP novelty experiment. It exists to verify that the
    implementation preserves pgmpy-style hard knowledge and to support a later
    journal extension.
    """
    if fraction <= 0:
        return None, 0, 0

    rng = random.Random(seed)
    true_edges = sorted(true_dag.edges(), key=repr)
    n_required = min(len(true_edges), max(1, round(fraction * len(true_edges))))
    required = set(rng.sample(true_edges, n_required))

    nodes = list(true_dag.nodes())
    absent_directions = [
        edge
        for edge in permutations(nodes, 2)
        if edge not in set(true_edges) and edge not in required
    ]
    n_forbidden = min(len(absent_directions), n_required)
    forbidden = set(rng.sample(absent_directions, n_forbidden))
    expert = ExpertKnowledge(required_edges=required, forbidden_edges=forbidden)
    return expert, len(required), len(forbidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["initialization", "repair", "all"],
        default="all",
    )
    parser.add_argument(
        "--models", nargs="+", default=["asia", "child", "insurance", "alarm"]
    )
    parser.add_argument(
        "--samples", nargs="+", type=int, default=[500, 2000, 10000]
    )
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("results/dag_recovery.csv")
    )
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--edge-prob", type=float, default=0.2)
    parser.add_argument(
        "--equivalent-sample-size",
        type=float,
        default=10.0,
        help="BDeu equivalent sample size; fixed across all benchmark arms.",
    )
    parser.add_argument(
        "--knowledge-fraction",
        type=float,
        default=0.0,
        help="Optional correct hard knowledge for validation/journal extensions.",
    )
    args = parser.parse_args()

    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if any(value < 1 for value in args.samples):
        parser.error("all --samples values must be positive")
    if not 0.0 <= args.knowledge_fraction <= 1.0:
        parser.error("--knowledge-fraction must lie in [0, 1]")
    if args.equivalent_sample_size <= 0:
        parser.error("--equivalent-sample-size must be positive")

    pgmpy_version = _installed_version("pgmpy")
    deap_version = _installed_version("deap")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(args.output)
    shd = SHD()
    arms = experiment_arms(args.experiment)

    for model_name in args.models:
        model = load_model(f"bnlearn/{model_name}")
        true_dag = model.copy()
        true_cpdag = true_dag.to_pdag()

        for n_samples in args.samples:
            for replicate in range(args.replicates):
                # One dataset seed per model/n/replicate. Every ablation arm sees
                # the exact same data and GeneticSearch master seed.
                seed = block_seed(model_name, n_samples, replicate)
                data = model.simulate(
                    n_samples=n_samples,
                    seed=seed,
                    show_progress=False,
                )
                # pgmpy's simulate() emits columns in an order that varies with
                # interpreter hash randomization. Chromosome loci are positional
                # in the column order, so canonicalize it; otherwise a resumed
                # or re-run benchmark pairs arms against a permuted search
                # space and results are not reproducible across invocations.
                data = data[sorted(data.columns, key=str)]
                expert, n_required, n_forbidden = constraint_sample(
                    true_dag,
                    args.knowledge_fraction,
                    seed ^ 0xA5A5A5A5,
                )

                # Deterministically shuffle execution order within each block so
                # runtime comparisons are not systematically tied to arm order.
                block_arms = list(arms)
                random.Random(seed ^ 0x6C8E9CF5).shuffle(block_arms)

                for arm in block_arms:
                    condition_row = {
                        "benchmark_protocol": BENCHMARK_PROTOCOL,
                        "software_version": SOFTWARE_VERSION,
                        "pgmpy_version": pgmpy_version,
                        "deap_version": deap_version,
                        "analysis_roles": "+".join(sorted(arm["analysis_roles"])),
                        "model": model_name,
                        "n_samples": n_samples,
                        "replicate": replicate,
                        "initialization": arm["initialization"],
                        "edge_selection": arm["edge_selection"],
                        "operation": arm["operation"],
                        "knowledge_fraction": args.knowledge_fraction,
                        "edge_prob": args.edge_prob,
                        "population_size": args.population_size,
                        "max_iter": args.max_iter,
                        "patience": args.patience,
                        "equivalent_sample_size": args.equivalent_sample_size,
                    }
                    if row_key(condition_row) in done:
                        continue

                    start = time.perf_counter()
                    # Construct a fresh score object for every arm so pgmpy's
                    # local-score cache is not warmed by an earlier treatment.
                    score = BDeu(
                        data,
                        equivalent_sample_size=args.equivalent_sample_size,
                    )
                    search = GeneticSearch(
                        scoring_method=score,
                        expert_knowledge=expert,
                        return_type="dag",
                        initialization=arm["initialization"],
                        population_size=args.population_size,
                        max_iter=args.max_iter,
                        patience=args.patience,
                        edge_prob=args.edge_prob,
                        repair_edge_selection=arm["edge_selection"],
                        repair_operation=arm["operation"],
                        random_state=seed,
                        show_progress=False,
                    ).fit(data)
                    elapsed = time.perf_counter() - start

                    est_cpdag = search.dag_.to_pdag()
                    dag_shd = shd(
                        true_causal_graph=true_dag,
                        est_causal_graph=search.dag_,
                    )
                    cpdag_distance = cpdag_pairwise_distance(true_cpdag, est_cpdag)
                    n_pairs = len(true_dag.nodes()) * (len(true_dag.nodes()) - 1) // 2
                    cpdag_distance_normalized = (
                        cpdag_distance / n_pairs if n_pairs else 0.0
                    )
                    skeleton = skeleton_precision_recall_f1(true_dag, search.dag_)
                    vstruct = v_structure_precision_recall_f1(true_dag, search.dag_)

                    row = {
                        **condition_row,
                        "seed": seed,
                        "dag_shd": dag_shd,
                        "cpdag_pairwise_distance": cpdag_distance,
                        "cpdag_pairwise_distance_normalized": cpdag_distance_normalized,
                        "skeleton_precision": skeleton.precision,
                        "skeleton_recall": skeleton.recall,
                        "skeleton_f1": skeleton.f1,
                        "vstructure_precision": vstruct.precision,
                        "vstructure_recall": vstruct.recall,
                        "vstructure_f1": vstruct.f1,
                        "true_edges": true_dag.number_of_edges(),
                        "estimated_edges": search.dag_.number_of_edges(),
                        "required_edges": n_required,
                        "forbidden_edges": n_forbidden,
                        "best_score": search.best_score_,
                        "generations": search.n_iter_,
                        "score_evaluations": search.n_score_evaluations_,
                        "initial_best_score": search.initial_population_summary_["best_score"],
                        "initial_mean_score": search.initial_population_summary_["mean_score"],
                        "initial_mean_edges": search.initial_population_summary_["mean_edges"],
                        "initial_min_edges": search.initial_population_summary_["min_edges"],
                        "initial_max_edges": search.initial_population_summary_["max_edges"],
                        "initial_unique_structures": search.initial_population_summary_["unique_structures"],
                        "initial_repair_calls": search.initialization_repair_stats_["calls"],
                        "initial_cycles_found": search.initialization_repair_stats_["cycles_found"],
                        "initial_reversals": search.initialization_repair_stats_["reversals"],
                        "initial_cycle_deletions": search.initialization_repair_stats_["cycle_deletions"],
                        "evolution_repair_calls": search.evolution_repair_stats_["calls"],
                        "evolution_cycles_found": search.evolution_repair_stats_["cycles_found"],
                        "evolution_reversals": search.evolution_repair_stats_["reversals"],
                        "evolution_cycle_deletions": search.evolution_repair_stats_["cycle_deletions"],
                        "repair_calls": search.repair_stats_["calls"],
                        "cycles_found": search.repair_stats_["cycles_found"],
                        "reversals": search.repair_stats_["reversals"],
                        "cycle_deletions": search.repair_stats_["cycle_deletions"],
                        "forbidden_deletions": search.repair_stats_["forbidden_deletions"],
                        "indegree_deletions": search.repair_stats_["indegree_deletions"],
                        "runtime_seconds": elapsed,
                    }
                    append_row(args.output, row)
                    done.add(row_key(row))
                    print(
                        row_key(row),
                        f"CPDAG-distance={cpdag_distance}",
                        f"skeleton-F1={skeleton.f1:.3f}",
                        f"{elapsed:.2f}s",
                    )


if __name__ == "__main__":
    main()
