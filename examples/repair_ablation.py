"""Run the four repair arms on the *same* sampled data and random seed.

This is deliberately a tiny demonstration. The paper benchmark should use
multiple independently generated datasets and report paired results across
replicates; see benchmarks/dag_recovery.py.
"""
from pgmpy.example_models import load_model

from bn_genetic_search import GeneticSearch
from bn_genetic_search.metrics import skeleton_precision_recall_f1

model = load_model("bnlearn/asia")
data = model.simulate(n_samples=2000, seed=42, show_progress=False)

for edge_selection in ("random", "mutual_info"):
    for operation in ("delete", "reverse_then_delete"):
        search = GeneticSearch(
            scoring_method="bdeu",
            return_type="dag",
            population_size=100,
            max_iter=100,
            repair_edge_selection=edge_selection,
            repair_operation=operation,
            random_state=2026,
            show_progress=False,
        ).fit(data)

        recovery = skeleton_precision_recall_f1(model, search.dag_)
        print(
            edge_selection,
            operation,
            f"skeleton F1={recovery.f1:.3f}",
            search.repair_stats_,
        )
