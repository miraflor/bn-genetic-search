"""Minimal end-to-end example using a pgmpy reference Bayesian network."""
from pgmpy.example_models import load_model

from bn_genetic_search import GeneticSearch

# pgmpy's bundled Asia network gives us a known data-generating DAG. Sampling
# from a known model is also the basis of the recovery experiments in
# benchmarks/dag_recovery.py.
model = load_model("bnlearn/asia")
data = model.simulate(n_samples=2000, seed=42, show_progress=False)

# The constructor deliberately follows pgmpy's causal-discovery vocabulary.
# Only population/search-specific settings are additional.
search = GeneticSearch(
    scoring_method="bdeu",
    return_type="pdag",
    population_size=100,
    max_iter=200,
    repair_edge_selection="mutual_info",
    repair_operation="reverse_then_delete",
    random_state=42,
    show_progress=True,
)
search.fit(data)

print(search.causal_graph_)
print("best structure score:", search.best_score_)
print("repair diagnostics:", search.repair_stats_)
