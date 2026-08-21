"""Evolutionary Bayesian-network structure learning using pgmpy + DEAP.

The public API deliberately follows pgmpy's current causal-discovery naming.
DEAP supplies the generic evolutionary operators; this package supplies the
Bayesian-network-specific representation, feasible initialization, constraint
handling, cycle repair, scoring bridge, and research diagnostics.
"""
from __future__ import annotations

import copy
import random
from collections.abc import Hashable
from dataclasses import asdict, dataclass
from statistics import fmean

import networkx as nx
import numpy as np
import pandas as pd
from deap import tools
from joblib import Parallel, delayed
from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge, HillClimbSearch
from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from sklearn.base import clone
from tqdm.auto import trange

# pgmpy 1.1.2 still exposes the private name _BaseCausalDiscovery. The current
# development branch follows the extension template and promotes the public
# BaseCausalDiscovery name. This compatibility import is intentionally narrow;
# an eventual upstream pgmpy PR should target the then-current dev API only.
try:  # pragma: no cover - branch depends on the installed pgmpy version
    from pgmpy.causal_discovery._base import BaseCausalDiscovery
except ImportError:  # pgmpy 1.1.2
    from pgmpy.causal_discovery._base import _BaseCausalDiscovery as BaseCausalDiscovery

from ._deap import make_toolbox
from ._encoding import code_length, decode_code, encode_dag
from ._initialization import sample_feasible_code, sample_raw_repaired_code
from ._repair import RepairStats, pairwise_mutual_information, repair_code, repair_graph
from ._rng import isolated_global_random, make_random_streams


@dataclass(frozen=True)
class GenerationRecord:
    """Compact per-generation diagnostics stored in :attr:`history_`."""

    generation: int
    best_score: float
    mean_score: float
    unique_scores: int
    unique_structures: int
    cache_size: int


class GeneticSearch(BaseCausalDiscovery):
    """Score-based causal discovery using evolutionary search over BN structures.

    ``GeneticSearch`` is a modernized descendant of a 2022 pgmpy+DEAP research
    implementation. It retains three ideas that are important for the associated
    paper: (i) direct ternary encoding of unordered node pairs, (ii) initialization
    directly in the feasible DAG space, and (iii) cycle-localized repair that can
    choose the weakest mutual-information edge and safely reverse it before
    falling back to deletion. Required/forbidden edges are treated as hard search-
    space constraints throughout initialization and repair.

    Generic evolutionary operations are delegated to DEAP. In particular,
    tournament selection, two-point crossover, uniform-integer mutation, fitness
    containers, and Hall-of-Fame tracking are not reimplemented here.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        pgmpy structure score to maximize. Accepted strings are whatever the
        installed pgmpy version registers (for discrete data, for example,
        ``"bdeu"`` or ``"bic-d"``).

    start_dag : DAG instance, default=None
        Optional DAG inserted into generation zero after constraint-aware repair.
        The name follows pgmpy's ``HillClimbSearch`` API.

    max_indegree : int or None, default=None
        Maximum number of parents allowed for each variable.

    expert_knowledge : ExpertKnowledge instance, default=None
        pgmpy structural knowledge. Required edges, forbidden edges, temporal
        restrictions, and search-space whitelists are resolved using pgmpy's own
        ``ExpertKnowledge`` conventions.

    return_type : {"dag", "pdag"}, default="pdag"
        Store either the best DAG or its Markov-equivalence-class PDAG/CPDAG in
        :attr:`causal_graph_`.

    initialization : {"feasible_dag", "random_chromosome_repair"}, default="feasible_dag"
        ``"feasible_dag"`` constructs generation-zero DAGs directly under the
        declared constraints, without post-hoc cycle repair. It is not claimed
        to sample uniformly from all admissible DAGs. ``"random_chromosome_repair"``
        creates unconstrained ternary
        chromosomes and repairs them; it is retained principally for the paper's
        initialization ablation.

    population_size : int, default=100
        Number of individuals in each generation.

    crossover_prob : float, default=0.5
        Probability of DEAP two-point crossover for each selected parent pair.

    mutation_prob : float, default=0.2
        Probability of invoking DEAP mutation on an offspring individual.

    gene_mutation_prob : float or None, default=None
        Per-locus probability passed to ``deap.tools.mutUniformInt``. ``None``
        uses ``1 / chromosome_length``.

    tournament_size : int, default=3
        Tournament size passed to ``deap.tools.selTournament``.

    elite_size : int, default=2
        Number of best individuals copied unchanged to the next generation.

    edge_prob : float, default=0.2
        Bernoulli probability for each admissible forward edge during feasible-
        DAG initialization.

    repair_edge_selection : {"mutual_info", "random"}, default="mutual_info"
        Rule for selecting an edge from a detected directed cycle. The
        ``"mutual_info"`` option is a categorical/discrete-data heuristic and
        is rejected for pgmpy scores that explicitly target continuous or mixed
        data.

    repair_operation : {"delete", "reverse_then_delete"}, default="reverse_then_delete"
        Delete the selected edge immediately, or first try a legal cycle-safe
        reversal and delete only if reversal is not admissible.

    refine : bool, default=False
        If True, run pgmpy's ``HillClimbSearch`` from the best evolutionary DAG.
        This reproduces the historical hybrid option but is deliberately off in
        the repair/initialization ablation experiments.

    epsilon : float, default=1e-4
        Improvement threshold used to reset the early-stopping patience counter.

    patience : int, default=20
        Stop after this many generations without improvement greater than
        ``epsilon``.

    max_iter : int, default=200
        Maximum number of evolutionary generations.

    n_jobs : int, default=1
        Parallel threads used to score previously unseen chromosomes.

    random_state : int or None, default=None
        Master seed. Independent child streams are used for initialization,
        initialization repair, DEAP variation, and evolutionary repair so that
        ablation treatments do not shift unrelated random draws.

    show_progress : bool, default=True
        Show a generation progress bar when pgmpy's global configuration permits.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        Learned graph in the requested return type.
    dag_ : DAG
        Best learned DAG before optional conversion to PDAG.
    adjacency_matrix_ : pandas.DataFrame
        Adjacency representation of :attr:`causal_graph_`.
    best_score_ : float
        pgmpy structure score of :attr:`dag_`.
    ga_score_ : float
        Evolutionary score before optional hill-climb refinement.
    history_ : list[GenerationRecord]
        Generation-level search diagnostics.
    repair_stats_ : dict
        Aggregate repair mechanism counts.
    deap_hall_of_fame_ : deap.tools.HallOfFame
        Best DEAP individual(s) seen during search.
    deap_logbook_ : deap.tools.Logbook
        Compact DEAP-native generation log.
    n_score_evaluations_ : int
        Number of unique chromosomes scored outside the cache.
    n_features_in_ : int
        Number of variables in the fitted data.
    feature_names_in_ : numpy.ndarray
        Variable names seen during fit.

    Notes
    -----
    The ternary representation, random DAG generation in general, hard structural
    restrictions, and generic genetic algorithms all have prior art. The paper
    therefore treats the proposed contribution as an integrated feasibility-
    preserving design and tests the effects of feasible-DAG initialization and
    the information-guided safe repair operator by controlled ablation.
    """

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        start_dag: DAG | None = None,
        max_indegree: int | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        return_type: str = "pdag",
        initialization: str = "feasible_dag",
        population_size: int = 100,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.2,
        gene_mutation_prob: float | None = None,
        tournament_size: int = 3,
        elite_size: int = 2,
        edge_prob: float = 0.2,
        repair_edge_selection: str = "mutual_info",
        repair_operation: str = "reverse_then_delete",
        refine: bool = False,
        epsilon: float = 1e-4,
        patience: int = 20,
        max_iter: int = 200,
        n_jobs: int = 1,
        random_state: int | None = None,
        show_progress: bool = True,
    ):
        # sklearn/pgmpy convention: constructor arguments are stored verbatim;
        # validation is deferred to fit.
        self.scoring_method = scoring_method
        self.start_dag = start_dag
        self.max_indegree = max_indegree
        self.expert_knowledge = expert_knowledge
        self.return_type = return_type
        self.initialization = initialization
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.edge_prob = edge_prob
        self.repair_edge_selection = repair_edge_selection
        self.repair_operation = repair_operation
        self.refine = refine
        self.epsilon = epsilon
        self.patience = patience
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.show_progress = show_progress

    # ------------------------------------------------------------------
    # pgmpy estimator entry point
    # ------------------------------------------------------------------
    def _fit(self, X: pd.DataFrame):
        self.variables_ = list(X.columns)
        self.n_features_in_ = len(self.variables_)
        self.feature_names_in_ = np.asarray(self.variables_, dtype=object)
        self._validate_hyperparameters()

        self.score_ = get_scoring_method(self.scoring_method, X)

        # Pairwise mutual information is used here as a *categorical* repair
        # heuristic. pgmpy structure scores advertise their supported datatype
        # through class tags; reject known continuous/mixed scores rather than
        # silently treating continuous measurements as category labels. Custom
        # score classes without a datatype tag remain the caller's responsibility.
        score_datatype = getattr(self.score_, "_tags", {}).get("supported_datatype")
        if (
            self.repair_edge_selection == "mutual_info"
            and score_datatype in {"continuous", "mixed"}
        ):
            raise ValueError(
                "repair_edge_selection='mutual_info' is intended for discrete "
                "data. Use repair_edge_selection='random' or provide a "
                "data-type-appropriate repair heuristic for continuous/mixed data."
            )

        expert, required, forbidden, search_space = self._prepare_expert_knowledge(X)

        # Mutual information is a repair heuristic, not the fitness function.
        edge_weights = (
            pairwise_mutual_information(X)
            if self.repair_edge_selection == "mutual_info"
            else {}
        )
        streams = make_random_streams(self.random_state)

        self.score_cache_: dict[tuple[int, ...], float] = {}
        self.history_: list[GenerationRecord] = []
        self.repair_stats_ = RepairStats()
        self.initialization_repair_stats_ = RepairStats()
        self.evolution_repair_stats_ = RepairStats()
        self.n_score_evaluations_ = 0

        chromosome_length = code_length(len(self.variables_))
        indpb = self.gene_mutation_prob
        if indpb is None:
            indpb = 1.0 / max(1, chromosome_length)

        self.toolbox_, individual_cls = make_toolbox(
            tournament_size=self.tournament_size,
            gene_mutation_prob=indpb,
        )
        self.deap_hall_of_fame_ = tools.HallOfFame(1)
        self.deap_logbook_ = tools.Logbook()
        self.deap_logbook_.header = [
            "gen",
            "best",
            "mean",
            "unique_scores",
            "unique_structures",
            "cache",
        ]

        population = self._initialize_population(
            individual_cls,
            initialization_rng=streams.initialization,
            initialization_repair_rng=streams.initialization_repair,
            required_edges=required,
            forbidden_edges=forbidden,
            search_space=search_space,
            edge_weights=edge_weights,
        )
        self._evaluate_population(population)
        self.deap_hall_of_fame_.update(population)
        self._record_generation(0, population)
        self.initial_population_summary_ = self._population_summary(population)

        best_seen = float(self.deap_hall_of_fame_[0].fitness.values[0])
        stagnant = 0

        # DEAP's standard operators use Python's global random module. Isolate
        # that state so the estimator is reproducible without polluting callers.
        with isolated_global_random(streams.deap_random_seed):
            if self.show_progress and config.SHOW_PROGRESS:
                generations = trange(1, int(self.max_iter) + 1, desc="GeneticSearch")
            else:
                generations = range(1, int(self.max_iter) + 1)

            for generation in generations:
                population = self._next_generation(
                    population,
                    repair_rng=streams.repair,
                    required_edges=required,
                    forbidden_edges=forbidden,
                    search_space=search_space,
                    edge_weights=edge_weights,
                )
                self.deap_hall_of_fame_.update(population)
                self._record_generation(generation, population)

                best_now = float(self.deap_hall_of_fame_[0].fitness.values[0])
                improvement = best_now - best_seen
                if improvement > self.epsilon:
                    best_seen = best_now
                    stagnant = 0
                else:
                    stagnant += 1
                if stagnant >= self.patience:
                    break

        best_individual = self.deap_hall_of_fame_[0]
        best_graph = self._to_pgmpy_dag(
            decode_code(list(best_individual), self.variables_)
        )
        self.ga_score_ = float(best_individual.fitness.values[0])

        if self.refine:
            # Historical GA -> local-search hybrid. Kept as an optional software
            # feature, not mixed into the paper's core initialization/repair tests.
            hc = HillClimbSearch(
                scoring_method=self.score_,
                start_dag=best_graph,
                max_indegree=self.max_indegree,
                expert_knowledge=expert,
                return_type="dag",
                show_progress=False,
            ).fit(X)
            best_graph = hc.causal_graph_

        self.dag_ = self._to_pgmpy_dag(best_graph, nodes=self.variables_)
        self.best_score_ = float(self.score_.score(self.dag_))
        self.n_iter_ = self.history_[-1].generation

        if self.return_type.lower() == "dag":
            self.causal_graph_ = self.dag_
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = self.dag_.to_pdag()
        else:  # already validated; defensive only
            raise ValueError("return_type must be one of: dag, pdag")

        # Use the fit-time column order rather than graph iteration order:
        # pgmpy graph constructors do not preserve node insertion order, and a
        # deterministic index makes the matrix directly comparable across runs.
        if hasattr(self.causal_graph_, "to_adjacency"):
            self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
                encoding="binary", nodelist=list(self.variables_)
            )
        else:  # pgmpy 1.1.2 compatibility
            self.adjacency_matrix_ = nx.to_pandas_adjacency(
                self.causal_graph_, nodelist=list(self.variables_), weight=1, dtype="int"
            )

        self.repair_stats_ = asdict(self.repair_stats_)
        self.initialization_repair_stats_ = asdict(self.initialization_repair_stats_)
        self.evolution_repair_stats_ = asdict(self.evolution_repair_stats_)
        return self

    # ------------------------------------------------------------------
    # Argument and expert-knowledge handling
    # ------------------------------------------------------------------
    def _validate_hyperparameters(self) -> None:
        if self.initialization not in {"feasible_dag", "random_chromosome_repair"}:
            raise ValueError(
                "initialization must be 'feasible_dag' or 'random_chromosome_repair'"
            )
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0 <= self.crossover_prob <= 1:
            raise ValueError("crossover_prob must lie in [0, 1]")
        if not 0 <= self.mutation_prob <= 1:
            raise ValueError("mutation_prob must lie in [0, 1]")
        if self.gene_mutation_prob is not None and not 0 <= self.gene_mutation_prob <= 1:
            raise ValueError("gene_mutation_prob must lie in [0, 1] or be None")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be in [0, population_size)")
        if not 0 <= self.edge_prob <= 1:
            raise ValueError("edge_prob must lie in [0, 1]")
        if self.repair_edge_selection not in {"mutual_info", "random"}:
            raise ValueError("repair_edge_selection must be 'mutual_info' or 'random'")
        if self.repair_operation not in {"delete", "reverse_then_delete"}:
            raise ValueError("repair_operation must be 'delete' or 'reverse_then_delete'")
        if str(self.return_type).lower() not in {"dag", "pdag"}:
            raise ValueError("return_type must be one of: dag, pdag")
        if self.max_indegree is not None and self.max_indegree < 0:
            raise ValueError("max_indegree must be non-negative or None")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")

    def _prepare_expert_knowledge(
        self, X: pd.DataFrame
    ) -> tuple[
        ExpertKnowledge,
        set[tuple[Hashable, Hashable]],
        set[tuple[Hashable, Hashable]],
        set[tuple[Hashable, Hashable]],
    ]:
        """Resolve pgmpy ``ExpertKnowledge`` without mutating the user's object."""
        if self.expert_knowledge is None:
            expert = ExpertKnowledge()
        elif hasattr(self.expert_knowledge, "get_params"):
            # Current pgmpy causal-discovery estimators use sklearn.clone so
            # fitted ``*_`` attributes land on a fresh object while constructor
            # parameters are preserved.
            expert = clone(self.expert_knowledge)
        else:
            # pgmpy 1.1.2 ExpertKnowledge predates sklearn estimator semantics.
            expert = copy.deepcopy(self.expert_knowledge)

        if hasattr(expert, "fit"):
            # Current pgmpy development API.
            expert.fit(X)
            required = set(expert.required_edges_)
            forbidden = set(expert.forbidden_edges_)
            search_space = set(expert.search_space_)
        else:
            # pgmpy 1.1.2 compatibility path, mirroring release semantics.
            if expert.search_space:
                expert.limit_search_space(X.columns)

            probe = DAG()
            probe.add_nodes_from(self.variables_)
            expert._validate_temporal_order(self.variables_)
            expert._orient_temporal_forbidden_edges(probe, only_edges=False)

            required = set(expert.required_edges)
            forbidden = set(expert.forbidden_edges)
            search_space = set(expert.search_space)

        if required & forbidden:
            raise ValueError(
                "expert knowledge marks an edge as both required and forbidden"
            )
        if search_space and not required.issubset(search_space):
            raise ValueError(
                "all required edges must belong to expert_knowledge.search_space"
            )

        required_dag = DAG()
        required_dag.add_nodes_from(self.variables_)
        required_dag.add_edges_from(required)
        if not nx.is_directed_acyclic_graph(required_dag):
            raise ValueError("required_edges create a directed cycle")
        if self.max_indegree is not None and any(
            required_dag.in_degree(node) > self.max_indegree
            for node in required_dag.nodes()
        ):
            raise ValueError("required_edges violate max_indegree")

        return expert, required, forbidden, search_space

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------
    def _initialize_population(
        self,
        individual_cls,
        *,
        initialization_rng: random.Random,
        initialization_repair_rng: random.Random,
        required_edges,
        forbidden_edges,
        search_space,
        edge_weights,
    ):
        population = []

        if self.start_dag is not None:
            if not isinstance(self.start_dag, DAG) or set(self.start_dag.nodes()) != set(
                self.variables_
            ):
                raise ValueError(
                    "start_dag must be a pgmpy DAG with exactly the variables in X"
                )
            start, stats = repair_graph(
                self.start_dag,
                edge_weights,
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
                search_space=search_space,
                max_indegree=self.max_indegree,
                edge_selection=self.repair_edge_selection,
                repair_operation=self.repair_operation,
                rng=initialization_repair_rng,
            )
            self.repair_stats_ += stats
            self.initialization_repair_stats_ += stats
            population.append(individual_cls(encode_dag(start, self.variables_)))

        while len(population) < self.population_size:
            if self.initialization == "feasible_dag":
                code = sample_feasible_code(
                    self.variables_,
                    edge_prob=self.edge_prob,
                    required_edges=required_edges,
                    forbidden_edges=forbidden_edges,
                    search_space=search_space,
                    max_indegree=self.max_indegree,
                    rng=initialization_rng,
                )
            else:
                code, stats = sample_raw_repaired_code(
                    self.variables_,
                    edge_weights=edge_weights,
                    required_edges=required_edges,
                    forbidden_edges=forbidden_edges,
                    search_space=search_space,
                    max_indegree=self.max_indegree,
                    edge_selection=self.repair_edge_selection,
                    repair_operation=self.repair_operation,
                    edge_prob=self.edge_prob,
                    rng=initialization_rng,
                    repair_rng=initialization_repair_rng,
                )
                self.repair_stats_ += stats
                self.initialization_repair_stats_ += stats
            population.append(individual_cls(code))
        return population

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------
    def _score_uncached(self, key: tuple[int, ...]) -> float:
        graph = self._to_pgmpy_dag(decode_code(key, self.variables_))
        return float(self.score_.score(graph))

    def _evaluate_population(self, population) -> None:
        """Evaluate only invalid DEAP fitnesses, with chromosome-level caching."""
        pending = [ind for ind in population if not ind.fitness.valid]
        if not pending:
            return

        unique: dict[tuple[int, ...], list] = {}
        for ind in pending:
            unique.setdefault(tuple(ind), []).append(ind)

        uncached = [key for key in unique if key not in self.score_cache_]
        if self.n_jobs == 1:
            raw = [self._score_uncached(key) for key in uncached]
        else:
            # Threads avoid copying the pgmpy score object and fitted data.
            raw = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._score_uncached)(key) for key in uncached
            )

        for key, value in zip(uncached, raw, strict=True):
            self.score_cache_[key] = float(value)
            self.n_score_evaluations_ += 1

        for key, individuals in unique.items():
            value = self.score_cache_[key]
            for ind in individuals:
                ind.fitness.values = (value,)

    # ------------------------------------------------------------------
    # Evolutionary generation: DEAP variation + BN-specific repair
    # ------------------------------------------------------------------
    def _next_generation(
        self,
        population,
        *,
        repair_rng: random.Random,
        required_edges,
        forbidden_edges,
        search_space,
        edge_weights,
    ):
        # Elitism and tournament selection are delegated to DEAP.
        elites = [self.toolbox_.clone(ind) for ind in tools.selBest(population, self.elite_size)]
        offspring = self.toolbox_.select(
            population, k=self.population_size - self.elite_size
        )
        offspring = list(map(self.toolbox_.clone, offspring))

        # DEAP two-point crossover. We invalidate fitness because the chromosome
        # may have changed even if subsequent repair happens to map it back to a
        # previously seen code; the cache will make that reevaluation cheap.
        for left, right in zip(offspring[::2], offspring[1::2]):
            # ``cxTwoPoint`` requires at least two loci. Networks with fewer
            # than three variables remain valid estimator inputs; they simply
            # have no meaningful two-point crossover operation.
            if len(left) >= 2 and random.random() < self.crossover_prob:
                self.toolbox_.mate(left, right)
                if left.fitness.valid:
                    del left.fitness.values
                if right.fitness.valid:
                    del right.fitness.values

        # DEAP uniform-integer mutation on ternary loci.
        for ind in offspring:
            if random.random() < self.mutation_prob:
                self.toolbox_.mutate(ind)
                if ind.fitness.valid:
                    del ind.fitness.values

            # A selected clone that underwent no variation was already feasible
            # in the parent population and needs no second repair pass. Only
            # invalidated offspring can have left the admissible DAG space.
            if ind.fitness.valid:
                continue

            # Variation is generic and may leave DAG space. The following step is
            # the BN-specific contribution: repair cycles and hard constraints.
            repaired, stats = repair_code(
                list(ind),
                self.variables_,
                edge_weights,
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
                search_space=search_space,
                max_indegree=self.max_indegree,
                edge_selection=self.repair_edge_selection,
                repair_operation=self.repair_operation,
                rng=repair_rng,
            )
            self.repair_stats_ += stats
            self.evolution_repair_stats_ += stats
            if repaired != list(ind):
                ind[:] = repaired

        self._evaluate_population(offspring)
        return elites + offspring

    def _record_generation(self, generation: int, population) -> None:
        scores = [float(ind.fitness.values[0]) for ind in population]
        record = GenerationRecord(
            generation=generation,
            best_score=max(scores),
            mean_score=fmean(scores),
            unique_scores=len(set(scores)),
            unique_structures=len({tuple(ind) for ind in population}),
            cache_size=len(self.score_cache_),
        )
        self.history_.append(record)
        self.deap_logbook_.record(
            gen=generation,
            best=record.best_score,
            mean=record.mean_score,
            unique_scores=record.unique_scores,
            unique_structures=record.unique_structures,
            cache=record.cache_size,
        )

    def _population_summary(self, population) -> dict[str, float | int]:
        """Summarize generation zero for the initialization ablation."""
        scores = [float(ind.fitness.values[0]) for ind in population]
        edge_counts = [
            decode_code(list(ind), self.variables_).number_of_edges()
            for ind in population
        ]
        return {
            "best_score": max(scores),
            "mean_score": fmean(scores),
            "mean_edges": fmean(edge_counts),
            "min_edges": min(edge_counts),
            "max_edges": max(edge_counts),
            "unique_structures": len({tuple(ind) for ind in population}),
        }

    @staticmethod
    def _to_pgmpy_dag(graph, nodes=None) -> DAG:
        # ``DAG.copy()`` in pgmpy 1.1.2 rebuilds the graph from its edge list,
        # which scrambles node order. Rebuild explicitly so fitted attributes
        # keep a deterministic node ordering (the fit-time column order when
        # ``nodes`` is given, otherwise the input graph's own order).
        dag = DAG()
        dag.add_nodes_from(graph.nodes() if nodes is None else nodes)
        dag.add_edges_from(graph.edges())
        return dag
