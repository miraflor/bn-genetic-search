# Testing layers

The repository distinguishes tests that can run without pgmpy from true integration tests.

1. **Encoding tests** verify the triangular index and exact DAG round trips.
2. **Repair tests** exercise delete, safe reversal, reversal fallback, required/forbidden edges,
   search-space restrictions, maximum indegree, and fixed-seed random repair.
3. **Evolution tests** verify that mutation changes the real chromosome (fixing the 2022 temporary
   list bug), crossover reproducibility, and tournament behavior.
4. **Recovery-metric tests** verify skeleton, v-structure, and CPDAG pair-state comparisons without
   relying on a particular pgmpy metric implementation.
5. **pgmpy integration tests** instantiate `GeneticSearch`, use pgmpy `BDeu` and `ExpertKnowledge`,
   and verify pgmpy-style fitted attributes. These require pgmpy to be installed.
6. **GitHub Actions** installs the declared package dependencies and runs all tests on supported
   Python versions.

The DAG-recovery benchmark is intentionally separate from unit tests because a recovery result is an
empirical research output, not a deterministic software invariant.
