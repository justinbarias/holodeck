"""HoloDeck test optimizer (`holodeck test optimize`).

Compounding coordinate-descent optimizer that alternates a numeric phase
(Optuna TPE over declared query-time axes) and a textual phase (Critic produces
a natural-language gradient, Applier rewrites the instructions), advancing a
``best_candidate`` baseline on every accepted improvement.

Submodules are imported lazily by consumers to keep this package's import graph
free of the ``holodeck.models`` cycle (``evaluations.optimizer`` references
``OptimizerConfig``).
"""
