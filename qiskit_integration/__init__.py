"""Qiskit integration for Quant Insight's Trend Portfolio Optimizer.

    from qiskit_integration import register
    register()          # adds backend="qiskit" and backend="qiskit-ibm" to /api/v1/optimize

Nothing in app/ is modified. See README.md in this folder.
"""
from .qiskit_qaoa import (  # noqa: F401
    CONFIG, configure, register, available, versions, presolve, build_reduced_qubo, clear_cache,
    qubo_to_ising, ising_diagonal, build_cost_operator, build_circuit,
    optimise_qaoa, sample_counts, run_qiskit, run_ibm, export_qasm3,
    depolarizing_noise_model, probabilities_numpy_reference,
)
