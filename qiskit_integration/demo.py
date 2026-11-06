"""Demo / smoke test:  python -m qiskit_integration.demo   (run from the project root)

Solves the built-in 10-trend portfolio three ways and prints a comparison:
exact branch and bound, the project's NumPy QAOA simulator, and Qiskit QAOA on Aer.
Options:  --noisy   add a depolarising noise model to the Qiskit sampling
          --reps N  QAOA depth (default 2)      --qasm   print the bound circuit as OpenQASM 3
"""
import argparse, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.optimizer import Model, Opt, exact, run_simulator
import qiskit_integration as qi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--noisy", action="store_true"); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--qasm", action="store_true"); a = ap.parse_args()
    qi.configure(reps=a.reps, noise_model=qi.depolarizing_noise_model() if a.noisy else None)
    M = Model(Opt(seed=a.seed)); (ef, eA), *_ = exact(M)[0]
    print(f"{M.n} trends, {M.nO} options, budget {M.b.budget:g}  |  {qi.versions()}")
    rows = [("exact (branch & bound)", {"objective": round(ef, 2), "ratio": 1.0, "verified": True, "ms": 0, "trends": [M.name(x) for x in M.sel(eA)]})]
    rows.append(("project NumPy QAOA", run_simulator(M, ef)))
    rows.append(("Qiskit QAOA" + (" (noisy)" if a.noisy else ""), qi.run_qiskit(M, ef)))
    print(f"\n{'backend':28}{'objective':>10}{'ratio':>8}{'verified':>10}{'ms':>8}")
    for name, r in rows: print(f"{name:28}{r.get('objective', '-'):>10}{r.get('ratio', '-'):>8}{str(r.get('verified', '-')):>10}{r.get('ms', '-'):>8}")
    q = rows[-1][1]
    print("\nQiskit run:", {k: q[k] for k in ("qubits", "shots", "depth", "two_qubit_gates", "reps", "raw_feasibility", "optimiser_evaluations") if k in q})
    print("Presolve:", q.get("presolve"), "| depth growth:", [(c["reps"], c["cvar"]) for c in q.get("convergence", [])], "| share of shots reaching optimum:", q.get("share_of_shots_reaching_optimum"))
    print("Chosen trends:", ", ".join(q["trends"]))
    if a.qasm: print("\n" + qi.export_qasm3(M, params=q["params"], reps=a.reps))


if __name__ == "__main__":
    main()
