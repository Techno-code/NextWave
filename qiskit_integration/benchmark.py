"""Benchmark:  python -m qiskit_integration.benchmark [--seeds 8] [--trends 10] [--budget 12]   (run from the project root)

For each random instance it runs: exact branch and bound (reference), the project's NumPy QAOA simulator, the project's annealer,
and Qiskit QAOA. It prints per-instance and mean approximation ratio (1.0 = the proven optimum) and wall time.
Qiskit is run cold (cache cleared) and then warm (cache hit), so you can see what the parameter cache saves.
"""
import argparse, statistics as st, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.optimizer import Model, Opt, exact, run_simulator, run_anneal
import qiskit_integration as qi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--budget", type=float, default=12)
    ap.add_argument("--trends", type=int, default=10); ap.add_argument("--reps", type=int, default=None); a = ap.parse_args()
    if a.reps: qi.configure(reps=a.reps)
    rows = []
    print(f"{'seed':>4} | {'numpy QAOA':>14} | {'anneal':>14} | {'Qiskit cold':>14} | {'Qiskit warm':>14} | qubits (p) evals")
    for seed in range(1, a.seeds + 1):
        M = Model(Opt(seed=seed, budget=a.budget)); M.T = M.T[:a.trends] if a.trends < len(M.T) else M.T
        M = Model(Opt(seed=seed, budget=a.budget, trends=M.T)); ef = exact(M)[0][0][0]
        n = run_simulator(M, ef); an = run_anneal(M, ef); qi.clear_cache(); c = qi.run_qiskit(M, ef); w = qi.run_qiskit(M, ef)
        if "skipped" in c: print(seed, c["skipped"]); continue
        rows.append((n, an, c, w))
        f = lambda r: f"{r['ratio']:.3f} {r['ms'] / 1000:6.1f}s"
        print(f"{seed:>4} | {f(n):>14} | {f(an):>14} | {f(c):>14} | {f(w):>14} | {c['qubits']} (p={c['reps']}) {c['optimiser_evaluations']}")
    if rows:
        m = lambda i, k: st.mean(r[i][k] for r in rows)
        print(f"\nmean ratio   numpy {m(0, 'ratio'):.3f}   anneal {m(1, 'ratio'):.3f}   Qiskit {m(2, 'ratio'):.3f}   (all verified: {all(r[i]['verified'] for r in rows for i in range(4))})")
        print(f"mean seconds numpy {m(0, 'ms') / 1000:.2f}    anneal {m(1, 'ms') / 1000:.2f}    Qiskit cold {m(2, 'ms') / 1000:.2f}   warm {m(3, 'ms') / 1000:.2f}")


if __name__ == "__main__":
    main()
