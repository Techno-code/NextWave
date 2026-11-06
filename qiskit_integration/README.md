# Qiskit integration for Quant Insight

Adds real [Qiskit](https://qiskit.org) QAOA to the Trend Portfolio Optimizer. It lives entirely in this folder: **no file in `app/` is changed.**

```
request → presolve (fewer qubits) → QUBO → Ising SparsePauliOp → native QAOA circuit
        → depth-growing optimiser (cached) → Aer ideal / noisy sampling, or IBM Quantum
        → decode → repair → classical verify → compare with the exact optimum
```

| Backend name | Runs on | Cost |
|---|---|---|
| `qiskit` | Qiskit Aer on your machine (ideal, or with a noise model) | free |
| `qiskit-ibm` | a real IBM Quantum device via `qiskit-ibm-runtime` | paid, guarded (see below) |

Every result is decoded, repaired and re-verified by the project's own classical checker (`Model.repair / ok / check`) and compared with the exact branch-and-bound reference, like the existing backends. **No quantum advantage is claimed.**

## What is advanced in this version

| Feature | What it does | Why |
|---|---|---|
| **Presolve** | Drops options that can never be in an optimal plan (gain ≤ 0, cost above budget) and caps the budget at what `max_picks` could ever spend | Every qubit removed halves the simulation. Proven safe: tests check on 25 random models that the exact optimum always survives |
| **Native circuit builder** | Builds the QAOA circuit straight from `H`/`RZ`/`RZZ`/`RX` gates, which Aer runs without transpiling | Setup fell from ~4 s to milliseconds. Tested equal to Qiskit's own `qaoa_ansatz` |
| **Depth growth (INTERP)** | Optimises depth 1, then starts depth 2 from the interpolated depth-1 angles, and so on; stops when an extra layer helps less than `grow_tol` | Far fewer optimiser evaluations than restarting at every depth, and the depth is chosen for you |
| **Parameter cache** | Identical problem + settings reuses the tuned angles (memory, or JSON on disk with `cache_dir`) | A repeat request costs one evaluation (~0.2 s) |
| **Time budget** | `time_budget_s` stops optimising and keeps the best angles so far | A request can never hang |
| **Pre-sorted CVaR** | Energy ordering is computed once, not on every evaluation | Cheaper inner loop |
| **Diagnostics** | `convergence` per depth, `presolve` savings, `share_of_shots_reaching_optimum`, `cache_hit`, depth and two-qubit gate count | You can see what the run did |
| **Noise + hardware path** | `depolarizing_noise_model()`, OpenQASM 3 export, IBM Runtime with cost guard rails | Standard Qiskit workflow |

## Install

```bash
pip install -r qiskit_integration/requirements-qiskit.txt
```

Qiskit is optional. Without it the rest of the app is unaffected, and the `qiskit` backend answers with a clear install message.

## Try it

```bash
python -m qiskit_integration.demo              # exact vs NumPy QAOA vs Qiskit QAOA
python -m qiskit_integration.demo --noisy      # with a depolarising noise model
python -m qiskit_integration.benchmark --seeds 8
python -m pytest qiskit_integration/tests -q   # 25 tests
```

## Use it from Python

```python
import qiskit_integration as qi
from app.optimizer import Model, Opt, exact

M = Model(Opt()); ef = exact(M)[0][0][0]
qi.configure(reps=3, shots=16384, cache_dir=".qiskit_cache")   # all optional
r = qi.run_qiskit(M, ef)               # same result shape as optimizer.run_simulator
print(r["ratio"], r["convergence"], r["presolve"])

qc, H, info = qi.build_circuit(M)      # the parametrised circuit and its cost operator
print(qi.export_qasm3(M, params=r["params"]))
```

## Turn on the API backend (one opt-in line)

If you want `POST /api/v1/optimize` to accept `"backend": "qiskit"` or `"qiskit-ibm"`, register the backends once at start-up (for example at the bottom of `app/main.py`):

```python
import qiskit_integration; qiskit_integration.register()
```

`compare_backends` still compares the simulator and annealer alongside.

## IBM Quantum (real hardware)

Angles are tuned on the free ideal simulator, then **one** sampling job is sent to the device. Same guard rails as the existing `hardware` backend:

- `QISKIT_IBM_COST_PER_RUN` must be set (the operator states the cost)
- the request must set `hardware_confirm: true` and a `hardware_max_cost` at or above it
- scheduled runs cannot use it (the app already blocks any request with `hardware_confirm`)
- credentials: `QISKIT_IBM_TOKEN` (or a saved Qiskit account), optional `QISKIT_IBM_INSTANCE`, `QISKIT_IBM_CHANNEL`, `QISKIT_IBM_BACKEND` (otherwise the least busy device with enough qubits)

**Honest status:** tested locally against a fake IBM device (`FakeSherbrooke`: transpile to native gates, then `SamplerV2` on a noisy simulation). **Never run on a real IBM device.**

## Settings (`qi.configure(...)`)

| Setting | Default | Meaning |
|---|---|---|
| `reps` | 3 | maximum QAOA depth (grown from 1) |
| `min_reps`, `grow_tol` | 1, 0.2 | never stop before `min_reps`; stop when a layer improves CVaR by less than this fraction |
| `alpha` | 0.1 | CVaR fraction (1.0 = plain expectation) |
| `presolve` | True | exact option and budget reduction |
| `cache`, `cache_dir` | True, None | reuse tuned angles; set a folder (or `QISKIT_CACHE_DIR`) to persist |
| `time_budget_s` | 60 | optimisation time limit |
| `restarts`, `maxiter` | 2, 60 | depth-1 random restarts; COBYLA iterations per start (depth 1 uses half) |
| `shots`, `top_k` | 16384, 1000 | final sampling shots; distinct bitstrings decoded and repaired |
| `qubit_cap` | 20 (`QISKIT_QUBIT_CAP`) | options + budget bits after presolve; larger requests are skipped with a message |
| `noise_model` | None | Aer `NoiseModel` for the final sampling only |
| `ibm_shots`, `ibm_optimization_level` | 4096, 3 | for `qiskit-ibm` |

## Measured results (not marketing)

`python -m qiskit_integration.benchmark --seeds 8`, default 10-trend problem, 14 qubits, one CPU core:

| | mean ratio to proven optimum | mean time |
|---|---|---|
| project NumPy QAOA | 0.957 | 0.36 s |
| project simulated annealing | 1.000 | 0.08 s |
| **Qiskit QAOA, cold** | **1.000** (all 8 seeds) | 5.1 s |
| **Qiskit QAOA, repeat request (cache)** | 1.000 | 0.2 s |

Version 1 of this folder needed about 10 s per run on the same problem (4 s of that was transpiling).

What this does and does not show:
- The Qiskit backend now finds the optimum on every seed, where the NumPy simulator misses on some. Much of that comes from sampling and decoding many more bitstrings, and from the depth growth, not from anything quantum.
- It is **not faster** than your NumPy simulator or the annealer, and the annealer is just as accurate here. Aer's per-evaluation cost (about 15 to 30 ms) is the floor; the savings come from needing fewer evaluations.
- Presolve does nothing on the default example (every option has positive gain); it pays off on problems with unaffordable or worthless options.
- Only a small share of shots decode to the exact optimum (`share_of_shots_reaching_optimum`, around 0.2 % here). The result comes from sampling many shots, then classical repair.

## How it was checked

- The Ising `SparsePauliOp` matches `build_qubo` on every bitstring tested, and its matrix diagonal matches.
- With identical angles, the Qiskit circuit and the project's NumPy simulator agree (max difference about 1e-19); the native circuit equals Qiskit's `qaoa_ansatz`; Aer equals the pure `quantum_info` fallback.
- The un-presolved reduced QUBO is identical to `optimizer.build_qubo`.
- Presolve never lost the optimum on 25 random models; trivial problems return a verified empty plan.
- Deterministic for a fixed seed; cache hits return identical answers (memory and disk); the time budget stops cleanly.
- Over-size and non-integer-cost problems are skipped with a message, not failed; the IBM guard rails refuse correctly.

## Limits

- Statevector simulation: memory doubles with every qubit (2^20 amplitudes is about 16 MB per state, but time per evaluation grows too).
- Angles are tuned noise-free; noise only affects the final sampling.
- Costs must be whole numbers (they set the budget slack bits), as for the existing simulator.
- `qiskit-ibm-runtime` 0.50+ prints a deprecation warning for `SamplerV2`; it still works, and the import will need updating when it is removed.
