# Evaluation report: dev suite

Harness v1.0. Cases: 150 (file 7b039b19c284dfdf, solver code 8f5d384e5ed3ff08). Development suite: public, so results here can be tuned against and prove little about unseen cases.

## Systems against ground truth

| System | Cases | Optimal | 95% CI | Mean gap % | Worst gap % | Infeasible | Claims match |
|---|---|---|---|---|---|---|---|
| engine_quick | 140 | 100.0% | 97.3% to 100.0% | 0.0 | 0.0 | 0 | 140/140 |
| engine_standard | 140 | 100.0% | 97.3% to 100.0% | 0.0 | 0.0 | 0 | 140/140 |
| engine_deep | 111 | 100.0% | 96.7% to 100.0% | 0.0 | 0.0 | 0 | 111/111 |
| greedy_value_per_cost | 140 | 60.7% | 52.4% to 68.4% | 2.275 | 24.78 | 0 | n/a |
| cheapest_first | 140 | 25.7% | 19.2% to 33.5% | 15.036 | 72.55 | 0 | n/a |
| highest_value_first | 140 | 57.1% | 48.9% to 65.0% | 3.155 | 30.36 | 0 | n/a |
| random_feasible | 140 | 7.9% | 4.4% to 13.5% | 21.053 | 44.03 | n/a | n/a |

Gap = (optimum - achieved value) / optimum. Baselines run on the correct option table; the engine runs on the text, so reading errors count against it.

## Paired comparison with baselines (same cases)

| Comparison | Engine better | Worse | Tied | Mean gap reduction % | 95% CI | Sign test p |
|---|---|---|---|---|---|---|
| engine_standard vs greedy_value_per_cost | 55 | 0 | 85 | 2.275 | 1.58 to 3.002 | 0.0 |
| engine_standard vs cheapest_first | 104 | 0 | 36 | 15.036 | 12.909 to 17.307 | 0.0 |
| engine_standard vs highest_value_first | 60 | 0 | 80 | 3.155 | 2.268 to 4.124 | 0.0 |
| engine_standard vs random_feasible | 129 | 0 | 11 | 21.053 | 18.969 to 23.023 | 0.0 |

## Input handling

- Malformed input rejected: 10 of 10.
- Options read correctly, by family: random 40/40, correlated 30/30, ties 15/15, fractional_costs 15/15, tight_budget 10/10, large 8/8, awkward_names 12/12, formatted_numbers 10/10.
- Options dropped without a warning: 0.

## Calibration

| Stated confidence | Cases | Optimal | 95% CI |
|---|---|---|---|
| deep:high | 65 | 100.0% | 94.4% to 100.0% |
| deep:medium | 46 | 100.0% | 92.3% to 100.0% |
| high | 140 | 100.0% | 97.3% to 100.0% |

Stability against fresh perturbations (111 Deep cases): reported mean 0.698, fresh mean 0.698, mean absolute error 0.02, mean signed error 0.0 (95% CI -0.00547 to 0.00584), per-case overclaims beyond noise 0. Signed error = reported minus fresh; positive means the engine claims more stability than a fresh test shows. Fresh rate = share of new +/-20% value perturbations under which the chosen set is still optimal by value; the engine's own figure counts only an identical set, so it should not exceed it. Per-case overclaims allow for sampling noise.

## What may be claimed

| Statement | Status |
|---|---|
| more accurate than simple baselines on unseen cases | not supported: run on the private suite; needs at least 200 answerable cases (has 140) |
| its 'high confidence' label is reliable (lower bound 90 percent or better) | not supported: needs the private suite and at least 100 high-confidence answers with a 95 percent lower bound of 90 percent |
| stated stability does not overclaim | supported |
| no silent loss of user input | supported |
| preferred by independent experts over other systems | not supported: no blinded study has been scored |
| better than other AI systems in general | not supported: this harness covers one structured problem type and no external systems; do not make this claim |
