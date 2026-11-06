"""Reference solvers for the 9 formulation families in business_problems_1M.csv.
Each function takes real instance data and returns an actual decision. Requires numpy and scipy (>=1.9).
Use the playbook for your row (solution_playbooks.csv) to pick the method; swap in OR-Tools/Gurobi at scale."""
import heapq, itertools, math
import numpy as np
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds, minimize

def knapsack(values, weights, capacity):
    """Exact 0/1 knapsack by DP (integer weights). Returns (best_value, chosen_indices)."""
    n = len(values); cap = int(capacity); w = [int(x) for x in weights]
    dp = np.zeros((n + 1, cap + 1))
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if w[i - 1] <= cap:
            dp[i, w[i - 1]:] = np.maximum(dp[i - 1, w[i - 1]:], dp[i - 1, :cap + 1 - w[i - 1]] + values[i - 1])
    chosen, c = [], cap
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]: chosen.append(i - 1); c -= w[i - 1]
    return float(dp[n, cap]), sorted(chosen)

def knapsack_greedy(values, weights, capacity):
    """Ratio greedy + best single item (>= 1/2 of optimum). Scales to millions of items."""
    order = sorted(range(len(values)), key=lambda i: values[i] / weights[i], reverse=True)
    tot_v = tot_w = 0; pick = []
    for i in order:
        if tot_w + weights[i] <= capacity: pick.append(i); tot_w += weights[i]; tot_v += values[i]
    single = max((i for i in range(len(values)) if weights[i] <= capacity), key=lambda i: values[i], default=None)
    if single is not None and values[single] > tot_v: return float(values[single]), [single]
    return float(tot_v), sorted(pick)

def multi_knapsack(values, A, b):
    """Multidimensional 0/1 knapsack via MIP. A is (k resources x n items), b is (k,) limits."""
    n = len(values)
    r = milp(c=-np.asarray(values, float), constraints=LinearConstraint(np.asarray(A, float), -np.inf, np.asarray(b, float)),
             integrality=np.ones(n), bounds=Bounds(0, 1))
    x = np.round(r.x).astype(int); return float(np.dot(values, x)), list(np.flatnonzero(x))

def assignment(cost, maximize=False):
    """Optimal one-to-one assignment (Hungarian). Returns (total, list of (row, col))."""
    cost = np.asarray(cost, float); r, c = linear_sum_assignment(cost, maximize=maximize)
    return float(cost[r, c].sum()), list(zip(r.tolist(), c.tolist()))

def set_cover_greedy(universe, sets, costs):
    """Greedy weighted set cover (within H(d) of optimal). sets: list of iterables."""
    left = set(universe); chosen = []; sets = [set(s) for s in sets]
    while left:
        i = max(range(len(sets)), key=lambda j: len(sets[j] & left) / costs[j] if sets[j] & left else -1)
        if not sets[i] & left: raise ValueError("universe cannot be covered")
        chosen.append(i); left -= sets[i]
    return sum(costs[i] for i in chosen), chosen

def facility_location(fixed, dist_demand_x_site, demand, k=None):
    """Capacity-free facility location via MIP. dist is (demands x sites). Optional cap k on open sites."""
    D = np.asarray(dist_demand_x_site, float); m, n = D.shape; d = np.asarray(demand, float)
    c = np.concatenate([np.asarray(fixed, float), (D * d[:, None]).ravel()])
    rows, lo, hi = [], [], []
    for i in range(m):  # each demand fully served
        a = np.zeros(n + m * n); a[n + i * n:n + (i + 1) * n] = 1; rows.append(a); lo.append(1); hi.append(1)
    for i in range(m):  # x_ij <= y_j
        for j in range(n):
            a = np.zeros(n + m * n); a[n + i * n + j] = 1; a[j] = -1; rows.append(a); lo.append(-np.inf); hi.append(0)
    if k is not None:
        a = np.zeros(n + m * n); a[:n] = 1; rows.append(a); lo.append(0); hi.append(k)
    integ = np.concatenate([np.ones(n), np.zeros(m * n)])
    r = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integ, bounds=Bounds(0, 1))
    return float(r.fun), list(np.flatnonzero(np.round(r.x[:n]) == 1))

def budget_allocation(marginal_fns, budget, step=1.0):
    """Greedy equimarginal allocation for separable concave responses.
    marginal_fns[j](s) = marginal return of option j when it already has spend s."""
    s = [0.0] * len(marginal_fns); heap = [(-f(0.0), j) for j, f in enumerate(marginal_fns)]; heapq.heapify(heap)
    left = budget
    while left >= step - 1e-12 and heap:
        g, j = heapq.heappop(heap)
        if -g <= 0: break
        s[j] += step; left -= step; heapq.heappush(heap, (-marginal_fns[j](s[j]), j))
    return s

def portfolio_min_variance(mu, cov, target_return, max_weight=1.0):
    """Long-only mean-variance (Markowitz) portfolio hitting a target expected return."""
    n = len(mu); mu = np.asarray(mu, float); cov = np.asarray(cov, float)
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}, {'type': 'ineq', 'fun': lambda w: w @ mu - target_return}]
    r = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), constraints=cons, bounds=[(0, max_weight)] * n, method='SLSQP')
    return r.x, float(r.fun)

def schedule_identical_machines(durations, m):
    """LPT list scheduling for makespan on m identical machines (within 4/3 of optimal)."""
    loads = [(0.0, k) for k in range(m)]; heapq.heapify(loads); plan = [[] for _ in range(m)]
    for j in sorted(range(len(durations)), key=lambda j: -durations[j]):
        l, k = heapq.heappop(loads); plan[k].append(j); heapq.heappush(loads, (l + durations[j], k))
    return max(l for l, _ in loads), plan

def wspt_order(weights, times):
    """Smith's rule: optimal single-machine order for total weighted completion time."""
    return sorted(range(len(times)), key=lambda j: times[j] / weights[j])

def vehicle_routes_savings(depot, pts, demand, cap):
    """Clarke-Wright savings for capacitated routing. depot and pts are (x,y). Returns routes as lists of point indices."""
    d = lambda a, b: math.dist(a, b); n = len(pts)
    routes = {i: [i] for i in range(n)}; where = {i: i for i in range(n)}; load = {i: demand[i] for i in range(n)}
    sav = sorted(((d(depot, pts[i]) + d(depot, pts[j]) - d(pts[i], pts[j]), i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)
    for s, i, j in sav:
        if s <= 0: break
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > cap: continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j: new = a + b
        elif a[0] == i and b[-1] == j: new = b + a
        elif a[0] == i and b[0] == j: new = a[::-1] + b
        elif a[-1] == i and b[-1] == j: new = a + b[::-1]
        else: continue
        routes[ri] = new; load[ri] += load[rj]; del routes[rj], load[rj]
        for p in new: where[p] = ri
    out = list(routes.values())
    return sum(d(depot, pts[r[0]]) + sum(d(pts[r[k]], pts[r[k + 1]]) for k in range(len(r) - 1)) + d(pts[r[-1]], depot) for r in out), out

if __name__ == "__main__":
    assert knapsack([60, 100, 120], [10, 20, 30], 50) == (220.0, [1, 2])
    assert knapsack_greedy([60, 100, 120], [10, 20, 30], 50)[0] >= 110
    assert multi_knapsack([10, 13, 7], [[3, 4, 2], [2, 1, 3]], [6, 5])[0] == 20.0
    assert assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])[0] == 5.0
    assert set_cover_greedy(range(1, 6), [[1, 2, 3], [2, 4], [3, 4], [4, 5]], [1, 1, 1, 1])[0] == 2
    assert facility_location([5, 5], [[1, 9], [9, 1]], [1, 1])[0] == 12.0
    assert sum(budget_allocation([lambda s: 5 - s, lambda s: 3 - s], 4)) == 4
    w, v = portfolio_min_variance([0.1, 0.05], [[0.04, 0], [0, 0.01]], 0.07); assert abs(w.sum() - 1) < 1e-6
    assert schedule_identical_machines([7, 5, 4, 3, 3], 2)[0] == 12  # LPT; optimum is 11, within the 4/3 bound
    assert wspt_order([1, 2], [4, 2]) == [1, 0]
    assert vehicle_routes_savings((0, 0), [(1, 0), (2, 0), (0, 5)], [1, 1, 1], 2)[1]
    print("all solver self-tests passed")
