"""Expert-written training sentences for each problem family, written from general operations-research
knowledge. They never appear in evaluation: scores are always measured on the real dataset rows only."""
import random
O = ["projects", "products", "suppliers", "campaigns", "experiments", "hires", "features", "vendors", "initiatives", "channels", "markets", "investments", "promotions", "contracts", "tools", "partners", "events", "ads"]
PE = ["employees", "nurses", "drivers", "technicians", "engineers", "sales reps", "agents", "volunteers", "interns"]
TA = ["projects", "shifts", "territories", "tickets", "machines", "classes", "clients", "orders", "tasks"]
SI = ["warehouses", "stores", "clinics", "hubs", "depots", "branches", "data centers", "service points", "stations"]
JO = ["production runs", "maintenance jobs", "deliveries", "releases", "meetings", "interviews", "surgeries", "launches", "orders"]
T = {
 "knapsack": ["Choose which {o} to fund from a long list", "Decide which {o} to pick within a fixed budget", "Rank {o} and select the best ones under a limit", "Select {o} to maximize return given limited resources", "Decide which {o} to drop to save cost"],
 "multi_dim_knapsack": ["Select {o} subject to budget, time and space limits", "Choose {o} while meeting several limits at once", "Pick {o} that satisfy quality standards and cost caps", "Select {o} under budget, headcount and capacity constraints"],
 "portfolio_selection": ["Pick {o} to balance return and risk", "Select {o} while avoiding overlap between them", "Build a diversified mix of {o}", "Choose {o} that balance cost and risk", "Select non-overlapping {o}"],
 "budget_allocation": ["Split a budget across {o} with diminishing returns", "Decide how much to spend on each of the {o}", "Allocate funds among {o}", "Set the level of spend for each of the {o}", "Distribute a fixed pool across {o}", "Set prices for {o}"],
 "assignment": ["Assign {p} to {t}", "Match {p} to {t} by skill", "Allocate {t} to {p} one to one", "Pair {p} with {t} to minimize cost"],
 "scheduling": ["Schedule {j} across the week", "Sequence {j} across machines", "Decide the timing of {j}", "Plan the best dates for {j}", "Build shift schedules that cover demand", "Order {j} to meet deadlines"],
 "facility_location": ["Choose locations for new {s}", "Decide where to open {s}", "Pick {s} to serve all customers", "Select sites for {s} to minimize distance", "Choose which {s} to open or close"],
 "set_cover": ["Pick {o} whose combined coverage reaches every segment", "Choose the fewest {o} that cover all requirements", "Select {o} so that every need is covered", "Choose {o} to cover the target audience at minimum cost"],
 "routing": ["Plan delivery routes for vehicles", "Order the stops to minimize distance", "Assign delivery orders to vehicles and routes", "Find the shortest route visiting all customers"]}
def synthetic(n=45, seed=1):
    R = random.Random(seed); out = []
    for fam, ts in T.items():
        for _ in range(n): out.append((R.choice(ts).format(o=R.choice(O), p=R.choice(PE), t=R.choice(TA), j=R.choice(JO), s=R.choice(SI)), fam))
    return out
