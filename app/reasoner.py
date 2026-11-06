"""Built-in reasoner: no external service. It reads a scenario, extracts quantities, matches it to an
expert playbook, ranks explanations or options by weighing wording against prior plausibility,
and computes simple implications from the numbers given. Rankings are heuristic, not measured."""
import math, re

def P(id, title, kind, cues, need, hyp, tests, actions, monitor, questions, calc=None):
    return dict(id=id, title=title, kind=kind, cues=cues, need=need, hyp=hyp, tests=tests, actions=actions, monitor=monitor, questions=questions, calc=calc)

def nums(t):
    t = t.lower(); out = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s?(?:%|percent)", t); out["pct"] = float(m.group(1)) if m else None
    d = [float(n) * {"day": 1, "week": 7, "month": 30}[u] for n, u in re.findall(r"(?:\b|^)(\d+(?:\.\d+)?)\s*[- ]?(day|week|month)s?", t)]
    for w, n in {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "twelve": 12}.items():
        d += [n * {"day": 1, "week": 7, "month": 30}[u] for u in re.findall(rf"\b{w}\s+(day|week|month)s?", t)]
    out["days"] = min(d) if d else None
    m = re.search(r"[$€£]\s?([\d,.]+)\s?(k|m|million|thousand)?", t)
    if m:
        try: out["money"] = float(m.group(1).replace(",", "")) * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(m.group(2), 1)
        except ValueError: pass
    return out

BOOK = [
 P("sales", "Declining sales", "diagnose", {"sales": 2, "revenue": 2, "orders": 1, "traffic": 1, "customers": 1}, [r"sales|revenue|orders|footfall|traffic", r"down|drop|declin|fell|lower|decreas|slump|slow"],
   [("Demand fell: season, trend shift or a competitor move", .28, {"season": 2, "competitor": 3, "market": 1, "trend": 2, "holiday": 2, "weather": 2}),
    ("Fewer customers are reaching you: traffic, ads, stock-outs or location", .27, {"traffic": 3, "ads": 2, "location": 3, "stock": 2, "website": 2, "footfall": 3, "visitors": 2, "near": 1}),
    ("Price or promotion changed", .17, {"price": 3, "discount": 2, "promotion": 2, "cost": 1, "expensive": 2}),
    ("Quality or service slipped, so visitors do not buy or do not return", .16, {"quality": 3, "complaint": 3, "review": 2, "service": 2, "refund": 2, "staff": 1}),
    ("A measurement or data problem", .12, {"report": 1, "tracking": 2, "system": 1, "dashboard": 1})],
   ["Split the drop by product, channel, location and week to see where it started", "Compare against last year and a similar unaffected location", "Check traffic and conversion separately: fewer visits or fewer buyers per visit?"],
   ["Fix the segment where the drop is concentrated before changing anything company-wide", "Run one cheap test (a price, promotion or staffing change) and compare to a control", "Ask 10 recent lost or lapsed customers why"],
   ["Weekly sales by segment", "Conversion rate", "Average order value"],
   ["When exactly did the decline begin, and what changed around then?", "Is it all products and places or only some?", "What are the weekly numbers before and after?"],
   lambda n: f"A {n['pct']:g} percent fall is large enough to trace to a specific segment; if one place or product explains most of it, the cause is local rather than market-wide." if n["pct"] else None),
 P("cash", "Cash shortfall and excess stock", "decide", {"inventory": 3, "stock": 2, "cash": 3, "shortfall": 3, "supplier": 2, "payment": 2, "payables": 2, "liquidity": 3, "loan": 1}, [r"cash|liquidity|shortfall|inventory|stock", r"."],
   [("Convert excess stock to cash: targeted markdowns, bundles, and sales to liquidators, slowest-moving items first", .30, {"inventory": 3, "stock": 3, "excess": 3, "overstock": 3}),
    ("Negotiate supplier terms: keep current terms, or trade a small discount or partial payment for time", .27, {"supplier": 3, "payment": 2, "early": 2, "terms": 3, "payables": 2}),
    ("Bridge the gap with short-term financing sized to the gap, only after the first two are exhausted", .17, {"loan": 3, "credit": 2, "bank": 2, "financing": 3, "borrow": 2}),
    ("Slow new purchasing and cut non-essential spend until stock normalises", .26, {"purchase": 2, "order": 1, "spend": 2, "cost": 1})],
   ["Rank stock by age and margin to find what can be discounted without losing money", "Build a weekly cash forecast to the deadline with and without each lever", "Ask the supplier what they would accept instead of early payment"],
   ["Start markdowns on the oldest stock now; do not wait for the deadline", "Offer the supplier a payment schedule backed by your cash forecast", "Freeze new orders for slow-moving lines"],
   ["Weekly cash balance versus forecast", "Stock cover in weeks", "Gross margin on cleared stock"],
   ["How large is the cash gap and when exactly is it due?", "What is your margin on the excess items?", "What would the supplier accept besides early payment?"],
   lambda n: (f"Clearing {n['pct']:g} percent excess in {n['days']:g} days means removing about {n['pct'] / max(n['days'] / 7, 1):.1f} percent of normal stock per week on top of regular sales." if n["pct"] and n["days"] else None)),
 P("supply", "Supply disruption", "decide", {"outage": 3, "supplier": 2, "shortage": 3, "delay": 2, "disruption": 3, "orders": 1, "expedite": 3, "customers": 1}, [r"outage|shortage|disruption|delay|stock-?out|late", r"."],
   [("Protect orders by customer value and contract penalty, expediting the highest-impact ones first", .32, {"expedite": 3, "orders": 2, "customers": 2, "penalty": 3, "contract": 2}),
    ("Find substitute suppliers or materials for the constrained input", .28, {"supplier": 2, "alternative": 3, "substitute": 3, "source": 2, "outage": 1}),
    ("Tell affected customers early with a firm revised date", .22, {"inform": 3, "customers": 2, "communicate": 3, "tell": 2, "notify": 3}),
    ("Rebalance production to the products that use available inputs", .18, {"production": 3, "manufactur": 2, "plant": 2, "line": 1})],
   ["List every open order with its customer, due date, penalty and the input it needs", "Check how long the outage lasts and how much stock remains", "Get quotes and lead times from one or two alternative sources"],
   ["Rank orders by penalty and customer value, and fulfil in that order", "Notify affected customers today with a realistic date", "Place a small test order with an alternative supplier"],
   ["Orders at risk and their penalty value", "Days of input remaining", "Customer response to notices"],
   ["How long is the outage expected to last?", "Which customers have penalties or are most valuable?", "Is an alternative supplier available and how fast?"]),
 P("capacity", "Scheduling and backlog", "decide", {"schedule": 3, "backlog": 3, "queue": 2, "capacity": 3, "rooms": 2, "staff": 2, "shifts": 2, "waiting": 2, "leave": 1}, [r"schedul|backlog|queue|capacity|waiting|shift", r"."],
   [("Prioritise by urgency and by the benefit per unit of scarce capacity, so urgent cases are never delayed", .34, {"urgent": 3, "priority": 3, "critical": 3, "emergency": 2}),
    ("Add capacity at the bottleneck: extend hours, cross-train or borrow staff", .26, {"leave": 2, "staff": 2, "surgeon": 2, "absent": 2, "short": 1}),
    ("Batch similar jobs to cut set-up time at the constrained resource", .20, {"setup": 3, "batch": 3, "turnover": 3, "prep": 2}),
    ("Reduce demand or defer lower-priority items with clear communication", .20, {"backlog": 2, "waiting": 2, "overload": 3})],
   ["Identify the single bottleneck resource and its true hours available", "Measure waiting time and urgency of the backlog", "Check set-up and idle time at the bottleneck"],
   ["Sequence work so the bottleneck is never idle and urgent items go first", "Use a simple rule (urgency, then oldest) and publish it", "Review weekly and adjust"],
   ["Waiting time by urgency class", "Bottleneck utilisation", "Cancellations and overruns"],
   ["How many jobs are waiting and how urgent are they?", "What hours is each resource available?", "Which constraints are fixed (rules, safety, contracts)?"]),
 P("prioritise", "Choosing among options with limited resources", "decide", {"ideas": 3, "options": 2, "pursue": 3, "prioritise": 3, "engineers": 2, "runway": 3, "cash": 1, "order": 1, "which": 1, "start-up": 3, "startup": 3}, [r"which|choose|pursue|prioriti|fund|ideas|options", r"cash|budget|engineer|runway|team|capacity|months|limited"],
   [("Start with the option that can show real customer evidence soonest and cheapest", .35, {"validate": 3, "test": 2, "quick": 2, "fast": 2, "customers": 1}),
    ("Prefer the option with the highest expected value per scarce resource (engineer-month or cash)", .30, {"engineers": 3, "value": 2, "revenue": 2, "return": 2}),
    ("Choose options that fit within your runway with a safety margin, even if the upside is smaller", .20, {"runway": 2, "cash": 1, "months": 1}),
    ("Keep one cheap experiment on a second option as a hedge", .15, {"hedge": 3, "risk": 2, "uncertain": 2})],
   ["Estimate for each option: cost, time to first evidence, and likely value", "Check which options fit within your runway with a safety margin", "Talk to 5 potential customers per option"],
   ["Run the cheapest decisive test on the leading option first", "Decide in advance what result would make you stop", "Revisit the ranking after each test"],
   ["Time to first customer evidence", "Cash months remaining", "Cost per learning"],
   ["What does each option cost and how long until evidence?", "How many months of cash and which people are available?", "What would make you stop an option?"],
   lambda n: (f"With about {n['days'] / 30:.0f} months of runway, favour work that shows evidence within the first third of that time ({n['days'] / 90:.1f} months)." if n["days"] and n["days"] >= 60 else None)),
 P("marketing", "Marketing and trend spend", "decide", {"marketing": 3, "campaign": 3, "trend": 2, "trending": 2, "topics": 2, "channel": 2, "audience": 2, "fund": 1, "budget": 1}, [r"marketing|campaign|trend|topics|channel|audience|advert", r"."],
   [("Fund the few trends with the best expected return after decay, and avoid ones with overlapping audiences", .36, {"trend": 3, "trending": 3, "topics": 3, "decay": 3, "overlap": 3}),
    ("Spread a small test budget across channels, then move money to what converts", .28, {"channel": 3, "test": 2, "budget": 2}),
    ("Concentrate on the audience segment most likely to buy", .22, {"audience": 3, "segment": 3, "customers": 1}),
    ("Hold a reserve for fast-rising opportunities", .14, {"reserve": 3, "flexible": 2})],
   ["Score each topic on expected reach, time to ship and audience overlap", "Check which topics your audience already engages with", "Estimate cost to produce each and the expected lifespan"],
   ["Fund the top few, test small, scale the winners", "Drop topics that duplicate an audience you already reach", "Use the Trend planner tab to optimise the mix"],
   ["Cost per engaged customer", "Time from idea to launch", "Overlap between campaigns"],
   ["What is the budget and the deadline?", "Who is the target audience?", "How long does each topic take to produce?"]),
 P("retention", "Customer churn and retention", "diagnose", {"churn": 3, "cancel": 3, "retention": 3, "leaving": 2, "subscribers": 2, "renewal": 3, "loyal": 2}, [r"churn|cancel|retention|renewal|leaving|lapsed", r"."],
   [("Onboarding or early value is weak, so users leave in the first weeks", .30, {"onboarding": 3, "new": 2, "trial": 2, "first": 1}),
    ("A competitor or a better price is pulling customers away", .22, {"competitor": 3, "price": 2, "cheaper": 3}),
    ("Product or service quality issues", .24, {"bug": 2, "quality": 3, "support": 2, "complaint": 3}),
    ("Wrong customers acquired, so poor fit", .14, {"campaign": 2, "discount": 2, "acquired": 2}),
    ("Billing or renewal friction", .10, {"billing": 3, "payment": 2, "renewal": 2, "failed": 2})],
   ["Compare churn by signup month and by customer type", "Read exit reasons and support tickets of recent leavers", "Look at usage in the weeks before leaving"],
   ["Fix the first-week experience for the segment that leaves most", "Reach out to at-risk customers before renewal", "Test a retention offer on a small group first"],
   ["Churn by cohort", "Activation rate", "Support tickets per customer"],
   ["Which customers leave and when?", "What do they say when they cancel?", "Did anything change before the rise?"]),
 P("cost", "Cost reduction", "decide", {"costs": 3, "cost": 2, "expenses": 3, "cut": 2, "savings": 3, "margin": 2, "overhead": 3, "waste": 2}, [r"cost|expens|overhead|margin|saving|waste", r"cut|reduc|lower|high|rising|squeez|save"],
   [("Remove waste in the largest cost lines first, by pareto, before touching anything customers see", .34, {"waste": 3, "largest": 2, "biggest": 2, "overhead": 2}),
    ("Renegotiate or consolidate suppliers and contracts", .28, {"supplier": 3, "contract": 3, "vendor": 3}),
    ("Improve efficiency in processes with high labour or rework", .22, {"labour": 3, "process": 2, "rework": 3, "automation": 3}),
    ("Raise price or mix where customers are least price-sensitive", .16, {"price": 3, "margin": 2, "mix": 3})],
   ["List costs by category and find the few that make up most of the total", "Compare supplier prices with quotes", "Measure rework and idle time"],
   ["Attack the top three cost lines with a named owner each", "Lock in savings with contracts only after checking quality", "Track savings monthly against a baseline"],
   ["Cost per unit", "Gross margin", "Savings delivered versus planned"],
   ["Which cost lines are the largest?", "What target and deadline apply?", "Which costs can customers not see changed?"]),
]

def reason(text):
    t = text.lower(); words = set(re.findall(r"[a-z\-]+", t)); n = nums(text); best = None
    for pb in BOOK:
        if not all(re.search(rx, t) for rx in pb["need"]): continue
        hit = [w for w in pb["cues"] if w in t]; sc = sum(pb["cues"][w] for w in hit)
        if sc >= 2 and (not best or sc > best[0]): best = (sc, pb, hit)
    if not best: return None
    sc, pb, hit = best
    post = [(name, prior * math.exp(.6 * sum(w for k, w in cue.items() if k in t))) for name, prior, cue in pb["hyp"]]
    z = sum(p for _, p in post); post = sorted(((n_, p / z) for n_, p in post), key=lambda x: -x[1])
    ins = pb["calc"](n) if pb["calc"] else None
    top = post[0]
    rec = (f"Most likely: {top[0]} ({top[1]:.0%}). Test first: {pb['tests'][0].lower()}." if pb["kind"] == "diagnose"
           else f"Start with: {top[0].lower()}. Then: {post[1][0].lower()}.")
    why = f"I read this as: {pb['title'].lower()}, from the words {', '.join(hit[:5])}. I ranked the possibilities by how plausible each usually is, adjusted for the wording of your request. These percentages are judgement-based, not measured."
    if ins: why += " " + ins
    return dict(recommendation=rec, why=why, alternatives=[f"{a} ({p:.0%})" for a, p in post[1:3]] + ["Next steps: " + "; ".join(pb["actions"][:2]) + "."],
                risks=["Ranking is heuristic and based on your wording only.", "Real data may point elsewhere; run the first test before acting."], monitor=pb["monitor"], questions=pb["questions"], tests=pb["tests"], title=pb["title"])
