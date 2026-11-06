# Labelling rubric (version 1)

Written so that two reviewers reach the same label from the problem text and its constraints.
Apply the first rule that fits.

- R6 routing: ordering stops or assigning orders to vehicle routes.
- R5 facility_location: choosing which sites, hubs, offices or stations to open, keep or close.
- R3 scheduling: deciding when, or in what order, work happens (dates, shifts, sequences, timetables).
- R4 assignment: matching people or resources one to one (or many to one) with tasks, slots or places, without deciding timing.
- R1 budget_allocation: deciding how much, a divisible amount or quantity split across options.
- R2 select a whole set of items, then:
  - R2d set_cover: the stated requirement is that the chosen items together cover every need or segment.
  - R2a portfolio_selection: overlap, conflicts, risk or diversification between the items is part of the problem.
  - R2b multi_dim_knapsack: two or more distinct resource limits are named (for example budget and staff).
  - R2c knapsack: one limit, no interactions.

## Status
- Reviewed labels live in data/label_review.csv. Original labels are kept in `formulation_family_original`.
- The review was done by one reviewer (the assistant) in one pass, after the first accuracy run. Treat the cleaned-label score as a development number, not as an independent measurement.
- A reviewer who did not see the original labels should relabel a random sample of at least 30 rows. Agreement with data/label_review.csv is the number to report.
