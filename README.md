This is a collection of automation scripts for implementing a 3-pass approach to
optimise planned maintenance of generation projects and budgets for hydro
projects.

The methodology of the 3-pass approach is as follows:
Pass 1:
- Yearly step/subproblem
- Daily (365) timepoints
- Thermal generators specified as fleets (gen_commit_cap)
- Planned maintenance provided as endogneous availability inputs
- Yearly hydro budgets
Output: Thermal maintenance scheduels and hydro budgets at a daily granularity

Pass 2:
- Monthly step/subproblem
- Daily (365) timepoints
- Thermal generators specified as fleets (gen_commit_cap)
- Planned maintenance schedules from Pass 1 specified as exogneous availability at daily granularity
- Daily hydro budgets from Pass 1 adjusted to monthly min/max constraints
Output: Final daily hydro budgets

Pass 3: Full production cost simulation with above daily allocation
- Daily step/subproblem
- 15-min (35040) timepoints
- Generators specified with exogenous availability at 15-min granularity based on:
  - planned maintenance determined in Pass 1, interleaved with
  - forced outage separately generated in a random manner for the whole year for each generator
- Daily hydro budgets modified as per output from Pass 2 with minimum capacity factor of 0 and maximum of 1 within the day
