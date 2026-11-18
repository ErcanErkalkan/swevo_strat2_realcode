# Model and evidence semantics

Euclidean distance is also travel time under normalized speed. A route pays vehicle activation cost plus distance. Tractive energy is a distance and load-distance proxy; delivery load decreases along the route and the depot return is unloaded. CO2 multiplies modeled energy by vehicle, destination-zone and regime factors; the return arc uses the last customer's zone. These quantities are model outputs, not traffic observations or measured fuel use.

Service starts after arrival and the opening of the customer's time window. Capacity, time-window and duty violations are evaluated alongside missing/duplicate coverage. The objective is the weighted average of normalized cost, energy and CO2, plus an overtime-ratio penalty. Infeasible solutions receive an additional penalty. `accepted` means all hard violations are at most 1e-9; `strict_duty` additionally requires zero overtime within that tolerance. Bounded overtime can therefore be accepted without being strict-duty compliant. All primary methods finish accepted and strict-duty compliant.

Public regimes retain the full depot horizon [0,H]. The model has route/regime choices; it does not maintain physical driver handoffs or persistent fleet identities. The 18-run synthetic suite exercises non-overlapping regime semantics at limited scale.

Comparator implementations share final evaluator semantics but have distinct search kernels. They are frozen internal implementations, not assertions about optimally tuned external HGS/ALNS/ILS software. Equal historical tuning effort is not established. `eval_budget` is an outer-step guard and omits nested evaluations. Common nominal time caps are 1800/3600/7200 seconds by tier; actual recorded times differ. `compute_wh` is elapsed time times assumed tier power (52/86/128 W), not meter data.

The ablation factors are conceptual families. In particular, the NoLNS condition disables periodic boundary intensification and the route-ALNS endgame together. A complete valid preregistered campaign is required before attributing system performance to these families; the current 48-row slice cannot establish such effects.
