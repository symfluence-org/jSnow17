# Snow17 0.3.0 equations and depletion-history corrections

Reference: [Anderson (2006), NOAA Snow17 description](https://www.weather.gov/media/owp/oh/hrl/docs/22snow17.pdf),
sections 3.3–3.5, equations 5–9 and the areal-depletion discussion on printed
pages 32–34. Tests are hand-derived equation/history fixtures, not outputs from
a complete independent operational Snow17 executable.

Seasonal peak accumulation is updated before melt; depletion uses the lesser
of SI and the peak. The default curve's positive-pack lower endpoint is .05;
an empty pack has zero area. Fresh snow above .2 mm/hour of interval depth
restores full effective area until 25% of that snowfall has melted. The
remaining 75% returns toward the pre-snow area. Continuing layers extend the
excursion; accumulation reaching three times the underlying pack resets its
seasonal peak. `effective_snow_area` exposes this history-aware diagnostic.

Equation tests cover rain heat, longwave/turbulent rain-on-snow energy,
atmospheric pressure with elevation, the .25 mm/hour rain threshold,
northern high-latitude seasonal melt adjustment, timestep-scaled ATI, heat
capacity and refreezing before drainage. NumPy/JAX history and conservation
checks supplement the equation fixtures. Use mm per interval and days for dt;
melt factors retain their documented six-hour units.

The state adds Wns, Ans, W100 and a fresh-excursion flag (defaults preserve old
six-argument construction). `swe` retains its legacy name but stores seasonal
peak SWE. Old checkpoint histories cannot be recovered automatically; replay
from a snow-free initial condition. Recalibrate after the numerical corrections.

Effective Snow17 area represents energy-effective melt area. It is not a
geometric exposed-glacier fraction or satellite optical snow observation.
The package still uses instantaneous liquid retention/drainage, not the full
operational Snow17 liquid transmission lag. These tests do not establish
complete operational equivalence, nor validity of any glacier coupling.

`return_energy=True` adds gross melt-energy equivalent and heat-deficit
consumption diagnostics (mm water equivalent) without changing the ordinary
Snow17 outputs. These diagnostics enable uniform-HRU snow/ice time allocation;
they are not geometric area measurements or a full energy-balance budget.
