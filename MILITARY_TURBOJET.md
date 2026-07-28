# Afterburning military turbojet optimization

The simulation is implemented in
[`optimize_military_turbojet.py`](optimize_military_turbojet.py). It uses
pyCycle components inside an OpenMDAO optimization problem.

## Formulation

The dry engine is sized at Mach 1.8 and 12,000 m to produce exactly 25 kN of
net thrust. The optimizer maximizes afterburner-on specific thrust,

\[
F_\mathrm{specific}=\frac{F_{n,\mathrm{wet}}}{\dot m_\mathrm{air}},
\]

using these design variables:

- Compressor pressure ratio: 3 to 30
- Turbine inlet temperature: 1,250 to 1,700 K

The afterburner fuel-air ratio is balanced to an exit temperature of 2,100 K.
The dry air mass flow is balanced to the 25 kN thrust requirement. Turbine
pressure ratio is balanced so compressor and turbine shaft power match.

The gradient-based optimizer is OpenMDAO's `ScipyOptimizeDriver` with SLSQP.
Central finite-difference total derivatives are used through fully converged
pyCycle cycles. Two gradient-based starts straddle a converging-diverging
nozzle regime transition, and the best converged feasible solution is kept.

Every modeled dry and wet flow-station area is constrained to

\[
A \leq \frac{\pi(0.7\ \mathrm{m})^2}{4}.
\]

## Run

```bash
MPLCONFIGDIR=/tmp/aeroprop-mpl OPENMDAO_REPORTS=0 \
  .venv/bin/python optimize_military_turbojet.py
```

For a quick local optimization, use `--single-start`. To evaluate the initial
model without optimization, use `--model-only`.

The machine-readable result is written to
[`military_turbojet_optimized.json`](military_turbojet_optimized.json).

## Optimized result

| Quantity | Value |
|---|---:|
| Compressor pressure ratio | 22.9245 |
| Turbine inlet temperature | 1,700.0 K |
| Dry air flow | 36.920 kg/s |
| Dry net thrust | 25.000 kN |
| Dry specific thrust | 677.143 N/(kg/s) |
| Afterburner exit temperature | 2,100.0 K |
| Wet net thrust | 43.605 kN |
| Wet specific thrust | 1,181.077 N/(kg/s) |
| Thrust augmentation | 1.744 |
| Maximum equivalent flow diameter | 0.5213 m |

All requested constraints are satisfied. The TIT limit is active; the
diameter constraint has approximately 0.179 m of margin.

## Modeling assumptions

- Tabular Jet-A thermodynamics
- AXI5 compressor map and LPT2269 turbine map, with extrapolation enabled
- Compressor/turbine efficiencies of 0.86/0.89
- 2% duct, 4% main-combustor, and 6% afterburner total-pressure loss
- Military-specification inlet recovery at the flight Mach number
- 9% total compressor bleed returned as turbine cooling flow
- A converging-diverging nozzle with a 0.99 velocity coefficient

pyCycle reports gas-path flow area, not structural casing diameter. The
reported diameter is therefore the equivalent circular flow diameter. Hub,
blockage, wall thickness, accessories, and installation clearance require a
separate mechanical sizing model.
