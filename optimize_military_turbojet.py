"""Gradient-based design of an afterburning military turbojet with pyCycle.

The dry engine is sized for 25 kN net thrust at Mach 1.8 and 12,000 m.
Compressor pressure ratio and turbine-inlet temperature are optimized to
maximize afterburning specific thrust.  A second, otherwise identical, cycle
evaluates the afterburner-on condition at the 2,100 K temperature limit.

Run:
    .venv/bin/python optimize_military_turbojet.py

Use ``--model-only`` to evaluate the initial design without optimization.
Use ``--single-start`` for a faster, local SLSQP run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import openmdao.api as om
import pycycle.api as pyc
from pycycle.thermo.tabular.thermo_add import ThermoAdd


DESIGN_MACH = 1.8
DESIGN_ALTITUDE_M = 12_000.0
DRY_THRUST_N = 25_000.0
MAX_T4_K = 1_700.0
MAX_T7_K = 2_100.0
MAX_DIAMETER_M = 0.7
MAX_FLOW_AREA_M2 = math.pi * MAX_DIAMETER_M**2 / 4.0

# Component assumptions used for both dry and afterburning operation.
COMPRESSOR_EFFICIENCY = 0.86
TURBINE_EFFICIENCY = 0.89
SHAFT_SPEED_RPM = 10_000.0

FLOW_STATIONS = (
    "inlet.Fl_O",
    "duct.Fl_O",
    "comp.Fl_O",
    "burner.Fl_O",
    "turb.Fl_O",
    "afterburner.Fl_O",
)

# Two starts straddle the CD-nozzle regime transition. Each run is a
# gradient-based SLSQP optimization; the best feasible local solution is kept.
OPTIMIZATION_STARTS = (
    (10.0, 1_650.0),
    (23.0, 1_700.0),
)


def install_numpy2_pycycle_compatibility() -> None:
    """Make pyCycle 4.4's tabular fuel mixer compatible with NumPy 2.

    pyCycle indexes the fuel-composition vector with an integer and then adds
    an OpenMDAO one-element array.  NumPy 2 correctly rejects that implicit
    array-to-scalar conversion.  A one-element slice retains the intended
    vector operation and remains complex-step differentiable.
    """

    if getattr(ThermoAdd, "_numpy2_compatible", False):
        return

    original_output_port_data = ThermoAdd.output_port_data

    def numpy2_compatible_output_port_data(self: ThermoAdd) -> None:
        original_output_port_data(self)
        if hasattr(self, "idx_compo"):
            index = self.idx_compo
            self.idx_compo = slice(index, index + 1)

    ThermoAdd.output_port_data = numpy2_compatible_output_port_data
    ThermoAdd._numpy2_compatible = True


install_numpy2_pycycle_compatibility()


class TurbojetCycle(pyc.Cycle):
    """Single-spool turbojet cycle with an optional active afterburner."""

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("size_to_thrust", default=False, types=bool)
        self.options.declare("afterburner_on", default=False, types=bool)

    def setup(self) -> None:
        # Tabular Jet-A properties are substantially faster and smoother for
        # repeated optimization evaluations than the equilibrium CEA model.
        self.options["thermo_method"] = "TABULAR"
        self.options["thermo_data"] = pyc.AIR_JETA_TAB_SPEC

        size_to_thrust = self.options["size_to_thrust"]
        afterburner_on = self.options["afterburner_on"]

        self.add_subsystem("fc", pyc.FlightConditions())
        self.add_subsystem("recovery", pyc.MilSpecRecovery())
        self.add_subsystem("inlet", pyc.Inlet())
        self.add_subsystem("duct", pyc.Duct())
        self.add_subsystem(
            "comp",
            pyc.Compressor(
                map_data=pyc.AXI5,
                bleed_names=["cool1", "cool2"],
                map_extrap=True,
            ),
            promotes_inputs=["Nmech"],
        )
        self.add_subsystem("burner", pyc.Combustor(fuel_type="FAR"))
        self.add_subsystem(
            "turb",
            pyc.Turbine(
                map_data=pyc.LPT2269,
                bleed_names=["cool1", "cool2"],
                map_extrap=True,
            ),
            promotes_inputs=["Nmech"],
        )
        self.add_subsystem("afterburner", pyc.Combustor(fuel_type="FAR"))
        self.add_subsystem(
            "nozzle",
            pyc.Nozzle(nozzType="CD", lossCoef="Cv", internal_solver=True),
        )
        self.add_subsystem(
            "shaft",
            pyc.Shaft(num_ports=2),
            promotes_inputs=["Nmech"],
        )
        self.add_subsystem(
            "perf",
            pyc.Performance(num_nozzles=1, num_burners=2),
        )

        self.connect("fc.MN", "recovery.MN")
        self.connect("recovery.ram_recovery", "inlet.ram_recovery")
        self.connect("duct.Fl_O:tot:P", "perf.Pt2")
        self.connect("comp.Fl_O:tot:P", "perf.Pt3")
        self.connect("burner.Wfuel", "perf.Wfuel_0")
        self.connect("afterburner.Wfuel", "perf.Wfuel_1")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("nozzle.Fg", "perf.Fg_0")
        self.connect("comp.trq", "shaft.trq_0")
        self.connect("turb.trq", "shaft.trq_1")
        self.connect("fc.Fl_O:stat:P", "nozzle.Ps_exhaust")

        balance = self.add_subsystem("balance", om.BalanceComp())
        if size_to_thrust:
            balance.add_balance(
                "W",
                val=55.0,
                units="kg/s",
                eq_units="N",
                lower=5.0,
                upper=150.0,
            )
            self.connect("balance.W", "inlet.Fl_I:stat:W")
            self.connect("perf.Fn", "balance.lhs:W")

        balance.add_balance(
            "FAR",
            val=0.025,
            eq_units="degK",
            lower=1.0e-4,
            upper=0.08,
        )
        self.connect("balance.FAR", "burner.Fl_I:FAR")
        self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")

        balance.add_balance(
            "turb_PR",
            val=3.0,
            lower=1.001,
            upper=12.0,
            eq_units="W",
            rhs_val=0.0,
        )
        self.connect("balance.turb_PR", "turb.PR")
        self.connect("shaft.pwr_net", "balance.lhs:turb_PR")

        if afterburner_on:
            balance.add_balance(
                "afterburner_FAR",
                val=0.025,
                eq_units="degK",
                lower=1.0e-6,
                upper=0.08,
            )
            self.connect(
                "balance.afterburner_FAR",
                "afterburner.Fl_I:FAR",
            )
            self.connect(
                "afterburner.Fl_O:tot:T",
                "balance.lhs:afterburner_FAR",
            )

        order = [
            "fc",
            "recovery",
            "inlet",
            "duct",
            "comp",
            "burner",
            "turb",
            "afterburner",
            "nozzle",
            "shaft",
            "perf",
            "balance",
        ]
        self.set_order(order)

        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
        self.pyc_connect_flow("inlet.Fl_O", "duct.Fl_I", connect_stat=False)
        self.pyc_connect_flow("duct.Fl_O", "comp.Fl_I", connect_stat=False)
        self.pyc_connect_flow("comp.Fl_O", "burner.Fl_I", connect_stat=False)
        self.pyc_connect_flow("burner.Fl_O", "turb.Fl_I", connect_stat=False)
        self.pyc_connect_flow(
            "turb.Fl_O",
            "afterburner.Fl_I",
            connect_stat=False,
        )
        self.pyc_connect_flow(
            "afterburner.Fl_O",
            "nozzle.Fl_I",
            connect_stat=False,
        )
        self.pyc_connect_flow(
            "comp.cool1",
            "turb.cool1",
            connect_stat=False,
        )
        self.pyc_connect_flow(
            "comp.cool2",
            "turb.cool2",
            connect_stat=False,
        )

        newton = self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True,
        )
        newton.options["atol"] = 1.0e-7
        newton.options["rtol"] = 1.0e-7
        newton.options["maxiter"] = 40
        newton.options["max_sub_solves"] = 80
        newton.options["iprint"] = -1
        newton.options["reraise_child_analysiserror"] = False
        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options["rho"] = 0.75
        newton.linesearch.options["iprint"] = -1
        self.linear_solver = om.DirectSolver(assemble_jac=True)

        super().setup()


class DesignMetrics(om.ExplicitComponent):
    """Dry/wet specific thrust and the thrust augmentation ratio."""

    def setup(self) -> None:
        self.add_input("dry_thrust", val=DRY_THRUST_N, units="N")
        self.add_input("wet_thrust", val=DRY_THRUST_N, units="N")
        self.add_input("airflow", val=55.0, units="kg/s")
        self.add_output("dry_specific_thrust", val=450.0, units="N*s/kg")
        self.add_output("wet_specific_thrust", val=700.0, units="N*s/kg")
        self.add_output("thrust_augmentation", val=1.0)
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs) -> None:
        dry_thrust = inputs["dry_thrust"]
        wet_thrust = inputs["wet_thrust"]
        airflow = inputs["airflow"]
        outputs["dry_specific_thrust"] = dry_thrust / airflow
        outputs["wet_specific_thrust"] = wet_thrust / airflow
        outputs["thrust_augmentation"] = wet_thrust / dry_thrust

    def compute_partials(self, inputs, partials) -> None:
        dry_thrust = inputs["dry_thrust"]
        wet_thrust = inputs["wet_thrust"]
        airflow = inputs["airflow"]
        partials["dry_specific_thrust", "dry_thrust"] = 1.0 / airflow
        partials["dry_specific_thrust", "wet_thrust"] = 0.0
        partials["dry_specific_thrust", "airflow"] = (
            -dry_thrust / airflow**2
        )
        partials["wet_specific_thrust", "dry_thrust"] = 0.0
        partials["wet_specific_thrust", "wet_thrust"] = 1.0 / airflow
        partials["wet_specific_thrust", "airflow"] = -wet_thrust / airflow**2
        partials["thrust_augmentation", "dry_thrust"] = (
            -wet_thrust / dry_thrust**2
        )
        partials["thrust_augmentation", "wet_thrust"] = 1.0 / dry_thrust
        partials["thrust_augmentation", "airflow"] = 0.0


class MilitaryTurbojetOptimization(om.Group):
    """Coupled dry-sizing and wet-performance optimization model."""

    def setup(self) -> None:
        design = self.add_subsystem("design", om.IndepVarComp())
        design.add_output("compressor_PR", val=10.0)
        design.add_output("T4", val=MAX_T4_K, units="degK")

        self.add_subsystem(
            "dry",
            TurbojetCycle(
                design=True,
                size_to_thrust=True,
                afterburner_on=False,
            ),
        )
        self.add_subsystem(
            "wet",
            TurbojetCycle(
                design=True,
                size_to_thrust=False,
                afterburner_on=True,
            ),
        )
        self.add_subsystem("metrics", DesignMetrics())

        self.connect("design.compressor_PR", "dry.comp.PR")
        self.connect("design.compressor_PR", "wet.comp.PR")
        self.connect("design.T4", "dry.balance.rhs:FAR")
        self.connect("design.T4", "wet.balance.rhs:FAR")
        self.connect("dry.balance.W", "wet.inlet.Fl_I:stat:W")

        self.connect("dry.perf.Fn", "metrics.dry_thrust")
        self.connect("wet.perf.Fn", "metrics.wet_thrust")
        self.connect("dry.inlet.Fl_O:stat:W", "metrics.airflow")

        self.add_design_var(
            "design.compressor_PR",
            lower=3.0,
            upper=30.0,
            ref=10.0,
        )
        self.add_design_var(
            "design.T4",
            lower=1_250.0,
            upper=MAX_T4_K,
            units="degK",
            ref=MAX_T4_K,
        )
        self.add_objective(
            "metrics.wet_specific_thrust",
            scaler=-1.0 / 1_000.0,
        )

        # Constrain each represented flow station in both operating modes
        # rather than only the engine face. This avoids hiding an oversized
        # combustor/afterburner section.
        for mode in ("dry", "wet"):
            for station in FLOW_STATIONS:
                self.add_constraint(
                    f"{mode}.{station}:stat:area",
                    upper=MAX_FLOW_AREA_M2,
                    units="m**2",
                    ref=MAX_FLOW_AREA_M2,
                    alias=(
                        f"diameter_limit_{mode}_"
                        f"{station.replace('.', '_')}"
                    ),
                )

        # Finite-difference total derivatives are used because pyCycle 4.4
        # predates NumPy 2. The optimizer is still gradient-based (SLSQP), and
        # each gradient is taken through a fully converged nonlinear cycle.
        self.approx_totals(
            method="fd",
            form="central",
            step=2.0e-4,
            step_calc="rel_avg",
        )


def set_cycle_defaults(prob: om.Problem, prefix: str, wet: bool) -> None:
    """Set common flight, component, and solver initial values."""

    prob.set_val(f"{prefix}.fc.MN", DESIGN_MACH)
    prob.set_val(f"{prefix}.fc.alt", DESIGN_ALTITUDE_M, units="m")
    prob.set_val(f"{prefix}.recovery.ram_recovery_base", 1.0)
    prob.set_val(f"{prefix}.Nmech", SHAFT_SPEED_RPM, units="rpm")
    prob.set_val(f"{prefix}.inlet.MN", 0.60)
    prob.set_val(f"{prefix}.duct.MN", 0.55)
    prob.set_val(f"{prefix}.comp.MN", 0.25)
    prob.set_val(f"{prefix}.burner.MN", 0.20)
    prob.set_val(f"{prefix}.turb.MN", 0.40)
    prob.set_val(f"{prefix}.afterburner.MN", 0.35)
    prob.set_val(f"{prefix}.comp.eff", COMPRESSOR_EFFICIENCY)
    prob.set_val(f"{prefix}.turb.eff", TURBINE_EFFICIENCY)
    prob.set_val(f"{prefix}.duct.dPqP", 0.02)
    prob.set_val(f"{prefix}.burner.dPqP", 0.04)
    prob.set_val(f"{prefix}.afterburner.dPqP", 0.06)
    prob.set_val(f"{prefix}.nozzle.Cv", 0.99)

    # Representative turbine-cooling extraction fractions.
    prob.set_val(f"{prefix}.comp.cool1:frac_W", 0.06)
    prob.set_val(f"{prefix}.comp.cool1:frac_P", 1.0)
    prob.set_val(f"{prefix}.comp.cool1:frac_work", 1.0)
    prob.set_val(f"{prefix}.comp.cool2:frac_W", 0.03)
    prob.set_val(f"{prefix}.comp.cool2:frac_P", 1.0)
    prob.set_val(f"{prefix}.comp.cool2:frac_work", 1.0)
    prob.set_val(f"{prefix}.turb.cool1:frac_P", 1.0)
    prob.set_val(f"{prefix}.turb.cool2:frac_P", 0.0)

    prob.set_val(f"{prefix}.balance.FAR", 0.025)
    prob.set_val(f"{prefix}.balance.turb_PR", 3.0)
    prob.set_val(f"{prefix}.fc.balance.Pt", 3.0, units="psi")
    prob.set_val(f"{prefix}.fc.balance.Tt", 500.0, units="degR")

    if wet:
        prob.set_val(
            f"{prefix}.balance.rhs:afterburner_FAR",
            MAX_T7_K,
            units="degK",
        )
        prob.set_val(f"{prefix}.balance.afterburner_FAR", 0.025)
    else:
        prob.set_val(f"{prefix}.afterburner.Fl_I:FAR", 0.0)
        prob.set_val(
            f"{prefix}.balance.rhs:W",
            DRY_THRUST_N,
            units="N",
        )
        prob.set_val(f"{prefix}.balance.W", 55.0, units="kg/s")


def build_problem() -> om.Problem:
    """Construct and initialize the optimization problem."""

    prob = om.Problem(model=MilitaryTurbojetOptimization())
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    prob.driver.options["maxiter"] = 80
    prob.driver.options["tol"] = 1.0e-7
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = []

    prob.setup()
    set_cycle_defaults(prob, "dry", wet=False)
    set_cycle_defaults(prob, "wet", wet=True)
    prob.set_solver_print(level=-1)
    return prob


def scalar(
    prob: om.Problem,
    name: str,
    units: str | None = None,
) -> float:
    """Return an OpenMDAO scalar output as a Python float."""

    return float(np.asarray(prob.get_val(name, units=units)).item())


def collect_results(prob: om.Problem) -> dict[str, Any]:
    """Collect optimized constraints and performance in SI units."""

    station_areas = {
        mode: {
            station: scalar(
                prob,
                f"{mode}.{station}:stat:area",
                "m**2",
            )
            for station in FLOW_STATIONS
        }
        for mode in ("dry", "wet")
    }
    station_diameters = {
        mode: {
            station: math.sqrt(4.0 * area / math.pi)
            for station, area in mode_areas.items()
        }
        for mode, mode_areas in station_areas.items()
    }
    flat_diameters = {
        f"{mode}.{station}": diameter
        for mode, mode_diameters in station_diameters.items()
        for station, diameter in mode_diameters.items()
    }

    dry_airflow = scalar(prob, "dry.inlet.Fl_O:stat:W", "kg/s")
    dry_thrust = scalar(prob, "dry.perf.Fn", "N")
    wet_thrust = scalar(prob, "wet.perf.Fn", "N")

    return {
        "flight_condition": {
            "mach": DESIGN_MACH,
            "altitude_m": DESIGN_ALTITUDE_M,
        },
        "design": {
            "compressor_pressure_ratio": scalar(
                prob,
                "design.compressor_PR",
            ),
            "turbine_inlet_temperature_K": scalar(
                prob,
                "design.T4",
                "degK",
            ),
            "turbine_pressure_ratio": scalar(
                prob,
                "dry.turb.PR",
            ),
            "airflow_kg_per_s": dry_airflow,
            "maximum_flowpath_diameter_m": max(flat_diameters.values()),
            "limiting_station": max(flat_diameters, key=flat_diameters.get),
        },
        "dry": {
            "net_thrust_N": dry_thrust,
            "specific_thrust_N_s_per_kg": dry_thrust / dry_airflow,
            "TSFC_kg_per_N_hr": scalar(
                prob,
                "dry.perf.TSFC",
                "kg/(h*N)",
            ),
            "fuel_flow_kg_per_s": scalar(
                prob,
                "dry.perf.Wfuel",
                "kg/s",
            ),
        },
        "afterburner_on": {
            "exit_temperature_K": scalar(
                prob,
                "wet.afterburner.Fl_O:tot:T",
                "degK",
            ),
            "net_thrust_N": wet_thrust,
            "specific_thrust_N_s_per_kg": wet_thrust / dry_airflow,
            "thrust_augmentation_ratio": wet_thrust / dry_thrust,
            "TSFC_kg_per_N_hr": scalar(
                prob,
                "wet.perf.TSFC",
                "kg/(h*N)",
            ),
            "total_fuel_flow_kg_per_s": scalar(
                prob,
                "wet.perf.Wfuel",
                "kg/s",
            ),
            "afterburner_fuel_air_ratio": scalar(
                prob,
                "wet.balance.afterburner_FAR",
            ),
        },
        "flowpath_diameters_m": station_diameters,
        "constraints": {
            "dry_thrust_target_N": DRY_THRUST_N,
            "T4_limit_K": MAX_T4_K,
            "T7_limit_K": MAX_T7_K,
            "diameter_limit_m": MAX_DIAMETER_M,
        },
    }


def print_results(results: dict[str, Any]) -> None:
    """Print a compact optimization report."""

    design = results["design"]
    dry = results["dry"]
    wet = results["afterburner_on"]

    print("\nOptimized military turbojet")
    print("---------------------------")
    print(
        f"Design point          : M {DESIGN_MACH:.1f}, "
        f"{DESIGN_ALTITUDE_M:,.0f} m"
    )
    print(
        f"Compressor PR         : "
        f"{design['compressor_pressure_ratio']:.4f}"
    )
    print(
        f"Turbine inlet T4      : "
        f"{design['turbine_inlet_temperature_K']:.2f} K"
    )
    print(
        f"Dry airflow           : "
        f"{design['airflow_kg_per_s']:.3f} kg/s"
    )
    print(f"Dry net thrust        : {dry['net_thrust_N'] / 1e3:.3f} kN")
    print(
        f"Dry specific thrust   : "
        f"{dry['specific_thrust_N_s_per_kg']:.3f} N/(kg/s)"
    )
    print(f"Dry TSFC              : {dry['TSFC_kg_per_N_hr']:.3f} kg/(N·hr)")
    print(
        f"Maximum flow diameter : "
        f"{design['maximum_flowpath_diameter_m']:.4f} m "
        f"at {design['limiting_station']}"
    )
    print(
        f"Afterburner exit T7   : "
        f"{wet['exit_temperature_K']:.2f} K"
    )
    print(
        f"Wet net thrust        : "
        f"{wet['net_thrust_N'] / 1e3:.3f} kN"
    )
    print(
        f"Wet specific thrust   : "
        f"{wet['specific_thrust_N_s_per_kg']:.3f} N/(kg/s)"
    )
    print(f"Wet TSFC              : {wet['TSFC_kg_per_N_hr']:.3f} kg/(N·hr)")
    print(
        f"Thrust augmentation   : "
        f"{wet['thrust_augmentation_ratio']:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Evaluate the initial design without running SLSQP.",
    )
    parser.add_argument(
        "--single-start",
        action="store_true",
        help="Run only the first SLSQP start (faster but less robust).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("military_turbojet_optimized.json"),
        help="JSON results path.",
    )
    args = parser.parse_args()

    # Avoid report generation during the many optimizer evaluations.
    os.environ.setdefault("OPENMDAO_REPORTS", "0")

    if args.model_only:
        prob = build_problem()
        prob.run_model()
        results = collect_results(prob)
    else:
        starts = (
            OPTIMIZATION_STARTS[:1]
            if args.single_start
            else OPTIMIZATION_STARTS
        )
        candidates: list[tuple[float, om.Problem, dict[str, Any]]] = []
        for start_pr, start_t4 in starts:
            print(
                f"\nStarting SLSQP from PR={start_pr:.1f}, "
                f"T4={start_t4:.0f} K"
            )
            candidate_prob = build_problem()
            candidate_prob.set_val("design.compressor_PR", start_pr)
            candidate_prob.set_val("design.T4", start_t4, units="degK")
            driver_result = candidate_prob.run_driver()
            if not driver_result.success:
                print("  This start did not converge; skipping it.")
                continue

            candidate_results = collect_results(candidate_prob)
            objective = candidate_results["afterburner_on"][
                "specific_thrust_N_s_per_kg"
            ]
            print(f"  Converged wet specific thrust: {objective:.3f} N/(kg/s)")
            candidates.append(
                (objective, candidate_prob, candidate_results)
            )

        if not candidates:
            raise RuntimeError("No SLSQP start converged.")

        _, prob, results = max(candidates, key=lambda candidate: candidate[0])
        results["optimization"] = {
            "method": "multistart SLSQP with central finite-difference totals",
            "objective": "maximize afterburner-on specific thrust",
            "starts": [
                {
                    "compressor_pressure_ratio": start_pr,
                    "T4_K": start_t4,
                }
                for start_pr, start_t4 in starts
            ],
            "converged_candidates": [
                {
                    "wet_specific_thrust_N_s_per_kg": objective,
                    "compressor_pressure_ratio": candidate_results["design"][
                        "compressor_pressure_ratio"
                    ],
                    "T4_K": candidate_results["design"][
                        "turbine_inlet_temperature_K"
                    ],
                }
                for objective, _, candidate_results in candidates
            ],
        }

    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print_results(results)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
