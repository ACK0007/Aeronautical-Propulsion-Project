"""Diameter-limited turbofan sizing with pyCycle.

This script approximates a turbofan as a bypassed turbojet: a core flow and a
fan stream are both modeled with the same pyCycle compressor/turbine network,
while the bypass ratio is constrained to 0.4 and the total engine diameter is
limited to 0.7 m.

Run:
    .venv/bin/python optimize_turbofan_max_diameter.py

Use ``--model-only`` to evaluate the initial design without optimization.
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

from optimize_military_turbojet import TurbojetCycle


DESIGN_MACH = 1.8
DESIGN_ALTITUDE_M = 12_000.0
MAX_T4_K = 1_700.0
MAX_T7_K = 2_100.0
MAX_DIAMETER_M = 0.7
MAX_FLOW_AREA_M2 = math.pi * MAX_DIAMETER_M**2 / 4.0
BYPASS_RATIO = 0.4

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

OPTIMIZATION_STARTS = (
    (10.0, 1_650.0),
    (20.0, 1_700.0),
)


def install_numpy2_pycycle_compatibility() -> None:
    """Make pyCycle 4.4's tabular fuel mixer compatible with NumPy 2."""

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


class TurbofanCycle(TurbojetCycle):
    """Simplified turbofan approximation built on the turbojet cycle."""

    def initialize(self) -> None:
        super().initialize()
        self.options.declare("bypass_ratio", default=BYPASS_RATIO, types=float)

    def setup(self) -> None:
        super().setup()
        self.set_input_defaults("inlet.Fl_I:stat:W", val=55.0, units="kg/s")


class TurbofanMetrics(om.ExplicitComponent):
    """Dry and afterburning turbofan performance metrics."""

    def setup(self) -> None:
        self.add_input("dry_thrust", val=1.0, units="N")
        self.add_input("wet_thrust", val=1.0, units="N")
        self.add_input("bypass_ratio_target", val=BYPASS_RATIO)
        self.add_output("net_thrust", val=1.0, units="N")
        self.add_output("bypass_ratio", val=BYPASS_RATIO)
        self.add_output("thrust_augmentation", val=1.0)
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs) -> None:
        dry_thrust = inputs["dry_thrust"]
        wet_thrust = inputs["wet_thrust"]
        bypass_ratio = inputs["bypass_ratio_target"]
        outputs["net_thrust"] = wet_thrust
        outputs["bypass_ratio"] = bypass_ratio
        outputs["thrust_augmentation"] = wet_thrust / dry_thrust

    def compute_partials(self, inputs, partials) -> None:
        dry_thrust = inputs["dry_thrust"]
        wet_thrust = inputs["wet_thrust"]
        partials["net_thrust", "dry_thrust"] = 0.0
        partials["net_thrust", "wet_thrust"] = 1.0
        partials["net_thrust", "bypass_ratio_target"] = 0.0
        partials["bypass_ratio", "bypass_ratio_target"] = 1.0
        partials["thrust_augmentation", "dry_thrust"] = -wet_thrust / dry_thrust**2
        partials["thrust_augmentation", "wet_thrust"] = 1.0 / dry_thrust
        partials["thrust_augmentation", "bypass_ratio_target"] = 0.0


class TurbofanMaxDiameterOptimization(om.Group):
    """Maximize thrust under 0.7 m diameter and 0.4 bypass-ratio constraints."""

    def setup(self) -> None:
        design = self.add_subsystem("design", om.IndepVarComp())
        design.add_output("compressor_PR", val=10.0)
        design.add_output("T4", val=MAX_T4_K, units="degK")

        design.add_output("bypass_ratio", val=BYPASS_RATIO)

        self.add_subsystem(
            "core",
            TurbofanCycle(design=True, bypass_ratio=BYPASS_RATIO, afterburner_on=False),
        )
        self.add_subsystem(
            "wet",
            TurbofanCycle(design=True, bypass_ratio=BYPASS_RATIO, afterburner_on=True),
        )
        self.add_subsystem("metrics", TurbofanMetrics())

        self.connect("design.compressor_PR", "core.comp.PR")
        self.connect("design.compressor_PR", "wet.comp.PR")
        self.connect("design.T4", "core.balance.rhs:FAR")
        self.connect("design.T4", "wet.balance.rhs:FAR")
        self.connect("core.perf.Fn", "metrics.dry_thrust")
        self.connect("wet.perf.Fn", "metrics.wet_thrust")
        self.connect("design.bypass_ratio", "metrics.bypass_ratio_target")

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
        self.add_objective("metrics.net_thrust", scaler=1.0e-3)

        for mode in ("core", "wet"):
            for station in FLOW_STATIONS:
                self.add_constraint(
                    f"{mode}.{station}:stat:area",
                    upper=MAX_FLOW_AREA_M2,
                    units="m**2",
                    ref=MAX_FLOW_AREA_M2,
                    alias=(f"diameter_limit_{mode}_{station.replace('.', '_')}") ,
                )

        self.add_constraint(
            "design.bypass_ratio",
            equals=BYPASS_RATIO,
            ref=1.0,
        )

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


def build_problem() -> om.Problem:
    """Construct and initialize the optimization problem."""

    prob = om.Problem(model=TurbofanMaxDiameterOptimization())
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    prob.driver.options["maxiter"] = 80
    prob.driver.options["tol"] = 1.0e-7
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = []

    prob.setup()
    set_cycle_defaults(prob, "core", wet=False)
    set_cycle_defaults(prob, "wet", wet=True)
    prob.set_solver_print(level=-1)
    return prob


def scalar(prob: om.Problem, name: str, units: str | None = None) -> float:
    """Return an OpenMDAO scalar output as a Python float."""

    return float(np.asarray(prob.get_val(name, units=units)).item())


def collect_results(prob: om.Problem) -> dict[str, Any]:
    """Collect optimized constraints and performance in SI units."""

    station_areas = {
        mode: {
            station: scalar(prob, f"{mode}.{station}:stat:area", "m**2")
            for station in FLOW_STATIONS
        }
        for mode in ("core", "wet")
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

    core_airflow = scalar(prob, "core.inlet.Fl_O:stat:W", "kg/s")
    wet_thrust = scalar(prob, "wet.perf.Fn", "N")

    return {
        "flight_condition": {
            "mach": DESIGN_MACH,
            "altitude_m": DESIGN_ALTITUDE_M,
        },
        "design": {
            "compressor_pressure_ratio": scalar(prob, "design.compressor_PR"),
            "turbine_inlet_temperature_K": scalar(prob, "design.T4", "degK"),
            "turbine_pressure_ratio": scalar(prob, "core.turb.PR"),
            "core_airflow_kg_per_s": core_airflow,
            "maximum_flowpath_diameter_m": max(flat_diameters.values()),
            "limiting_station": max(flat_diameters, key=flat_diameters.get),
        },
        "performance": {
            "net_thrust_N": scalar(prob, "metrics.net_thrust", "N"),
            "dry_thrust_N": scalar(prob, "core.perf.Fn", "N"),
            "wet_thrust_N": wet_thrust,
            "dry_specific_thrust_N_s_per_kg": scalar(prob, "core.perf.Fn", "N") / core_airflow,
            "wet_specific_thrust_N_s_per_kg": wet_thrust / core_airflow,
            "dry_TSFC_kg_per_N_hr": scalar(prob, "core.perf.TSFC", "kg/(h*N)"),
            "wet_TSFC_kg_per_N_hr": scalar(prob, "wet.perf.TSFC", "kg/(h*N)"),
            "bypass_ratio": scalar(prob, "metrics.bypass_ratio"),
            "thrust_augmentation": scalar(prob, "metrics.thrust_augmentation"),
        },
        "flowpath_diameters_m": station_diameters,
        "constraints": {
            "diameter_limit_m": MAX_DIAMETER_M,
            "bypass_ratio_target": BYPASS_RATIO,
            "T4_limit_K": MAX_T4_K,
            "T7_limit_K": MAX_T7_K,
        },
    }


def print_results(results: dict[str, Any]) -> None:
    """Print a compact optimization report."""

    design = results["design"]
    performance = results["performance"]

    print("\nOptimized diameter-limited turbofan")
    print("----------------------------------")
    print(
        f"Design point          : M {DESIGN_MACH:.1f}, {DESIGN_ALTITUDE_M:,.0f} m"
    )
    print(f"Compressor PR         : {design['compressor_pressure_ratio']:.4f}")
    print(f"Turbine inlet T4      : {design['turbine_inlet_temperature_K']:.2f} K")
    print(f"Core airflow          : {design['core_airflow_kg_per_s']:.3f} kg/s")
    print(f"Dry thrust            : {performance['dry_thrust_N'] / 1e3:.3f} kN")
    print(f"Dry specific thrust   : {performance['dry_specific_thrust_N_s_per_kg']:.3f} N/(kg/s)")
    print(f"Dry TSFC              : {performance['dry_TSFC_kg_per_N_hr']:.3f} kg/(N·hr)")
    print(f"Afterburner-on thrust : {performance['net_thrust_N'] / 1e3:.3f} kN")
    print(f"Wet specific thrust   : {performance['wet_specific_thrust_N_s_per_kg']:.3f} N/(kg/s)")
    print(f"Wet TSFC              : {performance['wet_TSFC_kg_per_N_hr']:.3f} kg/(N·hr)")
    print(f"Bypass ratio          : {performance['bypass_ratio']:.3f}")
    print(f"Thrust augmentation   : {performance['thrust_augmentation']:.3f}x")
    print(
        f"Maximum flow diameter : {design['maximum_flowpath_diameter_m']:.4f} m "
        f"at {design['limiting_station']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Evaluate the initial design without running SLSQP.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("turbofan_max_diameter_optimized.json"),
        help="JSON results path.",
    )
    args = parser.parse_args()

    os.environ.setdefault("OPENMDAO_REPORTS", "0")

    if args.model_only:
        prob = build_problem()
        prob.run_model()
        results = collect_results(prob)
    else:
        starts = OPTIMIZATION_STARTS
        candidates: list[tuple[float, om.Problem, dict[str, Any]]] = []
        for start_pr, start_t4 in starts:
            print(f"\nStarting SLSQP from PR={start_pr:.1f}, T4={start_t4:.0f} K")
            candidate_prob = build_problem()
            candidate_prob.set_val("design.compressor_PR", start_pr)
            candidate_prob.set_val("design.T4", start_t4, units="degK")
            driver_result = candidate_prob.run_driver()
            if not driver_result.success:
                print("  This start did not converge; skipping it.")
                continue
            candidate_results = collect_results(candidate_prob)
            objective = candidate_results["performance"]["net_thrust_N"]
            print(f"  Converged net thrust: {objective / 1e3:.3f} kN")
            candidates.append((objective, candidate_prob, candidate_results))

        if not candidates:
            print("  No SLSQP start converged; using the initial feasible model evaluation instead.")
            fallback_prob = build_problem()
            fallback_prob.set_val("design.compressor_PR", OPTIMIZATION_STARTS[0][0])
            fallback_prob.set_val("design.T4", OPTIMIZATION_STARTS[0][1], units="degK")
            fallback_prob.run_model()
            results = collect_results(fallback_prob)
            results["optimization"] = {
                "method": "fallback model evaluation",
                "objective": "maximize net thrust under 0.7 m diameter and 0.4 bypass ratio",
                "starts": [
                    {"compressor_pressure_ratio": start_pr, "T4_K": start_t4}
                    for start_pr, start_t4 in starts
                ],
                "converged_candidates": [],
            }
        else:
            _, prob, results = max(candidates, key=lambda candidate: candidate[0])
            results["optimization"] = {
                "method": "multistart SLSQP with central finite-difference totals",
                "objective": "maximize net thrust under 0.7 m diameter and 0.4 bypass ratio",
                "starts": [
                    {"compressor_pressure_ratio": start_pr, "T4_K": start_t4}
                    for start_pr, start_t4 in starts
                ],
                "converged_candidates": [
                    {
                        "net_thrust_N": objective,
                        "compressor_pressure_ratio": candidate_results["design"]["compressor_pressure_ratio"],
                        "T4_K": candidate_results["design"]["turbine_inlet_temperature_K"],
                    }
                    for objective, _, candidate_results in candidates
                ],
            }

    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print_results(results)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
