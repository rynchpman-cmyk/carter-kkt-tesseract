"""Coupled CCZ4 + Valencia GRHD + Theory 2.0 production-reference system.

The matter closure evolves total densitized Eulerian energy and momentum. Fluid
primitives are recovered after subtracting the instantaneous Proca and mismatch
stress. This makes target/repair exchange conservative without prescribing an
independent, frame-dependent exchange four-force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from .curved_proca import Theory2ProcaSystem
from .grhd import GRHDParameters, GRHDState, RecoveryReport, ValenciaGRHD
from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays
from .matter import FluidPrimitive, perfect_fluid_stress_energy
from .theory2 import (
    StressEnergy3p1,
    TargetEvolution,
    Theory2Parameters,
    combine_stress_energy,
    interaction_stress_energy,
    target_data,
)
from .io import load_checkpoint, save_checkpoint
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductionParameters:
    fixed_point_tolerance: float = 1.0e-10
    fixed_point_iterations: int = 30
    fixed_point_relaxation: float = 0.7
    atmosphere_gravitates: bool = False
    proca_cleaning_damping: float = 1.0
    coupled_rusanov_flux: bool = True

    def __post_init__(self) -> None:
        if self.fixed_point_tolerance <= 0.0 or self.fixed_point_iterations < 3:
            raise ValueError("fixed-point recovery controls are invalid")
        if not 0.0 < self.fixed_point_relaxation <= 1.0:
            raise ValueError("fixed_point_relaxation must lie in (0, 1]")
        if self.proca_cleaning_damping < 0.0:
            raise ValueError("proca_cleaning_damping must be nonnegative")


@dataclass
class TotalMatterState:
    D: Array
    momentum: Array
    energy: Array
    tracer: Array


@dataclass
class ProductionState:
    geometry: CCZ4State
    matter: TotalMatterState
    a: Array
    pi: Array
    astar: Array
    longitudinal: Array
    gauss_cleaning: Array
    time: float = 0.0


@dataclass(frozen=True)
class CoupledRecovery:
    fluid: FluidPrimitive
    fluid_conserved: GRHDState
    repair_stress: StressEnergy3p1
    total_stress: StressEnergy3p1
    report: RecoveryReport
    iterations: int
    residual: float


class ProductionSolver:
    def __init__(
        self,
        grid: PeriodicGrid,
        theory: Theory2Parameters | None = None,
        ccz4: CCZ4Parameters | None = None,
        grhd: GRHDParameters | None = None,
        production: ProductionParameters | None = None,
    ) -> None:
        self.grid = grid
        self.theory_parameters = theory or Theory2Parameters()
        self.ccz4 = CCZ4Solver(grid, ccz4)
        self.grhd = ValenciaGRHD(
            grid,
            grhd
            or GRHDParameters(gamma_ad=self.theory_parameters.gamma_ad),
        )
        self.production_parameters = production or ProductionParameters()
        self.proca = Theory2ProcaSystem(grid, self.theory_parameters)
        self.target = TargetEvolution(grid, self.theory_parameters)

    def _repair_stress(
        self,
        h: Array,
        fluid: FluidPrimitive,
        a: Array,
        pi: Array,
        astar: Array,
        longitudinal: Array,
    ) -> StressEnergy3p1:
        phi = self.proca.phi_from_longitudinal(
            h, longitudinal, fluid, astar
        )
        proca = self.proca.stress_energy_theory2(
            h, a, pi, fluid, astar, phi
        )
        interaction = interaction_stress_energy(
            h, fluid, a, phi, astar, self.theory_parameters
        )
        return combine_stress_energy(h, proca, interaction)

    def initialize(
        self,
        geometry: CCZ4State,
        fluid: FluidPrimitive,
        a: Array,
        pi: Array,
        astar: Array,
    ) -> ProductionState:
        h, _ = self.ccz4.physical_geometry(geometry)
        _, _, sqrt_h = inverse_metric(h)
        fluid_conserved = self.grhd.primitives_to_conserved(fluid, h)
        # Initialize the independent longitudinal variable from constraint-
        # satisfying data once; subsequent evolution never differentiates this
        # algebraic Gauss reconstruction through the fluid density.
        initial_phi = self.proca.constrained_phi(h, pi, fluid, astar)
        longitudinal = self.proca.longitudinal_density(
            h, fluid, astar, initial_phi
        )
        repair = self._repair_stress(
            h, fluid, a, pi, astar, longitudinal
        )
        total = TotalMatterState(
            D=fluid_conserved.D,
            momentum=fluid_conserved.momentum
            + sqrt_h[None, ...] * repair.momentum,
            energy=fluid_conserved.D
            + fluid_conserved.tau
            + sqrt_h * repair.rho,
            tracer=fluid_conserved.tracer,
        )
        return ProductionState(
            geometry, total, a.copy(), pi.copy(), astar.copy(),
            longitudinal, self.grid.zeros(), geometry.time,
        )

    def initialize_isolated_two_lobe(
        self,
        benchmark=None,
        ctt_parameters=None,
        picard_iterations: int = 3,
    ):
        """Solve isolated two-lobe constraints and build a production state.

        The returned pair is ``(state, initial_data)`` so the elliptic report is
        retained by the caller.
        """
        if getattr(self.grid, "is_periodic", True) or self.grid.ndim != 3:
            raise ValueError("isolated two-lobe data require a finite 3D grid")
        from .initial_data import constraint_solved_rotating_two_lobe_data

        initial = constraint_solved_rotating_two_lobe_data(
            self.grid,
            self.theory_parameters,
            benchmark,
            ctt_parameters,
            picard_iterations,
        )
        if not initial.gravitational_constraints_solved:
            raise RuntimeError(
                f"isolated CTT solve did not converge: {initial.constraint_report}"
            )
        geometry = self.ccz4.from_adm(initial.geometry)
        state = self.initialize(
            geometry,
            initial.fluid,
            initial.repair.a,
            initial.repair.pi,
            initial.repair.astar,
        )
        return state, initial

    def vacuum_state(self) -> ProductionState:
        geometry = self.ccz4.flat_state()
        fluid = FluidPrimitive(
            np.full(self.grid.shape, self.grhd.parameters.density_floor),
            np.full(self.grid.shape, self.grhd.parameters.internal_energy_floor),
            self.grid.zeros((3,)),
            self.grid.zeros(),
        )
        return self.initialize(
            geometry,
            fluid,
            self.grid.zeros((3,)),
            self.grid.zeros((3,)),
            self.grid.zeros((3,)),
        )

    def recover(self, state: ProductionState) -> CoupledRecovery:
        h, _ = self.ccz4.physical_geometry(state.geometry)
        _, _, sqrt_h = inverse_metric(h)
        # Start by treating all matter energy/momentum as fluid.
        initial_fluid_state = GRHDState(
            state.matter.D,
            state.matter.momentum,
            state.matter.energy - state.matter.D,
            state.matter.tracer,
            state.time,
        )
        fluid, report = self.grhd.conserved_to_primitives(initial_fluid_state, h)
        relaxation = self.production_parameters.fixed_point_relaxation
        residual = float("inf")
        repair = self._repair_stress(
            h, fluid, state.a, state.pi, state.astar, state.longitudinal
        )
        fluid_state = initial_fluid_state
        for iteration in range(1, self.production_parameters.fixed_point_iterations + 1):
            fluid_state = GRHDState(
                D=state.matter.D,
                momentum=state.matter.momentum
                - sqrt_h[None, ...] * repair.momentum,
                tau=state.matter.energy - sqrt_h * repair.rho - state.matter.D,
                tracer=state.matter.tracer,
                time=state.time,
            )
            candidate, report = self.grhd.conserved_to_primitives(fluid_state, h)
            density_scale = np.maximum(candidate.baryon_density, self.grhd.parameters.density_floor)
            internal_scale = np.maximum(candidate.specific_internal_energy, 1.0)
            residual = float(
                max(
                    np.max(np.abs(candidate.baryon_density - fluid.baryon_density) / density_scale),
                    np.max(
                        np.abs(candidate.specific_internal_energy - fluid.specific_internal_energy)
                        / internal_scale
                    ),
                    np.max(np.abs(candidate.velocity - fluid.velocity)),
                )
            )
            fluid = FluidPrimitive(
                relaxation * candidate.baryon_density
                + (1.0 - relaxation) * fluid.baryon_density,
                relaxation * candidate.specific_internal_energy
                + (1.0 - relaxation) * fluid.specific_internal_energy,
                relaxation * candidate.velocity + (1.0 - relaxation) * fluid.velocity,
                candidate.sigma,
            )
            repair = self._repair_stress(
                h, fluid, state.a, state.pi, state.astar, state.longitudinal
            )
            if residual < self.production_parameters.fixed_point_tolerance:
                break
        else:
            raise FloatingPointError(
                f"coupled primitive recovery failed to converge; residual={residual:.3e}"
            )

        # Finish on the exact candidate rather than its relaxed iterate.
        fluid_state = GRHDState(
            state.matter.D,
            state.matter.momentum - sqrt_h[None, ...] * repair.momentum,
            state.matter.energy - sqrt_h * repair.rho - state.matter.D,
            state.matter.tracer,
            state.time,
        )
        fluid, report = self.grhd.conserved_to_primitives(fluid_state, h)
        repair = self._repair_stress(
            h, fluid, state.a, state.pi, state.astar, state.longitudinal
        )
        fluid_part = perfect_fluid_stress_energy(
            self.grid, h, fluid, self.grhd.eos
        )
        if not self.production_parameters.atmosphere_gravitates:
            atmosphere = state.matter.D <= (
                sqrt_h * self.grhd.parameters.density_floor * (1.0 + 1.0e-10)
            )
            fluid_part = (
                np.where(atmosphere, 0.0, fluid_part[0]),
                np.where(atmosphere[None, ...], 0.0, fluid_part[1]),
                np.where(atmosphere[None, None, ...], 0.0, fluid_part[2]),
            )
        total = combine_stress_energy(
            h,
            fluid_part,
            (repair.rho, repair.momentum, repair.stress),
        )
        return CoupledRecovery(fluid, fluid_state, repair, total, report, iteration, residual)

    def _roll(self, field: Array, offset: int, axis: int) -> Array:
        if hasattr(self.grid, "shift"):
            return self.grid.shift(field, offset, axis)
        array_axis = field.ndim - self.grid.ndim + axis
        return np.roll(field, offset, axis=array_axis)

    def _coupled_speed_bound(
        self,
        recovery: CoupledRecovery,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
        a: Array,
        astar: Array,
        longitudinal: Array,
    ) -> Array:
        """Conservative local bound for the coupled Rusanov flux.

        The square-root mismatch factor intentionally overestimates the
        measured characteristic deformation.  It is a shock-capturing bound,
        not a replacement for the runtime hyperbolicity admissibility test.
        """
        fluid = recovery.fluid
        minus, plus = self.grhd._wave_speeds(fluid, h, lapse, shift, axis)
        h_inv, _, _ = inverse_metric(h)
        phi = self.proca.phi_from_longitudinal(
            h, longitudinal, fluid, astar
        )
        mismatch = a-astar
        mismatch_squared = np.einsum(
            "ij...,i...,j...->...", h_inv, mismatch, mismatch
        )
        phi_star = np.einsum("i...,i...->...", fluid.velocity, astar)
        mismatch_norm = np.maximum(
            mismatch_squared+(phi-phi_star)**2, 0.0
        )
        enthalpy = self.grhd.eos.specific_enthalpy(
            fluid.specific_internal_energy
        )
        deformation = np.sqrt(
            1.0+self.theory_parameters.coupling*mismatch_norm
            / np.maximum(enthalpy, 1.0e-14)
        )
        metric_light = np.sqrt(np.maximum(h_inv[axis, axis], 0.0))
        repair_bound = (
            np.abs(shift[axis])+lapse*metric_light*deformation
        )
        return np.maximum(
            np.maximum(np.abs(minus), np.abs(plus)), repair_bound
        )

    def _rusanov_interface(
        self, flux: Array, conserved: Array, speed: Array, axis: int
    ) -> Array:
        neighbor_flux = self._roll(flux, -1, axis)
        neighbor_state = self._roll(conserved, -1, axis)
        face_speed = np.maximum(speed, self._roll(speed, -1, axis))
        component_axes = conserved.ndim-self.grid.ndim
        face_speed = face_speed.reshape(
            (1,)*component_axes+self.grid.shape
        )
        return 0.5*(flux+neighbor_flux)-0.5*face_speed*(
            neighbor_state-conserved
        )

    def _total_matter_rhs(
        self,
        state: ProductionState,
        recovery: CoupledRecovery,
        h: Array,
        K: Array,
    ) -> tuple[Array, Array, Array, Array]:
        alpha = state.geometry.lapse
        beta = state.geometry.shift
        fluid = recovery.fluid
        fluid_state = recovery.fluid_conserved
        repair = recovery.repair_stress
        total = recovery.total_stress
        h_inv, _, sqrt_h = inverse_metric(h)
        dD = np.zeros_like(state.matter.D)
        dmomentum = np.zeros_like(state.matter.momentum)
        denergy = np.zeros_like(state.matter.energy)
        dtracer = np.zeros_like(state.matter.tracer)
        repair_momentum_up = np.einsum(
            "ij...,j...->i...", h_inv, repair.momentum
        )
        repair_stress_mixed = np.einsum(
            "ik...,kj...->ij...", h_inv, repair.stress
        )
        for axis in range(self.grid.ndim):
            if self.production_parameters.coupled_rusanov_flux:
                fluid_flux = self.grhd._flux(
                    fluid_state, fluid, h, alpha, beta, axis
                )
                flux_D, flux_momentum, flux_tau, flux_tracer = fluid_flux
                repair_energy_flux = sqrt_h * (
                    alpha * repair_momentum_up[axis]
                    - beta[axis] * repair.rho
                )
                repair_momentum_flux = sqrt_h[None, ...] * (
                    alpha[None, ...] * repair_stress_mixed[axis]
                    - beta[axis][None, ...] * repair.momentum
                )
                cell_fluxes = (
                    flux_D,
                    flux_momentum+repair_momentum_flux,
                    flux_D+flux_tau+repair_energy_flux,
                    flux_tracer,
                )
                conserved_parts = (
                    state.matter.D, state.matter.momentum,
                    state.matter.energy, state.matter.tracer,
                )
                speed = self._coupled_speed_bound(
                    recovery, h, alpha, beta, axis,
                    state.a, state.astar, state.longitudinal,
                )
                for derivative, cell_flux, conserved in zip(
                    (dD, dmomentum, denergy, dtracer),
                    cell_fluxes, conserved_parts,
                ):
                    interface_flux = self._rusanov_interface(
                        cell_flux, conserved, speed, axis
                    )
                    left_interface = self._roll(interface_flux, 1, axis)
                    derivative -= (
                        interface_flux-left_interface
                    ) / self.grid.spacing[axis]
                continue
            fluid_right, fluid_left = self.grhd.flux_pair(
                fluid, h, alpha, beta, axis
            )
            flux_D, flux_momentum, flux_tau, flux_tracer = fluid_right
            left_D, left_momentum, left_tau, left_tracer = fluid_left
            repair_energy_flux = sqrt_h * (
                alpha * repair_momentum_up[axis] - beta[axis] * repair.rho
            )
            repair_momentum_flux = sqrt_h[None, ...] * (
                alpha[None, ...] * repair_stress_mixed[axis]
                - beta[axis][None, ...] * repair.momentum
            )
            total_energy_flux = flux_D + flux_tau + repair_energy_flux
            total_momentum_flux = flux_momentum + repair_momentum_flux
            left_repair_energy = self._roll(repair_energy_flux, 1, axis)
            left_repair_momentum = self._roll(repair_momentum_flux, 1, axis)
            left_total_energy = left_D + left_tau + left_repair_energy
            left_total_momentum = left_momentum + left_repair_momentum
            dx = self.grid.spacing[axis]
            dD -= (flux_D - left_D) / dx
            dtracer -= (flux_tracer - left_tracer) / dx
            denergy -= (total_energy_flux - left_total_energy) / dx
            dmomentum -= (total_momentum_flux - left_total_momentum) / dx

        total_momentum_up = np.einsum(
            "ij...,j...->i...", h_inv, total.momentum
        )
        total_stress_up = np.einsum(
            "ik...,jl...,kl...->ij...", h_inv, h_inv, total.stress
        )
        gradient_alpha = self.grid.gradient(alpha)
        denergy += sqrt_h * (
            alpha * np.einsum("ij...,ij...->...", K, total_stress_up)
            - np.einsum("i...,i...->...", total_momentum_up, gradient_alpha)
        )
        for j in range(3):
            source = -total.rho * gradient_alpha[j]
            for i in range(3):
                if j < self.grid.ndim:
                    source += total.momentum[i] * self.grid.derivative(beta[i], j)
                for k in range(3):
                    if j < self.grid.ndim:
                        source += (
                            0.5
                            * alpha
                            * total_stress_up[i, k]
                            * self.grid.derivative(h[i, k], j)
                        )
            dmomentum[j] += sqrt_h * source
        return dD, dmomentum, denergy, dtracer

    def _pack(self, state: ProductionState) -> tuple[Array, ...]:
        g = state.geometry
        m = state.matter
        return (
            g.conformal_metric,
            g.conformal_A,
            g.conformal_factor,
            g.trace_K,
            g.theta,
            g.gamma_hat,
            g.lapse,
            g.shift,
            g.shift_driver,
            m.D,
            m.momentum,
            m.energy,
            m.tracer,
            state.a,
            state.pi,
            state.astar,
            state.longitudinal,
            state.gauss_cleaning,
        )

    def _unpack(self, values: tuple[Array, ...], time: float) -> ProductionState:
        geometry = CCZ4State(*values[:9], time)
        matter = TotalMatterState(*values[9:13])
        return ProductionState(
            geometry, matter, values[13], values[14], values[15],
            values[16], values[17], time,
        )

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        state = self._unpack(values, time)
        recovery = self.recover(state)
        h, K = self.ccz4.physical_geometry(state.geometry)
        matter_tuple = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        geometry_rhs = self.ccz4.rhs(time, values[:9], matter_tuple)
        matter_rhs = self._total_matter_rhs(state, recovery, h, K)
        da, dpi, dlongitudinal, dcleaning = self.proca.rhs_hyperbolic(
            h,
            state.a,
            state.pi,
            state.longitudinal,
            state.gauss_cleaning,
            recovery.fluid,
            state.astar,
            state.geometry.lapse,
            state.geometry.shift,
            self.production_parameters.proca_cleaning_damping,
        )
        (dastar,) = self.target.rhs(
            time, (state.astar,), h, recovery.fluid
        )
        if not getattr(self.grid, "is_periodic", True):
            from .boundary import RadiativeBoundary

            boundary = RadiativeBoundary(self.grid)
            da = boundary.radiative_rhs(state.a, da, 0.0, 1.0)
            dpi = boundary.radiative_rhs(state.pi, dpi, 0.0, 1.0)
            dastar = boundary.outflow_rhs(dastar)
            dlongitudinal = boundary.outflow_rhs(dlongitudinal)
            dcleaning = boundary.radiative_rhs(
                state.gauss_cleaning, dcleaning, 0.0, 1.0
            )
        return geometry_rhs + matter_rhs + (
            da, dpi, dastar, dlongitudinal, dcleaning
        )

    def step(self, state: ProductionState, dt: float) -> ProductionState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        evolved = rk4_arrays(self._pack(state), state.time, dt, self.rhs)
        result = self._unpack(evolved, state.time + dt)
        result.geometry = self.ccz4.project_algebraic(result.geometry)
        result.geometry.time = result.time
        # Recovery is a mandatory post-step admissibility check.
        self.recover(result)
        return result

    def recommended_dt(self, state: ProductionState, cfl: float = 0.1) -> float:
        recovery = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        bounds = [
            self._coupled_speed_bound(
                recovery, h, state.geometry.lapse, state.geometry.shift,
                axis, state.a, state.astar, state.longitudinal,
            )
            for axis in range(self.grid.ndim)
        ]
        max_speed = max(
            np.sqrt(2.0), *(float(np.max(bound)) for bound in bounds)
        )
        return cfl * min(self.grid.spacing) / (max_speed * np.sqrt(self.grid.ndim))

    def diagnostics(self, state: ProductionState) -> dict[str, float | int]:
        recovery = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        matter_tuple = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        ccz = self.ccz4.diagnostics(state.geometry, matter_tuple)
        phi = self.proca.phi_from_longitudinal(
            h, state.longitudinal, recovery.fluid, state.astar
        )
        gauss = self.proca.hyperbolic_gauss_constraint(
            h, state.pi, state.longitudinal
        )
        target = target_data(
            self.grid, h, recovery.fluid, state.astar, self.theory_parameters
        )
        h_inv = inverse_metric(h)[0]
        a_squared = np.einsum("ij...,i...,j...->...", h_inv, state.a, state.a)
        u_dot_A = lorentz = 1.0 / np.sqrt(
            1.0
            - np.einsum(
                "ij...,i...,j...->...",
                h,
                recovery.fluid.velocity,
                recovery.fluid.velocity,
            )
        )
        u_dot_A = lorentz * (
            -phi
            + np.einsum("i...,i...->...", recovery.fluid.velocity, state.a)
        )
        chi_sq = np.maximum(a_squared - phi**2 + u_dot_A**2, 0.0)
        causal_margin = recovery.fluid.baryon_density**2 / np.cosh(
            np.sqrt(chi_sq)
        ) ** 2
        coupled_speed_bound = max(
            float(np.max(self._coupled_speed_bound(
                recovery, h, state.geometry.lapse, state.geometry.shift,
                axis, state.a, state.astar, state.longitudinal,
            )))
            for axis in range(self.grid.ndim)
        )
        return {
            "time": state.time,
            "hamiltonian_l2": ccz.hamiltonian_l2,
            "momentum_l2": ccz.momentum_l2,
            "theta_l2": ccz.theta_l2,
            "z_l2": ccz.z_l2,
            "gauss_l2": float(np.sqrt(np.mean(gauss**2))),
            "gauss_cleaning_l2": float(
                np.sqrt(np.mean(state.gauss_cleaning**2))
            ),
            "maximum_coupled_speed_bound": coupled_speed_bound,
            "target_constraint_linf": float(np.max(np.abs(target.target_constraint))),
            "minimum_causal_margin": float(np.min(causal_margin)),
            "baryon_mass": float(np.sum(state.matter.D) * self.grid.cell_volume),
            "total_matter_energy": float(
                np.sum(state.matter.energy) * self.grid.cell_volume
            ),
            "recovery_iterations": recovery.iterations,
            "recovery_residual": recovery.residual,
            "recovery_failures": recovery.report.failed_cells,
        }


def save_production_state(
    path: str | Path,
    state: ProductionState,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    g = state.geometry
    m = state.matter
    return save_checkpoint(
        path,
        metadata={"format": "tesseract.production.v2", **(metadata or {})},
        conformal_metric=g.conformal_metric,
        conformal_A=g.conformal_A,
        conformal_factor=g.conformal_factor,
        trace_K=g.trace_K,
        theta=g.theta,
        gamma_hat=g.gamma_hat,
        lapse=g.lapse,
        shift=g.shift,
        shift_driver=g.shift_driver,
        D=m.D,
        total_momentum=m.momentum,
        total_energy=m.energy,
        tracer=m.tracer,
        a=state.a,
        pi=state.pi,
        astar=state.astar,
        longitudinal=state.longitudinal,
        gauss_cleaning=state.gauss_cleaning,
        time=state.time,
    )


def load_production_state(path: str | Path) -> tuple[ProductionState, dict[str, Any]]:
    arrays, metadata = load_checkpoint(path)
    if metadata.get("format") != "tesseract.production.v2":
        raise ValueError(
            "checkpoint is not production-state v2; legacy v1 lacks the "
            "independent longitudinal Proca variable"
        )
    time = float(arrays["time"])
    geometry = CCZ4State(
        arrays["conformal_metric"],
        arrays["conformal_A"],
        arrays["conformal_factor"],
        arrays["trace_K"],
        arrays["theta"],
        arrays["gamma_hat"],
        arrays["lapse"],
        arrays["shift"],
        arrays["shift_driver"],
        time,
    )
    matter = TotalMatterState(
        arrays["D"], arrays["total_momentum"], arrays["total_energy"], arrays["tracer"]
    )
    return (
        ProductionState(
            geometry, matter, arrays["a"], arrays["pi"], arrays["astar"],
            arrays["longitudinal"], arrays["gauss_cleaning"], time,
        ),
        metadata,
    )
