"""CCZ4 + entropy-based GRHD + Theory 3.2 production reference solver.

The evolved energy and momentum are those of matter plus both vector fields.
Vector/matter exchange is therefore internal to one conservative balance law.
The material reservoir is recovered from total energy after fixing baryon
density, thermal entropy, and the complete vector stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adm import inverse_metric
from .ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from .grhd import GRHDParameters, ValenciaGRHD
from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays
from .io import load_checkpoint, save_checkpoint
from .matter import FluidPrimitive, lorentz_factor
from .theory2 import StressEnergy3p1
from .theory3 import Theory3Parameters, Theory3System


@dataclass(frozen=True)
class Theory3ProductionParameters:
    recovery_tolerance: float = 1.0e-11
    recovery_iterations: int = 100
    cleaning_damping: float = 1.0
    minimum_total_enthalpy: float = 1.0e-13
    atmosphere_gravitates: bool = False
    flux_reconstruction: str = "piecewise_constant"
    reconstruction_theta: float = 1.5
    source_splitting: str = "lie"

    def __post_init__(self) -> None:
        if self.recovery_tolerance <= 0.0 or self.recovery_iterations < 20:
            raise ValueError("material recovery controls are invalid")
        if self.cleaning_damping < 0.0:
            raise ValueError("cleaning_damping must be nonnegative")
        if self.minimum_total_enthalpy <= 0.0:
            raise ValueError("minimum_total_enthalpy must be positive")
        if self.flux_reconstruction not in {
            "piecewise_constant",
            "muscl_mc",
            "weno5_z",
        }:
            raise ValueError(
                "flux_reconstruction must be piecewise_constant, muscl_mc, "
                "or weno5_z"
            )
        if not 1.0 <= self.reconstruction_theta <= 2.0:
            raise ValueError("reconstruction_theta must lie in [1, 2]")
        if self.source_splitting not in {"lie", "strang"}:
            raise ValueError("source_splitting must be lie or strang")


@dataclass
class Theory3MatterState:
    D: Array
    momentum: Array
    energy: Array
    entropy: Array
    tracer: Array


@dataclass
class Theory3ProductionState:
    geometry: CCZ4State
    matter: Theory3MatterState
    a: Array
    pi_A: Array
    b: Array
    pi_B: Array
    longitudinal_A: Array
    longitudinal_B: Array
    cleaning_A: Array
    cleaning_B: Array
    target_charge: Array
    target_current: Array
    time: float = 0.0


@dataclass(frozen=True)
class MaterialRecoveryReport:
    failed_cells: int
    atmosphere_cells: int
    maximum_pressure_residual: float
    minimum_total_enthalpy: float
    minimum_speed_gap: float


@dataclass(frozen=True)
class Theory3Recovery:
    fluid: FluidPrimitive
    reservoir_energy: Array
    entropy_per_baryon: Array
    total_enthalpy: Array
    sound_speed: Array
    material_stress: StressEnergy3p1
    vector_stress: StressEnergy3p1
    total_stress: StressEnergy3p1
    target_charge_eulerian: Array
    target_current_up: Array
    report: MaterialRecoveryReport


@dataclass(frozen=True)
class FiniteDampingReport:
    vector_energy_change: float = 0.0
    vector_momentum_change: tuple[float, float, float] = (0.0, 0.0, 0.0)
    irreversible_heat: float = 0.0
    total_energy_balance_error: float = 0.0
    maximum_gauss_change: float = 0.0


def _add_stress(
    first: StressEnergy3p1, second: StressEnergy3p1
) -> StressEnergy3p1:
    return StressEnergy3p1(
        first.rho + second.rho,
        first.momentum + second.momentum,
        first.stress + second.stress,
        first.trace + second.trace,
    )


class Theory3ProductionSolver:
    def __init__(
        self,
        grid: PeriodicGrid,
        theory: Theory3Parameters | None = None,
        ccz4: CCZ4Parameters | None = None,
        grhd: GRHDParameters | None = None,
        production: Theory3ProductionParameters | None = None,
    ) -> None:
        self.grid = grid
        self.theory_parameters = theory or Theory3Parameters()
        self.production_parameters = production or Theory3ProductionParameters()
        self.ccz4 = CCZ4Solver(grid, ccz4)
        self.grhd = ValenciaGRHD(
            grid,
            grhd or GRHDParameters(gamma_ad=self.theory_parameters.gamma_ad),
        )
        if abs(self.grhd.parameters.gamma_ad - self.theory_parameters.gamma_ad) > 1.0e-14:
            raise ValueError("Theory 3 and GRHD gamma_ad must agree")
        self.system = Theory3System(grid, self.theory_parameters)
        self.last_damping_report = FiniteDampingReport()

    @property
    def gamma_ad(self) -> float:
        return self.theory_parameters.gamma_ad

    def _specific_entropy(self, fluid: FluidPrimitive) -> Array:
        pressure = self.grhd.eos.pressure(
            fluid.baryon_density, fluid.specific_internal_energy
        )
        polytropic_constant = pressure / np.maximum(
            fluid.baryon_density**self.gamma_ad, 1.0e-300
        )
        return np.log(np.maximum(polytropic_constant, 1.0e-300)) / (
            self.gamma_ad - 1.0
        )

    def _material_stress(
        self, h: Array, fluid: FluidPrimitive, reservoir_energy: Array
    ) -> tuple[StressEnergy3p1, Array]:
        pressure = self.grhd.eos.pressure(
            fluid.baryon_density, fluid.specific_internal_energy
        )
        thermal_energy = self.grhd.eos.energy_density(
            fluid.baryon_density, fluid.specific_internal_energy
        )
        total_enthalpy = thermal_energy + reservoir_energy + pressure
        W = lorentz_factor(h, fluid.velocity)
        velocity_lower = np.einsum("ij...,j...->i...", h, fluid.velocity)
        prefactor = total_enthalpy * W**2
        rho = prefactor - pressure
        momentum = prefactor[None, ...] * velocity_lower
        stress = (
            prefactor[None, None, ...]
            * velocity_lower[:, None, ...]
            * velocity_lower[None, :, ...]
            + pressure[None, None, ...] * h
        )
        trace = np.einsum("ij...,ij...->...", inverse_metric(h)[0], stress)
        return StressEnergy3p1(rho, momentum, stress, trace), total_enthalpy

    def initialize(
        self,
        geometry: CCZ4State,
        fluid: FluidPrimitive,
        *,
        a: Array | None = None,
        pi_A: Array | None = None,
        b: Array | None = None,
        pi_B: Array | None = None,
        target_charge_eulerian: Array | None = None,
        target_relative_current_up: Array | None = None,
        reservoir_energy: Array | float = 0.0,
    ) -> Theory3ProductionState:
        h, _ = self.ccz4.physical_geometry(geometry)
        _, _, sqrt_h = inverse_metric(h)
        a = self.grid.zeros((3,)) if a is None else np.asarray(a, dtype=float)
        b = self.grid.zeros((3,)) if b is None else np.asarray(b, dtype=float)
        pi_A = self.grid.zeros((3,)) if pi_A is None else np.asarray(pi_A, dtype=float)
        pi_B = self.grid.zeros((3,)) if pi_B is None else np.asarray(pi_B, dtype=float)
        rho_D = (
            np.zeros(self.grid.shape)
            if target_charge_eulerian is None
            else np.asarray(target_charge_eulerian, dtype=float)
        )
        relative = (
            self.grid.zeros((3,))
            if target_relative_current_up is None
            else np.asarray(target_relative_current_up, dtype=float)
        )
        reservoir = np.broadcast_to(
            np.asarray(reservoir_energy, dtype=float), self.grid.shape
        ).copy()
        fluid_state = self.grhd.primitives_to_conserved(fluid, h)
        entropy = fluid_state.D * self._specific_entropy(fluid)
        material_stress, total_enthalpy = self._material_stress(
            h, fluid, reservoir
        )
        if np.any(total_enthalpy <= self.production_parameters.minimum_total_enthalpy):
            raise ValueError("initial total material enthalpy is not positive")
        longitudinal_A, longitudinal_B = self.system.initial_longitudinal_charges(
            h, pi_A, pi_B
        )
        target_charge = sqrt_h * rho_D
        target_current = sqrt_h[None, ...] * relative
        vector_stress = self.system.vector_stress(
            h,
            a,
            pi_A,
            b,
            pi_B,
            longitudinal_A,
            longitudinal_B,
            rho_D,
        )
        total = _add_stress(material_stress, vector_stress)
        matter = Theory3MatterState(
            D=fluid_state.D,
            momentum=sqrt_h[None, ...] * total.momentum,
            energy=sqrt_h * total.rho,
            entropy=entropy,
            tracer=fluid_state.tracer,
        )
        state = Theory3ProductionState(
            geometry,
            matter,
            a.copy(),
            pi_A.copy(),
            b.copy(),
            pi_B.copy(),
            longitudinal_A,
            longitudinal_B,
            self.grid.zeros(),
            self.grid.zeros(),
            target_charge,
            target_current,
            geometry.time,
        )
        self.recover(state)
        return state

    def vacuum_state(self) -> Theory3ProductionState:
        fluid = FluidPrimitive(
            np.full(self.grid.shape, self.grhd.parameters.density_floor),
            np.full(
                self.grid.shape, self.grhd.parameters.internal_energy_floor
            ),
            self.grid.zeros((3,)),
            self.grid.zeros(),
        )
        return self.initialize(self.ccz4.flat_state(), fluid=fluid)

    def _recover_material(
        self,
        h: Array,
        D_density: Array,
        momentum_down: Array,
        energy: Array,
        entropy_per_baryon: Array,
        sigma: Array,
    ) -> tuple[FluidPrimitive, Array, Array, Array, MaterialRecoveryReport]:
        h_inv = inverse_metric(h)[0]
        momentum_sq = np.einsum(
            "ij...,i...,j...->...", h_inv, momentum_down, momentum_down
        )
        momentum_norm = np.sqrt(np.maximum(momentum_sq, 0.0))
        atmosphere = (
            (D_density < 0.5 * self.grhd.parameters.density_floor)
            | ~np.isfinite(D_density)
            | ~np.isfinite(energy)
        )
        log_K = np.clip(
            (self.gamma_ad - 1.0) * entropy_per_baryon, -700.0, 700.0
        )
        K_entropy = np.exp(log_K)
        pressure_floor = (
            (self.gamma_ad - 1.0)
            * self.grhd.parameters.density_floor
            * self.grhd.parameters.internal_energy_floor
        )
        lower = np.maximum(pressure_floor, momentum_norm - energy + 1.0e-14)

        def residual(pressure: Array) -> tuple[Array, Array, Array]:
            denominator = np.maximum(energy + pressure, 1.0e-300)
            speed_sq = np.minimum(
                momentum_sq / denominator**2, 1.0 - 1.0e-13
            )
            W = 1.0 / np.sqrt(1.0 - speed_sq)
            density = D_density / W
            eos_pressure = K_entropy * np.maximum(density, 1.0e-300) ** self.gamma_ad
            return pressure - eos_pressure, density, W

        f_lower = residual(lower)[0]
        upper = np.maximum(
            1.0,
            lower + K_entropy * np.maximum(D_density, 1.0e-300) ** self.gamma_ad,
        )
        f_upper = residual(upper)[0]
        for _ in range(80):
            grow = (f_lower * f_upper > 0.0) & ~atmosphere
            if not np.any(grow):
                break
            upper = np.where(grow, 2.0 * upper, upper)
            f_upper = residual(upper)[0]
        bracket_failed = (f_lower * f_upper > 0.0) & ~atmosphere
        for _ in range(self.production_parameters.recovery_iterations):
            middle = 0.5 * (lower + upper)
            f_middle = residual(middle)[0]
            use_left = f_lower * f_middle <= 0.0
            upper = np.where(use_left, middle, upper)
            f_upper = np.where(use_left, f_middle, f_upper)
            lower = np.where(use_left, lower, middle)
            f_lower = np.where(use_left, f_lower, f_middle)
        pressure = 0.5 * (lower + upper)
        pressure_residual, density, W = residual(pressure)
        relative_residual = np.abs(pressure_residual) / np.maximum(
            pressure + np.abs(energy), 1.0
        )
        failed = (
            bracket_failed
            | ~np.isfinite(relative_residual)
            | (relative_residual > self.production_parameters.recovery_tolerance)
        )
        if np.any(atmosphere | failed):
            reset = atmosphere | failed
            density = np.where(reset, self.grhd.parameters.density_floor, density)
            pressure = np.where(reset, pressure_floor, pressure)
            W = np.where(reset, 1.0, W)
        else:
            reset = np.zeros(self.grid.shape, dtype=bool)
        velocity_up = np.einsum(
            "ij...,j...->i...", h_inv, momentum_down
        ) / np.maximum(energy + pressure, 1.0e-300)[None, ...]
        velocity_up[:, reset] = 0.0
        internal = pressure / (
            (self.gamma_ad - 1.0)
            * np.maximum(density, self.grhd.parameters.density_floor)
        )
        thermal_energy = density * (1.0 + internal)
        total_enthalpy = (energy + pressure) / W**2
        reservoir = total_enthalpy - thermal_energy - pressure
        reservoir = np.where(reset, 0.0, reservoir)
        total_enthalpy = np.where(
            reset, thermal_energy + pressure, total_enthalpy
        )
        sound_sq = np.clip(
            self.gamma_ad * pressure
            / np.maximum(total_enthalpy, 1.0e-300),
            0.0,
            1.0,
        )
        speed_gap = self.theory_parameters.target_speed - np.sqrt(sound_sq)
        enthalpy_failed = (
            total_enthalpy <= self.production_parameters.minimum_total_enthalpy
        ) & ~reset
        gap_failed = (
            speed_gap < self.theory_parameters.sound_speed_margin
        ) & ~reset
        failed |= enthalpy_failed | gap_failed
        report = MaterialRecoveryReport(
            failed_cells=int(np.count_nonzero(failed)),
            atmosphere_cells=int(np.count_nonzero(atmosphere)),
            maximum_pressure_residual=float(
                np.max(np.where(reset, 0.0, relative_residual))
            ),
            minimum_total_enthalpy=float(np.min(total_enthalpy)),
            minimum_speed_gap=float(np.min(speed_gap)),
        )
        fluid = FluidPrimitive(density, internal, velocity_up, sigma)
        return fluid, reservoir, total_enthalpy, np.sqrt(sound_sq), report

    def recover(self, state: Theory3ProductionState) -> Theory3Recovery:
        h, _ = self.ccz4.physical_geometry(state.geometry)
        _, _, sqrt_h = inverse_metric(h)
        rho_D = state.target_charge / sqrt_h
        vector_stress = self.system.vector_stress(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            rho_D,
        )
        material_energy = state.matter.energy / sqrt_h - vector_stress.rho
        material_momentum = (
            state.matter.momentum / sqrt_h[None, ...]
            - vector_stress.momentum
        )
        D_density = state.matter.D / sqrt_h
        entropy_per_baryon = state.matter.entropy / np.maximum(
            state.matter.D, 1.0e-300
        )
        sigma = state.matter.tracer / np.maximum(state.matter.D, 1.0e-300)
        fluid, reservoir, enthalpy, sound_speed, report = self._recover_material(
            h,
            D_density,
            material_momentum,
            material_energy,
            entropy_per_baryon,
            sigma,
        )
        if report.failed_cells:
            raise FloatingPointError(
                "Theory 3 material recovery left the admissible domain: "
                f"failed={report.failed_cells}, gap={report.minimum_speed_gap:.3e}, "
                f"enthalpy={report.minimum_total_enthalpy:.3e}"
            )
        material_stress, _ = self._material_stress(h, fluid, reservoir)
        if not self.production_parameters.atmosphere_gravitates:
            atmosphere = state.matter.D <= (
                sqrt_h
                * self.grhd.parameters.density_floor
                * (1.0 + 1.0e-10)
            )
            material_stress = StressEnergy3p1(
                np.where(atmosphere, 0.0, material_stress.rho),
                np.where(
                    atmosphere[None, ...], 0.0, material_stress.momentum
                ),
                np.where(
                    atmosphere[None, None, ...], 0.0, material_stress.stress
                ),
                np.where(atmosphere, 0.0, material_stress.trace),
            )
        total_stress = _add_stress(material_stress, vector_stress)
        rho_D, current_D = self.system.target_current(
            h, state.target_charge, state.target_current, fluid
        )
        return Theory3Recovery(
            fluid,
            reservoir,
            entropy_per_baryon,
            enthalpy,
            sound_speed,
            material_stress,
            vector_stress,
            total_stress,
            rho_D,
            current_D,
            report,
        )

    def _roll(self, field: Array, offset: int, axis: int) -> Array:
        if hasattr(self.grid, "shift"):
            return self.grid.shift(field, offset, axis)
        array_axis = field.ndim - self.grid.ndim + axis
        return np.roll(field, offset, axis=array_axis)

    def _rusanov_interface(
        self, flux: Array, conserved: Array, speed: Array, axis: int
    ) -> Array:
        neighbor_flux = self._roll(flux, -1, axis)
        neighbor_state = self._roll(conserved, -1, axis)
        face_speed = np.maximum(speed, self._roll(speed, -1, axis))
        component_axes = conserved.ndim - self.grid.ndim
        face_speed = face_speed.reshape((1,) * component_axes + self.grid.shape)
        if (
            self.production_parameters.flux_reconstruction
            == "piecewise_constant"
        ):
            return 0.5 * (flux + neighbor_flux) - 0.5 * face_speed * (
                neighbor_state - conserved
            )

        if self.production_parameters.flux_reconstruction == "muscl_mc":
            state_slope = self._mc_slope(conserved, axis)
            flux_slope = self._mc_slope(flux, axis)
            left_state = conserved + 0.5 * state_slope
            right_state = neighbor_state - 0.5 * self._roll(
                state_slope, -1, axis
            )
            left_flux = flux + 0.5 * flux_slope
            right_flux = neighbor_flux - 0.5 * self._roll(
                flux_slope, -1, axis
            )
        else:
            left_state, right_state = self._weno5_z_faces(
                conserved, axis
            )
            left_flux, right_flux = self._weno5_z_faces(flux, axis)
        return 0.5 * (left_flux + right_flux) - 0.5 * face_speed * (
            right_state - left_state
        )

    @staticmethod
    def _minmod(*values: Array) -> Array:
        """Return the componentwise minmod of equally shaped arrays."""
        stacked = np.stack(values, axis=0)
        same_sign = np.all(stacked > 0.0, axis=0) | np.all(
            stacked < 0.0, axis=0
        )
        magnitude = np.min(np.abs(stacked), axis=0)
        return np.where(same_sign, np.sign(stacked[0]) * magnitude, 0.0)

    def _mc_slope(self, field: Array, axis: int) -> Array:
        """Monotonized-central MUSCL slope in cell-average units.

        Reconstructing both the conserved state and its physical flux gives a
        second-order local Lax--Friedrichs interface on smooth solutions while
        retaining the old piecewise-constant path as an explicit control.
        """
        previous = self._roll(field, 1, axis)
        following = self._roll(field, -1, axis)
        backward = field - previous
        forward = following - field
        centered = 0.5 * (following - previous)
        theta = self.production_parameters.reconstruction_theta
        return self._minmod(theta * backward, centered, theta * forward)

    @staticmethod
    def _weno5_z_combine(
        v0: Array,
        v1: Array,
        v2: Array,
        v3: Array,
        v4: Array,
    ) -> Array:
        """Fifth-order WENO-Z reconstruction at the right stencil face."""
        q0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
        q1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
        q2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0
        beta0 = (
            (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2
            + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
        )
        beta1 = (
            (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2
            + 0.25 * (v1 - v3) ** 2
        )
        beta2 = (
            (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2
            + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2
        )
        tau5 = np.abs(beta0 - beta2)
        scale = np.maximum.reduce(
            (np.abs(v0), np.abs(v1), np.abs(v2), np.abs(v3), np.abs(v4))
        )
        epsilon = 1.0e-26 + 1.0e-12 * scale**2
        alpha0 = 0.1 * (1.0 + (tau5 / (beta0 + epsilon)) ** 2)
        alpha1 = 0.6 * (1.0 + (tau5 / (beta1 + epsilon)) ** 2)
        alpha2 = 0.3 * (1.0 + (tau5 / (beta2 + epsilon)) ** 2)
        normalization = alpha0 + alpha1 + alpha2
        return (
            alpha0 * q0 + alpha1 * q1 + alpha2 * q2
        ) / normalization

    def _weno5_z_faces(
        self, field: Array, axis: int
    ) -> tuple[Array, Array]:
        """Return left/right values at every cell's right interface."""
        im2 = self._roll(field, 2, axis)
        im1 = self._roll(field, 1, axis)
        ip1 = self._roll(field, -1, axis)
        ip2 = self._roll(field, -2, axis)
        ip3 = self._roll(field, -3, axis)
        left = self._weno5_z_combine(im2, im1, field, ip1, ip2)
        right = self._weno5_z_combine(ip3, ip2, ip1, field, im1)
        return left, right

    def _face_value(self, field: Array, axis: int, side: int) -> Array:
        array_axis = field.ndim - self.grid.ndim + axis
        slices: list[slice | int] = [slice(None)] * field.ndim
        slices[array_axis] = 0 if side < 0 else -1
        return field[tuple(slices)]

    def _set_face(
        self, field: Array, axis: int, side: int, value: Array
    ) -> None:
        array_axis = field.ndim - self.grid.ndim + axis
        slices: list[slice | int] = [slice(None)] * field.ndim
        slices[array_axis] = 0 if side < 0 else -1
        field[tuple(slices)] = value

    def _interface_divergence(
        self,
        interface: Array,
        axis: int,
        lower_flux: Array | None = None,
        upper_flux: Array | None = None,
    ) -> Array:
        right = np.array(interface, copy=True)
        if not getattr(self.grid, "is_periodic", True):
            if upper_flux is None or lower_flux is None:
                raise ValueError("nonperiodic divergence needs both boundary fluxes")
            self._set_face(right, axis, 1, upper_flux)
        left = self._roll(right, 1, axis)
        if not getattr(self.grid, "is_periodic", True):
            self._set_face(left, axis, -1, lower_flux)
        return -(right - left) / self.grid.spacing[axis]

    def _outgoing_boundary_flux(
        self,
        flux: Array,
        outward_indicator: Array,
        axis: int,
        side: int,
    ) -> Array:
        value = self._face_value(flux, axis, side)
        indicator = self._face_value(outward_indicator, axis, side)
        leading = value.ndim - indicator.ndim
        allow = side * indicator > 0.0
        return np.where(allow.reshape((1,) * leading + allow.shape), value, 0.0)

    def _target_characteristic_speeds(
        self,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> tuple[Array, Array]:
        h_inv = inverse_metric(h)[0]
        c2 = self.theory_parameters.target_speed**2
        speed_sq = np.einsum(
            "ij...,i...,j...->...", h, fluid.velocity, fluid.velocity
        )
        v_axis = fluid.velocity[axis]
        denominator = np.maximum(1.0 - speed_sq * c2, 1.0e-14)
        radical = (1.0 - speed_sq) * (
            h_inv[axis, axis] * (1.0 - speed_sq * c2)
            - v_axis**2 * (1.0 - c2)
        )
        common = v_axis * (1.0 - c2)
        root = self.theory_parameters.target_speed * np.sqrt(
            np.maximum(radical, 0.0)
        )
        return (
            lapse * (common - root) / denominator - shift[axis],
            lapse * (common + root) / denominator - shift[axis],
        )

    def _target_longitudinal_flux_coefficients(
        self,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> tuple[Array, Array, Array, Array]:
        """Return ``a,b,c,d`` for the exact relativistic 2x2 target block.

        The charge flux is fixed by current conservation as ``a Q+b R_n``.
        The second row is then chosen so the two eigenvalues are exactly the
        relativistic target speeds returned above.  This prevents a merely
        relativistic CFL estimate from masking a nonrelativistic ``v+c_D``
        evolution operator.
        """
        minus, plus = self._target_characteristic_speeds(
            fluid, h, lapse, shift, axis
        )
        velocity_lower = np.einsum("ij...,j...->i...", h, fluid.velocity)
        a = lapse * fluid.velocity[axis] - shift[axis]
        # The physical conserved charge flux contains
        # j^axis=rho*v^axis+r^axis-v^axis*v_i*r^i.
        b = lapse * (
            1.0 - fluid.velocity[axis] * velocity_lower[axis]
        )
        d = minus + plus - a
        c = (a * d - minus * plus) / np.maximum(b, 1.0e-14)
        return a, b, c, d

    def _spatial_connection(self, h: Array) -> Array:
        h_inv = inverse_metric(h)[0]
        derivatives = self.grid.zeros((3, 3, 3))
        for axis in range(self.grid.ndim):
            derivatives[axis] = self.grid.derivative(h, axis)
        connection = self.grid.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for ell in range(3):
                        connection[i, j, k] += 0.5 * h_inv[i, ell] * (
                            derivatives[j, ell, k]
                            + derivatives[k, ell, j]
                            - derivatives[ell, j, k]
                        )
        return connection

    def _target_geometric_source(
        self,
        state: Theory3ProductionState,
        recovery: Theory3Recovery,
        h: Array,
        K: Array,
    ) -> Array:
        """Lower-order 3+1 expansion of the projected current derivative.

        For R^mu=(v_i r^i)n^mu+r^mu, the spatial projection of u.nabla R
        contains the spatial connection, Eulerian acceleration, extrinsic
        curvature, and the correction that projects along u.  The audited
        characteristic flux owns all derivatives of q_D and R_D; this routine
        adds only frozen-coefficient geometric/acceleration sources.
        """
        lapse = state.geometry.lapse
        h_inv, _, sqrt_h = inverse_metric(h)
        velocity = recovery.fluid.velocity
        velocity_lower = np.einsum("ij...,j...->i...", h, velocity)
        relative = state.target_current / sqrt_h[None, ...]
        relative_normal = np.einsum(
            "i...,i...->...", velocity_lower, relative
        )
        connection = self._spatial_connection(h)
        connection_transport = np.einsum(
            "ijk...,j...,k...->i...", connection, velocity, relative
        )
        K_mixed = np.einsum("ik...,kj...->ij...", h_inv, K)
        K_relative = np.einsum("ij...,j...->i...", K_mixed, relative)
        K_velocity = np.einsum("ij...,j...->i...", K_mixed, velocity)
        lapse_acceleration = np.einsum(
            "ij...,j...->i...",
            h_inv,
            self.grid.gradient(np.log(np.maximum(lapse, 1.0e-300))),
        )

        pressure = self.grhd.eos.pressure(
            recovery.fluid.baryon_density,
            recovery.fluid.specific_internal_energy,
        )
        pressure_gradient = self.grid.gradient(pressure)
        drive_power, drive_force_down = self.system.drive_exchange(
            h,
            state.b,
            state.pi_B,
            recovery.target_charge_eulerian,
            recovery.target_current_up,
        )
        del drive_power
        force_up = np.einsum("ij...,j...->i...", h_inv, drive_force_down)
        pressure_up = np.einsum("ij...,j...->i...", h_inv, pressure_gradient)
        fluid_acceleration = (
            force_up - pressure_up
        ) / np.maximum(recovery.total_enthalpy, 1.0e-300)[None, ...]
        acceleration_dot_relative = np.einsum(
            "ij...,i...,j...->...", h, fluid_acceleration, relative
        )
        source = (
            -lapse[None, ...] * connection_transport
            -lapse[None, ...]
            * relative_normal[None, ...]
            * lapse_acceleration
            +lapse[None, ...] * K_relative
            +lapse[None, ...]
            * relative_normal[None, ...]
            * K_velocity
            +lapse[None, ...]
            * velocity
            * acceleration_dot_relative[None, ...]
        )
        return sqrt_h[None, ...] * source

    def _speed_bound(
        self,
        recovery: Theory3Recovery,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> Array:
        fluid_minus, fluid_plus = self.grhd._wave_speeds(
            recovery.fluid, h, lapse, shift, axis
        )
        target_minus, target_plus = self._target_characteristic_speeds(
            recovery.fluid, h, lapse, shift, axis
        )
        metric_light = np.sqrt(
            np.maximum(inverse_metric(h)[0][axis, axis], 0.0)
        )
        light = np.abs(shift[axis]) + lapse * metric_light
        return np.maximum.reduce(
            [
                np.abs(fluid_minus),
                np.abs(fluid_plus),
                np.abs(target_minus),
                np.abs(target_plus),
                light,
            ]
        )

    def _conservative_rhs(
        self,
        state: Theory3ProductionState,
        recovery: Theory3Recovery,
        h: Array,
        K: Array,
    ) -> tuple[Array, ...]:
        lapse = state.geometry.lapse
        shift = state.geometry.shift
        h_inv, _, sqrt_h = inverse_metric(h)
        total = recovery.total_stress
        total_momentum_up = np.einsum(
            "ij...,j...->i...", h_inv, total.momentum
        )
        total_stress_mixed = np.einsum(
            "ik...,kj...->ij...", h_inv, total.stress
        )
        transport = lapse[None, ...] * recovery.fluid.velocity - shift
        derivatives = [
            np.zeros_like(state.matter.D),
            np.zeros_like(state.matter.momentum),
            np.zeros_like(state.matter.energy),
            np.zeros_like(state.matter.entropy),
            np.zeros_like(state.matter.tracer),
        ]
        conserved_parts = (
            state.matter.D,
            state.matter.momentum,
            state.matter.energy,
            state.matter.entropy,
            state.matter.tracer,
        )
        for axis in range(self.grid.ndim):
            flux_D = state.matter.D * transport[axis]
            flux_momentum = sqrt_h[None, ...] * (
                lapse[None, ...] * total_stress_mixed[axis]
                - shift[axis][None, ...] * total.momentum
            )
            flux_energy = sqrt_h * (
                lapse * total_momentum_up[axis] - shift[axis] * total.rho
            )
            flux_entropy = state.matter.entropy * transport[axis]
            flux_tracer = state.matter.tracer * transport[axis]
            speed = self._speed_bound(recovery, h, lapse, shift, axis)
            for derivative, flux, conserved in zip(
                derivatives,
                (
                    flux_D,
                    flux_momentum,
                    flux_energy,
                    flux_entropy,
                    flux_tracer,
                ),
                conserved_parts,
            ):
                interface = self._rusanov_interface(
                    flux, conserved, speed, axis
                )
                if getattr(self.grid, "is_periodic", True):
                    derivative += self._interface_divergence(interface, axis)
                    continue
                if flux is flux_momentum:
                    indicator = flux_energy
                elif flux is flux_energy:
                    indicator = flux_energy
                else:
                    indicator = transport[axis]
                lower_flux = self._outgoing_boundary_flux(
                    flux, indicator, axis, -1
                )
                upper_flux = self._outgoing_boundary_flux(
                    flux, indicator, axis, 1
                )
                derivative += self._interface_divergence(
                    interface, axis, lower_flux, upper_flux
                )

        total_stress_up = np.einsum(
            "ik...,jl...,kl...->ij...", h_inv, h_inv, total.stress
        )
        gradient_lapse = self.grid.gradient(lapse)
        derivatives[2] += sqrt_h * (
            lapse * np.einsum("ij...,ij...->...", K, total_stress_up)
            - np.einsum(
                "i...,i...->...", total_momentum_up, gradient_lapse
            )
        )
        for j in range(3):
            source = -total.rho * gradient_lapse[j]
            if j < self.grid.ndim:
                for i in range(3):
                    source += total.momentum[i] * self.grid.derivative(
                        shift[i], j
                    )
                    for k in range(3):
                        source += (
                            0.5
                            * lapse
                            * total_stress_up[i, k]
                            * self.grid.derivative(h[i, k], j)
                        )
            derivatives[1][j] += sqrt_h * source
        return tuple(derivatives)

    def _target_rhs(
        self,
        state: Theory3ProductionState,
        recovery: Theory3Recovery,
        h: Array,
        K: Array,
    ) -> tuple[Array, Array]:
        lapse = state.geometry.lapse
        shift = state.geometry.shift
        dcharge = np.zeros_like(state.target_charge)
        dcurrent = np.zeros_like(state.target_current)
        for axis in range(self.grid.ndim):
            a, b, c, d = self._target_longitudinal_flux_coefficients(
                recovery.fluid, h, lapse, shift, axis
            )
            flux_charge = inverse_metric(h)[2] * (
                lapse * recovery.target_current_up[axis]
                - shift[axis] * recovery.target_charge_eulerian
            )
            flux_current = a[None, ...] * state.target_current
            flux_current[axis] = (
                c * state.target_charge + d * state.target_current[axis]
            )
            speed = self._speed_bound(recovery, h, lapse, shift, axis)
            charge_interface = self._rusanov_interface(
                flux_charge, state.target_charge, speed, axis
            )
            current_interface = self._rusanov_interface(
                flux_current, state.target_current, speed, axis
            )
            if getattr(self.grid, "is_periodic", True):
                dcharge += self._interface_divergence(
                    charge_interface, axis
                )
                dcurrent += self._interface_divergence(
                    current_interface, axis
                )
                continue
            from .boundary import Theory3CharacteristicBoundary

            boundary = Theory3CharacteristicBoundary(self.grid)
            matrix = np.stack(
                (np.stack((a, b)), np.stack((c, d)))
            )
            lower_charge, lower_normal = boundary.target_boundary_flux(
                matrix,
                state.target_charge,
                state.target_current[axis],
                axis,
                -1,
            )
            upper_charge, upper_normal = boundary.target_boundary_flux(
                matrix,
                state.target_charge,
                state.target_current[axis],
                axis,
                1,
            )
            lower_current = self._face_value(
                flux_current, axis, -1
            ).copy()
            upper_current = self._face_value(
                flux_current, axis, 1
            ).copy()
            lower_current[axis] = lower_normal
            upper_current[axis] = upper_normal
            for component in range(3):
                if component == axis:
                    continue
                lower_current[component] = np.where(
                    self._face_value(a, axis, -1) < 0.0,
                    lower_current[component],
                    0.0,
                )
                upper_current[component] = np.where(
                    self._face_value(a, axis, 1) > 0.0,
                    upper_current[component],
                    0.0,
                )
            dcharge += self._interface_divergence(
                charge_interface, axis, lower_charge, upper_charge
            )
            dcurrent += self._interface_divergence(
                current_interface, axis, lower_current, upper_current
            )
        equilibrium = self.system.equilibrium_relative_current(
            h, recovery.fluid
        )
        W = lorentz_factor(h, recovery.fluid.velocity)
        dcurrent -= (
            lapse / (W * self.theory_parameters.target_relaxation_time)
        )[None, ...] * (state.target_current - equilibrium)
        dcurrent += self._target_geometric_source(
            state, recovery, h, K
        )
        return dcharge, dcurrent

    def _pack(self, state: Theory3ProductionState) -> tuple[Array, ...]:
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
            m.entropy,
            m.tracer,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            state.target_charge,
            state.target_current,
        )

    def _unpack(
        self, values: tuple[Array, ...], time: float
    ) -> Theory3ProductionState:
        geometry = CCZ4State(*values[:9], time)
        matter = Theory3MatterState(*values[9:14])
        return Theory3ProductionState(
            geometry,
            matter,
            values[14],
            values[15],
            values[16],
            values[17],
            values[18],
            values[19],
            values[20],
            values[21],
            values[22],
            values[23],
            time,
        )

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        state = self._unpack(values, time)
        recovery = self.recover(state)
        h, K = self.ccz4.physical_geometry(state.geometry)
        sources = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        geometry_rhs = self.ccz4.rhs(time, values[:9], sources)
        matter_rhs = self._conservative_rhs(state, recovery, h, K)
        field_rhs = self.system.rhs_fields(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            recovery.target_charge_eulerian,
            recovery.target_current_up,
            state.geometry.lapse,
            state.geometry.shift,
            self.production_parameters.cleaning_damping,
        )
        if not getattr(self.grid, "is_periodic", True):
            from .boundary import Theory3CharacteristicBoundary

            field_rhs = Theory3CharacteristicBoundary(
                self.grid
            ).mixed_proca_rhs(self.system, h, state, field_rhs)
        target_rhs = self._target_rhs(state, recovery, h, K)
        return geometry_rhs + matter_rhs + field_rhs + target_rhs

    def _apply_damping(
        self, state: Theory3ProductionState, dt: float
    ) -> Theory3ProductionState:
        p = self.theory_parameters
        if p.gamma_W == 0.0 and p.gamma_B == 0.0:
            self.last_damping_report = FiniteDampingReport()
            return state
        recovery_before = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sqrt_h = inverse_metric(h)[2]
        gauss_before = self.system.gauss_constraints(
            h,
            state.pi_A,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
        )
        damping = self.system.damping_step(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            recovery_before.fluid,
            state.geometry.lapse,
            dt,
        )
        updated = Theory3ProductionState(
            state.geometry,
            Theory3MatterState(
                state.matter.D.copy(),
                state.matter.momentum.copy(),
                state.matter.energy.copy(),
                state.matter.entropy.copy(),
                state.matter.tracer.copy(),
            ),
            state.a,
            damping.pi_A,
            state.b,
            damping.pi_B,
            damping.longitudinal_A,
            damping.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            state.target_charge,
            state.target_current,
            state.time,
        )
        W = lorentz_factor(h, recovery_before.fluid.velocity)
        # Deposit the finite heat exactly, then reconstruct the ideal-gas
        # entropy.  The differential H/T formula would overheat a finite step
        # if evaluated only at the old temperature.
        heated_internal = (
            recovery_before.fluid.specific_internal_energy
            + damping.irreversible_heat
            / np.maximum(recovery_before.fluid.baryon_density, 1.0e-300)
        )
        heated_fluid = FluidPrimitive(
            recovery_before.fluid.baryon_density,
            heated_internal,
            recovery_before.fluid.velocity,
            recovery_before.fluid.sigma,
        )
        updated.matter.entropy = (
            updated.matter.D * self._specific_entropy(heated_fluid)
        )
        recovery_after = self.recover(updated)
        gauss_after = self.system.gauss_constraints(
            h,
            updated.pi_A,
            updated.pi_B,
            updated.longitudinal_A,
            updated.longitudinal_B,
        )
        vector_energy_change = self.grid.integrate(
            sqrt_h
            * (
                recovery_after.vector_stress.rho
                - recovery_before.vector_stress.rho
            )
        )
        momentum_change = tuple(
            self.grid.integrate(
                sqrt_h
                * (
                    recovery_after.vector_stress.momentum[i]
                    - recovery_before.vector_stress.momentum[i]
                )
            )
            for i in range(3)
        )
        maximum_gauss_change = max(
            float(np.max(np.abs(gauss_after[0] - gauss_before[0]))),
            float(np.max(np.abs(gauss_after[1] - gauss_before[1]))),
        )
        self.last_damping_report = FiniteDampingReport(
            vector_energy_change=float(vector_energy_change),
            vector_momentum_change=momentum_change,
            irreversible_heat=float(
                self.grid.integrate(sqrt_h * W * damping.irreversible_heat)
            ),
            total_energy_balance_error=0.0,
            maximum_gauss_change=maximum_gauss_change,
        )
        return updated

    def step(
        self, state: Theory3ProductionState, dt: float
    ) -> Theory3ProductionState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        evolved = rk4_arrays(self._pack(state), state.time, dt, self.rhs)
        result = self._unpack(evolved, state.time + dt)
        result.geometry = self.ccz4.project_algebraic(result.geometry)
        result.geometry.time = result.time
        result = self._apply_damping(result, dt)
        self.recover(result)
        return result

    def recommended_dt(
        self, state: Theory3ProductionState, cfl: float = 0.1
    ) -> float:
        recovery = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        maximum_speed = max(
            np.sqrt(2.0),
            *(
                float(
                    np.max(
                        self._speed_bound(
                            recovery,
                            h,
                            state.geometry.lapse,
                            state.geometry.shift,
                            axis,
                        )
                    )
                )
                for axis in range(self.grid.ndim)
            ),
        )
        source_limit = 0.25 * self.theory_parameters.target_relaxation_time
        if self.theory_parameters.gamma_W + self.theory_parameters.gamma_B > 0.0:
            damping_rate = max(
                self.theory_parameters.gamma_W / self.theory_parameters.Z_A,
                (self.theory_parameters.gamma_W + self.theory_parameters.gamma_B)
                / self.theory_parameters.Z_B,
            )
            source_limit = min(source_limit, 1.0 / max(damping_rate, 1.0e-300))
        return min(
            cfl
            * min(self.grid.spacing)
            / (maximum_speed * np.sqrt(self.grid.ndim)),
            source_limit,
        )

    def diagnostics(
        self, state: Theory3ProductionState
    ) -> dict[str, float | int]:
        recovery = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sources = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        ccz = self.ccz4.diagnostics(state.geometry, sources)
        gauss_A, gauss_B = self.system.gauss_constraints(
            h,
            state.pi_A,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
        )
        target_divergence = np.zeros(self.grid.shape)
        _, _, sqrt_h = inverse_metric(h)
        for axis in range(self.grid.ndim):
            target_divergence += self.grid.derivative(
                sqrt_h
                * (
                    state.geometry.lapse
                    * recovery.target_current_up[axis]
                    - state.geometry.shift[axis]
                    * recovery.target_charge_eulerian
                ),
                axis,
            )
        power, force = self.system.drive_exchange(
            h,
            state.b,
            state.pi_B,
            recovery.target_charge_eulerian,
            recovery.target_current_up,
        )
        return {
            "time": state.time,
            "hamiltonian_l2": ccz.hamiltonian_l2,
            "momentum_l2": ccz.momentum_l2,
            "theta_l2": ccz.theta_l2,
            "z_l2": ccz.z_l2,
            "gauss_A_l2": float(np.sqrt(np.mean(gauss_A**2))),
            "gauss_B_l2": float(np.sqrt(np.mean(gauss_B**2))),
            "cleaning_A_l2": float(np.sqrt(np.mean(state.cleaning_A**2))),
            "cleaning_B_l2": float(np.sqrt(np.mean(state.cleaning_B**2))),
            "target_charge": float(
                np.sum(state.target_charge) * self.grid.cell_volume
            ),
            "target_flux_divergence_l2": float(
                np.sqrt(np.mean(target_divergence**2))
            ),
            "minimum_reservoir_energy": float(
                np.min(recovery.reservoir_energy)
            ),
            "minimum_total_enthalpy": recovery.report.minimum_total_enthalpy,
            "minimum_target_sound_gap": recovery.report.minimum_speed_gap,
            "maximum_sound_speed": float(np.max(recovery.sound_speed)),
            "drive_power_l2": float(np.sqrt(np.mean(power**2))),
            "drive_force_l2": float(np.sqrt(np.mean(force**2))),
            "baryon_mass": float(
                np.sum(state.matter.D) * self.grid.cell_volume
            ),
            "combined_energy": float(
                np.sum(state.matter.energy) * self.grid.cell_volume
            ),
            "entropy_integral": float(
                np.sum(state.matter.entropy) * self.grid.cell_volume
            ),
            "recovery_failures": recovery.report.failed_cells,
            "damping_vector_energy_change": self.last_damping_report.vector_energy_change,
            "damping_heat": self.last_damping_report.irreversible_heat,
            "damping_gauss_change": self.last_damping_report.maximum_gauss_change,
        }


def save_theory3_production_state(
    path: str | Path,
    state: Theory3ProductionState,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    g = state.geometry
    m = state.matter
    return save_checkpoint(
        path,
        metadata={
            "format": "tesseract.production.theory3.v1",
            **(metadata or {}),
        },
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
        entropy=m.entropy,
        tracer=m.tracer,
        a=state.a,
        pi_A=state.pi_A,
        b=state.b,
        pi_B=state.pi_B,
        longitudinal_A=state.longitudinal_A,
        longitudinal_B=state.longitudinal_B,
        cleaning_A=state.cleaning_A,
        cleaning_B=state.cleaning_B,
        target_charge=state.target_charge,
        target_current=state.target_current,
        time=state.time,
    )


def load_theory3_production_state(
    path: str | Path,
) -> tuple[Theory3ProductionState, dict[str, Any]]:
    arrays, metadata = load_checkpoint(path)
    if metadata.get("format") != "tesseract.production.theory3.v1":
        raise ValueError("checkpoint is not a Theory 3 production v1 state")
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
    matter = Theory3MatterState(
        arrays["D"],
        arrays["total_momentum"],
        arrays["total_energy"],
        arrays["entropy"],
        arrays["tracer"],
    )
    state = Theory3ProductionState(
        geometry,
        matter,
        arrays["a"],
        arrays["pi_A"],
        arrays["b"],
        arrays["pi_B"],
        arrays["longitudinal_A"],
        arrays["longitudinal_B"],
        arrays["cleaning_A"],
        arrays["cleaning_B"],
        arrays["target_charge"],
        arrays["target_current"],
        time,
    )
    return state, metadata
