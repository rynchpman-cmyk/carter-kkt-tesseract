"""Valencia-formulation general-relativistic hydrodynamics.

The finite-volume backend uses MC reconstruction, relativistic HLL fluxes,
SSP-RK3 time stepping, ideal-gas primitive recovery, and explicit atmosphere
handling. Conserved variables are densitized by sqrt(gamma).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .grid import Array, PeriodicGrid
from .matter import FluidPrimitive, IdealGasEOS, lorentz_factor, perfect_fluid_stress_energy


@dataclass(frozen=True)
class GRHDParameters:
    gamma_ad: float = 5.0 / 3.0
    density_floor: float = 1.0e-12
    internal_energy_floor: float = 1.0e-10
    reconstruction_theta: float = 1.5
    recovery_tolerance: float = 1.0e-12
    recovery_iterations: int = 80

    def __post_init__(self) -> None:
        if self.gamma_ad <= 1.0:
            raise ValueError("gamma_ad must exceed one")
        if self.density_floor <= 0.0 or self.internal_energy_floor < 0.0:
            raise ValueError("atmosphere floors are invalid")
        if not 1.0 <= self.reconstruction_theta <= 2.0:
            raise ValueError("reconstruction_theta must lie in [1, 2]")
        if self.recovery_tolerance <= 0.0 or self.recovery_iterations < 10:
            raise ValueError("primitive recovery controls are invalid")


@dataclass
class GRHDState:
    D: Array
    momentum: Array
    tau: Array
    tracer: Array
    time: float = 0.0

    def copy(self) -> "GRHDState":
        return GRHDState(
            self.D.copy(), self.momentum.copy(), self.tau.copy(), self.tracer.copy(), self.time
        )


@dataclass(frozen=True)
class RecoveryReport:
    atmosphere_cells: int
    failed_cells: int
    maximum_residual: float


@dataclass(frozen=True)
class GRHDDiagnostics:
    time: float
    baryon_mass: float
    energy_integral: float
    momentum_integral: tuple[float, float, float]
    minimum_density: float
    maximum_density: float
    maximum_speed: float
    atmosphere_cells: int
    recovery_failures: int


Exchange = tuple[Array, Array]  # (-n_mu Q^mu, gamma_i_mu Q^mu)


def _minmod3(a: Array, b: Array, c: Array) -> Array:
    same = (np.sign(a) == np.sign(b)) & (np.sign(b) == np.sign(c))
    magnitude = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same, np.sign(a) * magnitude, 0.0)


class ValenciaGRHD:
    def __init__(self, grid: PeriodicGrid, parameters: GRHDParameters | None = None) -> None:
        self.grid = grid
        self.parameters = parameters or GRHDParameters()
        self.eos = IdealGasEOS(self.parameters.gamma_ad)
        self.last_recovery_report = RecoveryReport(0, 0, 0.0)

    def _axis(self, field: Array, spatial_axis: int) -> int:
        return field.ndim - self.grid.ndim + spatial_axis

    def _roll(self, field: Array, offset: int, spatial_axis: int) -> Array:
        if hasattr(self.grid, "shift"):
            return self.grid.shift(field, offset, spatial_axis)
        return np.roll(field, offset, axis=self._axis(field, spatial_axis))

    def primitives_to_conserved(self, fluid: FluidPrimitive, h: Array) -> GRHDState:
        _, _, sqrt_h = inverse_metric(h)
        W = lorentz_factor(h, fluid.velocity)
        pressure = self.eos.pressure(
            fluid.baryon_density, fluid.specific_internal_energy
        )
        enthalpy = self.eos.specific_enthalpy(fluid.specific_internal_energy)
        velocity_lower = np.einsum("ij...,j...->i...", h, fluid.velocity)
        D0 = fluid.baryon_density * W
        momentum0 = (
            fluid.baryon_density * enthalpy * W**2
        )[None, ...] * velocity_lower
        energy0 = fluid.baryon_density * enthalpy * W**2 - pressure
        return GRHDState(
            D=sqrt_h * D0,
            momentum=sqrt_h[None, ...] * momentum0,
            tau=sqrt_h * (energy0 - D0),
            tracer=sqrt_h * D0 * fluid.sigma,
        )

    def _pressure_residual(
        self, pressure: Array, D: Array, energy: Array, momentum_sq: Array
    ) -> tuple[Array, Array, Array, Array]:
        denominator = np.maximum(energy + pressure, 1.0e-300)
        velocity_sq = np.minimum(momentum_sq / denominator**2, 1.0 - 1.0e-14)
        W = 1.0 / np.sqrt(1.0 - velocity_sq)
        density = D / W
        enthalpy = denominator / np.maximum(D * W, 1.0e-300)
        eos_pressure = (
            (self.parameters.gamma_ad - 1.0)
            / self.parameters.gamma_ad
            * density
            * (enthalpy - 1.0)
        )
        return pressure - eos_pressure, density, W, enthalpy

    def conserved_to_primitives(
        self, state: GRHDState, h: Array
    ) -> tuple[FluidPrimitive, RecoveryReport]:
        h_inv, _, sqrt_h = inverse_metric(h)
        D = state.D / sqrt_h
        momentum = state.momentum / sqrt_h[None, ...]
        tau = state.tau / sqrt_h
        momentum_sq = np.einsum("ij...,i...,j...->...", h_inv, momentum, momentum)
        energy = tau + D
        atmosphere = (
            (D <= self.parameters.density_floor)
            | ~np.isfinite(D)
            | ~np.isfinite(energy)
            | (energy <= 0.0)
        )

        minimum_pressure = (
            (self.parameters.gamma_ad - 1.0)
            * self.parameters.density_floor
            * self.parameters.internal_energy_floor
        )
        lower = np.full(self.grid.shape, minimum_pressure)
        # Ensure E+p > |S| before evaluating the residual.
        lower = np.maximum(lower, np.sqrt(np.maximum(momentum_sq, 0.0)) - energy + 1.0e-14)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(
            1.0,
            (self.parameters.gamma_ad - 1.0) * np.maximum(energy, 0.0)
            + np.sqrt(np.maximum(momentum_sq, 0.0)),
        )
        f_lower = self._pressure_residual(lower, D, energy, momentum_sq)[0]
        f_upper = self._pressure_residual(upper, D, energy, momentum_sq)[0]
        for _ in range(30):
            needs_growth = (f_lower * f_upper > 0.0) & ~atmosphere
            if not np.any(needs_growth):
                break
            upper = np.where(needs_growth, 2.0 * upper, upper)
            f_upper = self._pressure_residual(upper, D, energy, momentum_sq)[0]

        bracket_failed = (f_lower * f_upper > 0.0) & ~atmosphere
        for _ in range(self.parameters.recovery_iterations):
            middle = 0.5 * (lower + upper)
            f_middle = self._pressure_residual(middle, D, energy, momentum_sq)[0]
            left_half = f_lower * f_middle <= 0.0
            upper = np.where(left_half, middle, upper)
            f_upper = np.where(left_half, f_middle, f_upper)
            lower = np.where(left_half, lower, middle)
            f_lower = np.where(left_half, f_lower, f_middle)
        pressure = 0.5 * (lower + upper)
        residual, density, W, enthalpy = self._pressure_residual(
            pressure, D, energy, momentum_sq
        )
        relative_residual = np.abs(residual) / np.maximum(pressure + np.abs(energy), 1.0)
        failed = bracket_failed | ~np.isfinite(relative_residual) | (
            relative_residual > 100.0 * self.parameters.recovery_tolerance
        )
        reset = atmosphere | failed

        density = np.where(reset, self.parameters.density_floor, density)
        pressure_floor = (
            (self.parameters.gamma_ad - 1.0)
            * density
            * self.parameters.internal_energy_floor
        )
        pressure = np.where(reset, pressure_floor, np.maximum(pressure, pressure_floor))
        internal = pressure / (
            (self.parameters.gamma_ad - 1.0) * np.maximum(density, self.parameters.density_floor)
        )
        velocity = np.einsum("ij...,j...->i...", h_inv, momentum) / np.maximum(
            energy + pressure, 1.0e-300
        )[None, ...]
        velocity[:, reset] = 0.0
        speed_sq = np.einsum("ij...,i...,j...->...", h, velocity, velocity)
        cap = np.sqrt(np.maximum(speed_sq, 0.0) / (1.0 - 1.0e-12))
        velocity /= np.maximum(cap, 1.0)[None, ...]
        sigma = np.where(state.D > 0.0, state.tracer / np.maximum(state.D, 1.0e-300), 0.0)
        sigma = np.where(reset, 0.0, sigma)
        fluid = FluidPrimitive(density, internal, velocity, sigma)
        report = RecoveryReport(
            atmosphere_cells=int(np.count_nonzero(atmosphere)),
            failed_cells=int(np.count_nonzero(failed)),
            maximum_residual=float(np.max(np.where(reset, 0.0, relative_residual))),
        )
        self.last_recovery_report = report
        return fluid, report

    def enforce_atmosphere(self, state: GRHDState, h: Array) -> GRHDState:
        fluid, _ = self.conserved_to_primitives(state, h)
        clean = self.primitives_to_conserved(fluid, h)
        clean.time = state.time
        return clean

    def _slope(self, field: Array, axis: int) -> Array:
        backward = field - self._roll(field, 1, axis)
        forward = self._roll(field, -1, axis) - field
        centered = 0.5 * (self._roll(field, -1, axis) - self._roll(field, 1, axis))
        theta = self.parameters.reconstruction_theta
        return _minmod3(theta * backward, centered, theta * forward)

    def _interface_primitives(
        self, fluid: FluidPrimitive, h_face: Array, axis: int
    ) -> tuple[FluidPrimitive, FluidPrimitive]:
        fields = (
            fluid.baryon_density,
            fluid.specific_internal_energy,
            fluid.velocity,
            fluid.sigma,
        )
        reconstructed: list[tuple[Array, Array]] = []
        for field in fields:
            slope = self._slope(field, axis)
            left = field + 0.5 * slope
            right = self._roll(field, -1, axis) - 0.5 * self._roll(slope, -1, axis)
            reconstructed.append((left, right))
        rho_l, rho_r = reconstructed[0]
        eps_l, eps_r = reconstructed[1]
        vel_l, vel_r = reconstructed[2]
        sig_l, sig_r = reconstructed[3]
        rho_l = np.maximum(rho_l, self.parameters.density_floor)
        rho_r = np.maximum(rho_r, self.parameters.density_floor)
        eps_l = np.maximum(eps_l, self.parameters.internal_energy_floor)
        eps_r = np.maximum(eps_r, self.parameters.internal_energy_floor)
        for velocity in (vel_l, vel_r):
            speed_sq = np.einsum("ij...,i...,j...->...", h_face, velocity, velocity)
            scale = np.sqrt(np.maximum(speed_sq, 0.0) / (1.0 - 1.0e-12))
            velocity /= np.maximum(scale, 1.0)[None, ...]
        return (
            FluidPrimitive(rho_l, eps_l, vel_l, sig_l),
            FluidPrimitive(rho_r, eps_r, vel_r, sig_r),
        )

    def _flux(
        self,
        conserved: GRHDState,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> tuple[Array, Array, Array, Array]:
        _, _, sqrt_h = inverse_metric(h)
        pressure = self.eos.pressure(fluid.baryon_density, fluid.specific_internal_energy)
        transport = lapse * fluid.velocity[axis] - shift[axis]
        flux_D = conserved.D * transport
        flux_momentum = conserved.momentum * transport[None, ...]
        flux_momentum[axis] += lapse * sqrt_h * pressure
        flux_tau = conserved.tau * transport + lapse * sqrt_h * pressure * fluid.velocity[axis]
        flux_tracer = conserved.tracer * transport
        return flux_D, flux_momentum, flux_tau, flux_tracer

    def _wave_speeds(
        self,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> tuple[Array, Array]:
        h_inv = inverse_metric(h)[0]
        pressure = self.eos.pressure(fluid.baryon_density, fluid.specific_internal_energy)
        enthalpy = self.eos.specific_enthalpy(fluid.specific_internal_energy)
        sound_sq = np.clip(
            self.parameters.gamma_ad * pressure / np.maximum(fluid.baryon_density * enthalpy, 1e-300),
            0.0,
            1.0 - 1.0e-12,
        )
        speed_sq = np.einsum("ij...,i...,j...->...", h, fluid.velocity, fluid.velocity)
        v_axis = fluid.velocity[axis]
        denominator = np.maximum(1.0 - speed_sq * sound_sq, 1.0e-14)
        radical = (1.0 - speed_sq) * (
            h_inv[axis, axis] * (1.0 - speed_sq * sound_sq)
            - v_axis**2 * (1.0 - sound_sq)
        )
        root = np.sqrt(np.maximum(radical, 0.0))
        common = v_axis * (1.0 - sound_sq)
        sound = np.sqrt(sound_sq)
        minus = lapse * (common - sound * root) / denominator - shift[axis]
        plus = lapse * (common + sound * root) / denominator - shift[axis]
        return minus, plus

    def _hll_flux(
        self,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> tuple[Array, Array, Array, Array]:
        h_face = 0.5 * (h + self._roll(h, -1, axis))
        lapse_face = 0.5 * (lapse + self._roll(lapse, -1, axis))
        shift_face = 0.5 * (shift + self._roll(shift, -1, axis))
        left, right = self._interface_primitives(fluid, h_face, axis)
        U_left = self.primitives_to_conserved(left, h_face)
        U_right = self.primitives_to_conserved(right, h_face)
        F_left = self._flux(U_left, left, h_face, lapse_face, shift_face, axis)
        F_right = self._flux(U_right, right, h_face, lapse_face, shift_face, axis)
        lm_left, lp_left = self._wave_speeds(left, h_face, lapse_face, shift_face, axis)
        lm_right, lp_right = self._wave_speeds(right, h_face, lapse_face, shift_face, axis)
        speed_left = np.minimum(0.0, np.minimum(lm_left, lm_right))
        speed_right = np.maximum(0.0, np.maximum(lp_left, lp_right))
        denominator = np.maximum(speed_right - speed_left, 1.0e-14)
        U_parts = (
            (U_left.D, U_right.D),
            (U_left.momentum, U_right.momentum),
            (U_left.tau, U_right.tau),
            (U_left.tracer, U_right.tracer),
        )
        results = []
        for (u_left, u_right), f_left, f_right in zip(U_parts, F_left, F_right):
            component_axes = u_left.ndim - self.grid.ndim
            s_l = speed_left.reshape((1,) * component_axes + self.grid.shape)
            s_r = speed_right.reshape((1,) * component_axes + self.grid.shape)
            den = denominator.reshape((1,) * component_axes + self.grid.shape)
            results.append(
                (s_r * f_left - s_l * f_right + s_l * s_r * (u_right - u_left)) / den
            )
        return tuple(results)  # type: ignore[return-value]

    def _set_edge(self, field: Array, axis: int, side: int, value: Array) -> None:
        array_axis = self._axis(field, axis)
        target = [slice(None)] * field.ndim
        source = [slice(None)] * value.ndim
        target[array_axis] = 0 if side < 0 else -1
        source[array_axis] = 0 if side < 0 else -1
        field[tuple(target)] = value[tuple(source)]

    def _outflow_boundary_flux(
        self,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
        side: int,
    ) -> tuple[Array, Array, Array, Array]:
        velocity = fluid.velocity.copy()
        array_axis = self._axis(velocity[axis], axis)
        edge = [slice(None)] * velocity[axis].ndim
        edge[array_axis] = 0 if side < 0 else -1
        normal_component = velocity[axis][tuple(edge)]
        # side=-1 has outward normal -e_axis; side=+1 has +e_axis.
        inward = side * normal_component < 0.0
        normal_component = np.where(inward, 0.0, normal_component)
        velocity[axis][tuple(edge)] = normal_component
        boundary_fluid = FluidPrimitive(
            fluid.baryon_density,
            fluid.specific_internal_energy,
            velocity,
            fluid.sigma,
        )
        conserved = self.primitives_to_conserved(boundary_fluid, h)
        return self._flux(conserved, boundary_fluid, h, lapse, shift, axis)

    def flux_pair(
        self,
        fluid: FluidPrimitive,
        h: Array,
        lapse: Array,
        shift: Array,
        axis: int,
    ) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
        """Return fluxes on each cell's right and left faces."""
        right = tuple(np.array(value, copy=True) for value in self._hll_flux(fluid, h, lapse, shift, axis))
        if getattr(self.grid, "is_periodic", True):
            left = tuple(self._roll(value, 1, axis) for value in right)
            return right, left
        lower = self._outflow_boundary_flux(fluid, h, lapse, shift, axis, -1)
        upper = self._outflow_boundary_flux(fluid, h, lapse, shift, axis, 1)
        for value, boundary_value in zip(right, upper):
            self._set_edge(value, axis, 1, boundary_value)
        left = tuple(self._roll(value, 1, axis) for value in right)
        for value, boundary_value in zip(left, lower):
            self._set_edge(value, axis, -1, boundary_value)
        return right, left

    def rhs(
        self,
        state: GRHDState,
        h: Array,
        K: Array,
        lapse: Array,
        shift: Array,
        exchange: Exchange | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        fluid, _ = self.conserved_to_primitives(state, h)
        dD = np.zeros_like(state.D)
        dmomentum = np.zeros_like(state.momentum)
        dtau = np.zeros_like(state.tau)
        dtracer = np.zeros_like(state.tracer)
        derivatives = (dD, dmomentum, dtau, dtracer)
        for axis in range(self.grid.ndim):
            right_fluxes, left_fluxes = self.flux_pair(fluid, h, lapse, shift, axis)
            for derivative, right_flux, left_flux in zip(
                derivatives, right_fluxes, left_fluxes
            ):
                derivative -= (right_flux - left_flux) / self.grid.spacing[axis]

        h_inv, _, sqrt_h = inverse_metric(h)
        rho_energy, momentum_lower, stress_lower = perfect_fluid_stress_energy(
            self.grid, h, fluid, self.eos
        )
        momentum_up = np.einsum("ij...,j...->i...", h_inv, momentum_lower)
        stress_up = np.einsum(
            "ik...,jl...,kl...->ij...", h_inv, h_inv, stress_lower
        )
        gradient_lapse = self.grid.gradient(lapse)
        dtau += sqrt_h * (
            lapse * np.einsum("ij...,ij...->...", K, stress_up)
            - np.einsum("i...,i...->...", momentum_up, gradient_lapse)
        )
        for j in range(3):
            source = -rho_energy * gradient_lapse[j]
            for i in range(3):
                source += momentum_lower[i] * (
                    self.grid.derivative(shift[i], j) if j < self.grid.ndim else 0.0
                )
                for k in range(3):
                    dh = self.grid.derivative(h[i, k], j) if j < self.grid.ndim else 0.0
                    source += 0.5 * lapse * stress_up[i, k] * dh
            dmomentum[j] += sqrt_h * source

        if exchange is not None:
            energy_exchange, momentum_exchange = exchange
            dtau += lapse * sqrt_h * energy_exchange
            dmomentum += lapse[None, ...] * sqrt_h[None, ...] * momentum_exchange
        return dD, dmomentum, dtau, dtracer

    def step(
        self,
        state: GRHDState,
        dt: float,
        h: Array,
        K: Array,
        lapse: Array,
        shift: Array,
        exchange: Exchange | None = None,
    ) -> GRHDState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        y0 = (state.D, state.momentum, state.tau, state.tracer)

        def add(y: tuple[Array, ...], factor: float, rhs: tuple[Array, ...]) -> tuple[Array, ...]:
            return tuple(value + factor * derivative for value, derivative in zip(y, rhs))

        k1 = self.rhs(state, h, K, lapse, shift, exchange)
        y1 = add(y0, dt, k1)
        s1 = GRHDState(*y1, state.time + dt)
        k2 = self.rhs(s1, h, K, lapse, shift, exchange)
        y2_euler = add(y1, dt, k2)
        y2 = tuple(0.75 * original + 0.25 * trial for original, trial in zip(y0, y2_euler))
        s2 = GRHDState(*y2, state.time + 0.5 * dt)
        k3 = self.rhs(s2, h, K, lapse, shift, exchange)
        y3_euler = add(y2, dt, k3)
        final = tuple(
            (1.0 / 3.0) * original + (2.0 / 3.0) * trial
            for original, trial in zip(y0, y3_euler)
        )
        result = GRHDState(*final, state.time + dt)
        return self.enforce_atmosphere(result, h)

    def diagnostics(self, state: GRHDState, h: Array) -> GRHDDiagnostics:
        fluid, report = self.conserved_to_primitives(state, h)
        speed_sq = np.einsum("ij...,i...,j...->...", h, fluid.velocity, fluid.velocity)
        return GRHDDiagnostics(
            time=state.time,
            baryon_mass=float(np.sum(state.D) * self.grid.cell_volume),
            energy_integral=float(np.sum(state.tau + state.D) * self.grid.cell_volume),
            momentum_integral=tuple(
                float(np.sum(state.momentum[i]) * self.grid.cell_volume) for i in range(3)
            ),
            minimum_density=float(np.min(fluid.baryon_density)),
            maximum_density=float(np.max(fluid.baryon_density)),
            maximum_speed=float(np.sqrt(np.max(speed_sq))),
            atmosphere_cells=report.atmosphere_cells,
            recovery_failures=report.failed_cells,
        )
