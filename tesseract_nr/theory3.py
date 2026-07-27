"""Theory 3.2 mixed-Proca and causal target-current formulation.

This module contains the local formulation pieces used by the CCZ4+GRHD
production driver.  It intentionally does not depend on the Theory 2 density-
dependent mass model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .backend import apply_constant_2x2_exponential
from .grid import Array, PeriodicGrid
from .matter import FluidPrimitive, lorentz_factor
from .theory2 import StressEnergy3p1


@dataclass(frozen=True)
class Theory3Parameters:
    Z_A: float = 1.0
    Z_B: float = 1.0
    mass_A: float = 0.5
    mass_B: float = 0.5
    mixing_mass_squared: float = 0.25
    target_speed: float = 0.95
    target_relaxation_time: float = 1.0
    target_strength: float = 0.0
    target_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    sound_speed_margin: float = 0.05
    gamma_ad: float = 5.0 / 3.0
    gamma_W: float = 0.0
    gamma_B: float = 0.0

    def __post_init__(self) -> None:
        if self.Z_A <= 0.0 or self.Z_B <= 0.0:
            raise ValueError("vector kinetic coefficients must be positive")
        if self.mass_A <= 0.0 or self.mass_B <= 0.0:
            raise ValueError("both Proca masses must be positive")
        if self.mixing_mass_squared < 0.0:
            raise ValueError("mixing_mass_squared must be nonnegative")
        if not 0.0 < self.target_speed < 1.0:
            raise ValueError("target_speed must lie in (0, 1)")
        if self.target_relaxation_time <= 0.0:
            raise ValueError("target_relaxation_time must be positive")
        if self.sound_speed_margin <= 0.0:
            raise ValueError("sound_speed_margin must be positive")
        if not 1.0 < self.gamma_ad < 2.0:
            raise ValueError("the production ideal-gas gamma must lie in (1, 2)")
        sound_ceiling = np.sqrt(self.gamma_ad - 1.0)
        if self.target_speed < sound_ceiling + self.sound_speed_margin:
            raise ValueError(
                "target_speed must exceed the ideal-gas sound-speed ceiling "
                "by sound_speed_margin"
            )
        if self.gamma_W < 0.0 or self.gamma_B < 0.0:
            raise ValueError("damping rates must be nonnegative")
        direction = np.asarray(self.target_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("target_direction must be a finite 3-vector")
        if self.target_strength != 0.0 and np.linalg.norm(direction) == 0.0:
            raise ValueError("a nonzero target strength needs a direction")

    @property
    def gauss_matrix(self) -> np.ndarray:
        mu2 = self.mixing_mass_squared
        return np.array(
            [
                [self.mass_A**2 + mu2 / self.Z_A, -mu2 / self.Z_A],
                [-mu2 / self.Z_B, self.mass_B**2 + mu2 / self.Z_B],
            ],
            dtype=float,
        )

    @property
    def gauss_determinant(self) -> float:
        return float(np.linalg.det(self.gauss_matrix))


@dataclass(frozen=True)
class DampingResult:
    pi_A: Array
    pi_B: Array
    longitudinal_A: Array
    longitudinal_B: Array
    irreversible_heat: Array
    electric_energy_change: Array


def _combine_stress(
    first: StressEnergy3p1, second: StressEnergy3p1
) -> StressEnergy3p1:
    return StressEnergy3p1(
        first.rho + second.rho,
        first.momentum + second.momentum,
        first.stress + second.stress,
        first.trace + second.trace,
    )


class Theory3System:
    """Two constant-mass Proca fields plus evolved longitudinal charges."""

    def __init__(
        self, grid: PeriodicGrid, parameters: Theory3Parameters | None = None
    ) -> None:
        self.grid = grid
        self.parameters = parameters or Theory3Parameters()
        self._K = self.parameters.gauss_matrix
        self._K_inv = np.linalg.inv(self._K)

    def field_strength(self, potential: Array) -> Array:
        field = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i < self.grid.ndim:
                    field[i, j] += self.grid.derivative(potential[j], i)
                if j < self.grid.ndim:
                    field[i, j] -= self.grid.derivative(potential[i], j)
        return field

    def electric_field(self, h: Array, pi: Array, Z: float) -> Array:
        sqrt_h = inverse_metric(h)[2]
        return -pi / (Z * sqrt_h)[None, ...]

    def magnetic_field(self, h: Array, potential: Array) -> Array:
        _, _, sqrt_h = inverse_metric(h)
        field = self.field_strength(potential)
        magnetic = self.grid.zeros((3,))
        magnetic[0] = field[1, 2] / sqrt_h
        magnetic[1] = field[2, 0] / sqrt_h
        magnetic[2] = field[0, 1] / sqrt_h
        return magnetic

    def reconstruct_phi(
        self,
        h: Array,
        longitudinal_A: Array,
        longitudinal_B: Array,
        target_charge_eulerian: Array,
    ) -> tuple[Array, Array]:
        sqrt_h = inverse_metric(h)[2]
        right_A = longitudinal_A / sqrt_h
        right_B = longitudinal_B / sqrt_h + target_charge_eulerian / self.parameters.Z_B
        phi_A = self._K_inv[0, 0] * right_A + self._K_inv[0, 1] * right_B
        phi_B = self._K_inv[1, 0] * right_A + self._K_inv[1, 1] * right_B
        return phi_A, phi_B

    def initial_longitudinal_charges(
        self, h: Array, pi_A: Array, pi_B: Array
    ) -> tuple[Array, Array]:
        """Translate constraint-satisfying initial electric data into Q_A,Q_B."""
        sqrt_h = inverse_metric(h)[2]
        electric_A = self.electric_field(h, pi_A, self.parameters.Z_A)
        electric_B = self.electric_field(h, pi_B, self.parameters.Z_B)
        divergence_A = self.grid.divergence(sqrt_h[None, ...] * electric_A) / sqrt_h
        divergence_B = self.grid.divergence(sqrt_h[None, ...] * electric_B) / sqrt_h
        return -sqrt_h * divergence_A, -sqrt_h * divergence_B

    def gauss_constraints(
        self,
        h: Array,
        pi_A: Array,
        pi_B: Array,
        longitudinal_A: Array,
        longitudinal_B: Array,
    ) -> tuple[Array, Array]:
        sqrt_h = inverse_metric(h)[2]
        electric_A = self.electric_field(h, pi_A, self.parameters.Z_A)
        electric_B = self.electric_field(h, pi_B, self.parameters.Z_B)
        div_A = self.grid.divergence(sqrt_h[None, ...] * electric_A) / sqrt_h
        div_B = self.grid.divergence(sqrt_h[None, ...] * electric_B) / sqrt_h
        return (
            div_A + longitudinal_A / sqrt_h,
            div_B + longitudinal_B / sqrt_h,
        )

    def _single_stress(
        self,
        h: Array,
        potential: Array,
        pi: Array,
        phi: Array,
        Z: float,
        mass_squared: float,
    ) -> StressEnergy3p1:
        h_inv, _, sqrt_h = inverse_metric(h)
        electric_up = -pi / (Z * sqrt_h)[None, ...]
        electric_down = np.einsum("ij...,j...->i...", h, electric_up)
        potential_up = np.einsum("ij...,j...->i...", h_inv, potential)
        field_lower = self.field_strength(potential)
        field_up = np.einsum(
            "ik...,jl...,kl...->ij...", h_inv, h_inv, field_lower
        )
        electric_sq = np.einsum("i...,i...->...", electric_down, electric_up)
        field_sq = np.einsum("ij...,ij...->...", field_lower, field_up)
        potential_sq = np.einsum("i...,i...->...", potential, potential_up)
        rho = Z * (
            0.5 * electric_sq
            + 0.25 * field_sq
            + 0.5 * mass_squared * (phi**2 + potential_sq)
        )
        momentum = Z * np.einsum(
            "j...,ij...->i...", electric_up, field_lower
        )
        momentum += Z * mass_squared * phi[None, ...] * potential
        magnetic_tensor = np.einsum(
            "ik...,kl...,jl...->ij...", field_lower, h_inv, field_lower
        )
        stress = Z * (
            -electric_down[:, None, ...] * electric_down[None, :, ...]
            + magnetic_tensor
            + h * (0.5 * electric_sq - 0.25 * field_sq)[None, None, ...]
            + mass_squared
            * (
                potential[:, None, ...] * potential[None, :, ...]
                - 0.5
                * h
                * (potential_sq - phi**2)[None, None, ...]
            )
        )
        trace = np.einsum("ij...,ij...->...", h_inv, stress)
        return StressEnergy3p1(rho, momentum, stress, trace)

    def vector_stress(
        self,
        h: Array,
        a: Array,
        pi_A: Array,
        b: Array,
        pi_B: Array,
        longitudinal_A: Array,
        longitudinal_B: Array,
        target_charge_eulerian: Array,
    ) -> StressEnergy3p1:
        p = self.parameters
        phi_A, phi_B = self.reconstruct_phi(
            h, longitudinal_A, longitudinal_B, target_charge_eulerian
        )
        stress_A = self._single_stress(
            h, a, pi_A, phi_A, p.Z_A, p.mass_A**2
        )
        stress_B = self._single_stress(
            h, b, pi_B, phi_B, p.Z_B, p.mass_B**2
        )
        h_inv = inverse_metric(h)[0]
        difference = a - b
        difference_up = np.einsum("ij...,j...->i...", h_inv, difference)
        difference_sq = np.einsum(
            "i...,i...->...", difference, difference_up
        )
        phi_difference = phi_A - phi_B
        mixing_rho = 0.5 * p.mixing_mass_squared * (
            phi_difference**2 + difference_sq
        )
        mixing_momentum = (
            p.mixing_mass_squared
            * phi_difference[None, ...]
            * difference
        )
        mixing_stress = p.mixing_mass_squared * (
            difference[:, None, ...] * difference[None, :, ...]
            - 0.5
            * h
            * (difference_sq - phi_difference**2)[None, None, ...]
        )
        mixing = StressEnergy3p1(
            mixing_rho,
            mixing_momentum,
            mixing_stress,
            np.einsum("ij...,ij...->...", h_inv, mixing_stress),
        )
        return _combine_stress(_combine_stress(stress_A, stress_B), mixing)

    def target_current(
        self,
        h: Array,
        target_charge_density: Array,
        relative_current_density: Array,
        fluid: FluidPrimitive,
    ) -> tuple[Array, Array]:
        """Return Eulerian rho_D and spatial j_D^i.

        The evolved relative current is densitized.  Its equilibrium is tied
        algebraically to matter; no fluid derivative enters the field source.
        """
        sqrt_h = inverse_metric(h)[2]
        rho = target_charge_density / sqrt_h
        relative = relative_current_density / sqrt_h[None, ...]
        velocity_lower = np.einsum("ij...,j...->i...", h, fluid.velocity)
        relative_normal = np.einsum(
            "i...,i...->...", velocity_lower, relative
        )
        # J=q u+R with u.R=0.  If r^i is the Eulerian spatial projection of
        # R, then rho_D=qW+v_i r^i and j_D^i=qWv^i+r^i.
        spatial = (
            (rho - relative_normal)[None, ...] * fluid.velocity + relative
        )
        return rho, spatial

    def target_comoving_decomposition(
        self,
        h: Array,
        target_charge_density: Array,
        relative_current_density: Array,
        fluid: FluidPrimitive,
    ) -> tuple[Array, Array, Array]:
        """Return comoving q_D, Eulerian r^i, and the u.R residual."""
        sqrt_h = inverse_metric(h)[2]
        rho = target_charge_density / sqrt_h
        relative = relative_current_density / sqrt_h[None, ...]
        velocity_lower = np.einsum("ij...,j...->i...", h, fluid.velocity)
        relative_normal = np.einsum(
            "i...,i...->...", velocity_lower, relative
        )
        W = lorentz_factor(h, fluid.velocity)
        comoving_charge = (rho - relative_normal) / W
        # R^mu=(v_i r^i)n^mu+r^mu, so u_mu R^mu vanishes algebraically.
        orthogonality = W * (-relative_normal + relative_normal)
        return comoving_charge, relative, orthogonality

    def equilibrium_relative_current(
        self, h: Array, fluid: FluidPrimitive
    ) -> Array:
        h_inv, _, sqrt_h = inverse_metric(h)
        direction_lower = np.asarray(
            self.parameters.target_direction, dtype=float
        ).reshape((3,) + (1,) * self.grid.ndim)
        direction_up = np.einsum("ij...,j...->i...", h_inv, direction_lower)
        norm = np.sqrt(
            np.maximum(
                np.einsum("ij...,i...,j...->...", h, direction_up, direction_up),
                1.0e-300,
            )
        )
        direction_up = direction_up / norm[None, ...]
        return (
            sqrt_h[None, ...]
            * self.parameters.target_strength
            * fluid.baryon_density[None, ...]
            * direction_up
        )

    def drive_exchange(
        self,
        h: Array,
        b: Array,
        pi_B: Array,
        target_charge_eulerian: Array,
        target_current_up: Array,
    ) -> tuple[Array, Array]:
        """Matter-frame 3+1 projections of +H^mu_nu J_D^nu.

        Returns Eulerian power ``-n_mu Q^mu`` and covariant spatial force.
        A positive stationary charge therefore receives ``+rho_D E_B_i``.
        """
        electric_up = self.electric_field(h, pi_B, self.parameters.Z_B)
        electric_down = np.einsum("ij...,j...->i...", h, electric_up)
        field_lower = self.field_strength(b)
        power = np.einsum("i...,i...->...", electric_down, target_current_up)
        force = target_charge_eulerian[None, ...] * electric_down
        force += np.einsum("ij...,j...->i...", field_lower, target_current_up)
        return power, force

    def rhs_fields(
        self,
        h: Array,
        a: Array,
        pi_A: Array,
        b: Array,
        pi_B: Array,
        longitudinal_A: Array,
        longitudinal_B: Array,
        cleaning_A: Array,
        cleaning_B: Array,
        target_charge_eulerian: Array,
        target_current_up: Array,
        lapse: Array,
        shift: Array,
        cleaning_damping: float,
    ) -> tuple[Array, ...]:
        if cleaning_damping < 0.0:
            raise ValueError("cleaning_damping must be nonnegative")
        p = self.parameters
        h_inv, _, sqrt_h = inverse_metric(h)
        phi_A, phi_B = self.reconstruct_phi(
            h, longitudinal_A, longitudinal_B, target_charge_eulerian
        )

        def potential_rhs(potential: Array, pi: Array, phi: Array, Z: float) -> Array:
            pi_lower = np.einsum("ij...,j...->i...", h, pi)
            result = lapse[None, ...] * pi_lower / (Z * sqrt_h)[None, ...]
            result -= self.grid.gradient(lapse * phi)
            for i in range(3):
                for j in range(self.grid.ndim):
                    result[i] += shift[j] * self.grid.derivative(potential[i], j)
                    if i < self.grid.ndim:
                        result[i] += potential[j] * self.grid.derivative(shift[j], i)
            return result

        da = potential_rhs(a, pi_A, phi_A, p.Z_A)
        db = potential_rhs(b, pi_B, phi_B, p.Z_B)
        a_up = np.einsum("ij...,j...->i...", h_inv, a)
        b_up = np.einsum("ij...,j...->i...", h_inv, b)

        def momentum_curl(potential: Array, Z: float) -> Array:
            field_lower = self.field_strength(potential)
            field_up = np.einsum(
                "ik...,jl...,kl...->ij...", h_inv, h_inv, field_lower
            )
            flux = Z * lapse[None, None, ...] * sqrt_h[None, None, ...] * field_up
            result = self.grid.zeros((3,))
            for i in range(3):
                for j in range(self.grid.ndim):
                    result[i] += self.grid.derivative(flux[j, i], j)
            return result

        dpi_A = momentum_curl(a, p.Z_A)
        dpi_B = momentum_curl(b, p.Z_B)
        mixing_up = a_up - b_up
        dpi_A -= lapse[None, ...] * sqrt_h[None, ...] * (
            p.Z_A * p.mass_A**2 * a_up + p.mixing_mass_squared * mixing_up
        )
        dpi_B -= lapse[None, ...] * sqrt_h[None, ...] * (
            p.Z_B * p.mass_B**2 * b_up - p.mixing_mass_squared * mixing_up
            - target_current_up
        )

        for dpi, source_pi, Z, cleaning in (
            (dpi_A, pi_A, p.Z_A, cleaning_A),
            (dpi_B, pi_B, p.Z_B, cleaning_B),
        ):
            gradient = self.grid.gradient(lapse * cleaning)
            dpi += Z * sqrt_h[None, ...] * np.einsum(
                "ij...,j...->i...", h_inv, gradient
            )
            shift_divergence = self.grid.divergence(shift)
            for i in range(3):
                dpi[i] += source_pi[i] * shift_divergence
                for j in range(self.grid.ndim):
                    dpi[i] += shift[j] * self.grid.derivative(source_pi[i], j)
                    if i < self.grid.ndim:
                        dpi[i] -= source_pi[j] * self.grid.derivative(shift[i], j)

        charge_A = longitudinal_A / sqrt_h
        charge_B = longitudinal_B / sqrt_h
        current_A = self._K[0, 0] * a_up + self._K[0, 1] * b_up
        current_B = (
            self._K[1, 0] * a_up
            + self._K[1, 1] * b_up
            - target_current_up / p.Z_B
        )
        dlong_A = np.zeros_like(longitudinal_A)
        dlong_B = np.zeros_like(longitudinal_B)
        for axis in range(self.grid.ndim):
            flux_A = sqrt_h * (
                lapse * current_A[axis] - shift[axis] * charge_A
            )
            flux_B = sqrt_h * (
                lapse * current_B[axis] - shift[axis] * charge_B
            )
            dlong_A -= self.grid.derivative(flux_A, axis)
            dlong_B -= self.grid.derivative(flux_B, axis)

        constraint_A, constraint_B = self.gauss_constraints(
            h, pi_A, pi_B, longitudinal_A, longitudinal_B
        )
        dclean_A = -lapse * (
            constraint_A + cleaning_damping * cleaning_A
        )
        dclean_B = -lapse * (
            constraint_B + cleaning_damping * cleaning_B
        )
        for axis in range(self.grid.ndim):
            dclean_A += shift[axis] * self.grid.derivative(cleaning_A, axis)
            dclean_B += shift[axis] * self.grid.derivative(cleaning_B, axis)
        return (
            da,
            dpi_A,
            db,
            dpi_B,
            dlong_A,
            dlong_B,
            dclean_A,
            dclean_B,
        )

    def _cross(self, h: Array, first: Array, second: Array) -> Array:
        _, _, sqrt_h = inverse_metric(h)
        first_down = np.einsum("ij...,j...->i...", h, first)
        second_down = np.einsum("ij...,j...->i...", h, second)
        result = self.grid.zeros((3,))
        result[0] = (first_down[1] * second_down[2] - first_down[2] * second_down[1]) / sqrt_h
        result[1] = (first_down[2] * second_down[0] - first_down[0] * second_down[2]) / sqrt_h
        result[2] = (first_down[0] * second_down[1] - first_down[1] * second_down[0]) / sqrt_h
        return result

    def _fluid_electric(
        self, h: Array, electric: Array, magnetic: Array, velocity: Array
    ) -> Array:
        W = lorentz_factor(h, velocity)
        dot = np.einsum("ij...,i...,j...->...", h, velocity, electric)
        return (
            W[None, ...] * (electric + self._cross(h, velocity, magnetic))
            - (W**2 / (W + 1.0))[None, ...] * velocity * dot[None, ...]
        )

    def _electric_from_fluid(
        self,
        h: Array,
        fluid_electric: Array,
        magnetic: Array,
        velocity: Array,
    ) -> Array:
        W = lorentz_factor(h, velocity)
        offset = W[None, ...] * self._cross(h, velocity, magnetic)
        value = fluid_electric - offset
        dot = np.einsum("ij...,i...,j...->...", h, velocity, value)
        return value / W[None, ...] + (
            W / (W + 1.0)
        )[None, ...] * velocity * dot[None, ...]

    def damping_step(
        self,
        h: Array,
        a: Array,
        pi_A: Array,
        b: Array,
        pi_B: Array,
        longitudinal_A: Array,
        longitudinal_B: Array,
        fluid: FluidPrimitive,
        lapse: Array,
        dt: float,
    ) -> DampingResult:
        """Exact frozen-fluid electric damping with compatible Gauss charges."""
        p = self.parameters
        if dt < 0.0:
            raise ValueError("dt must be nonnegative")
        if dt == 0.0 or (p.gamma_W == 0.0 and p.gamma_B == 0.0):
            zero = np.zeros(self.grid.shape)
            return DampingResult(
                pi_A.copy(), pi_B.copy(), longitudinal_A.copy(),
                longitudinal_B.copy(), zero, zero,
            )
        _, _, sqrt_h = inverse_metric(h)
        W = lorentz_factor(h, fluid.velocity)
        proper_dt = lapse * dt / W
        electric_A = self.electric_field(h, pi_A, p.Z_A)
        electric_B = self.electric_field(h, pi_B, p.Z_B)
        magnetic_A = self.magnetic_field(h, a)
        magnetic_B = self.magnetic_field(h, b)
        fluid_A = self._fluid_electric(
            h, electric_A, magnetic_A, fluid.velocity
        )
        fluid_B = self._fluid_electric(
            h, electric_B, magnetic_B, fluid.velocity
        )
        matrix = np.array(
            [
                [-p.gamma_W / p.Z_A, p.gamma_W / p.Z_A],
                [p.gamma_W / p.Z_B, -(p.gamma_W + p.gamma_B) / p.Z_B],
            ],
            dtype=float,
        )
        stacked = np.stack((fluid_A, fluid_B), axis=0)
        damped = apply_constant_2x2_exponential(
            matrix, stacked, proper_dt[None, ...]
        )
        damped_A, damped_B = damped[0], damped[1]
        old_quadratic = 0.5 * p.Z_A * np.einsum(
            "ij...,i...,j...->...", h, fluid_A, fluid_A
        ) + 0.5 * p.Z_B * np.einsum(
            "ij...,i...,j...->...", h, fluid_B, fluid_B
        )
        new_quadratic = 0.5 * p.Z_A * np.einsum(
            "ij...,i...,j...->...", h, damped_A, damped_A
        ) + 0.5 * p.Z_B * np.einsum(
            "ij...,i...,j...->...", h, damped_B, damped_B
        )
        heat = np.maximum(old_quadratic - new_quadratic, 0.0)
        updated_A = self._electric_from_fluid(
            h, damped_A, magnetic_A, fluid.velocity
        )
        updated_B = self._electric_from_fluid(
            h, damped_B, magnetic_B, fluid.velocity
        )
        new_pi_A = -p.Z_A * sqrt_h[None, ...] * updated_A
        new_pi_B = -p.Z_B * sqrt_h[None, ...] * updated_B
        delta_A = updated_A - electric_A
        delta_B = updated_B - electric_B
        new_long_A = longitudinal_A - self.grid.divergence(
            sqrt_h[None, ...] * delta_A
        )
        new_long_B = longitudinal_B - self.grid.divergence(
            sqrt_h[None, ...] * delta_B
        )
        return DampingResult(
            new_pi_A,
            new_pi_B,
            new_long_A,
            new_long_B,
            heat,
            new_quadratic - old_quadratic,
        )
