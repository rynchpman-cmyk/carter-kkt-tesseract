"""Proca Hamilton equations and stress tensor on an ADM slice."""

from __future__ import annotations

import numpy as np

from .adm import inverse_metric
from .grid import Array, PeriodicGrid
from .proca import ProcaParameters
from .matter import FluidPrimitive
from .theory2 import Theory2Parameters, source_data


class CurvedProcaSystem:
    def __init__(self, grid: PeriodicGrid, parameters: ProcaParameters | None = None) -> None:
        self.grid = grid
        self.parameters = parameters or ProcaParameters()

    def phi(self, pi: Array, sqrt_h: Array) -> Array:
        p = self.parameters
        # pi^i is a weight-one density, so its covariant divergence is partial_i pi^i.
        return self.grid.divergence(pi) / (p.Z * p.mass**2 * sqrt_h)

    def field_strength(self, a: Array) -> Array:
        F = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i < self.grid.ndim:
                    F[i, j] += self.grid.derivative(a[j], i)
                if j < self.grid.ndim:
                    F[i, j] -= self.grid.derivative(a[i], j)
        return F

    def rhs(
        self,
        h: Array,
        a: Array,
        pi: Array,
        lapse: Array,
        shift: Array,
    ) -> tuple[Array, Array]:
        p = self.parameters
        h_inv, _, sqrt_h = inverse_metric(h)
        phi = self.phi(pi, sqrt_h)
        pi_lower = np.einsum("ij...,j...->i...", h, pi)
        da = lapse[None, ...] * pi_lower / (p.Z * sqrt_h)[None, ...]
        da -= self.grid.gradient(lapse * phi)

        # Lie derivative of a covector.
        for i in range(3):
            for j in range(self.grid.ndim):
                da[i] += shift[j] * self.grid.derivative(a[i], j)
                if i < self.grid.ndim:
                    da[i] += a[j] * self.grid.derivative(shift[j], i)

        F_lower = self.field_strength(a)
        F_up = np.einsum("ik...,jl...,kl...->ij...", h_inv, h_inv, F_lower)
        flux = p.Z * lapse[None, None, ...] * sqrt_h[None, None, ...] * F_up
        dpi = self.grid.zeros((3,))
        for i in range(3):
            for j in range(self.grid.ndim):
                dpi[i] += self.grid.derivative(flux[j, i], j)
        a_up = np.einsum("ij...,j...->i...", h_inv, a)
        dpi -= p.Z * p.mass**2 * lapse[None, ...] * sqrt_h[None, ...] * a_up

        # Lie derivative of a contravariant vector density of weight one.
        shift_divergence = self.grid.divergence(shift)
        for i in range(3):
            dpi[i] += pi[i] * shift_divergence
            for j in range(self.grid.ndim):
                dpi[i] += shift[j] * self.grid.derivative(pi[i], j)
                if i < self.grid.ndim:
                    dpi[i] -= pi[j] * self.grid.derivative(shift[i], j)
        return da, dpi

    def stress_energy(self, h: Array, a: Array, pi: Array) -> tuple[Array, Array, Array]:
        p = self.parameters
        h_inv, _, sqrt_h = inverse_metric(h)
        phi = self.phi(pi, sqrt_h)
        # Theory 2.0 convention: E_i = F_{i mu} n^mu = -pi_i/(Z sqrt(h)).
        electric_up = -pi / (p.Z * sqrt_h)[None, ...]
        electric_down = np.einsum("ij...,j...->i...", h, electric_up)
        a_up = np.einsum("ij...,j...->i...", h_inv, a)
        F_lower = self.field_strength(a)
        F_up = np.einsum("ik...,jl...,kl...->ij...", h_inv, h_inv, F_lower)
        e2 = np.einsum("i...,i...->...", electric_down, electric_up)
        f2 = np.einsum("ij...,ij...->...", F_lower, F_up)
        a2 = np.einsum("i...,i...->...", a, a_up)
        rho = p.Z * (0.5 * e2 + 0.25 * f2 + 0.5 * p.mass**2 * (phi**2 + a2))
        momentum = p.Z * np.einsum("j...,ij...->i...", electric_up, F_lower)
        momentum += p.Z * p.mass**2 * phi[None, ...] * a

        stress = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                magnetic_tensor = np.zeros(self.grid.shape)
                for k in range(3):
                    magnetic_tensor += F_lower[i, k] * np.einsum(
                        "l...,l...->...", h_inv[k], F_lower[j]
                    )
                stress[i, j] = p.Z * (
                    -electric_down[i] * electric_down[j]
                    + magnetic_tensor
                    + h[i, j] * (0.5 * e2 - 0.25 * f2)
                    + p.mass**2
                    * (a[i] * a[j] - 0.5 * h[i, j] * (a2 - phi**2))
                )
        return rho, momentum, stress


class Theory2ProcaSystem(CurvedProcaSystem):
    """Density-driven Proca system with the quadratic Theory 2.0 interaction."""

    def __init__(self, grid: PeriodicGrid, parameters: Theory2Parameters | None = None) -> None:
        self.theory_parameters = parameters or Theory2Parameters()
        super().__init__(
            grid,
            ProcaParameters(self.theory_parameters.Z, self.theory_parameters.mass),
        )

    def constrained_phi(
        self,
        h: Array,
        pi: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
    ) -> Array:
        _, _, sqrt_h = inverse_metric(h)
        sources = source_data(h, fluid, astar_lower, self.theory_parameters)
        divergence_E = -self.grid.divergence(pi) / (
            self.theory_parameters.Z * sqrt_h
        )
        return (sources.rho_star - divergence_E) / sources.effective_mass_squared

    def longitudinal_density(
        self,
        h: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
        phi: Array,
    ) -> Array:
        """Densitized generalized-Proca longitudinal charge.

        ``q_A = M_eff^2 Phi-rho_star`` obeys the exact divergence identity
        ``nabla_mu(M_eff^2 A^mu-J_star^mu)=0``.  Evolving ``sqrt(h) q_A``
        avoids differentiating the algebraic Gauss reconstruction through a
        discontinuous density field.
        """
        _, _, sqrt_h = inverse_metric(h)
        sources = source_data(h, fluid, astar_lower, self.theory_parameters)
        return sqrt_h * (
            sources.effective_mass_squared * phi - sources.rho_star
        )

    def phi_from_longitudinal(
        self,
        h: Array,
        longitudinal_density: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
    ) -> Array:
        _, _, sqrt_h = inverse_metric(h)
        sources = source_data(h, fluid, astar_lower, self.theory_parameters)
        charge = longitudinal_density / sqrt_h
        return (charge + sources.rho_star) / sources.effective_mass_squared

    def hyperbolic_gauss_constraint(
        self,
        h: Array,
        pi: Array,
        longitudinal_density: Array,
    ) -> Array:
        _, _, sqrt_h = inverse_metric(h)
        divergence_E = -self.grid.divergence(pi) / (
            self.theory_parameters.Z * sqrt_h
        )
        return divergence_E + longitudinal_density / sqrt_h

    def gauss_constraint(
        self,
        h: Array,
        a: Array,
        pi: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
        phi: Array | None = None,
    ) -> Array:
        del a
        _, _, sqrt_h = inverse_metric(h)
        sources = source_data(h, fluid, astar_lower, self.theory_parameters)
        phi = (
            self.constrained_phi(h, pi, fluid, astar_lower) if phi is None else phi
        )
        divergence_E = -self.grid.divergence(pi) / (
            self.theory_parameters.Z * sqrt_h
        )
        return divergence_E + sources.effective_mass_squared * phi - sources.rho_star


    def rhs_theory2(
        self,
        h: Array,
        a: Array,
        pi: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
        lapse: Array,
        shift: Array,
    ) -> tuple[Array, Array]:
        p = self.theory_parameters
        h_inv, _, sqrt_h = inverse_metric(h)
        sources = source_data(h, fluid, astar_lower, p)
        phi = self.constrained_phi(h, pi, fluid, astar_lower)
        pi_lower = np.einsum("ij...,j...->i...", h, pi)
        da = lapse[None, ...] * pi_lower / (p.Z * sqrt_h)[None, ...]
        da -= self.grid.gradient(lapse * phi)
        for i in range(3):
            for j in range(self.grid.ndim):
                da[i] += shift[j] * self.grid.derivative(a[i], j)
                if i < self.grid.ndim:
                    da[i] += a[j] * self.grid.derivative(shift[j], i)

        F_lower = self.field_strength(a)
        F_up = np.einsum("ik...,jl...,kl...->ij...", h_inv, h_inv, F_lower)
        flux = p.Z * lapse[None, None, ...] * sqrt_h[None, None, ...] * F_up
        dpi = self.grid.zeros((3,))
        for i in range(3):
            for j in range(self.grid.ndim):
                dpi[i] += self.grid.derivative(flux[j, i], j)
        a_up = np.einsum("ij...,j...->i...", h_inv, a)
        astar_up = np.einsum("ij...,j...->i...", h_inv, astar_lower)
        dpi -= (
            p.Z
            * sources.effective_mass_squared[None, ...]
            * lapse[None, ...]
            * sqrt_h[None, ...]
            * a_up
        )
        dpi += (
            sources.target_coupling[None, ...]
            * lapse[None, ...]
            * sqrt_h[None, ...]
            * astar_up
        )

        shift_divergence = self.grid.divergence(shift)
        for i in range(3):
            dpi[i] += pi[i] * shift_divergence
            for j in range(self.grid.ndim):
                dpi[i] += shift[j] * self.grid.derivative(pi[i], j)
                if i < self.grid.ndim:
                    dpi[i] -= pi[j] * self.grid.derivative(shift[i], j)
        return da, dpi

    def rhs_hyperbolic(
        self,
        h: Array,
        a: Array,
        pi: Array,
        longitudinal_density: Array,
        gauss_cleaning: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
        lapse: Array,
        shift: Array,
        cleaning_damping: float = 1.0,
    ) -> tuple[Array, Array, Array, Array]:
        """Evolution without algebraically eliminating Phi through div(E).

        The longitudinal density is the time component of
        ``M_eff^2 A^mu-J_star^mu`` and is evolved conservatively.  The cleaning
        scalar propagates violations of ``div(E)+q_A=0`` and damps them.
        """
        if cleaning_damping < 0.0:
            raise ValueError("cleaning_damping must be nonnegative")
        p = self.theory_parameters
        h_inv, _, sqrt_h = inverse_metric(h)
        sources = source_data(h, fluid, astar_lower, p)
        phi = self.phi_from_longitudinal(
            h, longitudinal_density, fluid, astar_lower
        )
        pi_lower = np.einsum("ij...,j...->i...", h, pi)
        da = lapse[None, ...] * pi_lower / (p.Z * sqrt_h)[None, ...]
        da -= self.grid.gradient(lapse * phi)
        for i in range(3):
            for j in range(self.grid.ndim):
                da[i] += shift[j] * self.grid.derivative(a[i], j)
                if i < self.grid.ndim:
                    da[i] += a[j] * self.grid.derivative(shift[j], i)

        F_lower = self.field_strength(a)
        F_up = np.einsum("ik...,jl...,kl...->ij...", h_inv, h_inv, F_lower)
        flux = p.Z * lapse[None, None, ...] * sqrt_h[None, None, ...] * F_up
        dpi = self.grid.zeros((3,))
        for i in range(3):
            for j in range(self.grid.ndim):
                dpi[i] += self.grid.derivative(flux[j, i], j)
        a_up = np.einsum("ij...,j...->i...", h_inv, a)
        astar_up = np.einsum("ij...,j...->i...", h_inv, astar_lower)
        dpi -= (
            p.Z * sources.effective_mass_squared[None, ...]
            * lapse[None, ...] * sqrt_h[None, ...] * a_up
        )
        dpi += (
            sources.target_coupling[None, ...]
            * lapse[None, ...] * sqrt_h[None, ...] * astar_up
        )
        # E^i=-pi^i/(Z sqrt(h)); this sign launches the cleaning constraint
        # as a damped wave rather than an exponentially growing mode.
        cleaning_gradient = self.grid.gradient(lapse * gauss_cleaning)
        dpi += p.Z * sqrt_h[None, ...] * np.einsum(
            "ij...,j...->i...", h_inv, cleaning_gradient
        )

        shift_divergence = self.grid.divergence(shift)
        for i in range(3):
            dpi[i] += pi[i] * shift_divergence
            for j in range(self.grid.ndim):
                dpi[i] += shift[j] * self.grid.derivative(pi[i], j)
                if i < self.grid.ndim:
                    dpi[i] -= pi[j] * self.grid.derivative(shift[i], j)

        charge = longitudinal_density / sqrt_h
        current_up = (
            sources.effective_mass_squared[None, ...] * a_up
            - (sources.target_coupling / p.Z)[None, ...] * astar_up
        )
        longitudinal_flux = sqrt_h[None, ...] * (
            lapse[None, ...] * current_up - shift * charge[None, ...]
        )
        dlongitudinal = np.zeros_like(longitudinal_density)
        for axis in range(self.grid.ndim):
            # Use the same derivative family as div(pi), so the discrete
            # divergence identity preserves Gauss to roundoff.  The equation
            # is still conservative on periodic grids and, critically, never
            # differentiates a ratio containing M_eff.
            dlongitudinal -= self.grid.derivative(
                longitudinal_flux[axis], axis
            )

        constraint = self.hyperbolic_gauss_constraint(
            h, pi, longitudinal_density
        )
        dcleaning = -lapse * constraint - lapse * cleaning_damping * gauss_cleaning
        for axis in range(self.grid.ndim):
            dcleaning += shift[axis] * self.grid.derivative(gauss_cleaning, axis)
        return da, dpi, dlongitudinal, dcleaning

    def stress_energy_theory2(
        self,
        h: Array,
        a: Array,
        pi: Array,
        fluid: FluidPrimitive,
        astar_lower: Array,
        phi: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Bare Proca stress only; add interaction stress separately."""
        p = self.theory_parameters
        h_inv, _, sqrt_h = inverse_metric(h)
        phi = (
            self.constrained_phi(h, pi, fluid, astar_lower)
            if phi is None else phi
        )
        electric_up = -pi / (p.Z * sqrt_h)[None, ...]
        electric_down = np.einsum("ij...,j...->i...", h, electric_up)
        a_up = np.einsum("ij...,j...->i...", h_inv, a)
        F_lower = self.field_strength(a)
        F_up = np.einsum("ik...,jl...,kl...->ij...", h_inv, h_inv, F_lower)
        e2 = np.einsum("i...,i...->...", electric_down, electric_up)
        f2 = np.einsum("ij...,ij...->...", F_lower, F_up)
        a2 = np.einsum("i...,i...->...", a, a_up)
        rho = p.Z * (0.5 * e2 + 0.25 * f2 + 0.5 * p.mass**2 * (phi**2 + a2))
        momentum = p.Z * np.einsum("j...,ij...->i...", electric_up, F_lower)
        momentum += p.Z * p.mass**2 * phi[None, ...] * a
        stress = self.grid.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                magnetic_tensor = np.zeros(self.grid.shape)
                for k in range(3):
                    magnetic_tensor += F_lower[i, k] * np.einsum(
                        "l...,l...->...", h_inv[k], F_lower[j]
                    )
                stress[i, j] = p.Z * (
                    -electric_down[i] * electric_down[j]
                    + magnetic_tensor
                    + h[i, j] * (0.5 * e2 - 0.25 * f2)
                    + p.mass**2
                    * (a[i] * a[j] - 0.5 * h[i, j] * (a2 - phi**2))
                )
        return rho, momentum, stress
