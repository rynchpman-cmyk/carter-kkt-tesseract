"""Constrained spectral apparent-horizon diagnostics.

This module deliberately keeps the KKT evolution out of the spacetime right-
hand side.  It is a pseudo-time optimization used to locate marginally outer
trapped surfaces (MOTSs) on an already constructed ADM slice.  Physical time
continues to be supplied exclusively by the spacetime evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, pi, sqrt
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .adm import ADMState, christoffel, inverse_metric
from .grid import Array, CartesianGrid
from .nr_diagnostics import sphere_quadrature, trilinear_sample

if TYPE_CHECKING:
    from .ccz4 import CCZ4Solver, CCZ4State


Mode = tuple[int, int]


def spectral_modes(maximum_degree: int, include_dipole: bool = False) -> tuple[Mode, ...]:
    """Return the real-harmonic modes used by a star-shaped surface.

    The monopole is represented by the constant function one so its
    coefficient is exactly the logarithm of the reference radius.  Dipoles
    are omitted by default because they mostly duplicate a change of center.
    """

    if maximum_degree < 0:
        raise ValueError("maximum_degree must be non-negative")
    modes: list[Mode] = [(0, 0)]
    for ell in range(1, maximum_degree + 1):
        if ell == 1 and not include_dipole:
            continue
        modes.extend((ell, m) for m in range(-ell, ell + 1))
    return tuple(modes)


def _associated_legendre(ell: int, order: int, cosine: Array) -> Array:
    """Evaluate the associated Legendre polynomial with Condon--Shortley phase."""

    if not 0 <= order <= ell:
        raise ValueError("associated-Legendre indices require 0 <= order <= ell")
    cosine = np.asarray(cosine, dtype=float)
    p_mm = np.ones_like(cosine)
    if order:
        factor = 1.0
        root = np.sqrt(np.maximum(1.0 - cosine * cosine, 0.0))
        for _ in range(1, order + 1):
            p_mm *= -factor * root
            factor += 2.0
    if ell == order:
        return p_mm
    p_m1_m = (2 * order + 1) * cosine * p_mm
    if ell == order + 1:
        return p_m1_m
    previous, current = p_mm, p_m1_m
    for degree in range(order + 2, ell + 1):
        following = (
            (2 * degree - 1) * cosine * current
            - (degree + order - 1) * previous
        ) / (degree - order)
        previous, current = current, following
    return current


def real_spherical_harmonic(mode: Mode, theta: Array, phi: Array) -> Array:
    """Evaluate an orthonormal real spherical harmonic.

    Positive ``m`` denotes the cosine mode and negative ``m`` the sine mode.
    ``(0, 0)`` is special-cased to one for a convenient log-radius monopole.
    """

    ell, m = mode
    if ell < 0 or abs(m) > ell:
        raise ValueError(f"invalid spherical-harmonic mode {mode}")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if mode == (0, 0):
        return np.ones(np.broadcast_shapes(theta.shape, phi.shape))
    order = abs(m)
    polynomial = _associated_legendre(ell, order, np.cos(theta))
    normalization = exp(
        0.5
        * (
            np.log((2 * ell + 1) / (4.0 * pi))
            + lgamma(ell - order + 1)
            - lgamma(ell + order + 1)
        )
    )
    if m == 0:
        return normalization * polynomial
    angular = np.cos(order * phi) if m > 0 else np.sin(order * phi)
    return sqrt(2.0) * normalization * polynomial * angular


def evaluate_basis(modes: Iterable[Mode], theta: Array, phi: Array) -> Array:
    """Evaluate a mode sequence with the mode index on the leading axis."""

    return np.stack(
        [real_spherical_harmonic(mode, theta, phi) for mode in modes], axis=0
    )


def _axisymmetric_basis_derivatives(
    modes: Iterable[Mode], theta: Array, phi: Array
) -> tuple[Array, Array, Array]:
    """Return axisymmetric basis values and first/second theta derivatives."""

    values: list[Array] = []
    first: list[Array] = []
    second: list[Array] = []
    cosine = np.cos(theta)
    sine = np.sin(theta)
    denominator = np.maximum(1.0 - cosine * cosine, 1.0e-30)
    for mode in modes:
        ell, m = mode
        if m != 0:
            raise ValueError("direct Brill-Lindquist surfaces require m=0 modes")
        if mode == (0, 0):
            shape = np.broadcast_shapes(np.shape(theta), np.shape(phi))
            values.append(np.ones(shape))
            first.append(np.zeros(shape))
            second.append(np.zeros(shape))
            continue
        polynomial = _associated_legendre(ell, 0, cosine)
        previous = _associated_legendre(ell - 1, 0, cosine)
        derivative_x = ell * (cosine * polynomial - previous) / (
            cosine * cosine - 1.0
        )
        derivative_xx = (
            2.0 * cosine * derivative_x - ell * (ell + 1) * polynomial
        ) / denominator
        normalization = sqrt((2 * ell + 1) / (4.0 * pi))
        values.append(normalization * polynomial)
        first.append(-normalization * sine * derivative_x)
        second.append(
            normalization
            * (-cosine * derivative_x + sine * sine * derivative_xx)
        )
    return np.stack(values), np.stack(first), np.stack(second)


def _surface_basis_derivatives(
    modes: Iterable[Mode], theta: Array, phi: Array
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return basis values and all first/second angular derivatives.

    The calibrated axisymmetric path retains its analytic Legendre
    derivatives.  General real harmonics use centered differentiation of the
    analytic basis evaluator on the surface coordinates; quadrature nodes do
    not include the coordinate poles, so this is regular for the supported
    star-shaped representation.
    """

    modes = tuple(modes)
    if all(m == 0 for _, m in modes):
        values, theta_first, theta_second = _axisymmetric_basis_derivatives(
            modes, theta, phi
        )
        zeros = np.zeros_like(values)
        return values, theta_first, zeros, theta_second, zeros, zeros

    step = 2.0e-4
    values = evaluate_basis(modes, theta, phi)
    theta_plus = evaluate_basis(modes, theta + step, phi)
    theta_minus = evaluate_basis(modes, theta - step, phi)
    phi_plus = evaluate_basis(modes, theta, phi + step)
    phi_minus = evaluate_basis(modes, theta, phi - step)
    theta_first = (theta_plus - theta_minus) / (2.0 * step)
    phi_first = (phi_plus - phi_minus) / (2.0 * step)
    theta_second = (theta_plus - 2.0 * values + theta_minus) / (step * step)
    phi_second = (phi_plus - 2.0 * values + phi_minus) / (step * step)
    mixed = (
        evaluate_basis(modes, theta + step, phi + step)
        - evaluate_basis(modes, theta + step, phi - step)
        - evaluate_basis(modes, theta - step, phi + step)
        + evaluate_basis(modes, theta - step, phi - step)
    ) / (4.0 * step * step)
    return (
        values,
        theta_first,
        phi_first,
        theta_second,
        mixed,
        phi_second,
    )


@dataclass(frozen=True)
class KKTIteration:
    iteration: int
    merit: float
    spectral_residual_norm: float
    rms_expansion: float
    maximum_expansion: float
    projected_gradient_norm: float
    step_norm: float
    line_search_factor: float
    active_constraints: int


@dataclass(frozen=True)
class SpectralMOTSParameters:
    """Numerical contract for the constrained surface solve."""

    maximum_degree: int = 4
    modes: tuple[Mode, ...] | None = None
    include_dipole: bool = False
    n_theta: int = 14
    n_phi: int = 28
    minimum_radius: float = 0.1
    maximum_radius: float = 10.0
    maximum_mode_amplitude: float = 0.8
    expansion_tolerance: float = 1.0e-4
    projected_gradient_tolerance: float = 1.0e-7
    jacobian_step: float = 2.0e-4
    gauss_newton_damping: float = 1.0e-8
    trust_radius: float = 0.35
    maximum_iterations: int = 40
    maximum_line_searches: int = 12
    armijo_fraction: float = 1.0e-4

    def __post_init__(self) -> None:
        modes = (
            spectral_modes(self.maximum_degree, self.include_dipole)
            if self.modes is None
            else tuple(self.modes)
        )
        if not modes or modes[0] != (0, 0) or len(set(modes)) != len(modes):
            raise ValueError("modes must be unique and begin with the monopole (0, 0)")
        if any(ell < 0 or abs(m) > ell for ell, m in modes):
            raise ValueError("invalid spherical-harmonic mode")
        if not self.include_dipole and any(ell == 1 for ell, _ in modes):
            raise ValueError("dipole modes require include_dipole=True")
        if self.n_theta < 4 or self.n_phi < 8:
            raise ValueError("angular quadrature is too small")
        if not 0.0 < self.minimum_radius < self.maximum_radius:
            raise ValueError("radius bounds must be positive and ordered")
        if self.maximum_mode_amplitude <= 0.0:
            raise ValueError("maximum_mode_amplitude must be positive")
        if self.expansion_tolerance <= 0.0 or self.jacobian_step <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.gauss_newton_damping < 0.0 or self.trust_radius <= 0.0:
            raise ValueError("damping must be non-negative and trust radius positive")
        if self.maximum_iterations < 1 or self.maximum_line_searches < 1:
            raise ValueError("iteration limits must be positive")
        object.__setattr__(self, "modes", modes)


@dataclass(frozen=True)
class SpectralMOTSResult:
    found: bool
    reason: str
    center: tuple[float, float, float]
    modes: tuple[Mode, ...]
    coefficients: Array
    coordinate_radius_mean: float
    coordinate_radius_minimum: float
    coordinate_radius_maximum: float
    area: float
    irreducible_mass: float
    mean_expansion: float
    spectral_residual_norm: float
    rms_expansion: float
    maximum_expansion: float
    merit: float
    iterations: int
    projected_gradient_norm: float
    lower_multipliers: Array
    upper_multipliers: Array
    complementarity_residual: float
    active_lower: tuple[int, ...]
    active_upper: tuple[int, ...]
    response_eigenvalues: Array
    response_condition_number: float
    history: tuple[KKTIteration, ...]

    def diagnostics(self) -> dict[str, object]:
        """Return restart/log-friendly scalar and list diagnostics."""

        return {
            "found": self.found,
            "reason": self.reason,
            "center": list(self.center),
            "modes": [list(mode) for mode in self.modes],
            "coefficients": self.coefficients.tolist(),
            "coordinate_radius_mean": self.coordinate_radius_mean,
            "coordinate_radius_minimum": self.coordinate_radius_minimum,
            "coordinate_radius_maximum": self.coordinate_radius_maximum,
            "area": self.area,
            "irreducible_mass": self.irreducible_mass,
            "mean_expansion": self.mean_expansion,
            "spectral_residual_norm": self.spectral_residual_norm,
            "rms_expansion": self.rms_expansion,
            "maximum_expansion": self.maximum_expansion,
            "merit": self.merit,
            "iterations": self.iterations,
            "projected_gradient_norm": self.projected_gradient_norm,
            "lower_multipliers": self.lower_multipliers.tolist(),
            "upper_multipliers": self.upper_multipliers.tolist(),
            "complementarity_residual": self.complementarity_residual,
            "active_lower": list(self.active_lower),
            "active_upper": list(self.active_upper),
            "response_eigenvalues_real": self.response_eigenvalues.real.tolist(),
            "response_eigenvalues_imag": self.response_eigenvalues.imag.tolist(),
            "response_condition_number": self.response_condition_number,
        }


@dataclass(frozen=True)
class _SurfaceEvaluation:
    valid: bool
    reason: str
    expansion: Array
    radii: Array
    area_weights: Array
    normal_covector_norm: Array


class _StarSurfaceEvaluator:
    def __init__(
        self,
        grid: CartesianGrid,
        state: ADMState,
        center: tuple[float, float, float],
        parameters: SpectralMOTSParameters,
    ) -> None:
        self.grid = grid
        self.state = state
        self.center = np.asarray(center, dtype=float)
        self.parameters = parameters
        self.modes = parameters.modes
        self.h_inv, _, self.sqrt_h = inverse_metric(state.h)
        self.trace_K = np.einsum("ij...,ij...->...", self.h_inv, state.K)

        coordinates = grid.coordinates()
        displacement = np.stack(
            [coordinates[axis] - self.center[axis] for axis in range(3)]
        )
        self.grid_radius = np.sqrt(np.sum(displacement * displacement, axis=0))
        safe_radius = np.maximum(self.grid_radius, 1.0e-14)
        cosine = np.clip(displacement[2] / safe_radius, -1.0, 1.0)
        theta = np.arccos(cosine)
        phi = np.mod(np.arctan2(displacement[1], displacement[0]), 2.0 * pi)
        self.grid_basis = evaluate_basis(self.modes, theta, phi)

        quadrature = sphere_quadrature(
            1.0,
            (0.0, 0.0, 0.0),
            parameters.n_theta,
            parameters.n_phi,
        )
        self.directions = quadrature.normals
        self.weights = quadrature.weights
        self.surface_basis = evaluate_basis(
            self.modes, quadrature.theta, quadrature.phi
        )

    def _inside_patch(self, points: Array) -> bool:
        for axis in range(3):
            lower = self.grid.origin[axis] + 0.5 * self.grid.spacing[axis]
            upper = (
                self.grid.origin[axis]
                + self.grid.lengths[axis]
                - 0.5 * self.grid.spacing[axis]
            )
            if np.any(points[:, axis] < lower) or np.any(points[:, axis] > upper):
                return False
        return True

    def evaluate(self, coefficients: Array) -> _SurfaceEvaluation:
        coefficients = np.asarray(coefficients, dtype=float)
        log_radius_grid = np.einsum("m...,m->...", self.grid_basis, coefficients)
        if not np.all(np.isfinite(log_radius_grid)):
            return _SurfaceEvaluation(False, "non-finite surface", np.array([]), np.array([]), np.array([]), np.array([]))
        directional_radius = np.exp(np.clip(log_radius_grid, -50.0, 50.0))
        level_set = self.grid_radius / directional_radius - 1.0
        gradient = np.stack(
            [self.grid.derivative(level_set, axis) for axis in range(3)]
        )
        normal_norm = np.sqrt(
            np.maximum(
                np.einsum("ij...,i...,j...->...", self.h_inv, gradient, gradient),
                1.0e-30,
            )
        )
        normal_lower = gradient / normal_norm[None, ...]
        normal_upper = np.einsum("ij...,j...->i...", self.h_inv, normal_lower)
        divergence = self.grid.zeros()
        for axis in range(3):
            divergence += self.grid.derivative(
                self.sqrt_h * normal_upper[axis], axis
            ) / self.sqrt_h
        K_normal = np.einsum(
            "ij...,i...,j...->...", self.state.K, normal_upper, normal_upper
        )
        expansion_field = divergence + K_normal - self.trace_K

        surface_log_radius = np.einsum("mn,m->n", self.surface_basis, coefficients)
        radii = np.exp(surface_log_radius)
        if (
            np.min(radii) < self.parameters.minimum_radius
            or np.max(radii) > self.parameters.maximum_radius
        ):
            return _SurfaceEvaluation(False, "surface radius left admissible bounds", np.array([]), radii, np.array([]), np.array([]))
        points = self.center[None, :] + radii[:, None] * self.directions
        if not self._inside_patch(points):
            return _SurfaceEvaluation(False, "surface left interpolation patch", np.array([]), radii, np.array([]), np.array([]))
        expansion = trilinear_sample(self.grid, expansion_field, points)

        metric = np.moveaxis(trilinear_sample(self.grid, self.state.h, points), -1, 0)
        determinant = np.linalg.det(metric)
        sampled_normal_norm = trilinear_sample(self.grid, normal_norm, points)
        # F=r/h(Omega)-1 has partial_r F=1/h on the surface.  The coarea
        # formula therefore gives dA/dOmega=sqrt(det h) r^2 |dF|_h h.
        area_density = (
            np.sqrt(np.maximum(determinant, 0.0))
            * radii**3
            * sampled_normal_norm
        )
        area_weights = self.weights * area_density
        if not np.all(np.isfinite(expansion)) or not np.all(area_weights > 0.0):
            return _SurfaceEvaluation(False, "non-finite surface geometry", expansion, radii, area_weights, sampled_normal_norm)
        return _SurfaceEvaluation(True, "ok", expansion, radii, area_weights, sampled_normal_norm)


class _ParametricAxisymmetricSurfaceEvaluator:
    """Evaluate ``Theta_+`` from a general parametric radial graph.

    Unlike :class:`_StarSurfaceEvaluator`, this evaluator never differentiates
    a volume extension of the surface normal.  It differentiates the
    parametric embedding analytically and samples only ``gamma_ij``, ``K_ij``,
    and the Cartesian connection from the supplied ADM slice.  The resulting
    observer is therefore suitable for dynamical Cartesian snapshots while
    retaining the physical-normal stability normalization. The axisymmetric
    basis keeps analytic derivatives; general real harmonics use surface-local
    centered derivatives of the analytic harmonic evaluator.
    """

    def __init__(
        self,
        grid: CartesianGrid,
        state: ADMState,
        center: tuple[float, float, float],
        parameters: SpectralMOTSParameters,
        connection: Array | None = None,
    ) -> None:
        self.grid = grid
        self.state = state
        self.center = np.asarray(center, dtype=float)
        self.parameters = parameters
        self.modes = parameters.modes

        if connection is None:
            h_inv = inverse_metric(state.h)[0]
            self.connection = christoffel(grid, state.h, h_inv)
        else:
            self.connection = np.asarray(connection, dtype=float)
            if self.connection.shape != (3, 3, 3) + grid.shape:
                raise ValueError("precomputed Cartesian connection has wrong shape")
        quadrature = sphere_quadrature(
            1.0,
            (0.0, 0.0, 0.0),
            parameters.n_theta,
            parameters.n_phi,
        )
        self.directions = quadrature.normals
        self.weights = quadrature.weights
        self.theta = quadrature.theta
        self.phi = quadrature.phi
        (
            self.surface_basis,
            self.surface_basis_theta,
            self.surface_basis_phi,
            self.surface_basis_theta2,
            self.surface_basis_theta_phi,
            self.surface_basis_phi2,
        ) = _surface_basis_derivatives(
            self.modes, quadrature.theta, quadrature.phi
        )

        sine = np.sin(self.theta)
        cosine = np.cos(self.theta)
        cosine_phi = np.cos(self.phi)
        sine_phi = np.sin(self.phi)
        self.sine = sine
        self.cosine = cosine
        self.theta_direction = np.stack(
            (cosine * cosine_phi, cosine * sine_phi, -sine), axis=1
        )
        self.phi_direction = np.stack((-sine_phi, cosine_phi, np.zeros_like(sine)), axis=1)

    def _inside_patch(self, points: Array) -> bool:
        for axis in range(3):
            lower = self.grid.origin[axis] + 0.5 * self.grid.spacing[axis]
            upper = (
                self.grid.origin[axis]
                + self.grid.lengths[axis]
                - 0.5 * self.grid.spacing[axis]
            )
            if np.any(points[:, axis] < lower) or np.any(points[:, axis] > upper):
                return False
        return True

    def evaluate(self, coefficients: Array) -> _SurfaceEvaluation:
        coefficients = np.asarray(coefficients, dtype=float)
        log_radius = np.einsum("mn,m->n", self.surface_basis, coefficients)
        log_radius_theta = np.einsum(
            "mn,m->n", self.surface_basis_theta, coefficients
        )
        log_radius_theta2 = np.einsum(
            "mn,m->n", self.surface_basis_theta2, coefficients
        )
        log_radius_phi = np.einsum(
            "mn,m->n", self.surface_basis_phi, coefficients
        )
        log_radius_theta_phi = np.einsum(
            "mn,m->n", self.surface_basis_theta_phi, coefficients
        )
        log_radius_phi2 = np.einsum(
            "mn,m->n", self.surface_basis_phi2, coefficients
        )
        radii = np.exp(np.clip(log_radius, -50.0, 50.0))
        if (
            not np.all(np.isfinite(radii))
            or np.min(radii) < self.parameters.minimum_radius
            or np.max(radii) > self.parameters.maximum_radius
        ):
            return _SurfaceEvaluation(
                False,
                "surface radius left admissible bounds",
                np.array([]),
                radii,
                np.array([]),
                np.array([]),
            )
        radius_theta = radii * log_radius_theta
        radius_theta2 = radii * (
            log_radius_theta2 + log_radius_theta * log_radius_theta
        )
        radius_phi = radii * log_radius_phi
        radius_theta_phi = radii * (
            log_radius_theta_phi + log_radius_theta * log_radius_phi
        )
        radius_phi2 = radii * (
            log_radius_phi2 + log_radius_phi * log_radius_phi
        )
        points = self.center[None, :] + radii[:, None] * self.directions
        if not self._inside_patch(points):
            return _SurfaceEvaluation(
                False,
                "surface left interpolation patch",
                np.array([]),
                radii,
                np.array([]),
                np.array([]),
            )

        tangent_theta = (
            radius_theta[:, None] * self.directions
            + radii[:, None] * self.theta_direction
        )
        tangent_phi = (
            radius_phi[:, None] * self.directions
            + (radii * self.sine)[:, None] * self.phi_direction
        )
        second_theta_theta = (
            (radius_theta2 - radii)[:, None] * self.directions
            + (2.0 * radius_theta)[:, None] * self.theta_direction
        )
        second_theta_phi = (
            radius_theta_phi[:, None] * self.directions
            + radius_phi[:, None] * self.theta_direction
            + (radius_theta * self.sine + radii * self.cosine)[:, None]
            * self.phi_direction
        )
        second_phi_phi = (
            (radius_phi2 - radii * self.sine * self.sine)[:, None]
            * self.directions
            - (radii * self.sine * self.cosine)[:, None] * self.theta_direction
            + (2.0 * radius_phi * self.sine)[:, None] * self.phi_direction
        )

        metric = np.moveaxis(
            trilinear_sample(self.grid, self.state.h, points), -1, 0
        )
        extrinsic_curvature = np.moveaxis(
            trilinear_sample(self.grid, self.state.K, points), -1, 0
        )
        connection = np.moveaxis(
            trilinear_sample(self.grid, self.connection, points), -1, 0
        )
        try:
            metric_inverse = np.linalg.inv(metric)
        except np.linalg.LinAlgError:
            return _SurfaceEvaluation(
                False, "singular sampled metric", np.array([]), radii, np.array([]), np.array([])
            )

        normal_covector = np.cross(tangent_theta, tangent_phi)
        radial_orientation = np.einsum(
            "ni,ni->n", normal_covector, self.directions
        )
        normal_covector[radial_orientation < 0.0] *= -1.0
        normal_squared = np.einsum(
            "nij,ni,nj->n", metric_inverse, normal_covector, normal_covector
        )
        if np.any(normal_squared <= 0.0) or not np.all(np.isfinite(normal_squared)):
            return _SurfaceEvaluation(
                False, "invalid physical surface normal", np.array([]), radii, np.array([]), np.array([])
            )
        normal_lower = normal_covector / np.sqrt(normal_squared)[:, None]
        normal_upper = np.einsum("nij,nj->ni", metric_inverse, normal_lower)

        q_theta_theta = np.einsum(
            "nij,ni,nj->n", metric, tangent_theta, tangent_theta
        )
        q_theta_phi = np.einsum(
            "nij,ni,nj->n", metric, tangent_theta, tangent_phi
        )
        q_phi_phi = np.einsum(
            "nij,ni,nj->n", metric, tangent_phi, tangent_phi
        )
        q_determinant = q_theta_theta * q_phi_phi - q_theta_phi * q_theta_phi
        if np.any(q_determinant <= 0.0) or not np.all(np.isfinite(q_determinant)):
            return _SurfaceEvaluation(
                False, "singular induced surface metric", np.array([]), radii, np.array([]), np.array([])
            )
        q_inverse_theta_theta = q_phi_phi / q_determinant
        q_inverse_theta_phi = -q_theta_phi / q_determinant
        q_inverse_phi_phi = q_theta_theta / q_determinant

        covariant_second_theta_theta = second_theta_theta + np.einsum(
            "nijk,nj,nk->ni", connection, tangent_theta, tangent_theta
        )
        covariant_second_theta_phi = second_theta_phi + np.einsum(
            "nijk,nj,nk->ni", connection, tangent_theta, tangent_phi
        )
        covariant_second_phi_phi = second_phi_phi + np.einsum(
            "nijk,nj,nk->ni", connection, tangent_phi, tangent_phi
        )
        curvature_theta_theta = -np.einsum(
            "ni,ni->n", normal_lower, covariant_second_theta_theta
        )
        curvature_theta_phi = -np.einsum(
            "ni,ni->n", normal_lower, covariant_second_theta_phi
        )
        curvature_phi_phi = -np.einsum(
            "ni,ni->n", normal_lower, covariant_second_phi_phi
        )
        mean_curvature = (
            q_inverse_theta_theta * curvature_theta_theta
            + 2.0 * q_inverse_theta_phi * curvature_theta_phi
            + q_inverse_phi_phi * curvature_phi_phi
        )
        trace_extrinsic_curvature = np.einsum(
            "nij,nij->n", metric_inverse, extrinsic_curvature
        )
        normal_extrinsic_curvature = np.einsum(
            "nij,ni,nj->n", extrinsic_curvature, normal_upper, normal_upper
        )
        expansion = (
            mean_curvature
            + normal_extrinsic_curvature
            - trace_extrinsic_curvature
        )

        area_weights = self.weights * np.sqrt(q_determinant) / self.sine
        radial_normal_projection = np.einsum(
            "ni,ni->n", normal_lower, self.directions
        )
        normal_deformation_scale = radii * radial_normal_projection
        if (
            not np.all(np.isfinite(expansion))
            or not np.all(np.isfinite(area_weights))
            or not np.all(area_weights > 0.0)
            or not np.all(normal_deformation_scale > 0.0)
        ):
            return _SurfaceEvaluation(
                False,
                "non-finite parametric surface geometry",
                expansion,
                radii,
                area_weights,
                np.array([]),
            )
        # NormalMOTSStabilityOperator expects B/|dF| for one log-radius
        # coefficient.  Here that displacement is r B (s_i e_r^i).
        normal_covector_norm = 1.0 / normal_deformation_scale
        return _SurfaceEvaluation(
            True,
            "ok",
            expansion,
            radii,
            area_weights,
            normal_covector_norm,
        )


class _DirectBrillLindquistSurfaceEvaluator:
    """Evaluate the time-symmetric conformal MOTS equation on the surface."""

    def __init__(
        self,
        separation: float,
        total_bare_mass: float,
        parameters: SpectralMOTSParameters,
    ) -> None:
        if separation < 0.0 or total_bare_mass <= 0.0:
            raise ValueError("separation must be non-negative and mass positive")
        if any(m != 0 for _, m in parameters.modes):
            raise ValueError("direct Brill-Lindquist evaluator is axisymmetric")
        self.separation = float(separation)
        self.total_bare_mass = float(total_bare_mass)
        self.parameters = parameters
        self.modes = parameters.modes
        quadrature = sphere_quadrature(
            1.0,
            (0.0, 0.0, 0.0),
            parameters.n_theta,
            parameters.n_phi,
        )
        self.directions = quadrature.normals
        self.weights = quadrature.weights
        (
            self.surface_basis,
            self.surface_basis_theta,
            self.surface_basis_theta2,
        ) = _axisymmetric_basis_derivatives(
            self.modes, quadrature.theta, quadrature.phi
        )
        self.theta = quadrature.theta
        self.phi = quadrature.phi

    def evaluate(self, coefficients: Array) -> _SurfaceEvaluation:
        coefficients = np.asarray(coefficients, dtype=float)
        log_radius = np.einsum("mn,m->n", self.surface_basis, coefficients)
        log_radius_theta = np.einsum(
            "mn,m->n", self.surface_basis_theta, coefficients
        )
        log_radius_theta2 = np.einsum(
            "mn,m->n", self.surface_basis_theta2, coefficients
        )
        radii = np.exp(log_radius)
        if (
            not np.all(np.isfinite(radii))
            or np.min(radii) < self.parameters.minimum_radius
            or np.max(radii) > self.parameters.maximum_radius
        ):
            return _SurfaceEvaluation(
                False,
                "surface radius left admissible bounds",
                np.array([]),
                radii,
                np.array([]),
                np.array([]),
            )
        radius_theta = radii * log_radius_theta
        radius_theta2 = radii * (
            log_radius_theta2 + log_radius_theta * log_radius_theta
        )
        tangent_norm = np.sqrt(radii * radii + radius_theta * radius_theta)
        sine = np.sin(self.theta)
        cosine = np.cos(self.theta)
        cotangent = cosine / np.maximum(sine, 1.0e-30)
        flat_mean_curvature = (
            2.0 * radii * radii
            + 3.0 * radius_theta * radius_theta
            - radii * radius_theta2
            - (radius_theta / radii)
            * tangent_norm
            * tangent_norm
            * cotangent
        ) / (tangent_norm**3)

        points = radii[:, None] * self.directions
        half_separation = 0.5 * self.separation
        puncture_plus = np.array([0.0, 0.0, half_separation])
        puncture_minus = np.array([0.0, 0.0, -half_separation])
        displacement_plus = points - puncture_plus[None, :]
        displacement_minus = points - puncture_minus[None, :]
        distance_plus = np.linalg.norm(displacement_plus, axis=1)
        distance_minus = np.linalg.norm(displacement_minus, axis=1)
        if np.any(distance_plus <= 0.0) or np.any(distance_minus <= 0.0):
            return _SurfaceEvaluation(
                False,
                "surface intersected a Brill-Lindquist puncture",
                np.array([]),
                radii,
                np.array([]),
                np.array([]),
            )
        individual_mass = 0.5 * self.total_bare_mass
        conformal_factor = (
            1.0
            + individual_mass / (2.0 * distance_plus)
            + individual_mass / (2.0 * distance_minus)
        )
        conformal_gradient = -0.5 * individual_mass * (
            displacement_plus / distance_plus[:, None] ** 3
            + displacement_minus / distance_minus[:, None] ** 3
        )

        radial = self.directions
        theta_direction = np.stack(
            (
                cosine * np.cos(self.phi),
                cosine * np.sin(self.phi),
                -sine,
            ),
            axis=1,
        )
        flat_normal = (
            radii[:, None] * radial
            - radius_theta[:, None] * theta_direction
        ) / tangent_norm[:, None]
        normal_log_conformal = np.einsum(
            "ni,ni->n", flat_normal, conformal_gradient
        ) / conformal_factor
        expansion = conformal_factor ** (-2) * (
            flat_mean_curvature + 4.0 * normal_log_conformal
        )
        area_weights = (
            self.weights
            * conformal_factor**4
            * radii
            * tangent_norm
        )
        normal_covector_norm = (
            conformal_factor ** (-2) * tangent_norm / (radii * radii)
        )
        if (
            not np.all(np.isfinite(expansion))
            or not np.all(np.isfinite(area_weights))
            or not np.all(area_weights > 0.0)
        ):
            return _SurfaceEvaluation(
                False,
                "non-finite direct surface geometry",
                expansion,
                radii,
                area_weights,
                normal_covector_norm,
            )
        return _SurfaceEvaluation(
            True,
            "ok",
            expansion,
            radii,
            area_weights,
            normal_covector_norm,
        )


class SpectralMOTSFinder:
    """Locate a star-shaped MOTS with a bound-constrained KKT/Gauss--Newton flow."""

    def __init__(
        self,
        grid: CartesianGrid,
        parameters: SpectralMOTSParameters | None = None,
    ) -> None:
        if grid.ndim != 3:
            raise ValueError("spectral MOTS finding requires a 3D Cartesian grid")
        self.grid = grid
        self.parameters = parameters or SpectralMOTSParameters()

    def initial_coefficients(self, radius: float) -> Array:
        if not self.parameters.minimum_radius <= radius <= self.parameters.maximum_radius:
            raise ValueError("initial radius is outside the configured bounds")
        result = np.zeros(len(self.parameters.modes))
        result[0] = np.log(radius)
        return result

    def evaluator(
        self,
        state: ADMState,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> _StarSurfaceEvaluator:
        """Construct the legacy volume-level-set Cartesian observer."""

        return _StarSurfaceEvaluator(self.grid, state, center, self.parameters)

    def _bounds(self) -> tuple[Array, Array]:
        count = len(self.parameters.modes)
        lower = np.full(count, -self.parameters.maximum_mode_amplitude)
        upper = np.full(count, self.parameters.maximum_mode_amplitude)
        lower[0] = np.log(self.parameters.minimum_radius)
        upper[0] = np.log(self.parameters.maximum_radius)
        return lower, upper

    def _residual_jacobian(
        self, evaluator: _StarSurfaceEvaluator, coefficients: Array
    ) -> tuple[_SurfaceEvaluation, Array]:
        base = evaluator.evaluate(coefficients)
        if not base.valid:
            return base, np.empty((0, len(coefficients)))
        jacobian = np.empty((len(base.expansion), len(coefficients)))
        lower, upper = self._bounds()
        for index in range(len(coefficients)):
            step = self.parameters.jacobian_step * max(1.0, abs(coefficients[index]))
            plus = coefficients.copy()
            minus = coefficients.copy()
            plus[index] = min(upper[index], plus[index] + step)
            minus[index] = max(lower[index], minus[index] - step)
            forward = evaluator.evaluate(plus)
            backward = evaluator.evaluate(minus)
            if not forward.valid or not backward.valid or plus[index] == minus[index]:
                raise FloatingPointError(
                    f"cannot construct surface Jacobian for coefficient {index}"
                )
            jacobian[:, index] = (
                forward.expansion - backward.expansion
            ) / (plus[index] - minus[index])
        return base, jacobian

    @staticmethod
    def _projected_gradient(
        coefficients: Array, gradient: Array, lower: Array, upper: Array
    ) -> Array:
        result = gradient.copy()
        tolerance = 1.0e-10
        result[(coefficients <= lower + tolerance) & (gradient > 0.0)] = 0.0
        result[(coefficients >= upper - tolerance) & (gradient < 0.0)] = 0.0
        return result

    def find(
        self,
        state: ADMState,
        initial_radius: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_coefficients: Array | None = None,
    ) -> SpectralMOTSResult:
        evaluator = self.evaluator(state, center)
        return self._solve(
            evaluator, initial_radius, center, initial_coefficients
        )

    def _solve(
        self,
        evaluator: _StarSurfaceEvaluator,
        initial_radius: float,
        center: tuple[float, float, float],
        initial_coefficients: Array | None,
    ) -> SpectralMOTSResult:
        parameters = self.parameters
        coefficients = (
            self.initial_coefficients(initial_radius)
            if initial_coefficients is None
            else np.asarray(initial_coefficients, dtype=float).copy()
        )
        if coefficients.shape != (len(parameters.modes),):
            raise ValueError("initial_coefficients has the wrong shape")
        lower, upper = self._bounds()
        coefficients = np.clip(coefficients, lower, upper)
        history: list[KKTIteration] = []
        reason = "maximum iterations reached"
        final_evaluation: _SurfaceEvaluation | None = None
        final_jacobian = np.empty((0, len(coefficients)))
        final_gradient = np.full_like(coefficients, np.inf)
        basis = evaluator.surface_basis.T
        angular_mass = basis.T @ (evaluator.weights[:, None] * basis)
        angular_projection = np.linalg.solve(
            angular_mass, basis.T * evaluator.weights[None, :]
        )
        angular_cholesky = np.linalg.cholesky(angular_mass)

        for iteration in range(parameters.maximum_iterations + 1):
            try:
                evaluation, jacobian = self._residual_jacobian(evaluator, coefficients)
            except FloatingPointError as error:
                reason = str(error)
                break
            if not evaluation.valid:
                reason = evaluation.reason
                final_evaluation = evaluation
                break
            normalized_weights = evaluator.weights / (4.0 * pi)
            modal_residual = angular_projection @ evaluation.expansion
            modal_jacobian = angular_projection @ jacobian
            weighted_residual = angular_cholesky.T @ modal_residual / sqrt(4.0 * pi)
            weighted_jacobian = angular_cholesky.T @ modal_jacobian / sqrt(4.0 * pi)
            merit = 0.5 * float(weighted_residual @ weighted_residual)
            spectral_residual = sqrt(2.0 * merit)
            rms = sqrt(float(np.sum(normalized_weights * evaluation.expansion**2)))
            maximum = float(np.max(np.abs(evaluation.expansion)))
            gradient = weighted_jacobian.T @ weighted_residual
            projected = self._projected_gradient(coefficients, gradient, lower, upper)
            projected_norm = float(np.linalg.norm(projected, ord=np.inf))
            final_evaluation = evaluation
            final_jacobian = jacobian
            final_gradient = gradient
            if spectral_residual <= parameters.expansion_tolerance:
                reason = "expansion tolerance reached"
                history.append(
                    KKTIteration(iteration, merit, spectral_residual, rms, maximum, projected_norm, 0.0, 0.0, 0)
                )
                break
            if projected_norm <= parameters.projected_gradient_tolerance:
                reason = "KKT stationary point without a MOTS"
                history.append(
                    KKTIteration(iteration, merit, spectral_residual, rms, maximum, projected_norm, 0.0, 0.0, 0)
                )
                break
            if iteration == parameters.maximum_iterations:
                break

            hessian = weighted_jacobian.T @ weighted_jacobian
            hessian += parameters.gauss_newton_damping * np.eye(len(coefficients))
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = -projected
            outward_lower = (coefficients <= lower + 1.0e-10) & (step < 0.0)
            outward_upper = (coefficients >= upper - 1.0e-10) & (step > 0.0)
            step[outward_lower | outward_upper] = 0.0
            norm = float(np.linalg.norm(step))
            if norm > parameters.trust_radius:
                step *= parameters.trust_radius / norm
            if not np.all(np.isfinite(step)) or np.linalg.norm(step) == 0.0:
                step = -projected
                norm = float(np.linalg.norm(step))
                if norm > parameters.trust_radius:
                    step *= parameters.trust_radius / norm

            accepted = False
            factor = 1.0
            active_count = 0
            trial = coefficients
            for _ in range(parameters.maximum_line_searches):
                trial = np.clip(coefficients + factor * step, lower, upper)
                active_count = int(
                    np.count_nonzero(np.isclose(trial, lower, atol=1.0e-10))
                    + np.count_nonzero(np.isclose(trial, upper, atol=1.0e-10))
                )
                candidate = evaluator.evaluate(trial)
                displacement = trial - coefficients
                target = merit + parameters.armijo_fraction * float(gradient @ displacement)
                if candidate.valid:
                    candidate_modal = angular_projection @ candidate.expansion
                    candidate_residual = angular_cholesky.T @ candidate_modal / sqrt(4.0 * pi)
                    candidate_merit = 0.5 * float(candidate_residual @ candidate_residual)
                    if candidate_merit <= target or candidate_merit < merit:
                        accepted = True
                        break
                factor *= 0.5
            history.append(
                KKTIteration(
                    iteration,
                    merit,
                    spectral_residual,
                    rms,
                    maximum,
                    projected_norm,
                    float(np.linalg.norm(trial - coefficients)) if accepted else 0.0,
                    factor if accepted else 0.0,
                    active_count,
                )
            )
            if not accepted:
                reason = "line search could not reduce expansion residual"
                break
            coefficients = trial

        if final_evaluation is None or not final_evaluation.valid:
            empty = np.zeros(len(coefficients))
            return SpectralMOTSResult(
                False,
                reason,
                tuple(float(value) for value in center),
                parameters.modes,
                coefficients,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                len(history),
                np.inf,
                empty,
                empty,
                np.inf,
                (),
                (),
                np.array([], dtype=complex),
                np.inf,
                tuple(history),
            )

        normalized_weights = evaluator.weights / (4.0 * pi)
        residual = final_evaluation.expansion
        modal_residual = angular_projection @ residual
        weighted_modal_residual = angular_cholesky.T @ modal_residual / sqrt(4.0 * pi)
        merit = 0.5 * float(weighted_modal_residual @ weighted_modal_residual)
        spectral_residual = sqrt(2.0 * merit)
        rms = sqrt(float(np.sum(normalized_weights * residual * residual)))
        gradient = final_gradient
        projected = self._projected_gradient(coefficients, gradient, lower, upper)
        projected_norm = float(np.linalg.norm(projected, ord=np.inf))
        tolerance = 1.0e-8
        active_lower_mask = coefficients <= lower + tolerance
        active_upper_mask = coefficients >= upper - tolerance
        lower_multipliers = np.where(active_lower_mask, np.maximum(gradient, 0.0), 0.0)
        upper_multipliers = np.where(active_upper_mask, np.maximum(-gradient, 0.0), 0.0)
        stationarity = gradient - lower_multipliers + upper_multipliers
        complementarity = max(
            float(np.linalg.norm(stationarity, ord=np.inf)),
            float(np.max(np.abs(lower_multipliers * (coefficients - lower)))),
            float(np.max(np.abs(upper_multipliers * (upper - coefficients)))),
        )

        modal_response = angular_projection @ final_jacobian
        try:
            eigenvalues = np.linalg.eigvals(modal_response)
            condition = float(np.linalg.cond(modal_response))
        except np.linalg.LinAlgError:
            eigenvalues = np.full(len(coefficients), np.nan + 0.0j)
            condition = np.inf

        area = float(np.sum(final_evaluation.area_weights))
        found = spectral_residual <= parameters.expansion_tolerance
        return SpectralMOTSResult(
            found,
            reason,
            tuple(float(value) for value in center),
            parameters.modes,
            coefficients,
            float(np.sum(evaluator.weights * final_evaluation.radii) / (4.0 * pi)),
            float(np.min(final_evaluation.radii)),
            float(np.max(final_evaluation.radii)),
            area,
            sqrt(area / (16.0 * pi)),
            float(np.sum(evaluator.weights * residual) / (4.0 * pi)),
            spectral_residual,
            rms,
            float(np.max(np.abs(residual))),
            merit,
            len(history),
            projected_norm,
            lower_multipliers,
            upper_multipliers,
            complementarity,
            tuple(np.flatnonzero(active_lower_mask).tolist()),
            tuple(np.flatnonzero(active_upper_mask).tolist()),
            eigenvalues,
            condition,
            tuple(history),
        )


class ParametricMOTSFinder(SpectralMOTSFinder):
    """Parametric Cartesian MOTS observer for general star-shaped surfaces.

    Real spherical harmonics with all ``-ell <= m <= ell`` are supported.
    The embedding is differentiated on the angular surface and only the ADM
    metric, extrinsic curvature, and Cartesian connection are interpolated
    from the volume.  This avoids differentiating a volume extension of the
    surface normal and is the non-axisymmetric counterpart of the calibrated
    head-on observer.
    """

    def __init__(
        self,
        grid: CartesianGrid,
        parameters: SpectralMOTSParameters | None = None,
    ) -> None:
        super().__init__(grid, parameters)
        self._cached_state: ADMState | None = None
        self._cached_connection: Array | None = None

    def evaluator(  # type: ignore[override]
        self,
        state: ADMState,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> _ParametricAxisymmetricSurfaceEvaluator:
        if self._cached_state is not state or self._cached_connection is None:
            h_inv = inverse_metric(state.h)[0]
            self._cached_connection = christoffel(self.grid, state.h, h_inv)
            self._cached_state = state
        return _ParametricAxisymmetricSurfaceEvaluator(
            self.grid,
            state,
            center,
            self.parameters,
            self._cached_connection,
        )

    def clear_geometry_cache(self) -> None:
        self._cached_state = None
        self._cached_connection = None

    def normal_stability(
        self,
        state: ADMState,
        surface: SpectralMOTSResult,
        relative_step: float = 2.0e-4,
    ) -> NormalMOTSStabilityResult:
        operator = NormalMOTSStabilityOperator(self, relative_step)
        return operator.analyze_evaluator(
            self.evaluator(state, surface.center), surface
        )


class ParametricAxisymmetricMOTSFinder(ParametricMOTSFinder):
    """Corrected Cartesian MOTS observer for axisymmetric radial graphs.

    The input is an ordinary Cartesian ADM slice, so the same finder can be
    attached to reconstructed CCZ4 snapshots.  Only the surface
    parameterization is axisymmetric; the sampled metric and extrinsic
    curvature retain all Cartesian components.
    """

    def __init__(
        self,
        grid: CartesianGrid,
        parameters: SpectralMOTSParameters | None = None,
    ) -> None:
        super().__init__(grid, parameters)
        if any(m != 0 for _, m in self.parameters.modes):
            raise ValueError("parametric axisymmetric finder requires m=0 modes")


class DirectBrillLindquistMOTSFinder(SpectralMOTSFinder):
    """Axisymmetric MOTS finder using the direct conformal surface equation."""

    def __init__(
        self,
        parameters: SpectralMOTSParameters,
        total_bare_mass: float = 1.0,
    ) -> None:
        if total_bare_mass <= 0.0:
            raise ValueError("total_bare_mass must be positive")
        if any(m != 0 for _, m in parameters.modes):
            raise ValueError("direct Brill-Lindquist finder requires m=0 modes")
        self.grid = None
        self.parameters = parameters
        self.total_bare_mass = float(total_bare_mass)

    def evaluator(self, separation: float) -> _DirectBrillLindquistSurfaceEvaluator:
        return _DirectBrillLindquistSurfaceEvaluator(
            separation, self.total_bare_mass, self.parameters
        )

    def find(  # type: ignore[override]
        self,
        separation: float,
        initial_radius: float,
        initial_coefficients: Array | None = None,
    ) -> SpectralMOTSResult:
        return self._solve(
            self.evaluator(separation),
            initial_radius,
            (0.0, 0.0, 0.0),
            initial_coefficients,
        )

    def normal_stability(
        self,
        separation: float,
        surface: SpectralMOTSResult,
        relative_step: float = 2.0e-4,
    ) -> NormalMOTSStabilityResult:
        # The class is defined later in the module; name resolution occurs when
        # this method is called, after module initialization is complete.
        operator = NormalMOTSStabilityOperator(self, relative_step)
        return operator.analyze_evaluator(self.evaluator(separation), surface)


@dataclass(frozen=True)
class NormalMOTSStabilityResult:
    """Area-projected response to physical unit-normal surface deformation."""

    valid: bool
    reason: str
    modes: tuple[Mode, ...]
    operator_matrix: Array
    eigenvalues: Array
    principal_eigenvalue: complex
    operator_condition_number: float
    normalizer_condition_number: float
    normalizer_roundtrip_error: float
    finite_difference_relative_error: float
    area_adjoint_defect: float
    physical_step: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "modes": [list(mode) for mode in self.modes],
            "operator_matrix": self.operator_matrix.tolist(),
            "eigenvalues_real": self.eigenvalues.real.tolist(),
            "eigenvalues_imag": self.eigenvalues.imag.tolist(),
            "principal_eigenvalue_real": float(self.principal_eigenvalue.real),
            "principal_eigenvalue_imag": float(self.principal_eigenvalue.imag),
            "operator_condition_number": self.operator_condition_number,
            "normalizer_condition_number": self.normalizer_condition_number,
            "normalizer_roundtrip_error": self.normalizer_roundtrip_error,
            "finite_difference_relative_error": self.finite_difference_relative_error,
            "area_adjoint_defect": self.area_adjoint_defect,
            "physical_step": self.physical_step,
        }


class NormalMOTSStabilityOperator:
    """Discretize ``delta_f Theta_+`` for physical normal displacement ``f``.

    A log-radius coefficient perturbation ``dc`` moves the surface normally by
    ``f = delta(log h) / |dF|_gamma`` for ``F=r/h-1``.  The normalization map
    from coefficient perturbations to physical normal-deformation modes is
    built with the induced area measure and inverted before any expansion
    derivative is taken.  This removes the raw log-radius scaling from the
    reported spectrum.
    """

    def __init__(
        self,
        finder: SpectralMOTSFinder,
        relative_step: float = 2.0e-4,
    ) -> None:
        if relative_step <= 0.0:
            raise ValueError("relative_step must be positive")
        self.finder = finder
        self.relative_step = float(relative_step)

    @staticmethod
    def _invalid(
        modes: tuple[Mode, ...], reason: str, physical_step: float = np.nan
    ) -> NormalMOTSStabilityResult:
        count = len(modes)
        return NormalMOTSStabilityResult(
            False,
            reason,
            modes,
            np.empty((0, 0)),
            np.array([], dtype=complex),
            complex(np.nan, np.nan),
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            physical_step,
        )

    def analyze(
        self,
        state: ADMState,
        surface: SpectralMOTSResult,
    ) -> NormalMOTSStabilityResult:
        evaluator = self.finder.evaluator(state, surface.center)
        return self.analyze_evaluator(evaluator, surface)

    def analyze_evaluator(
        self,
        evaluator: _StarSurfaceEvaluator,
        surface: SpectralMOTSResult,
    ) -> NormalMOTSStabilityResult:
        parameters = self.finder.parameters
        if surface.modes != parameters.modes:
            return self._invalid(parameters.modes, "surface modes do not match finder")
        if surface.coefficients.shape != (len(parameters.modes),):
            return self._invalid(parameters.modes, "surface coefficients have wrong shape")
        base = evaluator.evaluate(surface.coefficients)
        if not base.valid:
            return self._invalid(parameters.modes, base.reason)

        basis = evaluator.surface_basis.T
        weighted_basis = base.area_weights[:, None] * basis
        area_mass = basis.T @ weighted_basis
        try:
            area_projection = np.linalg.solve(
                area_mass, basis.T * base.area_weights[None, :]
            )
        except np.linalg.LinAlgError:
            return self._invalid(parameters.modes, "singular induced-area mode mass")

        # A unit coefficient in log radius produces physical displacement
        # B_A/|dF|_gamma along the spatial unit normal.
        coefficient_normal_fields = (
            basis / base.normal_covector_norm[:, None]
        )
        normalizer = area_projection @ coefficient_normal_fields
        try:
            inverse_normalizer = np.linalg.inv(normalizer)
        except np.linalg.LinAlgError:
            return self._invalid(parameters.modes, "singular normal-deformation map")
        normalizer_condition = float(np.linalg.cond(normalizer))
        roundtrip = float(
            np.linalg.norm(
                normalizer @ inverse_normalizer - np.eye(len(parameters.modes)),
                ord=np.inf,
            )
        )
        physical_step = self.relative_step * surface.coordinate_radius_mean
        if not np.isfinite(physical_step) or physical_step <= 0.0:
            return self._invalid(parameters.modes, "invalid physical perturbation step")

        lower, upper = self.finder._bounds()

        def discretize(step: float) -> Array:
            operator = np.empty((len(parameters.modes), len(parameters.modes)))
            for mode_index in range(len(parameters.modes)):
                coefficient_direction = inverse_normalizer[:, mode_index]
                plus_coefficients = surface.coefficients + step * coefficient_direction
                minus_coefficients = surface.coefficients - step * coefficient_direction
                if (
                    np.any(plus_coefficients <= lower)
                    or np.any(plus_coefficients >= upper)
                    or np.any(minus_coefficients <= lower)
                    or np.any(minus_coefficients >= upper)
                ):
                    raise FloatingPointError(
                        "normal perturbation reached a coefficient constraint"
                    )
                plus = evaluator.evaluate(plus_coefficients)
                minus = evaluator.evaluate(minus_coefficients)
                if not plus.valid or not minus.valid:
                    failed = plus.reason if not plus.valid else minus.reason
                    raise FloatingPointError(
                        f"normal perturbation left the surface domain: {failed}"
                    )
                derivative = (plus.expansion - minus.expansion) / (2.0 * step)
                operator[:, mode_index] = area_projection @ derivative
            return operator

        try:
            coarse_operator = discretize(physical_step)
            operator = discretize(0.5 * physical_step)
        except FloatingPointError as error:
            return self._invalid(parameters.modes, str(error), physical_step)
        denominator = max(float(np.linalg.norm(operator)), 1.0e-30)
        finite_difference_error = float(
            np.linalg.norm(operator - coarse_operator) / denominator
        )
        try:
            eigenvalues = np.linalg.eigvals(operator)
            operator_condition = float(np.linalg.cond(operator))
        except np.linalg.LinAlgError:
            return self._invalid(
                parameters.modes, "normal stability eigensolve failed", physical_step
            )
        principal = complex(eigenvalues[np.argmin(eigenvalues.real)])
        area_adjoint = area_mass @ operator
        adjoint_denominator = max(float(np.linalg.norm(area_adjoint)), 1.0e-30)
        adjoint_defect = float(
            np.linalg.norm(area_adjoint - area_adjoint.T) / adjoint_denominator
        )
        return NormalMOTSStabilityResult(
            True,
            "ok",
            parameters.modes,
            operator,
            eigenvalues,
            principal,
            operator_condition,
            normalizer_condition,
            roundtrip,
            finite_difference_error,
            adjoint_defect,
            physical_step,
        )


@dataclass(frozen=True)
class MOTSBranchDefinition:
    name: str
    center: tuple[float, float, float]
    initial_radius: float
    initial_coefficients: Array | None = None


@dataclass(frozen=True)
class MOTSBranchObservation:
    name: str
    time: float
    result: SpectralMOTSResult
    newly_found: bool
    newly_lost: bool
    successful_observations: int
    failed_observations: int
    first_found_time: float | None

    def diagnostics(self) -> dict[str, object]:
        values = self.result.diagnostics()
        values.update(
            {
                "branch": self.name,
                "time": self.time,
                "newly_found": self.newly_found,
                "newly_lost": self.newly_lost,
                "successful_observations": self.successful_observations,
                "failed_observations": self.failed_observations,
                "first_found_time": self.first_found_time,
            }
        )
        return values


class MOTSBranchTracker:
    """Track independent MOTS solution branches without deleting failed branches."""

    def __init__(
        self,
        finder: SpectralMOTSFinder,
        branches: Iterable[MOTSBranchDefinition],
    ) -> None:
        self.finder = finder
        definitions = tuple(branches)
        if not definitions or len({branch.name for branch in definitions}) != len(definitions):
            raise ValueError("branch names must be non-empty and unique")
        self.definitions = definitions
        self._centers = {
            branch.name: tuple(float(value) for value in branch.center)
            for branch in definitions
        }
        self._coefficients: dict[str, Array] = {}
        self._was_found = {branch.name: False for branch in definitions}
        self._successes = {branch.name: 0 for branch in definitions}
        self._failures = {branch.name: 0 for branch in definitions}
        self._first_found: dict[str, float | None] = {
            branch.name: None for branch in definitions
        }

    def observe(self, state: ADMState) -> dict[str, MOTSBranchObservation]:
        observations: dict[str, MOTSBranchObservation] = {}
        for branch in self.definitions:
            seed = self._coefficients.get(branch.name, branch.initial_coefficients)
            result = self.finder.find(
                state,
                branch.initial_radius,
                self._centers[branch.name],
                seed,
            )
            previous = self._was_found[branch.name]
            if result.found:
                self._coefficients[branch.name] = result.coefficients.copy()
                self._successes[branch.name] += 1
                if self._first_found[branch.name] is None:
                    self._first_found[branch.name] = float(state.time)
            else:
                self._failures[branch.name] += 1
            observation = MOTSBranchObservation(
                branch.name,
                float(state.time),
                result,
                result.found and not previous,
                previous and not result.found,
                self._successes[branch.name],
                self._failures[branch.name],
                self._first_found[branch.name],
            )
            observations[branch.name] = observation
            self._was_found[branch.name] = result.found
        return observations

    def update_center(
        self, name: str, center: tuple[float, float, float]
    ) -> None:
        """Move a named branch's search center without discarding continuation."""

        if name not in self._centers:
            raise KeyError(f"unknown MOTS branch {name!r}")
        values = tuple(float(value) for value in center)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError("branch center must contain three finite values")
        self._centers[name] = values

    def seed_branch(self, name: str, coefficients: Array) -> None:
        """Replace one continuation seed while preserving branch counters."""

        if name not in self._centers:
            raise KeyError(f"unknown MOTS branch {name!r}")
        values = np.asarray(coefficients, dtype=float)
        if values.shape != (len(self.finder.parameters.modes),):
            raise ValueError("branch seed coefficients have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("branch seed coefficients must be finite")
        self._coefficients[name] = values.copy()

    def checkpoint(self) -> dict[str, object]:
        """Return the continuation state needed for restart-and-append runs."""

        return {
            "format": "tesseract.kkt_mots_branches.v1",
            "modes": [list(mode) for mode in self.finder.parameters.modes],
            "branches": {
                branch.name: {
                    "center": list(self._centers[branch.name]),
                    "coefficients": (
                        self._coefficients[branch.name].tolist()
                        if branch.name in self._coefficients
                        else None
                    ),
                    "was_found": self._was_found[branch.name],
                    "successes": self._successes[branch.name],
                    "failures": self._failures[branch.name],
                    "first_found_time": self._first_found[branch.name],
                }
                for branch in self.definitions
            },
        }

    def restore(self, payload: dict[str, object]) -> None:
        """Restore a checkpoint after validating its branch and spectral layout."""

        if payload.get("format") != "tesseract.kkt_mots_branches.v1":
            raise ValueError("unsupported KKT MOTS branch checkpoint")
        expected_modes = [list(mode) for mode in self.finder.parameters.modes]
        if payload.get("modes") != expected_modes:
            raise ValueError("checkpoint spectral modes do not match the finder")
        branch_payload = payload.get("branches")
        if not isinstance(branch_payload, dict) or set(branch_payload) != {
            branch.name for branch in self.definitions
        }:
            raise ValueError("checkpoint branch definitions do not match the tracker")
        count = len(self.finder.parameters.modes)
        for branch in self.definitions:
            values = branch_payload[branch.name]
            if not isinstance(values, dict):
                raise ValueError("invalid branch checkpoint record")
            raw_coefficients = values.get("coefficients")
            raw_center = values.get("center", list(branch.center))
            center = tuple(float(value) for value in raw_center)
            if len(center) != 3 or not np.all(np.isfinite(center)):
                raise ValueError("invalid checkpoint branch center")
            self._centers[branch.name] = center
            if raw_coefficients is None:
                self._coefficients.pop(branch.name, None)
            else:
                coefficients = np.asarray(raw_coefficients, dtype=float)
                if coefficients.shape != (count,) or not np.all(np.isfinite(coefficients)):
                    raise ValueError("invalid checkpoint surface coefficients")
                self._coefficients[branch.name] = coefficients.copy()
            self._was_found[branch.name] = bool(values.get("was_found", False))
            self._successes[branch.name] = int(values.get("successes", 0))
            self._failures[branch.name] = int(values.get("failures", 0))
            first_found = values.get("first_found_time")
            self._first_found[branch.name] = (
                None if first_found is None else float(first_found)
            )


@dataclass(frozen=True)
class CCZ4MOTSQualification:
    """Resolution and sign gates calibrated for one Cartesian snapshot."""

    cells_per_mean_radius: float
    minimum_cells_per_mean_radius: float
    principal_mode_real: float
    principal_mode_imaginary: float
    minimum_resolved_mode_magnitude: float
    resolution_gate_passed: bool
    principal_mode_sign_gate_passed: bool
    continuum_zero_qualified: bool
    reason: str

    def diagnostics(self) -> dict[str, object]:
        return {
            "cells_per_mean_radius": self.cells_per_mean_radius,
            "minimum_cells_per_mean_radius": self.minimum_cells_per_mean_radius,
            "principal_mode_real": self.principal_mode_real,
            "principal_mode_imaginary": self.principal_mode_imaginary,
            "minimum_resolved_mode_magnitude": self.minimum_resolved_mode_magnitude,
            "resolution_gate_passed": self.resolution_gate_passed,
            "principal_mode_sign_gate_passed": self.principal_mode_sign_gate_passed,
            "continuum_zero_qualified": self.continuum_zero_qualified,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CCZ4PhysicalNormalMOTSObservation:
    """One branch observation and its corrected unit-normal mode spectrum."""

    branch: MOTSBranchObservation
    normal_stability: NormalMOTSStabilityResult | None
    qualification: CCZ4MOTSQualification

    def diagnostics(self) -> dict[str, object]:
        values = self.branch.diagnostics()
        values["normal_stability"] = (
            None
            if self.normal_stability is None
            else self.normal_stability.diagnostics()
        )
        values["cartesian_qualification"] = self.qualification.diagnostics()
        return values


class CCZ4PhysicalNormalMOTSTracker:
    """Attach the corrected MOTS and mode observer to CCZ4 snapshots.

    Snapshot conversion is explicit and read-only: the CCZ4 fields are
    reconstructed as physical ``gamma_ij`` and ``K_ij``, passed to the
    parametric Cartesian finder, and never fed back into the evolution.
    Continuation state is delegated to :class:`MOTSBranchTracker`, so the
    observer can accompany restart-and-append campaigns.
    """

    def __init__(
        self,
        ccz4: CCZ4Solver,
        finder: ParametricAxisymmetricMOTSFinder,
        branches: Iterable[MOTSBranchDefinition],
        relative_step: float = 2.0e-4,
        minimum_cells_per_mean_radius: float = 16.0,
        minimum_resolved_mode_magnitude: float = 5.0e-3,
    ) -> None:
        if ccz4.grid is not finder.grid:
            raise ValueError("CCZ4 solver and MOTS finder must share one grid")
        if relative_step <= 0.0:
            raise ValueError("relative_step must be positive")
        if minimum_cells_per_mean_radius <= 0.0:
            raise ValueError("minimum_cells_per_mean_radius must be positive")
        if minimum_resolved_mode_magnitude <= 0.0:
            raise ValueError("minimum_resolved_mode_magnitude must be positive")
        self.ccz4 = ccz4
        self.finder = finder
        self.branches = MOTSBranchTracker(finder, branches)
        self.relative_step = float(relative_step)
        self.minimum_cells_per_mean_radius = float(
            minimum_cells_per_mean_radius
        )
        self.minimum_resolved_mode_magnitude = float(
            minimum_resolved_mode_magnitude
        )

    def qualify(
        self,
        observation: MOTSBranchObservation,
        stability: NormalMOTSStabilityResult | None,
    ) -> CCZ4MOTSQualification:
        if not observation.result.found or stability is None or not stability.valid:
            return CCZ4MOTSQualification(
                np.nan,
                self.minimum_cells_per_mean_radius,
                np.nan,
                np.nan,
                self.minimum_resolved_mode_magnitude,
                False,
                False,
                False,
                "no valid MOTS stability measurement",
            )
        spacing = max(float(value) for value in self.finder.grid.spacing)
        cells = observation.result.coordinate_radius_mean / spacing
        principal = stability.principal_eigenvalue
        resolution_passed = cells >= self.minimum_cells_per_mean_radius * (
            1.0 - 1.0e-12
        )
        sign_passed = (
            resolution_passed
            and abs(principal.real) >= self.minimum_resolved_mode_magnitude
            and abs(principal.imag) <= self.minimum_resolved_mode_magnitude
        )
        if not resolution_passed:
            reason = "surface is below the calibrated Cartesian resolution"
        elif not sign_passed:
            reason = "principal mode lies inside the calibrated grid-phase band"
        else:
            reason = "single-snapshot mode sign passed calibrated gates"
        return CCZ4MOTSQualification(
            float(cells),
            self.minimum_cells_per_mean_radius,
            float(principal.real),
            float(principal.imag),
            self.minimum_resolved_mode_magnitude,
            resolution_passed,
            sign_passed,
            False,
            reason,
        )

    def observe(
        self, state: CCZ4State
    ) -> dict[str, CCZ4PhysicalNormalMOTSObservation]:
        adm = self.ccz4.to_adm(state)
        branch_observations = self.branches.observe(adm)
        result: dict[str, CCZ4PhysicalNormalMOTSObservation] = {}
        for name, observation in branch_observations.items():
            stability = (
                self.finder.normal_stability(
                    adm, observation.result, self.relative_step
                )
                if observation.result.found
                else None
            )
            result[name] = CCZ4PhysicalNormalMOTSObservation(
                observation,
                stability,
                self.qualify(observation, stability),
            )
        self.finder.clear_geometry_cache()
        return result

    def update_center(
        self, name: str, center: tuple[float, float, float]
    ) -> None:
        self.branches.update_center(name, center)

    def seed_branch(self, name: str, coefficients: Array) -> None:
        self.branches.seed_branch(name, coefficients)

    def checkpoint(self) -> dict[str, object]:
        return {
            "format": "tesseract.ccz4_physical_normal_mots.v1",
            "relative_step": self.relative_step,
            "minimum_cells_per_mean_radius": self.minimum_cells_per_mean_radius,
            "minimum_resolved_mode_magnitude": self.minimum_resolved_mode_magnitude,
            "branches": self.branches.checkpoint(),
        }

    def restore(self, payload: dict[str, object]) -> None:
        if payload.get("format") != "tesseract.ccz4_physical_normal_mots.v1":
            raise ValueError("unsupported CCZ4 MOTS tracker checkpoint")
        relative_step = float(payload.get("relative_step", np.nan))
        if not np.isclose(relative_step, self.relative_step, rtol=0.0, atol=0.0):
            raise ValueError("checkpoint physical-normal step does not match tracker")
        minimum_cells = float(
            payload.get("minimum_cells_per_mean_radius", np.nan)
        )
        if not np.isclose(
            minimum_cells,
            self.minimum_cells_per_mean_radius,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("checkpoint Cartesian resolution gate does not match tracker")
        minimum_mode = float(
            payload.get("minimum_resolved_mode_magnitude", np.nan)
        )
        if not np.isclose(
            minimum_mode,
            self.minimum_resolved_mode_magnitude,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("checkpoint principal-mode gate does not match tracker")
        branch_payload = payload.get("branches")
        if not isinstance(branch_payload, dict):
            raise ValueError("invalid CCZ4 MOTS tracker branch checkpoint")
        self.branches.restore(branch_payload)
