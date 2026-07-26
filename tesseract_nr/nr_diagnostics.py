"""Quasi-local mass, apparent-horizon, and gravitational-wave diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import ADMState, christoffel, inverse_metric, ricci_tensor
from .grid import Array, CartesianGrid


def _levi_civita() -> Array:
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0
    return epsilon


def trilinear_sample(grid: CartesianGrid, field: Array, points: Array) -> Array:
    """Sample a component-leading field at Cartesian points shaped (N,3)."""
    if grid.ndim != 3:
        raise ValueError("trilinear sampling requires a 3D grid")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    field = np.asarray(field)
    leading_shape = field.shape[:-3]
    fractional = np.empty_like(points)
    indices = np.empty(points.shape, dtype=int)
    weights = np.empty_like(points)
    for axis in range(3):
        fractional[:, axis] = (
            (points[:, axis] - grid.origin[axis]) / grid.spacing[axis] - 0.5
        )
        indices[:, axis] = np.clip(
            np.floor(fractional[:, axis]).astype(int), 0, grid.shape[axis] - 2
        )
        weights[:, axis] = np.clip(
            fractional[:, axis] - indices[:, axis], 0.0, 1.0
        )
    result = np.zeros(leading_shape + (len(points),), dtype=field.dtype)
    for corner in range(8):
        selection = []
        weight = np.ones(len(points))
        for axis in range(3):
            bit = (corner >> axis) & 1
            selection.append(indices[:, axis] + bit)
            weight *= weights[:, axis] if bit else 1.0 - weights[:, axis]
        result += field[(..., selection[0], selection[1], selection[2])] * weight
    return result


@dataclass(frozen=True)
class SphereQuadrature:
    points: Array
    normals: Array
    theta: Array
    phi: Array
    weights: Array


def sphere_quadrature(
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_theta: int = 16,
    n_phi: int = 32,
) -> SphereQuadrature:
    if radius <= 0.0 or n_theta < 4 or n_phi < 8:
        raise ValueError("invalid extraction sphere or angular resolution")
    mu, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    phi_axis = 2.0 * np.pi * np.arange(n_phi) / n_phi
    mu_grid, phi_grid = np.meshgrid(mu, phi_axis, indexing="ij")
    theta = np.arccos(mu_grid)
    sin_theta = np.sqrt(np.maximum(1.0 - mu_grid**2, 0.0))
    normals = np.stack(
        (sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), mu_grid),
        axis=-1,
    ).reshape(-1, 3)
    points = np.asarray(center)[None, :] + radius * normals
    weights = np.repeat(mu_weights, n_phi) * (2.0 * np.pi / n_phi)
    return SphereQuadrature(points, normals, theta.ravel(), phi_grid.ravel(), weights)


@dataclass(frozen=True)
class HorizonResult:
    found: bool
    coordinate_radius: float
    area: float
    irreducible_mass: float
    expansion: float
    iterations: int


@dataclass(frozen=True)
class HawkingMassResult:
    """Hawking quasi-local mass of a closed coordinate-sphere surface."""

    coordinate_radius: float
    area: float
    areal_radius: float
    mass: float
    mean_outgoing_expansion: float
    mean_ingoing_expansion: float
    expansion_product_integral: float


class HawkingMassDiagnostic:
    """Evaluate a geometric mass aspect on nested closed two-surfaces.

    With the code's sign convention for ``K_ij``, define the surface mean
    curvature ``k=D_i s^i`` and tangential extrinsic-curvature trace
    ``p=K-K_ij s^i s^j``.  The unnormalised null expansions are
    ``theta_+=k-p`` and ``theta_-=-k-p``.  The Hawking mass is then

      sqrt(A/(16 pi)) [1 + integral(theta_+ theta_- dA)/(16 pi)].

    It vanishes for round spheres in Minkowski space and equals the
    Schwarzschild mass on spherical vacuum slices.  The scalar value is
    invariant for a fixed geometric surface; the coordinate-sphere foliation
    is an explicit extraction choice.
    """

    def __init__(self, grid: CartesianGrid, n_theta: int = 16, n_phi: int = 32) -> None:
        if grid.ndim != 3:
            raise ValueError("Hawking mass extraction requires a 3D grid")
        self.grid = grid
        self.n_theta = n_theta
        self.n_phi = n_phi

    def _surface_fields(
        self, state: ADMState, center: tuple[float, float, float]
    ) -> tuple[Array, Array]:
        h_inv, _, sqrt_h = inverse_metric(state.h)
        coordinates = self.grid.coordinates()
        displacement = np.stack(
            [coordinates[i] - center[i] for i in range(3)]
        )
        radius = np.sqrt(np.sum(displacement**2, axis=0))
        radial_covector = displacement / np.maximum(radius, 1.0e-14)[None, ...]
        normal_norm = np.sqrt(
            np.maximum(
                np.einsum(
                    "ij...,i...,j...->...",
                    h_inv,
                    radial_covector,
                    radial_covector,
                ),
                1.0e-30,
            )
        )
        normal_lower = radial_covector / normal_norm[None, ...]
        normal_upper = np.einsum("ij...,j...->i...", h_inv, normal_lower)
        mean_curvature = self.grid.zeros()
        for i in range(3):
            mean_curvature += (
                self.grid.derivative(sqrt_h * normal_upper[i], i) / sqrt_h
            )
        trace_K = np.einsum("ij...,ij...->...", h_inv, state.K)
        K_normal = np.einsum(
            "ij...,i...,j...->...", state.K, normal_upper, normal_upper
        )
        tangential_K = trace_K - K_normal
        outgoing = mean_curvature - tangential_K
        ingoing = -mean_curvature - tangential_K
        return outgoing, ingoing

    def evaluate(
        self,
        state: ADMState,
        radius: float,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> HawkingMassResult:
        sphere = sphere_quadrature(
            radius, center, self.n_theta, self.n_phi
        )
        outgoing_field, ingoing_field = self._surface_fields(state, center)
        outgoing = trilinear_sample(self.grid, outgoing_field, sphere.points)
        ingoing = trilinear_sample(self.grid, ingoing_field, sphere.points)
        h_samples = np.moveaxis(
            trilinear_sample(self.grid, state.h, sphere.points), -1, 0
        )
        determinant = np.linalg.det(h_samples)
        inverse = np.linalg.inv(h_samples)
        normal_norm = np.einsum(
            "ni,nij,nj->n", sphere.normals, inverse, sphere.normals
        )
        area_density = (
            np.sqrt(np.maximum(determinant * normal_norm, 0.0)) * radius**2
        )
        area_weights = sphere.weights * area_density
        area = float(np.sum(area_weights))
        product_integral = float(np.sum(area_weights * outgoing * ingoing))
        areal_radius = float(np.sqrt(area / (4.0 * np.pi)))
        mass = float(
            np.sqrt(area / (16.0 * np.pi))
            * (1.0 + product_integral / (16.0 * np.pi))
        )
        return HawkingMassResult(
            coordinate_radius=float(radius),
            area=area,
            areal_radius=areal_radius,
            mass=mass,
            mean_outgoing_expansion=float(np.sum(area_weights * outgoing) / area),
            mean_ingoing_expansion=float(np.sum(area_weights * ingoing) / area),
            expansion_product_integral=product_integral,
        )


class ApparentHorizonFinder:
    """Find a star-shaped spherical apparent horizon by expansion root search."""

    def __init__(self, grid: CartesianGrid, n_theta: int = 16, n_phi: int = 32) -> None:
        self.grid = grid
        self.n_theta = n_theta
        self.n_phi = n_phi

    def expansion_field(
        self, state: ADMState, center: tuple[float, float, float]
    ) -> Array:
        h_inv, _, sqrt_h = inverse_metric(state.h)
        coordinates = self.grid.coordinates()
        displacement = np.stack(
            [coordinates[i] - center[i] for i in range(3)]
        )
        radius = np.sqrt(np.sum(displacement**2, axis=0))
        covector = displacement / np.maximum(radius, 1e-14)[None, ...]
        norm = np.sqrt(np.maximum(np.einsum("ij...,i...,j...->...", h_inv, covector, covector), 1e-30))
        s_lower = covector / norm[None, ...]
        s_upper = np.einsum("ij...,j...->i...", h_inv, s_lower)
        divergence = self.grid.zeros()
        for i in range(3):
            divergence += self.grid.derivative(sqrt_h * s_upper[i], i) / sqrt_h
        trace_K = np.einsum("ij...,ij...->...", h_inv, state.K)
        K_ss = np.einsum("ij...,i...,j...->...", state.K, s_upper, s_upper)
        return divergence + K_ss - trace_K

    def mean_expansion(
        self, state: ADMState, radius: float, center=(0.0, 0.0, 0.0)
    ) -> float:
        sphere = sphere_quadrature(radius, center, self.n_theta, self.n_phi)
        values = trilinear_sample(self.grid, self.expansion_field(state, center), sphere.points)
        return float(np.sum(sphere.weights * values) / (4.0 * np.pi))

    def find(
        self,
        state: ADMState,
        minimum_radius: float,
        maximum_radius: float,
        center=(0.0, 0.0, 0.0),
        tolerance: float = 1e-4,
        maximum_iterations: int = 60,
    ) -> HorizonResult:
        lower, upper = float(minimum_radius), float(maximum_radius)
        f_lower = self.mean_expansion(state, lower, center)
        f_upper = self.mean_expansion(state, upper, center)
        if f_lower * f_upper > 0.0:
            return HorizonResult(False, np.nan, np.nan, np.nan, min(abs(f_lower), abs(f_upper)), 0)
        for iteration in range(1, maximum_iterations + 1):
            middle = 0.5 * (lower + upper)
            f_middle = self.mean_expansion(state, middle, center)
            if abs(f_middle) < tolerance or upper - lower < tolerance:
                break
            if f_lower * f_middle <= 0.0:
                upper, f_upper = middle, f_middle
            else:
                lower, f_lower = middle, f_middle
        sphere = sphere_quadrature(middle, center, self.n_theta, self.n_phi)
        h_samples = np.moveaxis(trilinear_sample(self.grid, state.h, sphere.points), -1, 0)
        # Induced area for coordinate sphere: sqrt(det h) sqrt(n_i h^ij n_j) r^2 dOmega.
        determinant = np.linalg.det(h_samples)
        inverse = np.linalg.inv(h_samples)
        normal_norm = np.einsum("ni,nij,nj->n", sphere.normals, inverse, sphere.normals)
        area_density = np.sqrt(np.maximum(determinant * normal_norm, 0.0)) * middle**2
        area = float(np.sum(sphere.weights * area_density))
        return HorizonResult(True, middle, area, np.sqrt(area / (16.0 * np.pi)), f_middle, iteration)


@dataclass(frozen=True)
class WaveExtraction:
    radius: float
    psi4: Array
    theta: Array
    phi: Array
    weights: Array
    modes_l2: dict[int, complex]


class Psi4Extractor:
    """Finite-radius vacuum Psi4 extraction using an orthonormal spatial dyad."""

    def __init__(self, grid: CartesianGrid, n_theta: int = 16, n_phi: int = 32) -> None:
        self.grid = grid
        self.n_theta = n_theta
        self.n_phi = n_phi

    def electric_magnetic_weyl(self, state: ADMState) -> tuple[Array, Array]:
        h_inv, _, sqrt_h = inverse_metric(state.h)
        gamma = christoffel(self.grid, state.h, h_inv)
        ricci = ricci_tensor(self.grid, state.h, h_inv, gamma)
        trace_K = np.einsum("ij...,ij...->...", h_inv, state.K)
        K_mixed = np.einsum("ik...,kj...->ij...", h_inv, state.K)
        electric = ricci + trace_K[None, None, ...] * state.K
        electric -= np.einsum("ik...,kj...->ij...", state.K, K_mixed)
        trace_e = np.einsum("ij...,ij...->...", h_inv, electric)
        electric -= state.h * (trace_e / 3.0)[None, None, ...]
        covariant_dK = self.grid.zeros((3, 3, 3))
        for k in range(3):
            for l in range(3):
                for j in range(3):
                    covariant_dK[k, l, j] = self.grid.derivative(state.K[l, j], k)
                    for m in range(3):
                        covariant_dK[k, l, j] -= gamma[m, k, l] * state.K[m, j]
                        covariant_dK[k, l, j] -= gamma[m, k, j] * state.K[l, m]
        symbol = _levi_civita()
        magnetic = np.einsum(
            "im...,mkl,klj...->ij...", state.h, symbol, covariant_dK
        ) / sqrt_h[None, None, ...]
        magnetic = 0.5 * (magnetic + np.swapaxes(magnetic, 0, 1))
        return electric, magnetic

    @staticmethod
    def _spin_minus_two_l2(theta: Array, phi: Array, m: int) -> Array:
        c, s = np.cos(theta), np.sin(theta)
        if m == 2:
            return np.sqrt(5 / (64 * np.pi)) * (1 + c) ** 2 * np.exp(2j * phi)
        if m == 1:
            return np.sqrt(5 / (16 * np.pi)) * s * (1 + c) * np.exp(1j * phi)
        if m == 0:
            return np.sqrt(15 / (32 * np.pi)) * s**2
        if m == -1:
            return np.sqrt(5 / (16 * np.pi)) * s * (1 - c) * np.exp(-1j * phi)
        if m == -2:
            return np.sqrt(5 / (64 * np.pi)) * (1 - c) ** 2 * np.exp(-2j * phi)
        raise ValueError("l=2 requires -2 <= m <= 2")

    def extract(self, state: ADMState, radius: float, center=(0.0, 0.0, 0.0)) -> WaveExtraction:
        sphere = sphere_quadrature(radius, center, self.n_theta, self.n_phi)
        electric, magnetic = self.electric_magnetic_weyl(state)
        E = np.moveaxis(trilinear_sample(self.grid, electric, sphere.points), -1, 0)
        B = np.moveaxis(trilinear_sample(self.grid, magnetic, sphere.points), -1, 0)
        h = np.moveaxis(trilinear_sample(self.grid, state.h, sphere.points), -1, 0)
        radial = sphere.normals.copy()
        radial /= np.sqrt(np.einsum("ni,nij,nj->n", radial, h, radial))[:, None]
        theta_vector = np.stack(
            (np.cos(sphere.theta) * np.cos(sphere.phi), np.cos(sphere.theta) * np.sin(sphere.phi), -np.sin(sphere.theta)), axis=1
        )
        theta_vector -= radial * np.einsum("ni,nij,nj->n", radial, h, theta_vector)[:, None]
        theta_vector /= np.sqrt(np.einsum("ni,nij,nj->n", theta_vector, h, theta_vector))[:, None]
        phi_vector = np.cross(radial, theta_vector)
        phi_vector /= np.sqrt(np.einsum("ni,nij,nj->n", phi_vector, h, phi_vector))[:, None]
        mbar = (theta_vector - 1j * phi_vector) / np.sqrt(2.0)
        psi4 = np.einsum("nij,ni,nj->n", E - 1j * B, mbar, mbar)
        modes = {
            m: complex(np.sum(sphere.weights * psi4 * np.conjugate(self._spin_minus_two_l2(sphere.theta, sphere.phi, m))))
            for m in range(-2, 3)
        }
        return WaveExtraction(radius, psi4, sphere.theta, sphere.phi, sphere.weights, modes)
