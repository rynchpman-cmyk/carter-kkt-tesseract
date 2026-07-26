"""Theory 3.3 two-current master-function material sector.

The implementation uses the zero-entrainment production slice of the M1
master function by default, while retaining the complete invariant stress and
momentum formulas.  The executable characteristic audit is obtained directly
from the two conservation laws and the two momentum-vorticity equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .adm import inverse_metric
from .grid import Array
from .matter import FluidPrimitive, lorentz_factor
from .theory2 import StressEnergy3p1


@dataclass(frozen=True)
class Theory33MasterParameters:
    gamma_ad: float = 5.0 / 3.0
    carrier_charge: float = 1.0
    carrier_K: float = 0.2
    carrier_sound_speed: float = 0.9
    chemical_susceptibility: float = 10.0
    target_fraction: float = 0.05
    entrainment_linear: float = 0.0
    entrainment_quadratic: float = 0.0
    entrainment_scale_fourth: float = 1.0
    relaxation_time: float = 1.0
    carrier_floor: float = 1.0e-10
    convexity_floor: float = 1.0e-10
    characteristic_condition_limit: float = 1.0e8
    maximum_relative_lorentz_factor: float = 2.0

    def __post_init__(self) -> None:
        if not 1.0 < self.gamma_ad < 2.0:
            raise ValueError("gamma_ad must lie in (1, 2)")
        if self.carrier_charge == 0.0:
            raise ValueError("minimal Theory 3.3 requires nonzero carrier_charge")
        if self.carrier_K <= 0.0:
            raise ValueError("carrier_K must be positive")
        if not 0.0 < self.carrier_sound_speed < 1.0:
            raise ValueError("carrier_sound_speed must lie in (0, 1)")
        if self.chemical_susceptibility <= 0.0:
            raise ValueError("chemical_susceptibility must be positive")
        if self.target_fraction < 0.0:
            raise ValueError("target_fraction must be nonnegative")
        if self.entrainment_linear < 0.0 or self.entrainment_quadratic < 0.0:
            raise ValueError("entrainment coefficients must be nonnegative")
        if self.entrainment_scale_fourth <= 0.0:
            raise ValueError("entrainment scale must be positive")
        if self.relaxation_time <= 0.0:
            raise ValueError("relaxation_time must be positive")
        if self.carrier_floor <= 0.0 or self.convexity_floor <= 0.0:
            raise ValueError("Theory 3.3 floors must be positive")
        if self.characteristic_condition_limit <= 1.0:
            raise ValueError("characteristic condition limit is invalid")
        if self.maximum_relative_lorentz_factor <= 1.0:
            raise ValueError("maximum relative Lorentz factor must exceed one")


@dataclass(frozen=True)
class CarrierPrimitive:
    number_density: Array
    velocity: Array


@dataclass(frozen=True)
class MasterState:
    entropy_per_baryon: Array
    temperature: Array
    relative_lorentz_factor: Array
    relative_invariant: Array
    lambda_mf: Array
    generalized_pressure: Array
    B_N: Array
    B_D: Array
    entrainment: Array
    chemical_N: Array
    chemical_D: Array
    thermodynamic_hessian_nn: Array
    thermodynamic_hessian_nd: Array
    thermodynamic_hessian_dd: Array
    stress: StressEnergy3p1
    carrier_momentum_down: Array
    carrier_momentum_normal: Array
    carrier_eulerian_density: Array
    source_charge_eulerian: Array
    source_current_up: Array
    drag_inertia: Array
    invariant_gradient: Array
    phase_two: Array
    phase_threshold: Array
    mobility_matrix: Array
    constitutive_model: str


@dataclass(frozen=True)
class CharacteristicAudit:
    speeds: np.ndarray
    maximum_absolute_speed: float
    eigenvector_condition: float
    legendre_minimum_eigenvalue: float
    thermodynamic_minimum_eigenvalue: float
    strongly_hyperbolic: bool
    causal: bool


class Theory33MasterFunction:
    def __init__(
        self,
        parameters: Theory33MasterParameters | None = None,
        constitutive_closure: Any | None = None,
    ) -> None:
        self.parameters = parameters or Theory33MasterParameters()
        if (
            constitutive_closure is not None
            and getattr(constitutive_closure, "parameters", self.parameters)
            != self.parameters
        ):
            raise ValueError(
                "frozen constitutive artifact parameters do not match "
                "the Theory 3.3 production parameters"
            )
        self.constitutive_closure = constitutive_closure
        self.constitutive_model = (
            "analytic_m1"
            if constitutive_closure is None
            else "qualified_frozen"
        )

    def specific_entropy(self, density: Array, internal: Array) -> Array:
        p = self.parameters
        pressure = (p.gamma_ad - 1.0) * density * internal
        K = pressure / np.maximum(density**p.gamma_ad, 1.0e-300)
        return np.log(np.maximum(K, 1.0e-300)) / (p.gamma_ad - 1.0)

    def thermal_pressure(self, density: Array, internal: Array) -> Array:
        return (self.parameters.gamma_ad - 1.0) * density * internal

    def thermal_energy(self, density: Array, internal: Array) -> Array:
        return density * (1.0 + internal)

    def lambda_from_invariants(
        self, n_squared: Array, d_squared: Array, x_squared: Array, entropy: Array
    ) -> Array:
        """Evaluate M1 directly from its four scalar action arguments."""
        p = self.parameters
        n = np.sqrt(np.maximum(n_squared, 1.0e-300))
        d = np.sqrt(np.maximum(d_squared, 1.0e-300))
        pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        thermal = n + pressure / (p.gamma_ad - 1.0)
        anchor = d - p.target_fraction * n
        rho0 = (
            thermal
            + p.carrier_K * d ** (1.0 + p.carrier_sound_speed**2)
            + 0.5 * anchor**2 / p.chemical_susceptibility
        )
        relative = x_squared - n * d
        return (
            -rho0
            - p.entrainment_linear * relative
            - 0.5
            * p.entrainment_quadratic
            * relative**2
            / p.entrainment_scale_fourth
        )

    def invariant_response(
        self,
        n_squared: Array,
        d_squared: Array,
        x_squared: Array,
        entropy: Array,
        *,
        phase_override: Array | bool | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return Lambda, its invariant gradient, hard phase, threshold, mobility.

        The gradient order is ``(n², d², x², entropy)``.  A supplied phase
        override freezes the nonsmooth classifier for a local Newton/Jacobian
        evaluation; the ordinary public evaluation always classifies the
        current state.
        """
        n2, d2, cross, sigma = np.broadcast_arrays(
            np.asarray(n_squared, dtype=float),
            np.asarray(d_squared, dtype=float),
            np.asarray(x_squared, dtype=float),
            np.asarray(entropy, dtype=float),
        )
        if self.constitutive_closure is not None:
            if phase_override is None:
                response = self.constitutive_closure.evaluate(
                    n2, d2, cross, sigma
                )
            else:
                response = self.constitutive_closure.evaluate_branch(
                    n2, d2, cross, sigma, phase_override
                )
            return (
                np.asarray(response.lambda_value, dtype=float),
                np.asarray(response.lambda_gradient, dtype=float),
                np.asarray(response.phase_two, dtype=bool),
                np.asarray(response.phase_threshold, dtype=float),
                np.asarray(response.mobility_matrix, dtype=float),
            )

        p = self.parameters
        n = np.sqrt(np.maximum(n2, 1.0e-300))
        d = np.sqrt(np.maximum(d2, 1.0e-300))
        pressure = np.exp((p.gamma_ad - 1.0) * sigma) * n**p.gamma_ad
        anchor = d - p.target_fraction * n
        rho_n = (
            1.0
            + p.gamma_ad
            * pressure
            / np.maximum((p.gamma_ad - 1.0) * n, 1.0e-300)
            - p.target_fraction * anchor / p.chemical_susceptibility
        )
        rho_d = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * d ** (p.carrier_sound_speed**2)
            + anchor / p.chemical_susceptibility
        )
        relative = cross - n * d
        entrainment = p.entrainment_linear + (
            p.entrainment_quadratic
            * relative
            / p.entrainment_scale_fourth
        )
        gradient = np.stack(
            [
                (-rho_n + entrainment * d)
                / np.maximum(2.0 * n, 1.0e-300),
                (-rho_d + entrainment * n)
                / np.maximum(2.0 * d, 1.0e-300),
                -entrainment,
                -pressure,
            ],
            axis=-1,
        )
        shape = n.shape
        return (
            self.lambda_from_invariants(n2, d2, cross, sigma),
            gradient,
            np.zeros(shape, dtype=bool),
            np.full(shape, np.inf),
            np.zeros(shape + (2, 2), dtype=float),
        )

    def _thermodynamic_hessian(
        self,
        n: Array,
        d: Array,
        gamma_rel: Array,
        entropy: Array,
        phase_two: Array,
    ) -> tuple[Array, Array, Array]:
        p = self.parameters
        if self.constitutive_closure is None:
            pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
            return (
                p.gamma_ad * pressure / np.maximum(n**2, 1.0e-300)
                + p.target_fraction**2 / p.chemical_susceptibility,
                np.full_like(n, -p.target_fraction / p.chemical_susceptibility),
                p.carrier_K
                * (1.0 + p.carrier_sound_speed**2)
                * p.carrier_sound_speed**2
                * d ** (p.carrier_sound_speed**2 - 1.0)
                + 1.0 / p.chemical_susceptibility,
            )

        def chemical(
            density: Array, carrier: Array
        ) -> tuple[Array, Array]:
            _, gradient, _, _, _ = self.invariant_response(
                density**2,
                carrier**2,
                density * carrier * gamma_rel,
                entropy,
                phase_override=phase_two,
            )
            rho_n = -(
                2.0 * density * gradient[..., 0]
                + carrier * gamma_rel * gradient[..., 2]
            )
            rho_d = -(
                2.0 * carrier * gradient[..., 1]
                + density * gamma_rel * gradient[..., 2]
            )
            return rho_n, rho_d

        step_n = 2.0e-5 * np.maximum(n, 1.0e-3)
        step_d = 2.0e-5 * np.maximum(d, 1.0e-4)
        rho_n_plus, rho_d_n_plus = chemical(n + step_n, d)
        rho_n_minus, rho_d_n_minus = chemical(
            np.maximum(n - step_n, 1.0e-300), d
        )
        rho_n_d_plus, rho_d_plus = chemical(n, d + step_d)
        rho_n_d_minus, rho_d_minus = chemical(
            n, np.maximum(d - step_d, 1.0e-300)
        )
        h_nn = (rho_n_plus - rho_n_minus) / (2.0 * step_n)
        h_dd = (rho_d_plus - rho_d_minus) / (2.0 * step_d)
        h_nd = 0.5 * (
            (rho_n_d_plus - rho_n_d_minus) / (2.0 * step_d)
            + (rho_d_n_plus - rho_d_n_minus) / (2.0 * step_n)
        )
        return h_nn, h_nd, h_dd

    def evaluate(
        self,
        h: Array,
        fluid: FluidPrimitive,
        carrier: CarrierPrimitive,
    ) -> MasterState:
        p = self.parameters
        n = np.asarray(fluid.baryon_density, dtype=float)
        d = np.asarray(carrier.number_density, dtype=float)
        if np.any(n <= 0.0) or np.any(d <= 0.0):
            raise ValueError("master-function densities must be positive")
        u = np.asarray(fluid.velocity, dtype=float)
        v = np.asarray(carrier.velocity, dtype=float)
        W_N = lorentz_factor(h, u)
        W_D = lorentz_factor(h, v)
        u_down = np.einsum("ij...,j...->i...", h, u)
        v_down = np.einsum("ij...,j...->i...", h, v)
        dot = np.einsum("ij...,i...,j...->...", h, u, v)
        gamma_rel = W_N * W_D * (1.0 - dot)
        gamma_rel = np.maximum(gamma_rel, 1.0)
        x2 = n * d * gamma_rel
        relative = x2 - n * d

        entropy = self.specific_entropy(n, fluid.specific_internal_energy)
        (
            lambda_mf,
            invariant_gradient,
            phase_two,
            phase_threshold,
            mobility_matrix,
        ) = self.invariant_response(
            n**2,
            d**2,
            x2,
            entropy,
        )
        B_N = -2.0 * invariant_gradient[..., 0]
        B_D = -2.0 * invariant_gradient[..., 1]
        entrainment = -invariant_gradient[..., 2]
        rho_n = B_N * n + entrainment * d
        rho_d = B_D * d + entrainment * n
        generalized_pressure = (
            lambda_mf
            + B_N * n**2
            + B_D * d**2
            + 2.0 * entrainment * x2
        )

        nW = n * W_N
        dW = d * W_D
        rho = (
            -generalized_pressure
            + B_N * nW**2
            + B_D * dW**2
            + 2.0 * entrainment * nW * dW
        )
        momentum = (
            B_N[None, ...] * nW[None, ...] ** 2 * u_down
            + B_D[None, ...] * dW[None, ...] ** 2 * v_down
            + entrainment[None, ...]
            * nW[None, ...]
            * dW[None, ...]
            * (u_down + v_down)
        )
        stress = generalized_pressure[None, None, ...] * h
        stress += (
            B_N[None, None, ...]
            * nW[None, None, ...] ** 2
            * u_down[:, None, ...]
            * u_down[None, :, ...]
        )
        stress += (
            B_D[None, None, ...]
            * dW[None, None, ...] ** 2
            * v_down[:, None, ...]
            * v_down[None, :, ...]
        )
        stress += (
            entrainment[None, None, ...]
            * nW[None, None, ...]
            * dW[None, None, ...]
            * (
                u_down[:, None, ...] * v_down[None, :, ...]
                + v_down[:, None, ...] * u_down[None, :, ...]
            )
        )
        h_inv = inverse_metric(h)[0]
        trace = np.einsum("ij...,ij...->...", h_inv, stress)

        chi_down = (
            B_D[None, ...] * dW[None, ...] * v_down
            + entrainment[None, ...] * nW[None, ...] * u_down
        )
        chi_normal = B_D * dW + entrainment * nW
        source_rho = p.carrier_charge * dW
        source_current = p.carrier_charge * dW[None, ...] * v

        rho_nn, rho_nd, rho_dd = self._thermodynamic_hessian(
            n,
            d,
            gamma_rel,
            entropy,
            phase_two,
        )
        pressure = self.thermal_pressure(n, fluid.specific_internal_energy)
        temperature = pressure / np.maximum(n, 1.0e-300)
        drag_inertia = np.maximum(B_D * d**2, 0.0)
        return MasterState(
            entropy_per_baryon=entropy,
            temperature=temperature,
            relative_lorentz_factor=gamma_rel,
            relative_invariant=relative,
            lambda_mf=lambda_mf,
            generalized_pressure=generalized_pressure,
            B_N=B_N,
            B_D=B_D,
            entrainment=entrainment,
            chemical_N=rho_n,
            chemical_D=rho_d,
            thermodynamic_hessian_nn=rho_nn,
            thermodynamic_hessian_nd=rho_nd,
            thermodynamic_hessian_dd=rho_dd,
            stress=StressEnergy3p1(rho, momentum, stress, trace),
            carrier_momentum_down=chi_down,
            carrier_momentum_normal=chi_normal,
            carrier_eulerian_density=dW,
            source_charge_eulerian=source_rho,
            source_current_up=source_current,
            drag_inertia=drag_inertia,
            invariant_gradient=invariant_gradient,
            phase_two=phase_two,
            phase_threshold=phase_threshold,
            mobility_matrix=mobility_matrix,
            constitutive_model=self.constitutive_model,
        )

    def conserved_carrier(
        self, h: Array, state: MasterState
    ) -> tuple[Array, Array]:
        sqrt_h = inverse_metric(h)[2]
        D_D = sqrt_h * state.carrier_eulerian_density
        P_D = D_D[None, ...] * state.carrier_momentum_down
        return D_D, P_D

    def drag_force_down(
        self,
        h: Array,
        fluid: FluidPrimitive,
        carrier: CarrierPrimitive,
        state: MasterState,
    ) -> tuple[Array, Array]:
        p = self.parameters
        W_N = lorentz_factor(h, fluid.velocity)
        W_D = lorentz_factor(h, carrier.velocity)
        u_down = np.einsum("ij...,j...->i...", h, fluid.velocity)
        v_down = np.einsum("ij...,j...->i...", h, carrier.velocity)
        resistance = state.drag_inertia / p.relaxation_time
        force = resistance[None, ...] * (
            W_N[None, ...] * u_down
            - state.relative_lorentz_factor[None, ...]
            * W_D[None, ...]
            * v_down
        )
        heat = resistance * (state.relative_lorentz_factor**2 - 1.0)
        return force, np.maximum(heat, 0.0)

    def equilibrium_audit(
        self, density: float, internal: float, carrier_density: float
    ) -> CharacteristicAudit:
        return self.characteristic_audit(
            density,
            internal,
            0.0,
            carrier_density,
            0.0,
        )

    def _flat_constitutive_vector(
        self,
        primitive: np.ndarray,
        entropy: float,
    ) -> np.ndarray:
        """Return (N0, mu_x, C0, chi_x, Nx, -mu_0, Cx, -chi_0)."""
        log_n, rapid_N, log_d, rapid_D = primitive
        n = float(np.exp(log_n))
        d = float(np.exp(log_d))
        p = self.parameters
        pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        internal = pressure / ((p.gamma_ad - 1.0) * n)
        u = float(np.tanh(rapid_N))
        v = float(np.tanh(rapid_D))
        WN = float(np.cosh(rapid_N))
        WD = float(np.cosh(rapid_D))
        h = np.eye(3).reshape(3, 3, 1)
        fluid = FluidPrimitive(
            np.array([n]),
            np.array([internal]),
            np.array([[u], [0.0], [0.0]]),
            np.array([0.0]),
        )
        carrier = CarrierPrimitive(
            np.array([d]), np.array([[v], [0.0], [0.0]])
        )
        state = self.evaluate(h, fluid, carrier)
        mu_x = state.B_N[0] * n * WN * u + state.entrainment[0] * d * WD * v
        mu_normal = state.B_N[0] * n * WN + state.entrainment[0] * d * WD
        chi_x = state.carrier_momentum_down[0, 0]
        chi_normal = state.carrier_momentum_normal[0]
        return np.array(
            [
                n * WN,
                mu_x,
                d * WD,
                chi_x,
                n * WN * u,
                mu_normal,
                d * WD * v,
                chi_normal,
            ],
            dtype=float,
        )

    @staticmethod
    def _numerical_jacobian(function, point: np.ndarray) -> np.ndarray:
        base = np.asarray(function(point), dtype=float)
        jac = np.empty((base.size, point.size), dtype=float)
        for column in range(point.size):
            step = 2.0e-6 * max(1.0, abs(float(point[column])))
            plus = point.copy()
            minus = point.copy()
            plus[column] += step
            minus[column] -= step
            jac[:, column] = (function(plus) - function(minus)) / (2.0 * step)
        return jac

    def characteristic_audit(
        self,
        density: float,
        internal: float,
        velocity: float,
        carrier_density: float,
        carrier_velocity: float,
    ) -> CharacteristicAudit:
        if density <= 0.0 or internal <= 0.0 or carrier_density <= 0.0:
            raise ValueError("characteristic audit needs positive densities and energy")
        if abs(velocity) >= 1.0 or abs(carrier_velocity) >= 1.0:
            raise ValueError("characteristic audit velocities must be subluminal")
        point = np.array(
            [
                np.log(density),
                np.arctanh(velocity),
                np.log(carrier_density),
                np.arctanh(carrier_velocity),
            ]
        )
        entropy = float(
            self.specific_entropy(
                np.array([density]), np.array([internal])
            )[0]
        )
        jac = self._numerical_jacobian(
            lambda value: self._flat_constitutive_vector(value, entropy), point
        )
        time_matrix = jac[:4]
        space_matrix = jac[4:]
        symbol = np.linalg.solve(time_matrix, space_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(symbol)
        imag = float(np.max(np.abs(eigenvalues.imag)))
        speeds = np.sort(eigenvalues.real)
        condition = float(np.linalg.cond(eigenvectors))

        h = np.eye(3).reshape(3, 3, 1)
        fluid = FluidPrimitive(
            np.array([density]),
            np.array([internal]),
            np.array([[velocity], [0.0], [0.0]]),
            np.array([0.0]),
        )
        carrier = CarrierPrimitive(
            np.array([carrier_density]),
            np.array([[carrier_velocity], [0.0], [0.0]]),
        )
        state = self.evaluate(h, fluid, carrier)
        legendre = np.array(
            [
                [state.B_N[0], state.entrainment[0]],
                [state.entrainment[0], state.B_D[0]],
            ]
        )
        thermo = np.array(
            [
                [state.thermodynamic_hessian_nn[0], state.thermodynamic_hessian_nd[0]],
                [state.thermodynamic_hessian_nd[0], state.thermodynamic_hessian_dd[0]],
            ]
        )
        legendre_min = float(np.min(np.linalg.eigvalsh(legendre)))
        thermo_min = float(np.min(np.linalg.eigvalsh(thermo)))
        strong = (
            imag < 1.0e-8
            and np.isfinite(condition)
            and condition < self.parameters.characteristic_condition_limit
            and legendre_min > self.parameters.convexity_floor
            and thermo_min > self.parameters.convexity_floor
        )
        maximum = float(np.max(np.abs(speeds)))
        return CharacteristicAudit(
            speeds,
            maximum,
            condition,
            legendre_min,
            thermo_min,
            strong,
            strong and maximum <= 1.0 + 1.0e-8,
        )
