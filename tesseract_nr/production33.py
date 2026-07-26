"""CCZ4 + mixed Proca + Theory 3.3 two-current production reference.

The Theory 3.2 state layout is reused for checkpoint and integrator
compatibility, but ``target_charge`` stores the densitized carrier source
charge and ``target_current`` stores the densitized carrier canonical
momentum.  Neither is a phenomenological reservoir variable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adm import inverse_metric
from .ccz4 import CCZ4Parameters, CCZ4State
from .grhd import GRHDParameters
from .grid import Array, PeriodicGrid
from .io import load_checkpoint, save_checkpoint
from .matter import FluidPrimitive, lorentz_factor
from .production3 import (
    FiniteDampingReport,
    Theory3MatterState,
    Theory3ProductionParameters,
    Theory3ProductionSolver,
    Theory3ProductionState,
    _add_stress,
)
from .theory3 import Theory3Parameters
from .theory33 import (
    CarrierPrimitive,
    MasterState,
    Theory33MasterFunction,
    Theory33MasterParameters,
)


@dataclass(frozen=True)
class Theory33RecoveryReport:
    failed_cells: int
    atmosphere_cells: int
    maximum_residual: float
    minimum_carrier_density: float
    maximum_relative_lorentz_factor: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float


@dataclass(frozen=True)
class Theory33Recovery:
    fluid: FluidPrimitive
    carrier: CarrierPrimitive
    master: MasterState
    entropy_per_baryon: Array
    sound_speed: Array
    material_stress: Any
    vector_stress: Any
    total_stress: Any
    target_charge_eulerian: Array
    target_current_up: Array
    report: Theory33RecoveryReport


class Theory33ProductionSolver(Theory3ProductionSolver):
    def __init__(
        self,
        grid: PeriodicGrid,
        master: Theory33MasterParameters | None = None,
        vector: Theory3Parameters | None = None,
        ccz4: CCZ4Parameters | None = None,
        grhd: GRHDParameters | None = None,
        production: Theory3ProductionParameters | None = None,
        constitutive_closure: Any | None = None,
    ) -> None:
        master_parameters = master or Theory33MasterParameters()
        if vector is None:
            vector = Theory3Parameters(
                target_speed=max(master_parameters.carrier_sound_speed, 0.9),
                target_relaxation_time=master_parameters.relaxation_time,
                target_strength=0.0,
                gamma_ad=master_parameters.gamma_ad,
            )
        if abs(vector.gamma_ad - master_parameters.gamma_ad) > 1.0e-14:
            raise ValueError("Theory 3.3 master and vector gamma_ad must agree")
        super().__init__(grid, vector, ccz4, grhd, production)
        self.master_parameters = master_parameters
        self.master = Theory33MasterFunction(
            master_parameters,
            constitutive_closure=constitutive_closure,
        )
        self.constitutive_model = self.master.constitutive_model
        self.constitutive_variant_digest = getattr(
            constitutive_closure,
            "variant_digest",
            None,
        )
        self._primitive_guess: Array | None = None
        self.last_drag_heat = 0.0
        self.last_drag_entropy_change = 0.0
        self.last_projection_entropy_change = 0.0
        self.last_projection_minimum_change = 0.0
        self.last_step_entropy_change = 0.0
        self.last_carrier_boundary_number_outflow = 0.0
        self.last_carrier_number_change = 0.0
        self.last_carrier_number_balance_residual = 0.0

    def _entropy_internal(self, density: float, entropy: float) -> float:
        p = self.master_parameters
        log_K = np.clip((p.gamma_ad - 1.0) * entropy, -700.0, 700.0)
        pressure = np.exp(log_K) * max(density, 1.0e-300) ** p.gamma_ad
        return pressure / (
            (p.gamma_ad - 1.0) * max(density, 1.0e-300)
        )

    def _cell_predictions(
        self,
        x: np.ndarray,
        h: np.ndarray,
        sqrt_h: float,
        entropy_per_baryon: float,
        phase_override: bool | None = None,
    ) -> np.ndarray:
        n = float(np.exp(np.clip(x[0], -700.0, 700.0)))
        q_N = np.asarray(x[1:4], dtype=float)
        d = float(np.exp(np.clip(x[4], -700.0, 700.0)))
        q_D = np.asarray(x[5:8], dtype=float)
        W_N = float(np.sqrt(1.0 + q_N @ h @ q_N))
        W_D = float(np.sqrt(1.0 + q_D @ h @ q_D))
        u = q_N / W_N
        v = q_D / W_D
        u_down = h @ u
        v_down = h @ v
        gamma_rel = max(W_N * W_D - q_N @ h @ q_D, 1.0)
        x2 = n * d * gamma_rel
        Lambda, gradient, _, _, _ = self.master.invariant_response(
            n**2,
            d**2,
            x2,
            entropy_per_baryon,
            phase_override=phase_override,
        )
        B_N = float(-2.0 * gradient[..., 0])
        B_D = float(-2.0 * gradient[..., 1])
        Q = float(-gradient[..., 2])
        Lambda = float(Lambda)
        Psi = Lambda + B_N * n**2 + B_D * d**2 + 2.0 * Q * x2
        nW = n * W_N
        dW = d * W_D
        energy = -Psi + B_N * nW**2 + B_D * dW**2 + 2.0 * Q * nW * dW
        momentum = (
            B_N * nW**2 * u_down
            + B_D * dW**2 * v_down
            + Q * nW * dW * (u_down + v_down)
        )
        chi = B_D * dW * v_down + Q * nW * u_down
        D_N = sqrt_h * nW
        D_D = sqrt_h * dW
        P_D = D_D * chi
        return np.concatenate(([D_N, D_D], momentum, [energy], P_D))

    def _cell_phase(
        self,
        x: np.ndarray,
        h: np.ndarray,
        entropy_per_baryon: float,
    ) -> bool | None:
        """Classify a primitive once for a piecewise-constant local solve."""
        if self.master.constitutive_closure is None:
            return None
        n = float(np.exp(np.clip(x[0], -700.0, 700.0)))
        d = float(np.exp(np.clip(x[4], -700.0, 700.0)))
        q_N = np.asarray(x[1:4], dtype=float)
        q_D = np.asarray(x[5:8], dtype=float)
        W_N = float(np.sqrt(1.0 + q_N @ h @ q_N))
        W_D = float(np.sqrt(1.0 + q_D @ h @ q_D))
        gamma_rel = max(W_N * W_D - q_N @ h @ q_D, 1.0)
        _, _, phase_two, _, _ = self.master.invariant_response(
            n**2,
            d**2,
            n * d * gamma_rel,
            entropy_per_baryon,
        )
        return bool(phase_two)

    @staticmethod
    def _cell_jacobian(function, point: np.ndarray) -> np.ndarray:
        base = function(point)
        jac = np.empty((base.size, point.size), dtype=float)
        for column in range(point.size):
            step = 2.0e-6 * max(1.0, abs(float(point[column])))
            plus = point.copy()
            minus = point.copy()
            plus[column] += step
            minus[column] -= step
            jac[:, column] = (function(plus) - function(minus)) / (2.0 * step)
        return jac

    def _recover_cells(
        self,
        h: Array,
        D_N: Array,
        D_D: Array,
        momentum: Array,
        energy: Array,
        P_D: Array,
        entropy_per_baryon: Array,
    ) -> tuple[FluidPrimitive, CarrierPrimitive, Array, int]:
        shape = self.grid.shape
        count = int(np.prod(shape))
        h_flat = h.reshape(3, 3, count)
        sqrt_flat = inverse_metric(h)[2].reshape(count)
        D_N_flat = D_N.reshape(count)
        D_D_flat = D_D.reshape(count)
        S_flat = momentum.reshape(3, count)
        E_flat = energy.reshape(count)
        P_flat = P_D.reshape(3, count)
        entropy_flat = entropy_per_baryon.reshape(count)
        guess_flat = None
        if self._primitive_guess is not None and self._primitive_guess.shape == (8,) + shape:
            guess_flat = self._primitive_guess.reshape(8, count)
        solved = np.empty((8, count), dtype=float)
        residuals = np.empty(count, dtype=float)
        failed = 0
        for cell in range(count):
            target = np.concatenate(
                (
                    [D_N_flat[cell], D_D_flat[cell]],
                    S_flat[:, cell],
                    [E_flat[cell]],
                    P_flat[:, cell],
                )
            )
            energy_scale = max(abs(E_flat[cell]), np.linalg.norm(S_flat[:, cell]), 1.0e-10)
            carrier_scale = max(abs(D_D_flat[cell]), 1.0e-10)
            momentum_scale = max(np.linalg.norm(P_flat[:, cell]), carrier_scale, 1.0e-10)
            scale = np.array(
                [
                    max(abs(D_N_flat[cell]), 1.0e-10),
                    carrier_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    momentum_scale,
                    momentum_scale,
                    momentum_scale,
                ]
            )
            if guess_flat is not None:
                x = guess_flat[:, cell].copy()
            else:
                n0 = max(D_N_flat[cell] / max(sqrt_flat[cell], 1.0e-300), self.grhd.parameters.density_floor)
                d0 = max(D_D_flat[cell] / max(sqrt_flat[cell], 1.0e-300), self.master_parameters.carrier_floor)
                denominator = max(E_flat[cell] + 1.0e-8, 1.0e-8)
                velocity = np.linalg.solve(h_flat[:, :, cell], S_flat[:, cell]) / denominator
                speed_sq = float(velocity @ h_flat[:, :, cell] @ velocity)
                if speed_sq >= 0.5:
                    velocity *= np.sqrt(0.5 / max(speed_sq, 1.0e-300))
                W0 = 1.0 / np.sqrt(1.0 - float(velocity @ h_flat[:, :, cell] @ velocity))
                q0 = W0 * velocity
                x = np.concatenate(([np.log(n0)], q0, [np.log(d0)], q0))

            phase_override = self._cell_phase(
                x,
                h_flat[:, :, cell],
                float(entropy_flat[cell]),
            )
            function = lambda value: self._cell_predictions(
                value,
                h_flat[:, :, cell],
                float(sqrt_flat[cell]),
                float(entropy_flat[cell]),
                phase_override,
            )
            best = np.inf
            for _ in range(self.production_parameters.recovery_iterations):
                prediction = function(x)
                residual = (prediction - target) / scale
                norm = float(np.max(np.abs(residual)))
                best = min(best, norm)
                if norm < max(self.production_parameters.recovery_tolerance, 2.0e-8):
                    break
                jac = self._cell_jacobian(function, x) / scale[:, None]
                delta = np.linalg.lstsq(jac, -residual, rcond=1.0e-12)[0]
                accepted = False
                for power in range(10):
                    trial = x + delta * (0.5**power)
                    trial[0] = np.clip(trial[0], -700.0, 700.0)
                    trial[4] = np.clip(trial[4], -700.0, 700.0)
                    trial_residual = (function(trial) - target) / scale
                    if np.max(np.abs(trial_residual)) < norm:
                        x = trial
                        accepted = True
                        break
                if not accepted:
                    break
            final = float(np.max(np.abs((function(x) - target) / scale)))
            residuals[cell] = final
            if not np.isfinite(final) or final > 2.0e-6:
                failed += 1
            final_phase = self._cell_phase(
                x,
                h_flat[:, :, cell],
                float(entropy_flat[cell]),
            )
            if (
                phase_override is not None
                and final_phase != phase_override
            ):
                failed += 1
            solved[:, cell] = x

        self._primitive_guess = solved.reshape((8,) + shape)
        n = np.exp(solved[0]).reshape(shape)
        d = np.exp(solved[4]).reshape(shape)
        q_N = solved[1:4].reshape((3,) + shape)
        q_D = solved[5:8].reshape((3,) + shape)
        W_N = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_N, q_N))
        W_D = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_D, q_D))
        velocity_N = q_N / W_N[None, ...]
        velocity_D = q_D / W_D[None, ...]
        internal = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            internal[index] = self._entropy_internal(float(n[index]), float(entropy_per_baryon[index]))
        sigma = np.zeros(shape)
        fluid = FluidPrimitive(n, internal, velocity_N, sigma)
        carrier = CarrierPrimitive(d, velocity_D)
        return fluid, carrier, residuals.reshape(shape), failed

    def _recover_cells_from_energy(
        self,
        h: Array,
        D_N: Array,
        D_D: Array,
        momentum: Array,
        energy: Array,
        P_D: Array,
        entropy_guess: Array,
    ) -> tuple[FluidPrimitive, CarrierPrimitive, Array, Array, int]:
        """Invert the nine independent material conserved variables.

        The production state also transports entropy.  At finite resolution
        that tenth value is redundant with total energy and the two-current
        master function, so Runge--Kutta stage states need a projection back
        to the constitutive manifold.  This square inversion leaves both
        currents, energy-momentum, and carrier canonical momentum untouched;
        it determines the thermodynamic entropy represented by those values.
        """
        shape = self.grid.shape
        count = int(np.prod(shape))
        h_flat = h.reshape(3, 3, count)
        sqrt_flat = inverse_metric(h)[2].reshape(count)
        D_N_flat = D_N.reshape(count)
        D_D_flat = D_D.reshape(count)
        S_flat = momentum.reshape(3, count)
        E_flat = energy.reshape(count)
        P_flat = P_D.reshape(3, count)
        entropy_flat = entropy_guess.reshape(count)
        guess_flat = None
        if self._primitive_guess is not None and self._primitive_guess.shape == (8,) + shape:
            guess_flat = self._primitive_guess.reshape(8, count)
        solved = np.empty((9, count), dtype=float)
        residuals = np.empty(count, dtype=float)
        failed = 0
        for cell in range(count):
            target = np.concatenate(
                (
                    [D_N_flat[cell], D_D_flat[cell]],
                    S_flat[:, cell],
                    [E_flat[cell]],
                    P_flat[:, cell],
                )
            )
            energy_scale = max(
                abs(E_flat[cell]), np.linalg.norm(S_flat[:, cell]), 1.0e-10
            )
            carrier_scale = max(abs(D_D_flat[cell]), 1.0e-10)
            momentum_scale = max(
                np.linalg.norm(P_flat[:, cell]), carrier_scale, 1.0e-10
            )
            scale = np.array(
                [
                    max(abs(D_N_flat[cell]), 1.0e-10),
                    carrier_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    momentum_scale,
                    momentum_scale,
                    momentum_scale,
                ]
            )
            if guess_flat is not None:
                primitive = guess_flat[:, cell]
                x = np.concatenate(
                    (
                        [primitive[0], entropy_flat[cell]],
                        primitive[1:4],
                        [primitive[4]],
                        primitive[5:8],
                    )
                )
            else:
                n0 = max(
                    D_N_flat[cell] / max(sqrt_flat[cell], 1.0e-300),
                    self.grhd.parameters.density_floor,
                )
                d0 = max(
                    D_D_flat[cell] / max(sqrt_flat[cell], 1.0e-300),
                    self.master_parameters.carrier_floor,
                )
                denominator = max(E_flat[cell] + 1.0e-8, 1.0e-8)
                velocity = (
                    np.linalg.solve(h_flat[:, :, cell], S_flat[:, cell])
                    / denominator
                )
                speed_sq = float(velocity @ h_flat[:, :, cell] @ velocity)
                if speed_sq >= 0.5:
                    velocity *= np.sqrt(0.5 / max(speed_sq, 1.0e-300))
                W0 = 1.0 / np.sqrt(
                    1.0 - float(velocity @ h_flat[:, :, cell] @ velocity)
                )
                q0 = W0 * velocity
                x = np.concatenate(
                    ([np.log(n0), entropy_flat[cell]], q0, [np.log(d0)], q0)
                )

            primitive_for_phase = np.concatenate(
                ([x[0]], x[2:5], [x[5]], x[6:9])
            )
            phase_override = self._cell_phase(
                primitive_for_phase,
                h_flat[:, :, cell],
                float(x[1]),
            )

            def function(value: np.ndarray) -> np.ndarray:
                primitive = np.concatenate(
                    (
                        [value[0]],
                        value[2:5],
                        [value[5]],
                        value[6:9],
                    )
                )
                return self._cell_predictions(
                    primitive,
                    h_flat[:, :, cell],
                    float(sqrt_flat[cell]),
                    float(value[1]),
                    phase_override,
                )

            for _ in range(self.production_parameters.recovery_iterations):
                residual = (function(x) - target) / scale
                norm = float(np.max(np.abs(residual)))
                if norm < max(self.production_parameters.recovery_tolerance, 2.0e-9):
                    break
                jac = self._cell_jacobian(function, x) / scale[:, None]
                delta = np.linalg.solve(jac, -residual)
                accepted = False
                for power in range(12):
                    trial = x + delta * (0.5**power)
                    trial[0] = np.clip(trial[0], -700.0, 700.0)
                    trial[5] = np.clip(trial[5], -700.0, 700.0)
                    if np.max(np.abs((function(trial) - target) / scale)) < norm:
                        x = trial
                        accepted = True
                        break
                if not accepted:
                    break
            final = float(np.max(np.abs((function(x) - target) / scale)))
            residuals[cell] = final
            if not np.isfinite(final) or final > 2.0e-7:
                failed += 1
            primitive_final = np.concatenate(
                ([x[0]], x[2:5], [x[5]], x[6:9])
            )
            final_phase = self._cell_phase(
                primitive_final,
                h_flat[:, :, cell],
                float(x[1]),
            )
            if (
                phase_override is not None
                and final_phase != phase_override
            ):
                failed += 1
            solved[:, cell] = x

        density = np.exp(solved[0]).reshape(shape)
        entropy = solved[1].reshape(shape)
        q_N = solved[2:5].reshape((3,) + shape)
        carrier_density = np.exp(solved[5]).reshape(shape)
        q_D = solved[6:9].reshape((3,) + shape)
        W_N = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_N, q_N))
        W_D = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_D, q_D))
        internal = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            internal[index] = self._entropy_internal(
                float(density[index]), float(entropy[index])
            )
        fluid = FluidPrimitive(
            density, internal, q_N / W_N[None, ...], np.zeros(shape)
        )
        carrier = CarrierPrimitive(
            carrier_density, q_D / W_D[None, ...]
        )
        self._primitive_guess = np.concatenate(
            (
                solved[0:1],
                solved[2:5],
                solved[5:6],
                solved[6:9],
            ),
            axis=0,
        ).reshape((8,) + shape)
        return (
            fluid,
            carrier,
            entropy,
            residuals.reshape(shape),
            failed,
        )

    def _project_to_master_manifold(
        self, state: Theory3ProductionState
    ) -> Theory3ProductionState:
        """Reconcile redundant entropy after a conservative RK stage."""
        h, _ = self.ccz4.physical_geometry(state.geometry)
        _, _, sqrt_h = inverse_metric(h)
        source_charge = state.target_charge / sqrt_h
        vector_stress = self.system.vector_stress(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            source_charge,
        )
        entropy_guess = state.matter.entropy / np.maximum(
            state.matter.D, 1.0e-300
        )
        fluid, carrier, entropy, residual, failed = self._recover_cells_from_energy(
            h,
            state.matter.D,
            state.target_charge / self.master_parameters.carrier_charge,
            state.matter.momentum / sqrt_h[None, ...] - vector_stress.momentum,
            state.matter.energy / sqrt_h - vector_stress.rho,
            state.target_current,
            entropy_guess,
        )
        if failed:
            raise FloatingPointError(
                "Theory 3.3 energy-manifold projection failed: "
                f"cells={failed}, residual={float(np.max(residual)):.3e}"
            )
        master = self.master.evaluate(h, fluid, carrier)
        if np.any(
            master.relative_lorentz_factor
            > self.master_parameters.maximum_relative_lorentz_factor
        ):
            raise FloatingPointError(
                "Theory 3.3 energy-manifold projection left the counterflow domain"
            )
        projected_entropy = state.matter.D * entropy
        correction = projected_entropy - state.matter.entropy
        self.last_projection_entropy_change += float(self.grid.integrate(correction))
        self.last_projection_minimum_change = min(
            self.last_projection_minimum_change, float(np.min(correction))
        )
        state.matter.entropy = projected_entropy
        return state

    def initialize(
        self,
        geometry: CCZ4State,
        fluid: FluidPrimitive,
        *,
        carrier: CarrierPrimitive | None = None,
        a: Array | None = None,
        pi_A: Array | None = None,
        b: Array | None = None,
        pi_B: Array | None = None,
    ) -> Theory3ProductionState:
        h, _ = self.ccz4.physical_geometry(geometry)
        _, _, sqrt_h = inverse_metric(h)
        a = self.grid.zeros((3,)) if a is None else np.asarray(a, dtype=float)
        b = self.grid.zeros((3,)) if b is None else np.asarray(b, dtype=float)
        pi_A = self.grid.zeros((3,)) if pi_A is None else np.asarray(pi_A, dtype=float)
        pi_B = self.grid.zeros((3,)) if pi_B is None else np.asarray(pi_B, dtype=float)
        if carrier is None:
            carrier = CarrierPrimitive(
                np.maximum(
                    self.master_parameters.target_fraction * fluid.baryon_density,
                    self.master_parameters.carrier_floor,
                ),
                np.array(fluid.velocity, copy=True),
            )
        master = self.master.evaluate(h, fluid, carrier)
        D_D, P_D = self.master.conserved_carrier(h, master)
        target_charge = self.master_parameters.carrier_charge * D_D
        longitudinal_A, longitudinal_B = self.system.initial_longitudinal_charges(h, pi_A, pi_B)
        vector_stress = self.system.vector_stress(
            h,
            a,
            pi_A,
            b,
            pi_B,
            longitudinal_A,
            longitudinal_B,
            master.source_charge_eulerian,
        )
        total = _add_stress(master.stress, vector_stress)
        W_N = lorentz_factor(h, fluid.velocity)
        D_N = sqrt_h * fluid.baryon_density * W_N
        entropy = D_N * master.entropy_per_baryon
        tracer = D_N * fluid.sigma
        matter = Theory3MatterState(
            D_N,
            sqrt_h[None, ...] * total.momentum,
            sqrt_h * total.rho,
            entropy,
            tracer,
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
            P_D,
            geometry.time,
        )
        q_N = W_N[None, ...] * fluid.velocity
        W_D = lorentz_factor(h, carrier.velocity)
        q_D = W_D[None, ...] * carrier.velocity
        self._primitive_guess = np.concatenate(
            (
                np.log(fluid.baryon_density)[None, ...],
                q_N,
                np.log(carrier.number_density)[None, ...],
                q_D,
            ),
            axis=0,
        )
        self.recover(state)
        return state

    def vacuum_state(self) -> Theory3ProductionState:
        fluid = FluidPrimitive(
            np.full(self.grid.shape, self.grhd.parameters.density_floor),
            np.full(self.grid.shape, self.grhd.parameters.internal_energy_floor),
            self.grid.zeros((3,)),
            self.grid.zeros(),
        )
        carrier = CarrierPrimitive(
            np.full(self.grid.shape, self.master_parameters.carrier_floor),
            self.grid.zeros((3,)),
        )
        return self.initialize(self.ccz4.flat_state(), fluid, carrier=carrier)

    def recover(self, state: Theory3ProductionState) -> Theory33Recovery:
        h, _ = self.ccz4.physical_geometry(state.geometry)
        _, _, sqrt_h = inverse_metric(h)
        source_charge = state.target_charge / sqrt_h
        vector_stress = self.system.vector_stress(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            source_charge,
        )
        material_energy = state.matter.energy / sqrt_h - vector_stress.rho
        material_momentum = state.matter.momentum / sqrt_h[None, ...] - vector_stress.momentum
        D_D = state.target_charge / self.master_parameters.carrier_charge
        entropy_per_baryon = state.matter.entropy / np.maximum(state.matter.D, 1.0e-300)
        fluid, carrier, residual, failed = self._recover_cells(
            h,
            state.matter.D,
            D_D,
            material_momentum,
            material_energy,
            state.target_current,
            entropy_per_baryon,
        )
        fluid = FluidPrimitive(
            fluid.baryon_density,
            fluid.specific_internal_energy,
            fluid.velocity,
            state.matter.tracer / np.maximum(state.matter.D, 1.0e-300),
        )
        master = self.master.evaluate(h, fluid, carrier)
        legendre_trace = master.B_N + master.B_D
        legendre_disc = np.sqrt(
            np.maximum((master.B_N - master.B_D) ** 2 + 4.0 * master.entrainment**2, 0.0)
        )
        legendre_min = 0.5 * (legendre_trace - legendre_disc)
        thermo_trace = master.thermodynamic_hessian_nn + master.thermodynamic_hessian_dd
        thermo_disc = np.sqrt(
            np.maximum(
                (master.thermodynamic_hessian_nn - master.thermodynamic_hessian_dd) ** 2
                + 4.0 * master.thermodynamic_hessian_nd**2,
                0.0,
            )
        )
        thermo_min = 0.5 * (thermo_trace - thermo_disc)
        failed += int(
            np.count_nonzero(
                (legendre_min <= self.master_parameters.convexity_floor)
                | (thermo_min <= self.master_parameters.convexity_floor)
                | (master.relative_lorentz_factor < 1.0 - 1.0e-10)
                | (
                    master.relative_lorentz_factor
                    > self.master_parameters.maximum_relative_lorentz_factor
                )
            )
        )
        report = Theory33RecoveryReport(
            failed,
            int(np.count_nonzero(state.matter.D <= sqrt_h * self.grhd.parameters.density_floor * (1.0 + 1.0e-10))),
            float(np.max(residual)),
            float(np.min(carrier.number_density)),
            float(np.max(master.relative_lorentz_factor)),
            float(np.min(legendre_min)),
            float(np.min(thermo_min)),
        )
        if failed:
            raise FloatingPointError(
                "Theory 3.3 multifluid recovery failed: "
                f"cells={failed}, residual={report.maximum_residual:.3e}, "
                f"legendre={report.minimum_legendre_eigenvalue:.3e}, "
                f"thermo={report.minimum_thermodynamic_eigenvalue:.3e}"
            )
        material_stress = master.stress
        if not self.production_parameters.atmosphere_gravitates:
            atmosphere = (
                state.matter.D
                <= sqrt_h
                * self.grhd.parameters.density_floor
                * (1.0 + 1.0e-10)
            ) & (
                carrier.number_density
                <= self.master_parameters.carrier_floor * (1.0 + 1.0e-10)
            )
            material_stress = type(master.stress)(
                np.where(atmosphere, 0.0, master.stress.rho),
                np.where(atmosphere[None, ...], 0.0, master.stress.momentum),
                np.where(atmosphere[None, None, ...], 0.0, master.stress.stress),
                np.where(atmosphere, 0.0, master.stress.trace),
            )
        total = _add_stress(material_stress, vector_stress)
        sound = np.sqrt(
            np.clip(
                self.master_parameters.gamma_ad
                * self.master.thermal_pressure(fluid.baryon_density, fluid.specific_internal_energy)
                / np.maximum(
                    self.master.thermal_energy(fluid.baryon_density, fluid.specific_internal_energy)
                    + self.master.thermal_pressure(fluid.baryon_density, fluid.specific_internal_energy),
                    1.0e-300,
                ),
                0.0,
                1.0,
            )
        )
        return Theory33Recovery(
            fluid,
            carrier,
            master,
            entropy_per_baryon,
            sound,
            material_stress,
            vector_stress,
            total,
            master.source_charge_eulerian,
            master.source_current_up,
            report,
        )

    def _speed_bound(self, recovery, h, lapse, shift, axis):
        metric_light = np.sqrt(np.maximum(inverse_metric(h)[0][axis, axis], 0.0))
        return np.abs(shift[axis]) + lapse * metric_light

    def _path_product(
        self, coefficient: Array, field: Array, axis: int
    ) -> Array:
        """Midpoint straight-path discretization of ``coefficient*d(field)``."""
        field_right = self._roll(field, -1, axis)
        field_left = self._roll(field, 1, axis)
        coefficient_right = 0.5 * (
            coefficient + self._roll(coefficient, -1, axis)
        )
        coefficient_left = 0.5 * (
            coefficient + self._roll(coefficient, 1, axis)
        )
        return (
            coefficient_right * (field_right - field)
            + coefficient_left * (field - field_left)
        ) / (2.0 * self.grid.spacing[axis])

    def _target_rhs(self, state, recovery, h, K):
        del K
        lapse = state.geometry.lapse
        shift = state.geometry.shift
        sqrt_h = inverse_metric(h)[2]
        transport_D = lapse[None, ...] * recovery.carrier.velocity - shift
        dcharge = np.zeros_like(state.target_charge)
        dmomentum = np.zeros_like(state.target_current)
        speed = None
        for axis in range(self.grid.ndim):
            flux_charge = state.target_charge * transport_D[axis]
            flux_momentum = state.target_current * transport_D[axis]
            speed = self._speed_bound(recovery, h, lapse, shift, axis)
            charge_interface = self._rusanov_interface(flux_charge, state.target_charge, speed, axis)
            momentum_interface = self._rusanov_interface(flux_momentum, state.target_current, speed, axis)
            if getattr(self.grid, "is_periodic", True):
                dcharge += self._interface_divergence(charge_interface, axis)
                dmomentum += self._interface_divergence(momentum_interface, axis)
            else:
                lower_charge = self._outgoing_boundary_flux(flux_charge, transport_D[axis], axis, -1)
                upper_charge = self._outgoing_boundary_flux(flux_charge, transport_D[axis], axis, 1)
                lower_momentum = self._outgoing_boundary_flux(flux_momentum, transport_D[axis], axis, -1)
                upper_momentum = self._outgoing_boundary_flux(flux_momentum, transport_D[axis], axis, 1)
                dcharge += self._interface_divergence(charge_interface, axis, lower_charge, upper_charge)
                dmomentum += self._interface_divergence(momentum_interface, axis, lower_momentum, upper_momentum)

        chi_down = recovery.master.carrier_momentum_down
        chi_zero = -lapse * recovery.master.carrier_momentum_normal + np.einsum(
            "i...,i...->...", shift, chi_down
        )
        carrier_density_eulerian = recovery.master.carrier_eulerian_density
        for component in range(self.grid.ndim):
            source = self._path_product(
                carrier_density_eulerian, chi_zero, component
            )
            for axis in range(self.grid.ndim):
                source += self._path_product(
                    carrier_density_eulerian * transport_D[axis],
                    chi_down[axis],
                    component,
                )
            dmomentum[component] += sqrt_h * source
        drive_power, drive_force = self.system.drive_exchange(
            h,
            state.b,
            state.pi_B,
            recovery.target_charge_eulerian,
            recovery.target_current_up,
        )
        del drive_power
        dmomentum += lapse[None, ...] * sqrt_h[None, ...] * drive_force
        return dcharge, dmomentum

    def carrier_boundary_number_outflow_rate(
        self,
        state: Theory3ProductionState,
        recovery: Theory33Recovery | None = None,
    ) -> float:
        """Return the outward carrier-number flux through all physical faces.

        This uses the same outgoing-only boundary flux as ``_target_rhs``.
        Positive values mean carrier number is leaving the Cartesian patch.
        """
        if getattr(self.grid, "is_periodic", True):
            return 0.0
        recovery = self.recover(state) if recovery is None else recovery
        h, _ = self.ccz4.physical_geometry(state.geometry)
        lapse = state.geometry.lapse
        shift = state.geometry.shift
        transport = lapse[None, ...] * recovery.carrier.velocity - shift
        charge_outflow = 0.0
        for axis in range(self.grid.ndim):
            flux = state.target_charge * transport[axis]
            lower = self._outgoing_boundary_flux(
                flux, transport[axis], axis, -1
            )
            upper = self._outgoing_boundary_flux(
                flux, transport[axis], axis, 1
            )
            face_area = self.grid.cell_volume / self.grid.spacing[axis]
            charge_outflow += face_area * float(np.sum(upper) - np.sum(lower))
        return charge_outflow / self.master_parameters.carrier_charge

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
        matter_rhs = list(self._conservative_rhs(state, recovery, h, K))
        target_charge_rhs, target_momentum_rhs = self._target_rhs(
            state, recovery, h, K
        )
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

            field_rhs = Theory3CharacteristicBoundary(self.grid).mixed_proca_rhs(
                self.system, h, state, field_rhs
            )
        return geometry_rhs + tuple(matter_rhs) + field_rhs + (
            target_charge_rhs,
            target_momentum_rhs,
        )

    def _apply_carrier_drag(
        self, state: Theory3ProductionState, dt: float
    ) -> Theory3ProductionState:
        """Apply an unconditionally stable local relative-rapidity decay.

        The collision solve preserves both number densities and total material
        energy-momentum exactly.  It solves for the entropy increase instead
        of assigning an energy residual to a reservoir.
        """
        before = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sqrt_h = inverse_metric(h)[2]
        shape = self.grid.shape
        count = int(np.prod(shape))
        h_flat = h.reshape(3, 3, count)
        sqrt_flat = sqrt_h.reshape(count)
        D_N_flat = state.matter.D.reshape(count)
        D_D_flat = (
            state.target_charge / self.master_parameters.carrier_charge
        ).reshape(count)
        material_energy = (
            state.matter.energy / sqrt_h - before.vector_stress.rho
        ).reshape(count)
        material_momentum = (
            state.matter.momentum / sqrt_h[None, ...]
            - before.vector_stress.momentum
        ).reshape(3, count)
        old_entropy = before.entropy_per_baryon.reshape(count)
        old_relative_gamma = before.master.relative_lorentz_factor.reshape(count)
        W_N = lorentz_factor(h, before.fluid.velocity).reshape(count)
        W_D = lorentz_factor(h, before.carrier.velocity).reshape(count)
        q_N_old = (W_N.reshape(shape)[None, ...] * before.fluid.velocity).reshape(3, count)
        q_D_old = (W_D.reshape(shape)[None, ...] * before.carrier.velocity).reshape(3, count)
        lapse_flat = state.geometry.lapse.reshape(count)
        solved = np.empty((9, count), dtype=float)
        entropy_increase = np.empty(count, dtype=float)
        for cell in range(count):
            # Below this threshold the available counterflow energy is smaller
            # than the nonlinear solve's entropy resolution.  Treating it as
            # exactly co-moving avoids manufacturing either sign of heat.
            if old_relative_gamma[cell] - 1.0 <= 1.0e-3:
                solved[:, cell] = np.concatenate(
                    (
                        [np.log(before.fluid.baryon_density.reshape(count)[cell])],
                        [old_entropy[cell]],
                        q_N_old[:, cell],
                        [np.log(before.carrier.number_density.reshape(count)[cell])],
                        q_D_old[:, cell],
                    )
                )
                entropy_increase[cell] = 0.0
                continue
            decay = float(
                np.exp(
                    -dt
                    * max(lapse_flat[cell], 0.0)
                    / self.master_parameters.relaxation_time
                )
            )
            relative_target = decay * (q_D_old[:, cell] - q_N_old[:, cell])
            target = np.concatenate(
                (
                    [D_N_flat[cell], D_D_flat[cell]],
                    material_momentum[:, cell],
                    [material_energy[cell]],
                    relative_target,
                )
            )
            energy_scale = max(
                abs(material_energy[cell]),
                np.linalg.norm(material_momentum[:, cell]),
                1.0e-10,
            )
            scale = np.array(
                [
                    max(abs(D_N_flat[cell]), 1.0e-10),
                    max(abs(D_D_flat[cell]), 1.0e-10),
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    1.0,
                    1.0,
                    1.0,
                ]
            )
            x = np.concatenate(
                (
                    [np.log(before.fluid.baryon_density.reshape(count)[cell])],
                    [old_entropy[cell]],
                    q_N_old[:, cell],
                    [np.log(before.carrier.number_density.reshape(count)[cell])],
                    q_D_old[:, cell],
                )
            )

            primitive_for_phase = np.concatenate(
                ([x[0]], x[2:5], [x[5]], x[6:9])
            )
            phase_override = self._cell_phase(
                primitive_for_phase,
                h_flat[:, :, cell],
                float(x[1]),
            )

            def function(value: np.ndarray) -> np.ndarray:
                primitive = np.concatenate(
                    (
                        [value[0]],
                        value[2:5],
                        [value[5]],
                        value[6:9],
                    )
                )
                conservative = self._cell_predictions(
                    primitive,
                    h_flat[:, :, cell],
                    float(sqrt_flat[cell]),
                    float(value[1]),
                    phase_override,
                )
                return np.concatenate(
                    (
                        conservative[:6],
                        value[6:9] - value[2:5],
                    )
                )

            for _ in range(self.production_parameters.recovery_iterations):
                residual = (function(x) - target) / scale
                norm = float(np.max(np.abs(residual)))
                if norm < 2.0e-9:
                    break
                jac = self._cell_jacobian(function, x) / scale[:, None]
                delta = np.linalg.solve(jac, -residual)
                accepted = False
                for power in range(12):
                    trial = x + delta * (0.5**power)
                    trial[0] = np.clip(trial[0], -700.0, 700.0)
                    trial[5] = np.clip(trial[5], -700.0, 700.0)
                    if np.max(np.abs((function(trial) - target) / scale)) < norm:
                        x = trial
                        accepted = True
                        break
                if not accepted:
                    break
            final = float(np.max(np.abs((function(x) - target) / scale)))
            if not np.isfinite(final) or final > 2.0e-7:
                raise FloatingPointError(
                    f"Theory 3.3 implicit drag solve failed in cell {cell}: {final:.3e}"
                )
            primitive_final = np.concatenate(
                ([x[0]], x[2:5], [x[5]], x[6:9])
            )
            final_phase = self._cell_phase(
                primitive_final,
                h_flat[:, :, cell],
                float(x[1]),
            )
            if (
                phase_override is not None
                and final_phase != phase_override
            ):
                raise FloatingPointError(
                    "Theory 3.3 implicit drag solve crossed the frozen "
                    f"constitutive branch in cell {cell}"
                )
            entropy_increase[cell] = x[1] - old_entropy[cell]
            if entropy_increase[cell] < -2.0e-9:
                raise FloatingPointError(
                    "Theory 3.3 drag collision map would decrease entropy: "
                    f"cell={cell}, delta_sigma={entropy_increase[cell]:.3e}, "
                    f"gamma_rel={old_relative_gamma[cell]:.9f}, decay={decay:.9f}"
                )
            solved[:, cell] = x

        density = np.exp(solved[0]).reshape(shape)
        entropy = solved[1].reshape(shape)
        q_N = solved[2:5].reshape((3,) + shape)
        carrier_density = np.exp(solved[5]).reshape(shape)
        q_D = solved[6:9].reshape((3,) + shape)
        W_N_new = np.sqrt(
            1.0 + np.einsum("ij...,i...,j...->...", h, q_N, q_N)
        )
        W_D_new = np.sqrt(
            1.0 + np.einsum("ij...,i...,j...->...", h, q_D, q_D)
        )
        fluid = FluidPrimitive(
            density,
            np.vectorize(self._entropy_internal)(density, entropy),
            q_N / W_N_new[None, ...],
            before.fluid.sigma,
        )
        carrier = CarrierPrimitive(
            carrier_density, q_D / W_D_new[None, ...]
        )
        master = self.master.evaluate(h, fluid, carrier)
        _, carrier_momentum = self.master.conserved_carrier(h, master)
        updated = Theory3ProductionState(
            state.geometry,
            Theory3MatterState(
                state.matter.D,
                state.matter.momentum,
                state.matter.energy,
                state.matter.D * entropy,
                state.matter.tracer,
            ),
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            state.target_charge,
            carrier_momentum,
            state.time,
        )
        self._primitive_guess = np.concatenate(
            (
                np.log(density)[None, ...],
                q_N,
                np.log(carrier_density)[None, ...],
                q_D,
            ),
            axis=0,
        )
        self.recover(updated)
        entropy_change_density = state.matter.D * entropy_increase.reshape(shape)
        self.last_drag_entropy_change = float(
            self.grid.integrate(entropy_change_density)
        )
        self.last_drag_heat = float(
            self.grid.integrate(
                sqrt_h
                * before.master.temperature
                * before.fluid.baryon_density
                * entropy_increase.reshape(shape)
            )
        )
        return updated

    def step(
        self, state: Theory3ProductionState, dt: float
    ) -> Theory3ProductionState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        entropy_before_step = float(self.grid.integrate(state.matter.entropy))
        self.last_projection_entropy_change = 0.0
        self.last_projection_minimum_change = 0.0
        values = self._pack(state)
        time = state.time

        def stage(
            base: tuple[Array, ...],
            slope: tuple[Array, ...],
            factor: float,
            stage_time: float,
        ) -> tuple[Array, ...]:
            candidate = self._unpack(
                tuple(y + factor * dt * k for y, k in zip(base, slope)),
                stage_time,
            )
            candidate.geometry = self.ccz4.project_algebraic(candidate.geometry)
            candidate.geometry.time = stage_time
            candidate = self._project_to_master_manifold(candidate)
            return self._pack(candidate)

        # Classical RK4 with a constitutive projection at every internal
        # stage.  The projection changes only redundant entropy; all variables
        # entering the conservative and canonical-momentum balances are left
        # exactly as produced by RK4.
        k1 = tuple(self.rhs(time, values))
        y2 = stage(values, k1, 0.5, time + 0.5 * dt)
        k2 = tuple(self.rhs(time + 0.5 * dt, y2))
        y3 = stage(values, k2, 0.5, time + 0.5 * dt)
        k3 = tuple(self.rhs(time + 0.5 * dt, y3))
        y4 = stage(values, k3, 1.0, time + dt)
        k4 = tuple(self.rhs(time + dt, y4))
        charge_to_number = 1.0 / self.master_parameters.carrier_charge
        stage_number_change = (
            dt
            * charge_to_number
            * self.grid.integrate(k1[22] + 2.0 * k2[22] + 2.0 * k3[22] + k4[22])
            / 6.0
        )
        evolved = tuple(
            y + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
            for y, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
        result = self._unpack(evolved, time + dt)
        result.geometry = self.ccz4.project_algebraic(result.geometry)
        result.geometry.time = result.time
        result = self._project_to_master_manifold(result)
        result = self._apply_carrier_drag(result, dt)
        result = self._apply_damping(result, dt)
        self.recover(result)
        actual_number_change = charge_to_number * self.grid.integrate(
            result.target_charge - state.target_charge
        )
        self.last_carrier_boundary_number_outflow = -stage_number_change
        self.last_carrier_number_change = actual_number_change
        self.last_carrier_number_balance_residual = (
            actual_number_change + self.last_carrier_boundary_number_outflow
        )
        self.last_step_entropy_change = (
            float(self.grid.integrate(result.matter.entropy))
            - entropy_before_step
        )
        if (
            getattr(self.grid, "is_periodic", True)
            and self.last_step_entropy_change < -1.0e-10
        ):
            raise FloatingPointError(
                "Theory 3.3 projected finite-volume step decreased total entropy: "
                f"delta={self.last_step_entropy_change:.3e}"
            )
        return result

    def _apply_damping(self, state: Theory3ProductionState, dt: float) -> Theory3ProductionState:
        p = self.theory_parameters
        if p.gamma_W == 0.0 and p.gamma_B == 0.0:
            self.last_damping_report = FiniteDampingReport()
            return state
        before = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sqrt_h = inverse_metric(h)[2]
        gauss_before = self.system.gauss_constraints(
            h, state.pi_A, state.pi_B, state.longitudinal_A, state.longitudinal_B
        )
        damping = self.system.damping_step(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            before.fluid,
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
        heated_internal = before.fluid.specific_internal_energy + damping.irreversible_heat / np.maximum(
            before.fluid.baryon_density, 1.0e-300
        )
        heated = FluidPrimitive(
            before.fluid.baryon_density,
            heated_internal,
            before.fluid.velocity,
            before.fluid.sigma,
        )
        updated.matter.entropy = updated.matter.D * self.master.specific_entropy(
            heated.baryon_density, heated.specific_internal_energy
        )
        after = self.recover(updated)
        gauss_after = self.system.gauss_constraints(
            h, updated.pi_A, updated.pi_B, updated.longitudinal_A, updated.longitudinal_B
        )
        self.last_damping_report = FiniteDampingReport(
            vector_energy_change=float(
                self.grid.integrate(sqrt_h * (after.vector_stress.rho - before.vector_stress.rho))
            ),
            vector_momentum_change=tuple(
                self.grid.integrate(
                    sqrt_h * (after.vector_stress.momentum[i] - before.vector_stress.momentum[i])
                )
                for i in range(3)
            ),
            irreversible_heat=float(self.grid.integrate(sqrt_h * damping.irreversible_heat)),
            total_energy_balance_error=0.0,
            maximum_gauss_change=max(
                float(np.max(np.abs(gauss_after[0] - gauss_before[0]))),
                float(np.max(np.abs(gauss_after[1] - gauss_before[1]))),
            ),
        )
        return updated

    def recommended_dt(self, state: Theory3ProductionState, cfl: float = 0.1) -> float:
        self.recover(state)
        light_dt = cfl * min(self.grid.spacing) / np.sqrt(2.0 * self.grid.ndim)
        return min(light_dt, 0.2 * self.master_parameters.relaxation_time)

    def diagnostics(self, state: Theory3ProductionState) -> dict[str, object]:
        recovery = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sources = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        ccz = self.ccz4.diagnostics(state.geometry, sources)
        gauss_A, gauss_B = self.system.gauss_constraints(
            h, state.pi_A, state.pi_B, state.longitudinal_A, state.longitudinal_B
        )
        D_D = state.target_charge / self.master_parameters.carrier_charge
        return {
            "time": state.time,
            "hamiltonian_l2": ccz.hamiltonian_l2,
            "momentum_l2": ccz.momentum_l2,
            "theta_l2": ccz.theta_l2,
            "z_l2": ccz.z_l2,
            "gauss_A_l2": float(np.sqrt(np.mean(gauss_A**2))),
            "gauss_B_l2": float(np.sqrt(np.mean(gauss_B**2))),
            "baryon_mass": float(np.sum(state.matter.D) * self.grid.cell_volume),
            "carrier_number": float(np.sum(D_D) * self.grid.cell_volume),
            "target_charge": float(np.sum(state.target_charge) * self.grid.cell_volume),
            "combined_energy": float(np.sum(state.matter.energy) * self.grid.cell_volume),
            "entropy_integral": float(np.sum(state.matter.entropy) * self.grid.cell_volume),
            "minimum_carrier_density": recovery.report.minimum_carrier_density,
            "maximum_relative_lorentz_factor": recovery.report.maximum_relative_lorentz_factor,
            "minimum_legendre_eigenvalue": recovery.report.minimum_legendre_eigenvalue,
            "minimum_thermodynamic_eigenvalue": recovery.report.minimum_thermodynamic_eigenvalue,
            "constitutive_model": recovery.master.constitutive_model,
            "constitutive_phase_two_fraction": float(
                np.mean(recovery.master.phase_two)
            ),
            "constitutive_minimum_switching_margin": float(
                np.min(
                    np.abs(
                        recovery.master.relative_lorentz_factor
                        - 1.0
                        - recovery.master.phase_threshold
                    )
                )
            )
            if np.all(np.isfinite(recovery.master.phase_threshold))
            else None,
            "constitutive_variant_digest": (
                self.constitutive_variant_digest or ""
            ),
            "recovery_maximum_residual": recovery.report.maximum_residual,
            "recovery_failures": recovery.report.failed_cells,
            "damping_vector_energy_change": self.last_damping_report.vector_energy_change,
            "damping_heat": self.last_damping_report.irreversible_heat,
            "damping_gauss_change": self.last_damping_report.maximum_gauss_change,
            "drag_heat": self.last_drag_heat,
            "drag_entropy_change": self.last_drag_entropy_change,
            "projection_entropy_change": self.last_projection_entropy_change,
            "projection_minimum_entropy_density_change": self.last_projection_minimum_change,
            "step_entropy_change": self.last_step_entropy_change,
            "carrier_boundary_number_outflow": self.last_carrier_boundary_number_outflow,
            "carrier_number_change": self.last_carrier_number_change,
            "carrier_number_balance_residual": self.last_carrier_number_balance_residual,
        }


def save_theory33_production_state(
    path: str | Path,
    state: Theory3ProductionState,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    g = state.geometry
    m = state.matter
    return save_checkpoint(
        path,
        metadata={"format": "tesseract.production.theory33.v1", **(metadata or {})},
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
        carrier_source_charge=state.target_charge,
        carrier_canonical_momentum=state.target_current,
        time=state.time,
    )


def load_theory33_production_state(
    path: str | Path,
) -> tuple[Theory3ProductionState, dict[str, Any]]:
    arrays, metadata = load_checkpoint(path)
    if metadata.get("format") != "tesseract.production.theory33.v1":
        raise ValueError("checkpoint is not a Theory 3.3 production v1 state")
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
    return (
        Theory3ProductionState(
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
            arrays["carrier_source_charge"],
            arrays["carrier_canonical_momentum"],
            time,
        ),
        metadata,
    )
