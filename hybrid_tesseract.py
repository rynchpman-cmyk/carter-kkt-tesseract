"""Differentiable Carter/KKT simulator with a learned master function.

The discrete branch, parity, boundary, and KKT active-set decisions remain
hard. Within the selected region, PyTorch differentiates the exact continuous
calculation, including the mixed derivatives introduced by dM/dz.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor, nn


torch.set_default_dtype(torch.float64)

DIM = 4
SAFE_EPS = 1.0e-12

H_J_PLUS = 0x0
H_J_MINUS = 0x1
H_G = 0x2
H_GAMMA = 0x3
H_Q_LIMITED = 0x4
H_THETA_STAR = 0x5
H_LAMBDA = 0x6
H_ENERGY = 0x7
H_J_LIMITED = 0x8
H_RESIDUAL = 0x9
H_X_PLUS = 0xA
H_X_MINUS = 0xB
H_X_CROSS = 0xC
H_KKT = 0xD
H_CLOSURE = 0xE
H_M = 0xF


def tesseract_coord(vertex: int, *, device=None, dtype=None) -> Tensor:
    return torch.tensor(
        [
            1.0 if vertex & 0x8 else -1.0,
            1.0 if vertex & 0x4 else -1.0,
            1.0 if vertex & 0x2 else -1.0,
            1.0 if vertex & 0x1 else -1.0,
        ],
        device=device,
        dtype=dtype,
    )


def vertex_phase(vertex: int) -> float:
    h = tesseract_coord(vertex)
    return float(h @ torch.tensor([0.31, 0.47, 0.73, 1.13]))


def smooth_abs(x: Tensor) -> Tensor:
    return torch.sqrt(x.square() + 1.0e-8)


class CarterMasterFunction(nn.Module):
    """Smooth learned scalar Lambda_phi(X+, X-, Xcross, Gamma, q).

    A trainable linear path gives a stable constitutive initialization. A
    nonlinear residual MLP learns the rest of the master function. Both paths
    are learned, so Carter momenta are always derivatives of one scalar.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 1)
        self.residual = nn.Sequential(
            nn.Linear(5, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.20, 0.20, 0.10, 0.10, 0.10]])
            )
            self.linear.bias.fill_(0.50)
            nn.init.normal_(self.residual[-1].weight, std=2.0e-3)
            self.residual[-1].bias.zero_()

    def forward(self, features: Tensor) -> Tensor:
        return (self.linear(features) + self.residual(features)).squeeze(-1)


class _SelectedBoxQPSolve(torch.autograd.Function):
    """Implicit VJP for one frozen box-QP active set.

    The hard 81-way search is intentionally outside this function.  Once an
    active state is selected, its continuous solution is A x = b.  The
    backward pass solves A^T lambda = dL/dx and contracts the selected KKT
    residual, so no gradient flows through candidate ranking or tie-breaking.
    """

    @staticmethod
    def forward(
        ctx,
        hessian: Tensor,
        linear: Tensor,
        lo: Tensor,
        hi: Tensor,
        active_state: Tensor,
    ) -> Tensor:
        free = active_state.eq(0)
        identity = torch.eye(
            hessian.shape[-1],
            device=hessian.device,
            dtype=hessian.dtype,
        ).expand_as(hessian)
        matrix = torch.where(free[..., None], hessian, identity)
        bound = torch.where(active_state.lt(0), lo, hi)
        rhs = torch.where(free, -linear, bound)
        solution = torch.linalg.solve(matrix, rhs[..., None]).squeeze(-1)
        ctx.save_for_backward(matrix, solution, active_state)
        return solution

    @staticmethod
    def backward(ctx, grad_solution: Tensor):
        matrix, solution, active_state = ctx.saved_tensors
        adjoint = torch.linalg.solve(
            matrix.transpose(-1, -2),
            grad_solution[..., None],
        ).squeeze(-1)
        free = active_state.eq(0)
        grad_hessian = torch.where(
            free[..., None],
            -adjoint[..., :, None] * solution[..., None, :],
            torch.zeros_like(matrix),
        )
        grad_linear = torch.where(
            free, -adjoint, torch.zeros_like(adjoint)
        )
        grad_lo = torch.where(
            active_state.lt(0), adjoint, torch.zeros_like(adjoint)
        )
        grad_hi = torch.where(
            active_state.gt(0), adjoint, torch.zeros_like(adjoint)
        )
        return grad_hessian, grad_linear, grad_lo, grad_hi, None


@dataclass
class KKTResult:
    x: Tensor
    objective: Tensor
    active_state: Tensor
    active_mask: Tensor
    valid: Tensor
    max_violation: Tensor
    feasibility_margin: Tensor
    condition_number: Tensor


@dataclass
class TesseractState:
    j_plus: Tensor
    j_minus: Tensor
    metric: Tensor
    gamma: Tensor
    x_plus: Tensor
    x_minus: Tensor
    x_cross: Tensor
    q_limited: Tensor
    theta: Tensor
    lambda_value: Tensor
    pi_plus: Tensor
    pi_minus: Tensor
    stress: Tensor
    energy: Tensor
    j_limited: Tensor
    residual: Tensor
    free_energy: Tensor
    kkt: KKTResult
    classifier_mask: Tensor
    classifier_margin: Tensor


@dataclass
class StepDiagnostics:
    sigma: Tensor
    iota: Tensor
    kkt_valid: Tensor
    kkt_max_violation: Tensor
    kkt_feasibility_margin: Tensor
    kkt_condition_number: Tensor
    classifier_margin: Tensor
    parity_margin: Tensor
    switch_margin: Tensor
    local_gain: Tensor | None
    active_mask: Tensor
    classifier_mask: Tensor


class HybridTesseract(nn.Module):
    """Hard-constraint differentiable version of the Tesseract recurrence."""

    def __init__(
        self,
        *,
        master_width: int = 32,
        learn_dynamics: bool = True,
        learn_target: bool = False,
        boundary_certificates: bool = True,
    ) -> None:
        super().__init__()
        self.master = CarterMasterFunction(master_width)

        self.c = nn.Parameter(torch.tensor(0.25), requires_grad=learn_dynamics)
        self.kappa = nn.Parameter(torch.tensor(0.10), requires_grad=learn_dynamics)
        self.alpha = nn.Parameter(torch.tensor(0.05), requires_grad=learn_dynamics)
        self.zeta = nn.Parameter(torch.tensor(0.01), requires_grad=learn_dynamics)
        self.omega_delta = nn.Parameter(
            torch.tensor(0.50), requires_grad=learn_dynamics
        )

        self.register_buffer("chi", torch.tensor(0.02))
        self.register_buffer("epsilon_n", torch.tensor(1.0e-6))
        self.register_buffer("crossing_tolerance", torch.tensor(1.0e-5))
        self.register_buffer("branch_split", torch.tensor(0.05))
        self.register_buffer("omega_e", torch.tensor(1.0))
        self.register_buffer("omega_t", torch.tensor(0.10))
        self.register_buffer("omega_pi", torch.tensor(0.10))
        self.register_buffer("omega_a", torch.tensor(0.10))

        self.target_t = nn.Parameter(
            torch.zeros(DIM, DIM), requires_grad=learn_target
        )
        states = torch.tensor(list(product((-1, 0, 1), repeat=4)))
        powers = torch.tensor([1, 3, 9, 27])
        masks = ((states + 1) * powers).sum(dim=-1)
        order = torch.argsort(masks)
        self.register_buffer("kkt_states", states[order])
        self.register_buffer("kkt_masks", masks[order])
        self.boundary_certificates = boundary_certificates

    def current_field(
        self, z: Tensor, vertex: int, sigma: Tensor, orientation: float
    ) -> Tensor:
        h = tesseract_coord(vertex, device=z.device, dtype=z.dtype)
        components = []
        phase0 = vertex_phase(vertex)
        for index in range(DIM):
            a = float(index + 1)
            phase = phase0 + orientation * 0.29 * a
            base = 0.52 * h[index] + orientation * 0.11 * a
            slope = orientation * (0.055 + 0.018 * a)
            amplitude = 0.13 + 0.015 * a
            frequency = 0.47 + 0.19 * a
            value = (
                base
                + slope * z
                + amplitude * torch.sin(frequency * z + phase)
                + self.branch_split
                * sigma
                * h[index]
                * torch.tanh(1.65 * z + phase)
            )
            components.append(value)
        return torch.stack(components, dim=-1)

    def metric_field(self, z: Tensor, sigma: Tensor) -> Tensor:
        phase = vertex_phase(H_G)
        g00 = -(
            1.28
            + 0.015 * sigma
            + 0.075 * torch.tanh(0.72 * z + phase)
        )
        g11 = 1.04 + 0.045 * torch.sin(0.61 * z + phase * 0.7)
        g22 = 1.09 + 0.038 * torch.cos(0.53 * z + phase * 0.9)
        g33 = 1.13 + 0.042 * torch.tanh(0.83 * z + phase * 1.1)
        g01 = 0.014 * torch.sin(0.39 * z + phase + 0.2 * sigma)
        g12 = 0.011 * torch.cos(0.44 * z + phase * 0.8)
        g23 = 0.009 * torch.sin(0.57 * z + phase * 1.2 - 0.15 * sigma)
        zero = torch.zeros_like(z)
        rows = [
            torch.stack([g00, g01, zero, zero], dim=-1),
            torch.stack([g01, g11, g12, zero], dim=-1),
            torch.stack([zero, g12, g22, g23], dim=-1),
            torch.stack([zero, zero, g23, g33], dim=-1),
        ]
        return torch.stack(rows, dim=-2)

    def gamma_field(self, z: Tensor, sigma: Tensor) -> Tensor:
        h = tesseract_coord(H_GAMMA, device=z.device, dtype=z.dtype)
        phase = vertex_phase(H_GAMMA)
        return (
            1.0
            + 0.035 * sigma * h[3]
            + 0.19 * torch.tanh(0.78 * z + phase)
            + 0.035 * h[1] * torch.sin(0.31 * z + phase * 0.5)
        )

    @staticmethod
    def invariant(metric: Tensor, left: Tensor, right: Tensor) -> Tensor:
        return -torch.einsum("bi,bij,bj->b", left, metric, right)

    def build_kkt_problem(
        self,
        z: Tensor,
        gamma: Tensor,
        x_plus: Tensor,
        x_minus: Tensor,
        x_cross: Tensor,
        sigma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        phase = vertex_phase(H_KKT)
        target = torch.stack(
            [
                0.56
                * torch.tanh(
                    0.43 * x_cross
                    + 0.24 * gamma
                    + 0.12 * z
                    + phase * 0.08
                )
                + 0.075 * sigma,
                0.43
                * torch.tanh(
                    0.36 * x_plus - 0.11 * x_minus + phase * 0.04
                ),
                0.43
                * torch.tanh(
                    0.36 * x_minus - 0.11 * x_plus - phase * 0.03
                ),
                0.31
                * torch.tanh(
                    0.29 * (x_plus - x_minus)
                    + 0.18 * z
                    + phase * 0.06
                )
                + 0.045 * sigma,
            ],
            dim=-1,
        )
        lo = torch.stack(
            [
                -0.32 + 0.025 * torch.tanh(0.48 * z),
                -0.46 + 0.018 * torch.sin(0.37 * z + phase),
                -0.46 + 0.017 * torch.cos(0.41 * z + phase * 0.7),
                -0.40 + 0.020 * torch.tanh(0.52 * z + phase * 0.5),
            ],
            dim=-1,
        )
        hi = torch.stack(
            [
                0.32 + 0.025 * torch.tanh(0.48 * z),
                0.46 + 0.018 * torch.sin(0.37 * z + phase),
                0.46 + 0.017 * torch.cos(0.41 * z + phase * 0.7),
                0.40 + 0.020 * torch.tanh(0.52 * z + phase * 0.5),
            ],
            dim=-1,
        )
        rows = []
        for i in range(4):
            row = []
            for j in range(4):
                if i == j:
                    fi = float(i + 1)
                    value = (
                        1.42
                        + 0.24 * fi
                        + 0.070 * gamma.square()
                        + 0.022 * fi * smooth_abs(x_cross)
                    )
                else:
                    separation = float(abs(i - j) + 1)
                    frequency = 0.28 + 0.055 * float(i + j + 1)
                    value = (0.055 / separation) * torch.tanh(
                        frequency * z
                        + phase * 0.13 * float(i + j + 1)
                    )
                row.append(value)
            rows.append(torch.stack(row, dim=-1))
        hessian = torch.stack(rows, dim=-2)
        linear = -torch.einsum("bij,bj->bi", hessian, target)
        return hessian, linear, lo, hi

    def solve_bounded_kkt(
        self, hessian: Tensor, linear: Tensor, lo: Tensor, hi: Tensor, sigma: Tensor
    ) -> KKTResult:
        # Candidate enumeration and ranking are exact hard decisions.  Keeping
        # them out of autograd also avoids retaining 81 dense solve graphs.
        with torch.no_grad():
            batch = hessian.shape[0]
            states = self.kkt_states
            candidates = states.shape[0]
            free = states.eq(0)[None, :, :, None]
            h_rows = hessian[:, None, :, :].expand(
                batch, candidates, -1, -1
            )
            identity = torch.eye(
                4, device=hessian.device, dtype=hessian.dtype
            )
            identity = identity[None, None, :, :].expand_as(h_rows)
            matrix = torch.where(free, h_rows, identity)

            lower_values = lo[:, None, :].expand(batch, candidates, -1)
            upper_values = hi[:, None, :].expand(batch, candidates, -1)
            bound_values = torch.where(
                states[None, :, :].lt(0), lower_values, upper_values
            )
            rhs = torch.where(
                states[None, :, :].eq(0),
                -linear[:, None, :].expand(batch, candidates, -1),
                bound_values,
            )
            solution = torch.linalg.solve(
                matrix.reshape(-1, 4, 4), rhs.reshape(-1, 4, 1)
            ).reshape(batch, candidates, 4)
            gradient = (
                torch.einsum("bij,bnj->bni", hessian, solution)
                + linear[:, None, :]
            )

            feasibility_tol = 2.0e-5
            within_box = (
                solution.ge(lo[:, None, :] - feasibility_tol)
                & solution.le(hi[:, None, :] + feasibility_tol)
            )
            stationarity = torch.where(
                states[None, :, :].eq(0),
                gradient.abs().le(feasibility_tol),
                torch.ones_like(within_box),
            )
            lower_dual = torch.where(
                states[None, :, :].lt(0),
                gradient.ge(-feasibility_tol),
                torch.ones_like(within_box),
            )
            upper_dual = torch.where(
                states[None, :, :].gt(0),
                gradient.le(feasibility_tol),
                torch.ones_like(within_box),
            )
            valid_candidates = (
                within_box & stationarity & lower_dual & upper_dual
            ).all(dim=-1)
            candidate_objective = (
                torch.einsum("bi,bni->bn", linear, solution)
                + 0.5
                * torch.einsum(
                    "bni,bij,bnj->bn", solution, hessian, solution
                )
            )

            # Reproduce the shader's one-sided active-state tie preference.
            positive_preference = torch.where(
                states.gt(0), 0, torch.where(states.eq(0), 1, 2)
            )
            negative_preference = torch.where(
                states.lt(0), 0, torch.where(states.eq(0), 1, 2)
            )
            places = torch.tensor(
                [1, 3, 9, 27], device=hessian.device, dtype=torch.long
            )
            positive_score = (positive_preference * places).sum(dim=-1)
            negative_score = (negative_preference * places).sum(dim=-1)
            tie_score = torch.where(
                sigma[:, None].gt(0),
                positive_score[None, :],
                negative_score[None, :],
            )
            ranked_objective = (
                candidate_objective
                + tie_score.to(candidate_objective.dtype) * 1.0e-12
            )
            ranked_objective = ranked_objective.masked_fill(
                ~valid_candidates, torch.inf
            )
            best_index = ranked_objective.argmin(dim=-1)
            best_state = states[best_index]
            valid = valid_candidates.any(dim=-1)

        best_x = _SelectedBoxQPSolve.apply(
            hessian, linear, lo, hi, best_state
        )
        best_objective = (
            torch.einsum("bi,bi->b", linear, best_x)
            + 0.5
            * torch.einsum("bi,bij,bj->b", best_x, hessian, best_x)
        )

        primal_violation = torch.maximum(
            (lo - best_x).clamp_min(0).amax(dim=-1),
            (best_x - hi).clamp_min(0).amax(dim=-1),
        )
        best_gradient = torch.einsum("bij,bj->bi", hessian, best_x) + linear
        free_violation = torch.where(
            best_state.eq(0), best_gradient.abs(), torch.zeros_like(best_gradient)
        ).amax(dim=-1)
        lower_violation = torch.where(
            best_state.lt(0),
            (-best_gradient).clamp_min(0),
            torch.zeros_like(best_gradient),
        ).amax(dim=-1)
        upper_violation = torch.where(
            best_state.gt(0),
            best_gradient.clamp_min(0),
            torch.zeros_like(best_gradient),
        ).amax(dim=-1)
        max_violation = torch.stack(
            [primal_violation, free_violation, lower_violation, upper_violation],
            dim=-1,
        ).amax(dim=-1)
        free_margin = torch.minimum(best_x - lo, hi - best_x)
        dual_margin = torch.where(
            best_state.lt(0),
            best_gradient,
            torch.where(best_state.gt(0), -best_gradient, free_margin),
        )
        feasibility_margin = torch.where(
            best_state.eq(0), free_margin, dual_margin
        ).amin(dim=-1)
        free = best_state.eq(0)
        identity = torch.eye(
            4, device=hessian.device, dtype=hessian.dtype
        ).expand_as(hessian)
        selected_matrix = torch.where(free[..., None], hessian, identity)
        condition_number = torch.linalg.cond(selected_matrix)
        return KKTResult(
            x=best_x,
            objective=best_objective,
            active_state=best_state,
            active_mask=self.kkt_masks[best_index],
            valid=valid,
            max_violation=max_violation,
            feasibility_margin=feasibility_margin,
            condition_number=condition_number,
        )

    def _primitive_and_kkt(self, z: Tensor, sigma: Tensor):
        j_plus = self.current_field(z, H_J_PLUS, sigma, +1.0)
        j_minus = self.current_field(z, H_J_MINUS, sigma, -1.0)
        metric = self.metric_field(z, sigma)
        gamma = self.gamma_field(z, sigma)
        x_plus = self.invariant(metric, j_plus, j_plus)
        x_minus = self.invariant(metric, j_minus, j_minus)
        x_cross = self.invariant(metric, j_plus, j_minus)
        problem = self.build_kkt_problem(
            z, gamma, x_plus, x_minus, x_cross, sigma
        )
        kkt = self.solve_bounded_kkt(*problem, sigma)
        return j_plus, j_minus, metric, gamma, x_plus, x_minus, x_cross, kkt

    def evaluate(self, z: Tensor, sigma: Tensor) -> TesseractState:
        (
            j_plus,
            j_minus,
            metric,
            gamma,
            x_plus,
            x_minus,
            x_cross,
            kkt,
        ) = self._primitive_and_kkt(z, sigma)
        q_limited = kkt.x[:, 0]
        theta = torch.stack(
            [kkt.x[:, 1], kkt.x[:, 2], kkt.x[:, 3], torch.zeros_like(z)],
            dim=-1,
        )

        features = torch.stack(
            [x_plus, x_minus, x_cross, gamma, q_limited], dim=-1
        )
        lambda_value = self.master(features)
        lambda_partials = torch.autograd.grad(
            lambda_value.sum(), features, create_graph=True
        )[0]
        b_plus = -2.0 * lambda_partials[:, 0]
        b_minus = -2.0 * lambda_partials[:, 1]
        entrainment_a = -lambda_partials[:, 2]
        j_plus_cov = torch.einsum("bij,bj->bi", metric, j_plus)
        j_minus_cov = torch.einsum("bij,bj->bi", metric, j_minus)
        pi_plus = (
            b_plus[:, None] * j_plus_cov
            + entrainment_a[:, None] * j_minus_cov
        )
        pi_minus = (
            b_minus[:, None] * j_minus_cov
            + entrainment_a[:, None] * j_plus_cov
        )
        psi = (
            lambda_value
            - (j_plus * pi_plus).sum(dim=-1)
            - (j_minus * pi_minus).sum(dim=-1)
        )
        identity = torch.eye(4, device=z.device, dtype=z.dtype)
        stress = (
            psi[:, None, None] * identity[None, :, :]
            + torch.einsum("bi,bj->bij", j_plus, pi_plus)
            + torch.einsum("bi,bj->bij", j_minus, pi_minus)
        )

        phase_energy = vertex_phase(H_ENERGY)
        energy = (
            0.5 * (smooth_abs(x_plus) + smooth_abs(x_minus))
            + 0.25 * x_cross.square()
            + 0.10 * (gamma - 1.0).square()
            + 0.05 * theta[:, :3].square().sum(dim=-1)
            + 0.0025 * phase_energy
        )
        phase_limited = vertex_phase(H_J_LIMITED)
        j_limited = (
            q_limited
            + 0.50 * theta[:, 0] * theta[:, 1]
            - 0.25 * theta[:, 2]
            + 0.10 * torch.sin(z + theta[:, 2] + phase_limited * 0.1)
        )
        phase_residual = vertex_phase(H_RESIDUAL)
        residual = (
            gamma
            + q_limited * x_cross
            - theta[:, :3].mean(dim=-1)
            + 0.003 * phase_residual
        )
        stress_delta = stress - self.target_t[None, :, :]
        stress_term = stress_delta.square().sum(dim=(-2, -1))
        delta_pi = pi_plus - pi_minus
        momentum_term = torch.einsum(
            "bi,bij,bj->b", delta_pi, metric, delta_pi
        )
        free_energy = (
            self.omega_e * energy
            + 0.5 * self.omega_t * stress_term
            + 0.5 * self.omega_pi * momentum_term
            + self.omega_a * entrainment_a * x_cross
        )

        margins = torch.stack(
            [z, gamma - 1.0, x_cross, q_limited - theta[:, 0]], dim=-1
        )
        bits = torch.tensor([1, 2, 4, 8], device=z.device, dtype=torch.long)
        classifier_mask = (margins.ge(0).long() * bits).sum(dim=-1)
        return TesseractState(
            j_plus=j_plus,
            j_minus=j_minus,
            metric=metric,
            gamma=gamma,
            x_plus=x_plus,
            x_minus=x_minus,
            x_cross=x_cross,
            q_limited=q_limited,
            theta=theta,
            lambda_value=lambda_value,
            pi_plus=pi_plus,
            pi_minus=pi_minus,
            stress=stress,
            energy=energy,
            j_limited=j_limited,
            residual=residual,
            free_energy=free_energy,
            kkt=kkt,
            classifier_mask=classifier_mask,
            classifier_margin=margins,
        )

    def _certificate(self, z: Tensor, sigma: Tensor):
        values = self._primitive_and_kkt(z, sigma)
        gamma, x_cross, kkt = values[3], values[6], values[7]
        q = kkt.x[:, 0]
        theta0 = kkt.x[:, 1]
        margins = torch.stack([z, gamma - 1.0, x_cross, q - theta0], dim=-1)
        bits = torch.tensor([1, 2, 4, 8], device=z.device, dtype=torch.long)
        classifier = (margins.ge(0).long() * bits).sum(dim=-1)
        return kkt.active_mask, classifier, margins

    def _boundary_crossed(
        self,
        z0: Tensor,
        z1: Tensor,
        sigma: Tensor,
        state0: TesseractState,
    ) -> Tensor:
        with torch.no_grad():
            tolerance = self.crossing_tolerance
            u0 = 1.5 * (z0 + self.c.detach())
            u1 = 1.5 * (z1 + self.c.detach())
            maximum = torch.maximum(u0.abs(), u1.abs())
            minimum = torch.where(
                u0 * u1 <= 0,
                torch.zeros_like(u0),
                torch.minimum(u0.abs(), u1.abs()),
            )
            low_cell = torch.floor((minimum - tolerance).clamp_min(0)).long()
            high_cell = torch.floor(maximum + tolerance).long()
            crossed = low_cell.ne(high_cell) | ~torch.isfinite(z1)

            midpoint = 0.5 * (z0 + z1)
            mid_active, mid_classifier, mid_margins = self._certificate(
                midpoint, sigma
            )
            end_active, end_classifier, end_margins = self._certificate(z1, sigma)

            def changed(
                active_a: Tensor,
                classifier_a: Tensor,
                margins_a: Tensor,
                active_b: Tensor,
                classifier_b: Tensor,
                margins_b: Tensor,
            ) -> Tensor:
                margin_crossed = (
                    margins_a.abs().le(tolerance)
                    | margins_b.abs().le(tolerance)
                    | margins_a.lt(0).ne(margins_b.lt(0))
                ).any(dim=-1)
                return (
                    active_a.ne(active_b)
                    | classifier_a.ne(classifier_b)
                    | margin_crossed
                )

            crossed |= changed(
                state0.kkt.active_mask,
                state0.classifier_mask,
                state0.classifier_margin,
                mid_active,
                mid_classifier,
                mid_margins,
            )
            crossed |= changed(
                mid_active,
                mid_classifier,
                mid_margins,
                end_active,
                end_classifier,
                end_margins,
            )
            return crossed.to(z0.dtype)

    def _recurrence(
        self,
        z: Tensor,
        d_j_limited: Tensor,
        residual: Tensor,
        d_free_energy: Tensor,
        jump_free_energy: Tensor,
        iota: Tensor,
    ) -> Tensor:
        u = 1.5 * (z + self.c)
        parity = torch.remainder(torch.floor(u.abs()), 2.0).detach()
        normalization = torch.sqrt(
            (z.square() + self.c.square() + self.epsilon_n).clamp_min(SAFE_EPS)
        )
        correction = (
            self.kappa * u
            - self.alpha * d_j_limited
            + self.chi * residual
            - self.zeta
            * (d_free_energy + self.omega_delta * iota * jump_free_energy)
        )
        return u.square() + self.c + (parity / normalization) * correction

    def step(
        self,
        z: Tensor,
        *,
        detach_diagnostics: bool = True,
        measure_local_gain: bool = False,
    ) -> tuple[Tensor, StepDiagnostics]:
        if z.ndim != 1:
            raise ValueError("z must be a one-dimensional batch tensor")
        if not z.requires_grad:
            z = z.requires_grad_()
        sigma = torch.where(
            z.detach().ge(0), torch.ones_like(z), -torch.ones_like(z)
        )
        active = self.evaluate(z, sigma)
        opposite = self.evaluate(z, -sigma)
        d_j_limited = torch.autograd.grad(
            active.j_limited.sum(), z, create_graph=True
        )[0]
        d_free_energy = torch.autograd.grad(
            active.free_energy.sum(), z, create_graph=True
        )[0]
        jump_free_energy = sigma * (
            active.free_energy - opposite.free_energy
        )
        provisional = self._recurrence(
            z,
            d_j_limited,
            active.residual,
            d_free_energy,
            jump_free_energy,
            torch.zeros_like(z),
        )
        if self.boundary_certificates:
            iota = self._boundary_crossed(
                z.detach(), provisional.detach(), sigma.detach(), active
            )
        else:
            iota = torch.zeros_like(z)
        result = self._recurrence(
            z,
            d_j_limited,
            active.residual,
            d_free_energy,
            jump_free_energy,
            iota,
        )
        parity_coordinate = (1.5 * (z + self.c)).abs()
        parity_cell = torch.floor(parity_coordinate).detach()
        parity_margin = torch.minimum(
            parity_coordinate - parity_cell,
            parity_cell + 1.0 - parity_coordinate,
        )
        kkt_margin = torch.minimum(
            active.kkt.feasibility_margin,
            opposite.kkt.feasibility_margin,
        )
        kkt_condition = torch.maximum(
            active.kkt.condition_number,
            opposite.kkt.condition_number,
        )
        classifier_margin = torch.minimum(
            active.classifier_margin.abs().amin(dim=-1),
            opposite.classifier_margin.abs().amin(dim=-1),
        )
        switch_margin = torch.stack(
            [kkt_margin, classifier_margin, parity_margin], dim=-1
        ).amin(dim=-1)
        local_gain = None
        if measure_local_gain:
            local_gain = torch.autograd.grad(
                result.sum(),
                z,
                create_graph=not detach_diagnostics,
                retain_graph=True,
            )[0]

        def diagnostic(value: Tensor) -> Tensor:
            return value.detach() if detach_diagnostics else value

        diagnostics = StepDiagnostics(
            sigma=sigma.detach(),
            iota=iota.detach(),
            kkt_valid=(
                active.kkt.valid & opposite.kkt.valid
            ).detach(),
            kkt_max_violation=torch.maximum(
                active.kkt.max_violation,
                opposite.kkt.max_violation,
            ).detach(),
            kkt_feasibility_margin=diagnostic(kkt_margin),
            kkt_condition_number=kkt_condition.detach(),
            classifier_margin=diagnostic(classifier_margin),
            parity_margin=diagnostic(parity_margin),
            switch_margin=diagnostic(switch_margin),
            local_gain=(
                None if local_gain is None else diagnostic(local_gain)
            ),
            active_mask=active.kkt.active_mask.detach(),
            classifier_mask=active.classifier_mask.detach(),
        )
        return result, diagnostics

    def rollout(
        self,
        initial_z: Tensor,
        steps: int,
        *,
        detach_diagnostics: bool = True,
        measure_local_gain: bool = False,
    ) -> tuple[Tensor, list[StepDiagnostics]]:
        if steps < 1:
            raise ValueError("steps must be positive")
        z = initial_z
        trajectory = [z]
        diagnostics = []
        for _ in range(steps):
            z, step_diagnostics = self.step(
                z,
                detach_diagnostics=detach_diagnostics,
                measure_local_gain=measure_local_gain,
            )
            trajectory.append(z)
            diagnostics.append(step_diagnostics)
        return torch.stack(trajectory, dim=0), diagnostics
