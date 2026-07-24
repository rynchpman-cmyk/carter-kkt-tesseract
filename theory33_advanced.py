"""Frozen-data Theory 3.3 qualification, replay, and event-aware BPTT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np

from run_theory33_hybrid import (
    DEFAULT_TARGET,
    validate_broad_basin,
    validate_global_safety,
)
from theory33_experiments import (
    CertifiedNeuralKappaController,
    ConvexDriftResidual,
)
from theory33_hybrid import (
    BroadBasinKappaController,
    Theory33HybridTesseract,
    Theory33MasterFunction,
    Theory33Parameters,
)
from theory33_kinetic_data import KineticDataConfig, sha256_file


def _float(value: Tensor | float) -> float:
    if isinstance(value, Tensor):
        return float(value.detach())
    return float(value)


def _state_dict_to_json(module: nn.Module) -> dict[str, Any]:
    return {
        name: value.detach().cpu().tolist()
        for name, value in module.state_dict().items()
    }


def _load_json_state(module: nn.Module, state: dict[str, Any]) -> None:
    reference = module.state_dict()
    converted = {
        name: torch.as_tensor(value, dtype=reference[name].dtype)
        for name, value in state.items()
    }
    module.load_state_dict(converted, strict=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _split_indices(
    count: int,
    *,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    train_end = int(0.70 * count)
    validation_end = int(0.85 * count)
    return (
        order[:train_end],
        order[train_end:validation_end],
        order[validation_end:],
    )


class KineticMobilityNetwork(nn.Module):
    """Scalar-context network whose output is a PSD 2x2 mobility."""

    def __init__(self, width: int = 12) -> None:
        super().__init__()
        self.width = int(width)
        self.input_layer = nn.Linear(3, width)
        self.output_layer = nn.Linear(width, 3)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.zeros_(self.output_layer.weight)
        with torch.no_grad():
            self.output_layer.bias.copy_(
                torch.tensor([-1.5, 0.0, -1.0])
            )

    def cholesky(self, context: Tensor) -> Tensor:
        hidden = torch.tanh(self.input_layer(context))
        raw = self.output_layer(hidden)
        diagonal0 = F.softplus(raw[:, 0]) + 1.0e-6
        diagonal1 = F.softplus(raw[:, 2]) + 1.0e-6
        zero = torch.zeros_like(diagonal0)
        return torch.stack(
            [
                torch.stack([diagonal0, zero], dim=-1),
                torch.stack([raw[:, 1], diagonal1], dim=-1),
            ],
            dim=-2,
        )

    def forward(self, context: Tensor) -> Tensor:
        factor = self.cholesky(context)
        return factor @ factor.transpose(-1, -2)


class CovariantPSDMobility(nn.Module):
    """Two-sector PSD mobility times a spacetime-orthogonal projector."""

    def __init__(self, network: KineticMobilityNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(
        self,
        context: Tensor,
        metric: Tensor,
        four_velocity: Tensor,
        carrier_velocity: Tensor,
        force_amplitudes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mobility = self.network(context)
        inverse_metric = torch.linalg.inv(metric)
        projector = inverse_metric + torch.einsum(
            "bi,bj->bij", four_velocity, four_velocity
        )
        relative_gamma = -torch.einsum(
            "bi,bij,bj->b",
            four_velocity,
            metric,
            carrier_velocity,
        )
        relative_direction = (
            carrier_velocity - relative_gamma[:, None] * four_velocity
        )
        relative_norm = torch.sqrt(
            torch.einsum(
                "bi,bij,bj->b",
                relative_direction,
                metric,
                relative_direction,
            ).clamp_min(0.0)
        )
        coordinate_axis = torch.zeros_like(four_velocity)
        coordinate_axis[:, 1] = 1.0
        projector_mixed = projector @ metric
        fallback = torch.einsum(
            "bij,bj->bi", projector_mixed, coordinate_axis
        )
        fallback_norm = torch.sqrt(
            torch.einsum(
                "bi,bij,bj->b", fallback, metric, fallback
            ).clamp_min(1.0e-18)
        )
        normalized_relative = relative_direction / relative_norm[
            :, None
        ].clamp_min(1.0e-18)
        normalized_fallback = fallback / fallback_norm[:, None]
        spatial_vector = torch.where(
            (relative_norm > 1.0e-9)[:, None],
            normalized_relative,
            normalized_fallback,
        )
        spatial_covector = torch.einsum(
            "bij,bj->bi", metric, spatial_vector
        )
        force_covectors = (
            force_amplitudes[:, :, None] * spatial_covector[:, None, :]
        )
        projected_forces = torch.einsum(
            "buv,bjv->bju", projector, force_covectors
        )
        flux = -torch.einsum(
            "bij,bju->biu", mobility, projected_forces
        )
        entropy_production = -torch.einsum(
            "biu,biu->b", force_covectors, flux
        )
        return flux, entropy_production, mobility


def normalize_context(
    density: Tensor,
    temperature: Tensor,
    fraction: Tensor,
    config: KineticDataConfig,
) -> Tensor:
    density_coordinate = (
        2.0
        * (density - config.density_minimum)
        / (config.density_maximum - config.density_minimum)
        - 1.0
    )
    log_temperature = torch.log(temperature.clamp_min(1.0e-12))
    log_minimum = math.log(config.temperature_minimum)
    log_maximum = math.log(config.temperature_maximum)
    temperature_coordinate = (
        2.0 * (log_temperature - log_minimum) / (log_maximum - log_minimum)
        - 1.0
    )
    fraction_coordinate = (
        2.0
        * (fraction - config.fraction_minimum)
        / (config.fraction_maximum - config.fraction_minimum)
        - 1.0
    )
    return torch.stack(
        [
            density_coordinate,
            temperature_coordinate,
            fraction_coordinate,
        ],
        dim=-1,
    ).clamp(-1.5, 1.5)


class MobilityTheory33(Theory33HybridTesseract):
    """Theory33 relaxation generated by a covariant PSD mobility."""

    def __init__(
        self,
        mobility: CovariantPSDMobility,
        *,
        data_config: KineticDataConfig,
        response_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mobility = mobility
        self.data_config = data_config
        self.register_buffer(
            "mobility_response_scale", torch.tensor(response_scale)
        )

    def evaluate(self, z: Tensor, sigma: Tensor):
        state = super().evaluate(z, sigma)
        p = self.theory33_parameters
        fraction = state.carrier_density / state.number_density.clamp_min(
            1.0e-12
        )
        context = normalize_context(
            state.number_density,
            state.temperature,
            fraction,
            self.data_config,
        )
        chemical_force = (
            state.carrier_density - p.target_fraction * state.number_density
        ) / (
            p.chemical_susceptibility
            * state.number_density.clamp_min(1.0e-12)
        )
        drag_force = (
            state.carrier_rapidity - state.baryon_rapidity
        ) / p.relaxation_time
        force_amplitudes = torch.stack(
            [chemical_force, drag_force], dim=-1
        )
        four_velocity = (
            state.j_plus / state.number_density[:, None].clamp_min(1.0e-12)
        )
        carrier_velocity = (
            state.j_minus
            / state.carrier_density[:, None].clamp_min(1.0e-12)
        )
        flux, entropy_production, mobility_matrix = self.mobility(
            context,
            state.metric,
            four_velocity,
            carrier_velocity,
            force_amplitudes,
        )
        response_amplitudes = torch.einsum(
            "bij,bj->bi", mobility_matrix, force_amplitudes
        )
        state.residual = (
            -self.mobility_response_scale
            * response_amplitudes.sum(dim=-1)
        )
        state.mobility_flux = flux
        state.entropy_production = entropy_production
        state.mobility_matrix = mobility_matrix
        return state


@dataclass(frozen=True)
class ContextualPhaseThreshold:
    base: float
    temperature_coefficient: float
    density_coefficient: float
    fraction_coefficient: float
    logistic_width: float

    def tensor(self) -> Tensor:
        return torch.tensor(
            [
                self.base,
                self.temperature_coefficient,
                self.density_coefficient,
                self.fraction_coefficient,
            ]
        )


class QualifiedPhaseMaster(Theory33MasterFunction):
    """Analytic M1 plus a frozen convex residual above a learned phase edge."""

    def __init__(
        self,
        parameters: Theory33Parameters,
        drift_network: ConvexDriftResidual,
        phase: ContextualPhaseThreshold,
        *,
        energy_scale: float,
        drift_scale: float = 0.016,
        data_config: KineticDataConfig = KineticDataConfig(),
    ) -> None:
        super().__init__(parameters, learn_residual=False)
        self.drift_network = drift_network
        self.data_config = data_config
        self.register_buffer(
            "phase_coefficients", phase.tensor()
        )
        self.register_buffer("energy_scale", torch.tensor(energy_scale))
        self.register_buffer("drift_scale", torch.tensor(drift_scale))

    def relative_drift(self, features: Tensor) -> Tensor:
        n2, d2, cross, _entropy = features.unbind(dim=-1)
        product = torch.sqrt((n2 * d2).clamp_min(1.0e-24))
        return (cross / product - 1.0).clamp_min(0.0)

    def threshold(self, features: Tensor) -> Tensor:
        n2, d2, _cross, entropy = features.unbind(dim=-1)
        density = torch.sqrt(n2.clamp_min(1.0e-24))
        carrier = torch.sqrt(d2.clamp_min(1.0e-24))
        fraction = carrier / density.clamp_min(1.0e-12)
        p = self.parameters
        pressure = (
            torch.exp((p.gamma_ad - 1.0) * entropy)
            * density.pow(p.gamma_ad)
        )
        temperature = pressure / density.clamp_min(1.0e-12)
        context = normalize_context(
            density, temperature, fraction, self.data_config
        )
        return (
            self.phase_coefficients[0]
            + self.phase_coefficients[1] * context[:, 1]
            + self.phase_coefficients[2] * context[:, 0]
            + self.phase_coefficients[3] * context[:, 2]
        )

    def forward(self, features: Tensor) -> Tensor:
        base = self.base_lambda(features)
        drift = self.relative_drift(features)
        threshold = self.threshold(features)
        y = drift / self.drift_scale
        threshold_y = threshold / self.drift_scale
        phase_energy = self.energy_scale * (
            self.drift_network(y)
            - self.drift_network(threshold_y)
        )
        phase_two = (drift >= threshold).detach()
        return base - torch.where(
            phase_two, phase_energy, torch.zeros_like(phase_energy)
        )


def load_transport_tensors(
    path: Path,
    config: KineticDataConfig,
) -> dict[str, Tensor]:
    rows = _read_csv(path)
    density = torch.tensor([float(row["density"]) for row in rows])
    temperature = torch.tensor(
        [float(row["temperature"]) for row in rows]
    )
    fraction = torch.tensor(
        [float(row["carrier_fraction"]) for row in rows]
    )
    drift = torch.tensor([float(row["relative_drift"]) for row in rows])
    mobility = torch.tensor(
        [
            [
                [float(row["mobility_00"]), float(row["mobility_01"])],
                [float(row["mobility_01"]), float(row["mobility_11"])],
            ]
            for row in rows
        ]
    )
    drift_energy = torch.tensor(
        [float(row["drift_energy"]) for row in rows]
    )
    return {
        "context": normalize_context(
            density, temperature, fraction, config
        ),
        "density": density,
        "temperature": temperature,
        "fraction": fraction,
        "drift": drift,
        "mobility": mobility,
        "drift_energy": drift_energy,
    }


def train_mobility_network(
    data: dict[str, Tensor],
    *,
    epochs: int = 1200,
) -> tuple[KineticMobilityNetwork, dict[str, Any]]:
    torch.manual_seed(3305)
    # Mobility does not depend on the repeated drift coordinate.
    unique = torch.arange(0, data["context"].shape[0], 11)
    context = data["context"][unique]
    target = data["mobility"][unique]
    train, validation, test = _split_indices(
        context.shape[0], seed=3305
    )
    network = KineticMobilityNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=8.0e-3)
    scale = target[train].abs().mean(dim=0).clamp_min(1.0e-4)
    for _ in range(epochs):
        prediction = network(context[train])
        loss = (
            (prediction - target[train]) / scale
        ).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def metrics(indices: Tensor) -> dict[str, float]:
        prediction = network(context[indices]).detach()
        difference = prediction - target[indices]
        relative = difference.abs() / target[indices].abs().clamp_min(1.0e-4)
        minimum_eigenvalue = torch.linalg.eigvalsh(prediction)[..., 0]
        return {
            "rmse": _float(difference.square().mean().sqrt()),
            "maximum_relative_error": _float(relative.max()),
            "minimum_eigenvalue": _float(minimum_eigenvalue.min()),
        }

    return network, {
        "epochs": epochs,
        "unique_contexts": int(context.shape[0]),
        "train": metrics(train),
        "validation": metrics(validation),
        "test": metrics(test),
    }


def train_drift_network_from_data(
    data: dict[str, Tensor],
    *,
    config: KineticDataConfig,
    epochs: int = 900,
) -> tuple[ConvexDriftResidual, dict[str, Any]]:
    torch.manual_seed(3306)
    drift = data["drift"] / config.drift_maximum
    target = data["drift_energy"] / data["mobility"][:, 1, 1]
    train, validation, test = _split_indices(drift.numel(), seed=3306)
    network = ConvexDriftResidual()
    optimizer = torch.optim.Adam(network.parameters(), lr=6.0e-3)
    for _ in range(epochs):
        train_drift = drift[train].detach().clone().requires_grad_(True)
        prediction = network(train_drift)
        loss = (prediction - target[train]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def metrics(indices: Tensor) -> dict[str, float]:
        value = network(drift[indices]).detach()
        difference = value - target[indices]
        return {
            "rmse": _float(difference.square().mean().sqrt()),
            "maximum_absolute_error": _float(difference.abs().max()),
        }

    probe = torch.linspace(0.0, 1.0, 513).requires_grad_(True)
    value = network(probe)
    first = torch.autograd.grad(value.sum(), probe, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), probe)[0]
    return network, {
        "epochs": epochs,
        "rows": int(drift.numel()),
        "train": metrics(train),
        "validation": metrics(validation),
        "test": metrics(test),
        "origin_value": _float(value[0]),
        "origin_derivative": _float(first[0]),
        "minimum_second_derivative": _float(second.min()),
    }


def _phase_design(
    rows: Iterable[dict[str, str]],
    config: KineticDataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(rows)
    drift = np.asarray(
        [float(row["relative_drift"]) for row in rows]
    )
    temperature = np.asarray([float(row["temperature"]) for row in rows])
    density = np.asarray([float(row["density"]) for row in rows])
    fraction = np.asarray(
        [float(row["carrier_fraction"]) for row in rows]
    )
    temperature_coordinate = (
        2.0
        * (
            np.log(temperature)
            - math.log(config.temperature_minimum)
        )
        / (
            math.log(config.temperature_maximum)
            - math.log(config.temperature_minimum)
        )
        - 1.0
    )
    density_coordinate = (
        2.0
        * (density - config.density_minimum)
        / (config.density_maximum - config.density_minimum)
        - 1.0
    )
    fraction_coordinate = (
        2.0
        * (fraction - config.fraction_minimum)
        / (config.fraction_maximum - config.fraction_minimum)
        - 1.0
    )
    design = np.stack(
        [
            np.ones_like(drift),
            drift / 0.01,
            temperature_coordinate,
            density_coordinate,
            fraction_coordinate,
        ],
        axis=-1,
    )
    label = np.asarray([int(row["phase_label"]) for row in rows])
    trajectory = np.asarray(
        [int(row["trajectory_id"]) for row in rows]
    )
    return design, label.astype(np.float64), trajectory


def _fit_logistic_irls(
    design: np.ndarray,
    label: np.ndarray,
    *,
    ridge: float = 1.0e-6,
    iterations: int = 80,
) -> np.ndarray:
    def solve_small(matrix: Tensor, vector: Tensor) -> Tensor:
        augmented = torch.cat(
            [matrix.clone(), vector[:, None].clone()], dim=1
        )
        size = matrix.shape[0]
        for pivot in range(size):
            selected = pivot + int(
                torch.argmax(augmented[pivot:, pivot].abs())
            )
            if selected != pivot:
                saved = augmented[pivot].clone()
                augmented[pivot] = augmented[selected]
                augmented[selected] = saved
            divisor = augmented[pivot, pivot].clone()
            if abs(_float(divisor)) < 1.0e-18:
                raise RuntimeError("singular phase-threshold fit")
            augmented[pivot] /= divisor
            for row in range(size):
                if row == pivot:
                    continue
                augmented[row] -= (
                    augmented[row, pivot] * augmented[pivot]
                )
        return augmented[:, -1]

    design_tensor = torch.from_numpy(design)
    label_tensor = torch.from_numpy(label)
    coefficient = torch.zeros(design.shape[1])
    penalty = torch.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        linear = (design_tensor @ coefficient).clamp(-40.0, 40.0)
        probability = torch.sigmoid(linear)
        weight = (
            probability * (1.0 - probability)
        ).clamp_min(1.0e-8)
        hessian = (
            design_tensor.T @ (weight[:, None] * design_tensor)
            + penalty
        )
        gradient = (
            design_tensor.T @ (label_tensor - probability)
            - penalty @ coefficient
        )
        step = solve_small(hessian, gradient)
        coefficient += step
        if _float(torch.sqrt((step * step).sum())) < 1.0e-10:
            break
    return coefficient.numpy()


def _threshold_from_logistic(
    coefficient: np.ndarray,
) -> ContextualPhaseThreshold:
    drift_coefficient = coefficient[1]
    if drift_coefficient <= 0.0:
        raise RuntimeError("learned phase direction is not increasing")
    conversion = -0.01 / drift_coefficient
    return ContextualPhaseThreshold(
        base=float(conversion * coefficient[0]),
        temperature_coefficient=float(conversion * coefficient[2]),
        density_coefficient=float(conversion * coefficient[3]),
        fraction_coefficient=float(conversion * coefficient[4]),
        logistic_width=float(0.01 / drift_coefficient),
    )


def _classification_metrics(
    coefficient: np.ndarray,
    design: np.ndarray,
    label: np.ndarray,
) -> dict[str, float]:
    design_tensor = torch.from_numpy(design)
    coefficient_tensor = torch.from_numpy(coefficient)
    label_tensor = torch.from_numpy(label)
    linear = (design_tensor @ coefficient_tensor).clamp(-40.0, 40.0)
    probability = torch.sigmoid(linear)
    nll = F.binary_cross_entropy(probability, label_tensor)
    return {
        "negative_log_likelihood": _float(nll),
        "brier_score": _float(
            (probability - label_tensor).square().mean()
        ),
        "accuracy": _float(
            ((probability >= 0.5) == label_tensor.bool())
            .to(torch.float64)
            .mean()
        ),
    }


def fit_phase_threshold_with_uncertainty(
    path: Path,
    *,
    config: KineticDataConfig,
    bootstrap_samples: int = 200,
) -> tuple[ContextualPhaseThreshold, dict[str, Any]]:
    rows = _read_csv(path)
    design, label, trajectory = _phase_design(rows, config)
    train_mask = trajectory % 10 <= 6
    validation_mask = trajectory % 10 == 7
    test_mask = trajectory % 10 >= 8
    coefficient = _fit_logistic_irls(
        design[train_mask], label[train_mask]
    )
    phase = _threshold_from_logistic(coefficient)

    rng = np.random.default_rng(3307)
    train_trajectories = np.unique(trajectory[train_mask])
    samples = []
    for _ in range(bootstrap_samples):
        selected = rng.choice(
            train_trajectories,
            size=train_trajectories.size,
            replace=True,
        )
        indices = np.concatenate(
            [np.flatnonzero(trajectory == item) for item in selected]
        )
        fitted = _fit_logistic_irls(design[indices], label[indices])
        samples.append(asdict(_threshold_from_logistic(fitted)))

    uncertainty: dict[str, dict[str, float]] = {}
    for field in asdict(phase):
        values = np.asarray([sample[field] for sample in samples])
        uncertainty[field] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }

    return phase, {
        "bootstrap_samples": bootstrap_samples,
        "trajectory_split": {
            "train": int(np.unique(trajectory[train_mask]).size),
            "validation": int(
                np.unique(trajectory[validation_mask]).size
            ),
            "test": int(np.unique(trajectory[test_mask]).size),
        },
        "point_estimate": asdict(phase),
        "uncertainty": uncertainty,
        "train": _classification_metrics(
            coefficient, design[train_mask], label[train_mask]
        ),
        "validation": _classification_metrics(
            coefficient,
            design[validation_mask],
            label[validation_mask],
        ),
        "test": _classification_metrics(
            coefficient, design[test_mask], label[test_mask]
        ),
    }


def sha256_json_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def train_serializable_kappa(
    *,
    epochs: int = 800,
    samples_per_branch: int = 1024,
) -> tuple[CertifiedNeuralKappaController, dict[str, Any]]:
    """Train the certified controller while retaining its frozen weights."""
    torch.manual_seed(3301)
    teacher = BroadBasinKappaController()
    controller = CertifiedNeuralKappaController()
    base_kappa = torch.tensor(-0.6352141942408)
    intervals = (
        controller.negative_interval,
        controller.positive_interval,
    )
    z = torch.cat(
        [
            torch.linspace(lower, upper, samples_per_branch)
            for lower, upper in intervals
        ]
    )
    with torch.no_grad():
        target = teacher(z, base_kappa) - base_kappa
        controller.negative.anchor.copy_(target[0])
        controller.positive.anchor.copy_(target[samples_per_branch])
    optimizer = torch.optim.Adam(controller.parameters(), lr=8.0e-3)
    for _ in range(epochs):
        prediction = controller.residual(z)
        value_loss = (prediction - target).square().mean()
        endpoints = torch.tensor(
            [0, samples_per_branch - 1, samples_per_branch, z.numel() - 1]
        )
        endpoint_loss = (
            prediction[endpoints] - target[endpoints]
        ).square().mean()
        loss = value_loss + 4.0 * endpoint_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    prediction = controller.residual(z).detach()
    audit_model = Theory33HybridTesseract(
        learn_dynamics=False, boundary_certificates=False
    )
    audit_model.kappa_controller = controller
    basin = validate_broad_basin(
        audit_model,
        steps=128,
        target=DEFAULT_TARGET,
        points=129,
        characteristic_points=9,
    )
    return controller, {
        "epochs": epochs,
        "samples_per_branch": samples_per_branch,
        "rmse": _float((prediction - target).square().mean().sqrt()),
        "maximum_error": _float((prediction - target).abs().max()),
        "certified_slope_bound": _float(
            controller.negative.slope_bound
        ),
        "audit": basin,
    }


def explicit_margin_audit(
    model: Theory33HybridTesseract,
    *,
    points: int = 65,
    steps: int = 128,
    characteristic_points: int = 9,
    lyapunov_growth_limit: float = 0.65,
    characteristic_margin_floor: float = 0.01,
) -> dict[str, Any]:
    """Audit every requested qualification margin explicitly."""
    z = torch.linspace(-1.0, 1.0, points)
    previous_lyapunov = (z - DEFAULT_TARGET).square()
    metrics: dict[str, Any] = {
        "points": points,
        "steps": steps,
        "kkt_valid": True,
        "physical_valid": True,
        "finite": True,
        "min_kkt_margin": math.inf,
        "min_classifier_margin": math.inf,
        "min_parity_margin": math.inf,
        "min_switching_margin": math.inf,
        "min_legendre_margin": math.inf,
        "min_thermodynamic_margin": math.inf,
        "min_causal_screen_margin": math.inf,
        "min_timelike_margin": math.inf,
        "min_metric_signature_margin": math.inf,
        "max_lyapunov_growth": -math.inf,
        "max_abs_state": _float(z.abs().max()),
    }
    original_certificates = model.boundary_certificates
    model.boundary_certificates = False
    try:
        for _ in range(steps):
            probe = z.detach().clone().requires_grad_(True)
            z, diagnostics = model.step(probe)
            z = z.detach()
            lyapunov = (z - DEFAULT_TARGET).square()
            metrics["kkt_valid"] &= bool(diagnostics.kkt_valid.all())
            metrics["physical_valid"] &= bool(
                diagnostics.physical_valid.all()
            )
            metrics["finite"] &= bool(torch.isfinite(z).all())
            for key, value in (
                ("min_kkt_margin", diagnostics.kkt_feasibility_margin),
                ("min_classifier_margin", diagnostics.classifier_margin),
                ("min_parity_margin", diagnostics.parity_margin),
                ("min_switching_margin", diagnostics.switch_margin),
                ("min_legendre_margin", diagnostics.legendre_margin),
                (
                    "min_thermodynamic_margin",
                    diagnostics.thermodynamic_margin,
                ),
                (
                    "min_causal_screen_margin",
                    diagnostics.causal_screen_margin,
                ),
                ("min_timelike_margin", diagnostics.timelike_margin),
                (
                    "min_metric_signature_margin",
                    diagnostics.metric_signature_margin,
                ),
            ):
                metrics[key] = min(metrics[key], _float(value.min()))
            metrics["max_lyapunov_growth"] = max(
                metrics["max_lyapunov_growth"],
                _float((lyapunov - previous_lyapunov).max()),
            )
            metrics["max_abs_state"] = max(
                metrics["max_abs_state"], _float(z.abs().max())
            )
            previous_lyapunov = lyapunov
    finally:
        model.boundary_certificates = original_certificates

    maximum_characteristic_speed = 0.0
    characteristic_valid = True
    characteristic_grid = torch.linspace(
        -1.0,
        max(1.05, metrics["max_abs_state"]),
        characteristic_points,
    )
    for sigma_value in (-1.0, 1.0):
        probe = characteristic_grid.clone().requires_grad_(True)
        sigma = torch.full_like(probe, sigma_value)
        state = model.evaluate(probe, sigma)
        for audit in model.characteristic_audits(state):
            characteristic_valid &= audit.causal
            maximum_characteristic_speed = max(
                maximum_characteristic_speed,
                audit.maximum_absolute_speed,
            )
    metrics["characteristic_valid"] = characteristic_valid
    metrics["max_characteristic_speed"] = maximum_characteristic_speed
    metrics["characteristic_margin"] = (
        1.0 - maximum_characteristic_speed
    )
    metrics["lyapunov_margin"] = (
        lyapunov_growth_limit - metrics["max_lyapunov_growth"]
    )
    metrics["terminal_max_error"] = _float(
        (z - DEFAULT_TARGET).abs().max()
    )
    metrics["converged_fraction"] = _float(
        ((z - DEFAULT_TARGET).abs() < 1.0e-10)
        .to(torch.float64)
        .mean()
    )
    global_safety = validate_global_safety(
        model,
        steps=32,
        target=DEFAULT_TARGET,
        exponent_points=9,
    )
    metrics["global_safety"] = global_safety
    thresholds = {
        "kkt": 1.0e-6,
        "legendre": 1.0e-8,
        "thermodynamic": 1.0e-8,
        "switching": 1.0e-6,
        "characteristic": characteristic_margin_floor,
        "lyapunov": 0.0,
    }
    metrics["thresholds"] = thresholds
    metrics["qualified"] = (
        metrics["kkt_valid"]
        and metrics["physical_valid"]
        and metrics["finite"]
        and characteristic_valid
        and metrics["min_kkt_margin"] > thresholds["kkt"]
        and metrics["min_legendre_margin"] > thresholds["legendre"]
        and (
            metrics["min_thermodynamic_margin"]
            > thresholds["thermodynamic"]
        )
        and metrics["min_switching_margin"] > thresholds["switching"]
        and (
            metrics["characteristic_margin"]
            > thresholds["characteristic"]
        )
        and metrics["lyapunov_margin"] > thresholds["lyapunov"]
        and metrics["max_abs_state"] < 1.051
        and metrics["converged_fraction"] >= 0.99
        and bool(global_safety["finite"])
        and bool(global_safety["kkt_valid"])
        and bool(global_safety["physical_valid"])
        and float(global_safety["maximum_outer_gradient"]) == 0.0
    )
    return metrics


def qualify_residual_scale(
    drift_network: ConvexDriftResidual,
    phase: ContextualPhaseThreshold,
    *,
    data_config: KineticDataConfig,
    mobility_network: KineticMobilityNetwork | None = None,
    kappa_controller: CertifiedNeuralKappaController | None = None,
    minimum_scale: float = 1.0e-10,
    maximum_scale: float = 1.0e-6,
    bisection_steps: int = 8,
    audit_points: int = 65,
    characteristic_points: int = 9,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Maximize Carter residual scale subject to explicit safety margins."""
    parameters = Theory33Parameters()
    attempts: list[dict[str, Any]] = []

    def audit_scale(scale: float) -> dict[str, Any]:
        master = QualifiedPhaseMaster(
            parameters,
            drift_network,
            phase,
            energy_scale=scale,
            data_config=data_config,
        )
        if mobility_network is None:
            model = Theory33HybridTesseract(
                learn_dynamics=False,
                boundary_certificates=False,
            )
        else:
            model = MobilityTheory33(
                CovariantPSDMobility(mobility_network),
                data_config=data_config,
                learn_dynamics=False,
                boundary_certificates=False,
            )
        model.master = master
        if kappa_controller is not None:
            model.kappa_controller = kappa_controller
        audit = explicit_margin_audit(
            model,
            points=audit_points,
            characteristic_points=characteristic_points,
        )
        attempts.append(
            {
                "energy_scale": scale,
                "qualified": bool(audit["qualified"]),
                "min_kkt_margin": audit["min_kkt_margin"],
                "min_legendre_margin": audit["min_legendre_margin"],
                "min_thermodynamic_margin": (
                    audit["min_thermodynamic_margin"]
                ),
                "min_switching_margin": audit["min_switching_margin"],
                "characteristic_margin": audit["characteristic_margin"],
                "lyapunov_margin": audit["lyapunov_margin"],
                "max_characteristic_speed": (
                    audit["max_characteristic_speed"]
                ),
            }
        )
        return audit

    lower = minimum_scale
    lower_audit = audit_scale(lower)
    if not bool(lower_audit["qualified"]):
        raise RuntimeError("minimum Carter residual scale did not qualify")
    upper = lower
    upper_audit = lower_audit
    while upper < maximum_scale:
        candidate = min(2.0 * upper, maximum_scale)
        candidate_audit = audit_scale(candidate)
        if not bool(candidate_audit["qualified"]):
            upper = candidate
            upper_audit = candidate_audit
            break
        lower = candidate
        lower_audit = candidate_audit
        upper = candidate
        upper_audit = candidate_audit
        if candidate == maximum_scale:
            return lower, attempts, lower_audit
    if bool(upper_audit["qualified"]):
        return upper, attempts, upper_audit
    log_lower = math.log(lower)
    log_upper = math.log(upper)
    for _ in range(bisection_steps):
        candidate = math.exp(0.5 * (log_lower + log_upper))
        candidate_audit = audit_scale(candidate)
        if bool(candidate_audit["qualified"]):
            lower = candidate
            lower_audit = candidate_audit
            log_lower = math.log(candidate)
        else:
            log_upper = math.log(candidate)
    return lower, attempts, lower_audit


def _trajectory_objective_and_signature(
    model: Theory33HybridTesseract,
    initial: Tensor,
    *,
    steps: int,
    derivative: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    z0 = initial.detach().clone().requires_grad_(True)
    z = z0
    accumulated = torch.zeros_like(z)
    signatures = []
    original_certificates = model.boundary_certificates
    model.boundary_certificates = False
    try:
        for _ in range(steps):
            parity_cell = torch.floor(
                (1.5 * (z.detach() + model.c.detach())).abs()
            ).to(torch.long)
            sigma = torch.where(
                z.detach() >= 0.0,
                torch.ones_like(z),
                -torch.ones_like(z),
            )
            phase_code = torch.zeros_like(parity_cell)
            if isinstance(model.master, QualifiedPhaseMaster):
                state = model.evaluate(z, sigma)
                features = torch.stack(
                    [
                        state.x_plus,
                        state.x_minus,
                        state.x_cross,
                        state.entropy_per_baryon,
                    ],
                    dim=-1,
                )
                phase_code = (
                    model.master.relative_drift(features)
                    >= model.master.threshold(features)
                ).to(torch.long)
            z, diagnostics = model.step(
                z,
                detach_diagnostics=False,
            )
            signatures.append(
                torch.stack(
                    [
                        parity_cell,
                        diagnostics.active_mask,
                        diagnostics.classifier_mask,
                        diagnostics.sigma.to(torch.long),
                        diagnostics.outer_safety_gate.to(torch.long),
                        phase_code,
                    ],
                    dim=-1,
                )
            )
            accumulated = accumulated + (z - DEFAULT_TARGET).square()
        objective = (
            0.5 * (z - DEFAULT_TARGET).square()
            + 0.02 * accumulated / steps
        )
        objective_derivative = None
        if derivative:
            objective_derivative = torch.autograd.grad(
                objective.sum(), z0
            )[0]
        signature = torch.cat(signatures, dim=-1).detach()
        return objective.detach(), objective_derivative, signature
    finally:
        model.boundary_certificates = original_certificates


def _randomized_smoothing_estimate(
    model: Theory33HybridTesseract,
    *,
    center: float,
    smoothing_scale: float,
    steps: int,
    samples: int,
    batch_size: int = 256,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(3310 + steps)
    epsilon = torch.randn(samples, generator=generator)
    contributions = []
    for start in range(0, samples, batch_size):
        batch = epsilon[start : start + batch_size]
        plus, _, _ = _trajectory_objective_and_signature(
            model,
            center + smoothing_scale * batch,
            steps=steps,
            derivative=False,
        )
        minus, _, _ = _trajectory_objective_and_signature(
            model,
            center - smoothing_scale * batch,
            steps=steps,
            derivative=False,
        )
        contributions.append(
            (plus - minus) * batch / (2.0 * smoothing_scale)
        )
    values = torch.cat(contributions)
    return _float(values.mean()), _float(
        values.std(unbiased=True) / math.sqrt(values.numel())
    )


def compare_multistep_estimators(
    model: Theory33HybridTesseract,
    *,
    center: float = -2.0 / 3.0 - 0.05,
    smoothing_scale: float = 0.01,
    horizons: tuple[int, ...] = (1, 4, 8, 16),
    grid_points: int = 1537,
    randomized_samples: int = 2048,
) -> dict[str, Any]:
    """Compare pathwise and event-location derivatives over many steps."""
    coordinate = torch.linspace(
        center - 4.0 * smoothing_scale,
        center + 4.0 * smoothing_scale,
        grid_points,
    )
    standardized = (coordinate - center) / smoothing_scale
    density = torch.exp(-0.5 * standardized.square()) / (
        math.sqrt(2.0 * math.pi) * smoothing_scale
    )
    normalization = torch.trapezoid(density, coordinate)
    records = []
    for steps in horizons:
        loss, branch_derivative, signature = (
            _trajectory_objective_and_signature(
                model,
                coordinate,
                steps=steps,
                derivative=True,
            )
        )
        assert branch_derivative is not None
        ordinary = torch.trapezoid(
            branch_derivative * density, coordinate
        ) / normalization
        reference = torch.trapezoid(
            loss
            * (coordinate - center)
            * density
            / smoothing_scale**2,
            coordinate,
        ) / normalization
        changed = (signature[1:] != signature[:-1]).any(dim=-1)
        changed_indices = torch.nonzero(changed).flatten()
        jump_terms = []
        spacing = _float(coordinate[1] - coordinate[0])
        for index in changed_indices:
            left = int(index)
            right = left + 1
            branch_change = (
                0.5
                * (
                    _float(branch_derivative[left])
                    + _float(branch_derivative[right])
                )
                * spacing
            )
            jump = _float(loss[right] - loss[left]) - branch_change
            boundary = 0.5 * _float(
                coordinate[left] + coordinate[right]
            )
            boundary_density = (
                math.exp(
                    -0.5
                    * ((boundary - center) / smoothing_scale) ** 2
                )
                / (math.sqrt(2.0 * math.pi) * smoothing_scale)
                / _float(normalization)
            )
            if abs(jump) >= 1.0e-5:
                jump_terms.append(
                    {
                        "initial_event_location": boundary,
                        "objective_jump": jump,
                        "density_weight": boundary_density,
                        "derivative_contribution": (
                            jump * boundary_density
                        ),
                    }
                )
        jump_contribution = sum(
            item["derivative_contribution"] for item in jump_terms
        )
        ordinary_value = _float(ordinary)
        # A conservative/Clarke trace changes only the derivative assigned
        # at the measure-zero event itself under a continuous Gaussian. It
        # therefore equals the ordinary pathwise integral unless a separate
        # value-jump impulse is supplied.
        conservative = ordinary_value
        jump_aware = ordinary_value + jump_contribution
        randomized, randomized_standard_error = (
            _randomized_smoothing_estimate(
                model,
                center=center,
                smoothing_scale=smoothing_scale,
                steps=steps,
                samples=randomized_samples,
            )
        )
        reference_value = _float(reference)
        records.append(
            {
                "steps": steps,
                "event_signature_changes": int(changed_indices.numel()),
                "material_jump_count": len(jump_terms),
                "ordinary": ordinary_value,
                "conservative": conservative,
                "randomized_smoothing": randomized,
                "randomized_standard_error": randomized_standard_error,
                "jump_aware": jump_aware,
                "reference": reference_value,
                "ordinary_error": abs(ordinary_value - reference_value),
                "conservative_error": abs(
                    conservative - reference_value
                ),
                "randomized_error": abs(
                    randomized - reference_value
                ),
                "jump_aware_error": abs(
                    jump_aware - reference_value
                ),
                "event_location_terms": jump_terms,
            }
        )
    return {
        "center": center,
        "smoothing_distribution": "Gaussian",
        "smoothing_scale": smoothing_scale,
        "grid_points": grid_points,
        "randomized_samples": randomized_samples,
        "records": records,
        "jump_aware_best_every_horizon": all(
            item["jump_aware_error"]
            < item["ordinary_error"]
            for item in records
            if item["material_jump_count"] > 0
        ),
    }


def mobility_covariance_audit(
    network: KineticMobilityNetwork,
) -> dict[str, float | bool]:
    operator = CovariantPSDMobility(network)
    context = torch.tensor(
        [
            [-0.5, -0.25, 0.1],
            [0.2, 0.4, -0.3],
            [0.8, -0.7, 0.6],
        ]
    )
    rapidity = torch.tensor([-0.4, 0.1, 0.55])
    carrier_rapidity = rapidity + torch.tensor([0.15, -0.2, 0.25])
    velocity = torch.stack(
        [
            torch.cosh(rapidity),
            torch.sinh(rapidity),
            torch.zeros_like(rapidity),
            torch.zeros_like(rapidity),
        ],
        dim=-1,
    )
    carrier_velocity = torch.stack(
        [
            torch.cosh(carrier_rapidity),
            torch.sinh(carrier_rapidity),
            torch.zeros_like(rapidity),
            torch.zeros_like(rapidity),
        ],
        dim=-1,
    )
    metric = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0])).repeat(
        context.shape[0], 1, 1
    )
    forces = torch.tensor(
        [[0.03, -0.2], [-0.02, 0.15], [0.01, 0.3]]
    )
    flux, entropy, mobility = operator(
        context, metric, velocity, carrier_velocity, forces
    )
    boost_rapidity = 0.37
    boost = torch.eye(4)
    boost[0, 0] = math.cosh(boost_rapidity)
    boost[0, 1] = math.sinh(boost_rapidity)
    boost[1, 0] = math.sinh(boost_rapidity)
    boost[1, 1] = math.cosh(boost_rapidity)
    boosted_velocity = torch.einsum("ij,bj->bi", boost, velocity)
    boosted_carrier = torch.einsum(
        "ij,bj->bi", boost, carrier_velocity
    )
    boosted_flux, boosted_entropy, _ = operator(
        context,
        metric,
        boosted_velocity,
        boosted_carrier,
        forces,
    )
    expected_flux = torch.einsum("ij,bsj->bsi", boost, flux)
    covariance_error = _float((boosted_flux - expected_flux).abs().max())
    entropy_invariance_error = _float(
        (boosted_entropy - entropy).abs().max()
    )
    minimum_eigenvalue = _float(
        torch.linalg.eigvalsh(mobility)[..., 0].min()
    )
    minimum_entropy = _float(entropy.min())
    return {
        "covariance_error": covariance_error,
        "entropy_invariance_error": entropy_invariance_error,
        "minimum_mobility_eigenvalue": minimum_eigenvalue,
        "minimum_entropy_production": minimum_entropy,
        "passed": (
            covariance_error < 1.0e-10
            and entropy_invariance_error < 1.0e-10
            and minimum_eigenvalue > 0.0
            and minimum_entropy >= -1.0e-12
        ),
    }


def build_composed_frozen_model(
    *,
    controller: CertifiedNeuralKappaController,
    mobility_network: KineticMobilityNetwork,
    drift_network: ConvexDriftResidual,
    phase: ContextualPhaseThreshold,
    energy_scale: float,
    data_config: KineticDataConfig,
) -> MobilityTheory33:
    model = MobilityTheory33(
        CovariantPSDMobility(mobility_network),
        data_config=data_config,
        learn_dynamics=False,
        boundary_certificates=False,
    )
    model.kappa_controller = controller
    model.master = QualifiedPhaseMaster(
        Theory33Parameters(),
        drift_network,
        phase,
        energy_scale=energy_scale,
        data_config=data_config,
    )
    return model


def _probe_outputs(
    controller: CertifiedNeuralKappaController,
    mobility_network: KineticMobilityNetwork,
    drift_network: ConvexDriftResidual,
    phase: ContextualPhaseThreshold,
) -> dict[str, Any]:
    controller_z = torch.tensor([-0.95, -0.80, 0.70, 0.90])
    context = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-0.2, 0.1, 0.4],
            [0.5, -0.4, 0.8],
            [1.0, 1.0, 1.0],
        ]
    )
    drift = torch.tensor([0.0, 0.2, 0.6, 1.0])
    return {
        "controller_z": controller_z.tolist(),
        "controller_residual": (
            controller.residual(controller_z).detach().tolist()
        ),
        "mobility_context": context.tolist(),
        "mobility_matrix": mobility_network(context).detach().tolist(),
        "drift_coordinate": drift.tolist(),
        "drift_energy": drift_network(drift).detach().tolist(),
        "phase": asdict(phase),
    }


def create_frozen_artifact(
    *,
    data_directory: Path,
    controller: CertifiedNeuralKappaController,
    mobility_network: KineticMobilityNetwork,
    drift_network: ConvexDriftResidual,
    phase: ContextualPhaseThreshold,
    energy_scale: float,
    metrics: dict[str, Any],
    data_config: KineticDataConfig,
) -> dict[str, Any]:
    variants = {
        "controller": {
            "architecture": "certified-monotone-lattice",
            "width": 24,
            "state": _state_dict_to_json(controller),
        },
        "mobility": {
            "architecture": "context-mlp-cholesky-psd",
            "width": mobility_network.width,
            "state": _state_dict_to_json(mobility_network),
        },
        "drift_master": {
            "architecture": "convex-quadratic-softplus-bregman",
            "width": 10,
            "state": _state_dict_to_json(drift_network),
            "energy_scale": energy_scale,
            "drift_scale": data_config.drift_maximum,
        },
        "phase": asdict(phase),
    }
    artifact = {
        "format": "theory33-frozen-constitutive-variants",
        "version": 1,
        "dtype": "float64",
        "data_config": asdict(data_config),
        "datasets": {
            "transport": {
                "path": "data/theory33_kinetic_transport.csv",
                "sha256": sha256_file(
                    data_directory / "theory33_kinetic_transport.csv"
                ),
            },
            "phase": {
                "path": "data/theory33_noisy_phase_trajectories.csv",
                "sha256": sha256_file(
                    data_directory
                    / "theory33_noisy_phase_trajectories.csv"
                ),
            },
        },
        "variants": variants,
        "variant_digest": sha256_json_payload(variants),
        "probe_outputs": _probe_outputs(
            controller, mobility_network, drift_network, phase
        ),
        "metrics": metrics,
    }
    return artifact


def write_frozen_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )


def load_frozen_variants(
    artifact: dict[str, Any],
) -> tuple[
    CertifiedNeuralKappaController,
    KineticMobilityNetwork,
    ConvexDriftResidual,
    ContextualPhaseThreshold,
    float,
    KineticDataConfig,
]:
    if artifact.get("format") != "theory33-frozen-constitutive-variants":
        raise ValueError("unexpected frozen Theory 3.3 artifact format")
    variants = artifact["variants"]
    if sha256_json_payload(variants) != artifact["variant_digest"]:
        raise ValueError("frozen variant digest mismatch")
    config = KineticDataConfig(**artifact["data_config"])
    controller = CertifiedNeuralKappaController(
        width=int(variants["controller"]["width"])
    )
    _load_json_state(controller, variants["controller"]["state"])
    mobility = KineticMobilityNetwork(
        width=int(variants["mobility"]["width"])
    )
    _load_json_state(mobility, variants["mobility"]["state"])
    drift = ConvexDriftResidual(
        width=int(variants["drift_master"]["width"])
    )
    _load_json_state(drift, variants["drift_master"]["state"])
    phase = ContextualPhaseThreshold(**variants["phase"])
    energy_scale = float(variants["drift_master"]["energy_scale"])
    return controller, mobility, drift, phase, energy_scale, config


def replay_frozen_artifact(
    path: Path,
    *,
    audit_points: int = 65,
    characteristic_points: int = 9,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    (
        controller,
        mobility,
        drift,
        phase,
        energy_scale,
        config,
    ) = load_frozen_variants(artifact)
    for dataset in artifact["datasets"].values():
        dataset_path = path.parent / dataset["path"]
        if sha256_file(dataset_path) != dataset["sha256"]:
            raise ValueError(f"dataset digest mismatch: {dataset_path}")
    replay_probe = _probe_outputs(controller, mobility, drift, phase)
    recorded_probe = artifact["probe_outputs"]

    def maximum_nested_error(left: Any, right: Any) -> float:
        left_tensor = torch.as_tensor(left)
        right_tensor = torch.as_tensor(right)
        return _float((left_tensor - right_tensor).abs().max())

    probe_errors = {
        "controller": maximum_nested_error(
            replay_probe["controller_residual"],
            recorded_probe["controller_residual"],
        ),
        "mobility": maximum_nested_error(
            replay_probe["mobility_matrix"],
            recorded_probe["mobility_matrix"],
        ),
        "drift": maximum_nested_error(
            replay_probe["drift_energy"],
            recorded_probe["drift_energy"],
        ),
    }
    model = build_composed_frozen_model(
        controller=controller,
        mobility_network=mobility,
        drift_network=drift,
        phase=phase,
        energy_scale=energy_scale,
        data_config=config,
    )
    audit = explicit_margin_audit(
        model,
        points=audit_points,
        characteristic_points=characteristic_points,
    )
    covariance = mobility_covariance_audit(mobility)
    passed = (
        max(probe_errors.values()) == 0.0
        and bool(audit["qualified"])
        and bool(covariance["passed"])
    )
    return {
        "passed": passed,
        "fit_executed": False,
        "probe_errors": probe_errors,
        "audit": audit,
        "mobility_covariance": covariance,
    }
