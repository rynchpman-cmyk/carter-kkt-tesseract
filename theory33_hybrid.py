"""Self-contained physical two-current mode for the Carter/KKT Tesseract.

This module deliberately has no dependency on an external solver. It
implements only the local, covariant two-current constitutive closure needed
by the public Tesseract:

* future-timelike baryon and carrier currents built from KKT primitives;
* the analytic M1 master function;
* Carter conjugate momenta, generalized pressure, and Hilbert stress;
* convexity, timelike-domain, and characteristic diagnostics; and
* the original hard KKT/parity recurrence around that physical closure.

The complete finite-volume, Proca, and spacetime evolution systems are outside
the scope of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from hybrid_tesseract import (
    SAFE_EPS,
    CarterMasterFunction,
    HybridTesseract,
    StepDiagnostics,
    TesseractState,
)


@dataclass(frozen=True)
class Theory33Parameters:
    """Parameters of the public M1 two-current constitutive family."""

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
    density_reference: float = 0.1
    carrier_fraction_reference: float = 0.05
    entropy_reference: float = 1.0
    entropy_span: float = 0.25
    density_floor: float = 1.0e-10
    convexity_floor: float = 1.0e-10
    maximum_relative_lorentz_factor: float = 2.0
    characteristic_condition_limit: float = 1.0e8

    def __post_init__(self) -> None:
        if not 1.0 < self.gamma_ad < 2.0:
            raise ValueError("gamma_ad must lie in (1, 2)")
        if self.carrier_charge == 0.0:
            raise ValueError("carrier_charge must be nonzero")
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
            raise ValueError("entrainment_scale_fourth must be positive")
        if self.relaxation_time <= 0.0:
            raise ValueError("relaxation_time must be positive")
        if self.density_reference <= 0.0:
            raise ValueError("density_reference must be positive")
        if self.carrier_fraction_reference <= 0.0:
            raise ValueError("carrier_fraction_reference must be positive")
        if self.entropy_span < 0.0:
            raise ValueError("entropy_span must be nonnegative")
        if self.density_floor <= 0.0 or self.convexity_floor <= 0.0:
            raise ValueError("physical floors must be positive")
        if self.maximum_relative_lorentz_factor <= 1.0:
            raise ValueError("maximum relative Lorentz factor must exceed one")
        if self.characteristic_condition_limit <= 1.0:
            raise ValueError("characteristic condition limit is invalid")


@dataclass(frozen=True)
class CharacteristicAudit:
    speeds: Tensor
    maximum_absolute_speed: float
    eigenvector_condition: float
    legendre_minimum_eigenvalue: float
    thermodynamic_minimum_eigenvalue: float
    strongly_hyperbolic: bool
    causal: bool


@dataclass
class Theory33TesseractState(TesseractState):
    number_density: Tensor
    carrier_density: Tensor
    entropy_per_baryon: Tensor
    baryon_rapidity: Tensor
    carrier_rapidity: Tensor
    relative_lorentz_factor: Tensor
    relative_invariant: Tensor
    generalized_pressure: Tensor
    b_baryon: Tensor
    b_carrier: Tensor
    entrainment: Tensor
    chemical_baryon: Tensor
    chemical_carrier: Tensor
    temperature: Tensor
    energy_density: Tensor
    legendre_margin: Tensor
    thermodynamic_margin: Tensor
    causal_screen_margin: Tensor
    timelike_margin: Tensor
    metric_signature_margin: Tensor
    physical_valid: Tensor


@dataclass
class Theory33StepDiagnostics(StepDiagnostics):
    physical_valid: Tensor
    legendre_margin: Tensor
    thermodynamic_margin: Tensor
    causal_screen_margin: Tensor
    timelike_margin: Tensor
    metric_signature_margin: Tensor
    maximum_relative_lorentz_factor: Tensor
    current_normalization_error: Tensor
    characteristic_valid: Tensor | None
    maximum_characteristic_speed: Tensor | None


class BroadBasinKappaController(nn.Module):
    """Compact fitted state dependence for the odd-cell capture gain.

    Part 6 permits a step-dependent ``kappa_n``.  The controller represents
    its residual over the constant ``kappa`` as two degree-five Chebyshev
    series, one for each odd interval reachable from ``z in [-1, 1]``.
    The coefficients are a deterministic least-squares projection of the
    exact one-step capture gain.  Keeping this controller outside the master
    function preserves the qualified M1 characteristic structure.
    """

    negative_interval = (-1.0, -2.0 / 3.0 - 0.05)
    positive_interval = (2.0 / 3.0 - 0.05, 1.05)
    negative_residual = (
        0.2431000319387511,
        -0.5872255154526649,
        -0.004016857234897285,
        -0.0017696005661933711,
        -0.00015602168925102296,
        -0.00001359424888399637,
    )
    positive_residual = (
        -0.1277798606082578,
        -0.6691702417812642,
        -0.021897173207727234,
        -0.001599039698013725,
        0.00019723263036147492,
        -0.000024669618214050533,
    )

    def __init__(self, *, learnable: bool = False) -> None:
        super().__init__()
        self.negative_coefficients = nn.Parameter(
            torch.tensor(self.negative_residual),
            requires_grad=learnable,
        )
        self.positive_coefficients = nn.Parameter(
            torch.tensor(self.positive_residual),
            requires_grad=learnable,
        )

    @staticmethod
    def _normalized_coordinate(
        value: Tensor, interval: tuple[float, float]
    ) -> Tensor:
        lower, upper = interval
        return (
            2.0 * (value - lower) / (upper - lower) - 1.0
        ).clamp(-1.0, 1.0)

    @staticmethod
    def _chebyshev_series(value: Tensor, coefficients: Tensor) -> Tensor:
        terms = [torch.ones_like(value), value]
        for _ in range(2, coefficients.numel()):
            terms.append(2.0 * value * terms[-1] - terms[-2])
        basis = torch.stack(terms[: coefficients.numel()], dim=-1)
        return torch.einsum("bi,i->b", basis, coefficients)

    def forward(self, z: Tensor, base_kappa: Tensor) -> Tensor:
        negative_coordinate = self._normalized_coordinate(
            z, self.negative_interval
        )
        positive_coordinate = self._normalized_coordinate(
            z, self.positive_interval
        )
        negative = self._chebyshev_series(
            negative_coordinate, self.negative_coefficients
        )
        positive = self._chebyshev_series(
            positive_coordinate, self.positive_coefficients
        )
        residual = torch.where(z < 0.0, negative, positive)
        return base_kappa + residual


class Theory33MasterFunction(nn.Module):
    """Analytic M1 master plus an opt-in, zero-initialized neural residual."""

    def __init__(
        self,
        parameters: Theory33Parameters,
        *,
        residual_width: int = 32,
        learn_residual: bool = False,
        maximum_residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if maximum_residual_scale < 0.0:
            raise ValueError("maximum_residual_scale must be nonnegative")
        self.parameters = parameters
        self.maximum_residual_scale = float(maximum_residual_scale)
        self.residual = CarterMasterFunction(residual_width)
        self.raw_residual_scale = nn.Parameter(
            torch.tensor(0.0), requires_grad=learn_residual
        )
        for parameter in self.residual.parameters():
            parameter.requires_grad_(learn_residual)

    @property
    def residual_scale(self) -> Tensor:
        return self.maximum_residual_scale * torch.tanh(
            self.raw_residual_scale
        )

    def base_lambda(self, features: Tensor) -> Tensor:
        """Return M1 for ``(n², d², x², entropy)``."""
        p = self.parameters
        n_squared, d_squared, x_squared, entropy = features.unbind(dim=-1)
        n = torch.sqrt(n_squared.clamp_min(SAFE_EPS))
        d = torch.sqrt(d_squared.clamp_min(SAFE_EPS))
        pressure = torch.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        thermal = n + pressure / (p.gamma_ad - 1.0)
        anchor = d - p.target_fraction * n
        rho0 = (
            thermal
            + p.carrier_K * d ** (1.0 + p.carrier_sound_speed**2)
            + 0.5 * anchor.square() / p.chemical_susceptibility
        )
        relative = x_squared - n * d
        return (
            -rho0
            - p.entrainment_linear * relative
            - 0.5
            * p.entrainment_quadratic
            * relative.square()
            / p.entrainment_scale_fourth
        )

    def residual_features(self, features: Tensor) -> Tensor:
        p = self.parameters
        n_squared, d_squared, x_squared, entropy = features.unbind(dim=-1)
        n = torch.sqrt(n_squared.clamp_min(SAFE_EPS))
        d = torch.sqrt(d_squared.clamp_min(SAFE_EPS))
        gamma_relative = x_squared / (n * d).clamp_min(SAFE_EPS)
        return torch.stack(
            [
                torch.log(n / p.density_reference),
                torch.log(
                    d
                    / (
                        p.density_reference
                        * p.carrier_fraction_reference
                    )
                ),
                gamma_relative - 1.0,
                entropy - p.entropy_reference,
                d / n.clamp_min(SAFE_EPS) - p.target_fraction,
            ],
            dim=-1,
        )

    def forward(self, features: Tensor) -> Tensor:
        base = self.base_lambda(features)
        residual = self.residual(self.residual_features(features))
        return base + self.residual_scale * residual

    def thermodynamics(
        self,
        n: Tensor,
        d: Tensor,
        entropy: Tensor,
    ) -> dict[str, Tensor]:
        """Analytic M1 thermodynamics at fixed entropy."""
        p = self.parameters
        pressure = torch.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        thermal = n + pressure / (p.gamma_ad - 1.0)
        anchor = d - p.target_fraction * n
        carrier_energy = p.carrier_K * d ** (
            1.0 + p.carrier_sound_speed**2
        )
        chemical_energy = (
            0.5 * anchor.square() / p.chemical_susceptibility
        )
        chemical_baryon = (
            1.0
            + p.gamma_ad
            * pressure
            / ((p.gamma_ad - 1.0) * n).clamp_min(SAFE_EPS)
            - p.target_fraction
            * anchor
            / p.chemical_susceptibility
        )
        chemical_carrier = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * d ** (p.carrier_sound_speed**2)
            + anchor / p.chemical_susceptibility
        )
        hessian_nn = (
            p.gamma_ad * pressure / n.square().clamp_min(SAFE_EPS)
            + p.target_fraction**2 / p.chemical_susceptibility
        )
        hessian_nd = torch.full_like(
            n, -p.target_fraction / p.chemical_susceptibility
        )
        hessian_dd = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * p.carrier_sound_speed**2
            * d ** (p.carrier_sound_speed**2 - 1.0)
            + 1.0 / p.chemical_susceptibility
        )
        thermal_energy = thermal
        baryon_sound_squared = (
            p.gamma_ad
            * pressure
            / (thermal_energy + pressure).clamp_min(SAFE_EPS)
        )
        return {
            "pressure": pressure,
            "thermal_energy": thermal_energy,
            "carrier_energy": carrier_energy,
            "chemical_energy": chemical_energy,
            "anchor": anchor,
            "chemical_baryon": chemical_baryon,
            "chemical_carrier": chemical_carrier,
            "hessian_nn": hessian_nn,
            "hessian_nd": hessian_nd,
            "hessian_dd": hessian_dd,
            "temperature": pressure / n.clamp_min(SAFE_EPS),
            "baryon_sound_squared": baryon_sound_squared,
        }

    def _flat_constitutive_vector(
        self, primitive: Tensor, entropy: Tensor
    ) -> Tensor:
        """Return the 1D time and space principal constitutive vectors."""
        log_n, rapidity_n, log_d, rapidity_d = primitive.unbind()
        n = torch.exp(log_n)
        d = torch.exp(log_d)
        n0 = n * torch.cosh(rapidity_n)
        nx = n * torch.sinh(rapidity_n)
        d0 = d * torch.cosh(rapidity_d)
        dx = d * torch.sinh(rapidity_d)
        n_squared = n.square()
        d_squared = d.square()
        x_squared = n * d * torch.cosh(rapidity_n - rapidity_d)
        invariants = torch.stack(
            [n_squared, d_squared, x_squared, entropy]
        )
        lambda_value = self(invariants[None, :])[0]
        partials = torch.autograd.grad(
            lambda_value,
            invariants,
            create_graph=True,
            retain_graph=True,
        )[0]
        b_n = -2.0 * partials[0]
        b_d = -2.0 * partials[1]
        entrainment = -partials[2]
        mu_x = b_n * nx + entrainment * dx
        chi_x = b_d * dx + entrainment * nx
        minus_mu_0 = b_n * n0 + entrainment * d0
        minus_chi_0 = b_d * d0 + entrainment * n0
        return torch.stack(
            [n0, mu_x, d0, chi_x, nx, minus_mu_0, dx, minus_chi_0]
        )

    def characteristic_audit(
        self,
        density: float,
        entropy: float,
        baryon_rapidity: float,
        carrier_density: float,
        carrier_rapidity: float,
    ) -> CharacteristicAudit:
        """Build the exact local 1D principal symbol by differentiation."""
        if density <= 0.0 or carrier_density <= 0.0:
            raise ValueError("characteristic audit requires positive densities")
        point = torch.tensor(
            [
                math.log(density),
                baryon_rapidity,
                math.log(carrier_density),
                carrier_rapidity,
            ],
            dtype=self.raw_residual_scale.dtype,
            device=self.raw_residual_scale.device,
            requires_grad=True,
        )
        entropy_tensor = torch.tensor(
            entropy,
            dtype=point.dtype,
            device=point.device,
        )
        jacobian = torch.autograd.functional.jacobian(
            lambda value: self._flat_constitutive_vector(
                value, entropy_tensor
            ),
            point,
            create_graph=False,
            vectorize=True,
        )
        time_matrix = jacobian[:4]
        space_matrix = jacobian[4:]
        symbol = torch.linalg.solve(time_matrix, space_matrix)
        eigenvalues, eigenvectors = torch.linalg.eig(symbol)
        speeds = torch.sort(eigenvalues.real).values.detach()
        maximum_imaginary = float(eigenvalues.imag.abs().max())
        condition = float(torch.linalg.cond(eigenvectors))

        primitive = point.detach().clone().requires_grad_(True)
        log_n, rapidity_n, log_d, rapidity_d = primitive.unbind()
        n = torch.exp(log_n)
        d = torch.exp(log_d)
        invariants = torch.stack(
            [
                n.square(),
                d.square(),
                n * d * torch.cosh(rapidity_n - rapidity_d),
                entropy_tensor,
            ]
        )
        lambda_value = self(invariants[None, :])[0]
        partials = torch.autograd.grad(lambda_value, invariants)[0]
        b_n = -2.0 * partials[0]
        b_d = -2.0 * partials[1]
        entrainment = -partials[2]
        legendre = torch.stack(
            [
                torch.stack([b_n, entrainment]),
                torch.stack([entrainment, b_d]),
            ]
        )
        thermo = self.thermodynamics(
            n[None], d[None], entropy_tensor[None]
        )
        hessian = torch.stack(
            [
                torch.stack(
                    [
                        thermo["hessian_nn"][0],
                        thermo["hessian_nd"][0],
                    ]
                ),
                torch.stack(
                    [
                        thermo["hessian_nd"][0],
                        thermo["hessian_dd"][0],
                    ]
                ),
            ]
        )
        legendre_minimum = float(
            torch.linalg.eigvalsh(legendre)[0].detach()
        )
        thermodynamic_minimum = float(
            torch.linalg.eigvalsh(hessian)[0].detach()
        )
        p = self.parameters
        strong = (
            maximum_imaginary < 1.0e-8
            and math.isfinite(condition)
            and condition < p.characteristic_condition_limit
            and legendre_minimum > p.convexity_floor
            and thermodynamic_minimum > p.convexity_floor
        )
        maximum_speed = float(speeds.abs().max())
        return CharacteristicAudit(
            speeds=speeds,
            maximum_absolute_speed=maximum_speed,
            eigenvector_condition=condition,
            legendre_minimum_eigenvalue=legendre_minimum,
            thermodynamic_minimum_eigenvalue=thermodynamic_minimum,
            strongly_hyperbolic=strong,
            causal=strong and maximum_speed <= 1.0 + 1.0e-8,
        )


class Theory33HybridTesseract(HybridTesseract):
    """Hard-KKT Tesseract whose Carter sector is a physical M1 material."""

    def __init__(
        self,
        *,
        parameters: Theory33Parameters | None = None,
        master_width: int = 32,
        learn_dynamics: bool = True,
        learn_target: bool = False,
        learn_master_residual: bool = False,
        broad_basin_conditioning: bool = True,
        learn_conditioner: bool = False,
        boundary_certificates: bool = True,
    ) -> None:
        super().__init__(
            master_width=master_width,
            learn_dynamics=learn_dynamics,
            learn_target=learn_target,
            boundary_certificates=boundary_certificates,
        )
        self.theory33_parameters = parameters or Theory33Parameters()
        self.master = Theory33MasterFunction(
            self.theory33_parameters,
            residual_width=master_width,
            learn_residual=learn_master_residual,
        )
        self.broad_basin_conditioning = broad_basin_conditioning
        self.kappa_controller = BroadBasinKappaController(
            learnable=learn_conditioner
        )
        # Use the same intrinsic stable even-cell core as the conditioned
        # public baseline. Odd-cell capture is supplied by the physical
        # correction terms. These gains are an independent one-step physical
        # capture initialization; they do not load the baseline neural master.
        with torch.no_grad():
            self.c.fill_(0.05)
            self.kappa.fill_(-0.6352141942408)
            self.alpha.fill_(4.846934670314316)
            self.zeta.fill_(5.229051999290102)
            self.omega_delta.zero_()

    def _recurrence(
        self,
        z: Tensor,
        d_j_limited: Tensor,
        residual: Tensor,
        d_free_energy: Tensor,
        jump_free_energy: Tensor,
        iota: Tensor,
    ) -> Tensor:
        """Apply the literal recurrence with fitted state-dependent kappa."""
        u = 1.5 * (z + self.c)
        parity = torch.remainder(torch.floor(u.abs()), 2.0).detach()
        normalization = torch.sqrt(
            (z.square() + self.c.square() + self.epsilon_n).clamp_min(
                SAFE_EPS
            )
        )
        kappa = (
            self.kappa_controller(z, self.kappa)
            if self.broad_basin_conditioning
            else self.kappa
        )
        correction = (
            kappa * u
            - self.alpha * d_j_limited
            + self.chi * residual
            - self.zeta
            * (d_free_energy + self.omega_delta * iota * jump_free_energy)
        )
        return u.square() + self.c + (parity / normalization) * correction

    def load_compatible_dynamics(
        self, checkpoint: str | Path | dict[str, Any]
    ) -> list[str]:
        """Load non-master parameters from a baseline hybrid checkpoint."""
        if isinstance(checkpoint, (str, Path)):
            document = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        else:
            document = checkpoint
        source = document.get("model_state", document)
        destination = self.state_dict()
        compatible = {
            name: value
            for name, value in source.items()
            if not name.startswith("master.")
            and name in destination
            and destination[name].shape == value.shape
        }
        self.load_state_dict(compatible, strict=False)
        return sorted(compatible)

    def _latent_and_kkt(
        self, z: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Construct a smooth latent KKT problem without synthetic currents."""
        gamma_latent = self.gamma_field(z, sigma)
        x_plus_latent = 0.42 + 0.08 * torch.tanh(0.71 * z + 0.13 * sigma)
        x_minus_latent = 0.16 + 0.04 * torch.tanh(
            -0.53 * z + 0.09 * sigma
        )
        gamma_relative_latent = 1.08 + 0.04 * torch.tanh(
            0.37 * z - 0.11 * sigma
        )
        x_cross_latent = (
            torch.sqrt(
                (x_plus_latent * x_minus_latent).clamp_min(SAFE_EPS)
            )
            * gamma_relative_latent
        )
        problem = self.build_kkt_problem(
            z,
            gamma_latent,
            x_plus_latent,
            x_minus_latent,
            x_cross_latent,
            sigma,
        )
        kkt = self.solve_bounded_kkt(*problem, sigma)
        return (
            gamma_latent,
            x_plus_latent,
            x_minus_latent,
            x_cross_latent,
            kkt,
        )

    def _physical_currents(
        self,
        z: Tensor,
        sigma: Tensor,
        metric: Tensor,
        gamma_latent: Tensor,
        kkt: Any,
    ) -> dict[str, Tensor]:
        p = self.theory33_parameters
        log_density = kkt.x[:, 0]
        log_fraction = kkt.x[:, 1]
        baryon_rapidity = kkt.x[:, 2]
        carrier_rapidity = kkt.x[:, 3]
        density = p.density_reference * torch.exp(log_density)
        carrier_density = (
            density
            * p.carrier_fraction_reference
            * torch.exp(log_fraction)
        )
        entropy = (
            p.entropy_reference
            + p.entropy_span * torch.tanh(gamma_latent - 1.0)
        )

        spatial_metric = metric[:, 1:, 1:]
        shift_down = metric[:, 0, 1:]
        shift_up = torch.linalg.solve(
            spatial_metric, shift_down[..., None]
        ).squeeze(-1)
        lapse_squared = (
            -metric[:, 0, 0]
            + torch.einsum("bi,bi->b", shift_down, shift_up)
        )
        lapse = torch.sqrt(lapse_squared.clamp_min(SAFE_EPS))
        direction = torch.zeros(
            z.shape[0],
            3,
            dtype=z.dtype,
            device=z.device,
        )
        direction[:, 0] = 1.0
        direction_norm = torch.sqrt(
            torch.einsum(
                "bi,bij,bj->b",
                direction,
                spatial_metric,
                direction,
            ).clamp_min(SAFE_EPS)
        )
        direction = direction / direction_norm[:, None]

        def four_velocity(rapidity: Tensor) -> Tensor:
            lorentz = torch.cosh(rapidity)
            spatial = (
                torch.sinh(rapidity)[:, None] * direction
                - (lorentz / lapse)[:, None] * shift_up
            )
            return torch.cat(
                [(lorentz / lapse)[:, None], spatial], dim=-1
            )

        baryon_velocity = four_velocity(baryon_rapidity)
        carrier_velocity = four_velocity(carrier_rapidity)
        baryon_current = density[:, None] * baryon_velocity
        carrier_current = carrier_density[:, None] * carrier_velocity
        return {
            "density": density,
            "carrier_density": carrier_density,
            "entropy": entropy,
            "baryon_rapidity": baryon_rapidity,
            "carrier_rapidity": carrier_rapidity,
            "baryon_current": baryon_current,
            "carrier_current": carrier_current,
            "lapse": lapse,
        }

    def _primitive_and_kkt(self, z: Tensor, sigma: Tensor):
        gamma_latent, _, _, _, kkt = self._latent_and_kkt(z, sigma)
        metric = self.metric_field(z, sigma)
        physical = self._physical_currents(
            z, sigma, metric, gamma_latent, kkt
        )
        baryon_current = physical["baryon_current"]
        carrier_current = physical["carrier_current"]
        x_plus = self.invariant(metric, baryon_current, baryon_current)
        x_minus = self.invariant(metric, carrier_current, carrier_current)
        x_cross = self.invariant(metric, baryon_current, carrier_current)
        relative_gamma = x_cross / torch.sqrt(
            (x_plus * x_minus).clamp_min(SAFE_EPS)
        )
        return (
            baryon_current,
            carrier_current,
            metric,
            relative_gamma,
            x_plus,
            x_minus,
            x_cross,
            kkt,
        )

    def evaluate(self, z: Tensor, sigma: Tensor) -> Theory33TesseractState:
        gamma_latent, _, _, _, kkt = self._latent_and_kkt(z, sigma)
        metric = self.metric_field(z, sigma)
        physical = self._physical_currents(
            z, sigma, metric, gamma_latent, kkt
        )
        j_plus = physical["baryon_current"]
        j_minus = physical["carrier_current"]
        density = physical["density"]
        carrier_density = physical["carrier_density"]
        entropy = physical["entropy"]
        baryon_rapidity = physical["baryon_rapidity"]
        carrier_rapidity = physical["carrier_rapidity"]
        lapse = physical["lapse"]

        x_plus = self.invariant(metric, j_plus, j_plus)
        x_minus = self.invariant(metric, j_minus, j_minus)
        x_cross = self.invariant(metric, j_plus, j_minus)
        relative_gamma = x_cross / torch.sqrt(
            (x_plus * x_minus).clamp_min(SAFE_EPS)
        )
        relative_invariant = x_cross - torch.sqrt(
            (x_plus * x_minus).clamp_min(SAFE_EPS)
        )
        invariant_features = torch.stack(
            [x_plus, x_minus, x_cross, entropy], dim=-1
        )
        lambda_value = self.master(invariant_features)
        lambda_partials = torch.autograd.grad(
            lambda_value.sum(),
            invariant_features,
            create_graph=True,
        )[0]
        b_plus = -2.0 * lambda_partials[:, 0]
        b_minus = -2.0 * lambda_partials[:, 1]
        entrainment = -lambda_partials[:, 2]
        j_plus_cov = torch.einsum("bij,bj->bi", metric, j_plus)
        j_minus_cov = torch.einsum("bij,bj->bi", metric, j_minus)
        pi_plus = (
            b_plus[:, None] * j_plus_cov
            + entrainment[:, None] * j_minus_cov
        )
        pi_minus = (
            b_minus[:, None] * j_minus_cov
            + entrainment[:, None] * j_plus_cov
        )
        generalized_pressure = (
            lambda_value
            - torch.einsum("bi,bi->b", j_plus, pi_plus)
            - torch.einsum("bi,bi->b", j_minus, pi_minus)
        )
        identity = torch.eye(4, device=z.device, dtype=z.dtype)
        stress = (
            generalized_pressure[:, None, None] * identity[None, :, :]
            + torch.einsum("bi,bj->bij", j_plus, pi_plus)
            + torch.einsum("bi,bj->bij", j_minus, pi_minus)
        )

        inverse_metric = torch.linalg.inv(metric)
        stress_contravariant = torch.einsum(
            "bij,bjk->bik", stress, inverse_metric
        )
        normal_down = torch.zeros_like(j_plus)
        normal_down[:, 0] = -lapse
        energy_density = torch.einsum(
            "bi,bij,bj->b",
            normal_down,
            stress_contravariant,
            normal_down,
        )

        thermodynamics = self.master.thermodynamics(
            density, carrier_density, entropy
        )
        legendre_matrix = torch.stack(
            [
                torch.stack([b_plus, entrainment], dim=-1),
                torch.stack([entrainment, b_minus], dim=-1),
            ],
            dim=-2,
        )
        legendre_minimum = torch.linalg.eigvalsh(legendre_matrix)[:, 0]
        thermodynamic_matrix = torch.stack(
            [
                torch.stack(
                    [
                        thermodynamics["hessian_nn"],
                        thermodynamics["hessian_nd"],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        thermodynamics["hessian_nd"],
                        thermodynamics["hessian_dd"],
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        thermodynamic_minimum = torch.linalg.eigvalsh(
            thermodynamic_matrix
        )[:, 0]
        p = self.theory33_parameters
        legendre_margin = legendre_minimum - p.convexity_floor
        thermodynamic_margin = (
            thermodynamic_minimum - p.convexity_floor
        )
        maximum_sound_speed = torch.sqrt(
            torch.maximum(
                thermodynamics["baryon_sound_squared"],
                torch.full_like(
                    density, p.carrier_sound_speed**2
                ),
            ).clamp_min(0.0)
        )
        causal_screen_margin = 1.0 - maximum_sound_speed
        timelike_margin = torch.minimum(
            torch.minimum(x_plus, x_minus),
            torch.minimum(
                relative_gamma - 1.0,
                p.maximum_relative_lorentz_factor - relative_gamma,
            ),
        )
        metric_eigenvalues = torch.linalg.eigvalsh(metric)
        metric_signature_margin = torch.minimum(
            -metric_eigenvalues[:, 0], metric_eigenvalues[:, 1]
        )
        physical_valid = (
            (density > p.density_floor)
            & (carrier_density > p.density_floor)
            & (thermodynamics["temperature"] > 0.0)
            & (legendre_margin > 0.0)
            & (thermodynamic_margin > 0.0)
            & (causal_screen_margin >= -1.0e-8)
            & (timelike_margin >= -1.0e-10)
            & (metric_signature_margin > 0.0)
            & torch.isfinite(energy_density)
            & (energy_density > 0.0)
        )

        source_charge = (
            p.carrier_charge
            * carrier_density
            * torch.cosh(carrier_rapidity)
        )
        source_charge_scale = (
            abs(p.carrier_charge)
            * p.density_reference
            * p.carrier_fraction_reference
        )
        chemical_residual = (
            thermodynamics["anchor"]
            / (
                p.chemical_susceptibility
                * density.clamp_min(SAFE_EPS)
            )
        )
        drag_residual = (
            carrier_rapidity - baryon_rapidity
        ) / p.relaxation_time
        residual = -(chemical_residual + drag_residual)
        stress_delta = (
            stress - self.target_t[None, :, :]
        ) / p.density_reference
        stress_term = stress_delta.square().sum(dim=(-2, -1))
        free_energy = (
            energy_density / p.density_reference
            + 0.5 * self.omega_t * stress_term
            + self.omega_pi * (relative_gamma - 1.0)
            + thermodynamics["chemical_energy"] / p.density_reference
        )
        theta = torch.stack(
            [kkt.x[:, 1], kkt.x[:, 2], kkt.x[:, 3], torch.zeros_like(z)],
            dim=-1,
        )
        margins = torch.stack(
            [
                density - p.density_floor,
                carrier_density - p.density_floor,
                legendre_margin,
                thermodynamic_margin,
            ],
            dim=-1,
        )
        bits = torch.tensor([1, 2, 4, 8], device=z.device, dtype=torch.long)
        classifier_mask = (margins.ge(0).long() * bits).sum(dim=-1)
        return Theory33TesseractState(
            j_plus=j_plus,
            j_minus=j_minus,
            metric=metric,
            gamma=relative_gamma,
            x_plus=x_plus,
            x_minus=x_minus,
            x_cross=x_cross,
            q_limited=kkt.x[:, 0],
            theta=theta,
            lambda_value=lambda_value,
            pi_plus=pi_plus,
            pi_minus=pi_minus,
            stress=stress,
            energy=energy_density,
            j_limited=source_charge / source_charge_scale,
            residual=residual,
            free_energy=free_energy,
            kkt=kkt,
            classifier_mask=classifier_mask,
            classifier_margin=margins,
            number_density=density,
            carrier_density=carrier_density,
            entropy_per_baryon=entropy,
            baryon_rapidity=baryon_rapidity,
            carrier_rapidity=carrier_rapidity,
            relative_lorentz_factor=relative_gamma,
            relative_invariant=relative_invariant,
            generalized_pressure=generalized_pressure,
            b_baryon=b_plus,
            b_carrier=b_minus,
            entrainment=entrainment,
            chemical_baryon=thermodynamics["chemical_baryon"],
            chemical_carrier=thermodynamics["chemical_carrier"],
            temperature=thermodynamics["temperature"],
            energy_density=energy_density,
            legendre_margin=legendre_margin,
            thermodynamic_margin=thermodynamic_margin,
            causal_screen_margin=causal_screen_margin,
            timelike_margin=timelike_margin,
            metric_signature_margin=metric_signature_margin,
            physical_valid=physical_valid,
        )

    def _certificate(
        self, z: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Boundary certificates run under no_grad in the base class. Re-enable
        # local differentiation because Carter coefficients are derivatives of
        # the scalar master function, then detach the hard certificate.
        with torch.enable_grad():
            probe = z.detach().clone().requires_grad_(True)
            state = self.evaluate(probe, sigma.detach())
        return (
            state.kkt.active_mask.detach(),
            state.classifier_mask.detach(),
            state.classifier_margin.detach(),
        )

    def characteristic_audits(
        self, state: Theory33TesseractState
    ) -> list[CharacteristicAudit]:
        audits = []
        for index in range(state.number_density.shape[0]):
            audits.append(
                self.master.characteristic_audit(
                    float(state.number_density[index].detach()),
                    float(state.entropy_per_baryon[index].detach()),
                    float(state.baryon_rapidity[index].detach()),
                    float(state.carrier_density[index].detach()),
                    float(state.carrier_rapidity[index].detach()),
                )
            )
        return audits

    def step(
        self,
        z: Tensor,
        *,
        detach_diagnostics: bool = True,
        measure_local_gain: bool = False,
        audit_characteristics: bool = False,
    ) -> tuple[Tensor, Theory33StepDiagnostics]:
        result, diagnostics = super().step(
            z,
            detach_diagnostics=detach_diagnostics,
            measure_local_gain=measure_local_gain,
        )
        probe = z if z.requires_grad else z.requires_grad_()
        sigma = torch.where(
            probe.detach().ge(0),
            torch.ones_like(probe),
            -torch.ones_like(probe),
        )
        active = self.evaluate(probe, sigma)
        opposite = self.evaluate(probe, -sigma)

        metric_norm_plus = torch.einsum(
            "bi,bij,bj->b", active.j_plus, active.metric, active.j_plus
        )
        metric_norm_minus = torch.einsum(
            "bi,bij,bj->b", active.j_minus, active.metric, active.j_minus
        )
        normalization_error = torch.maximum(
            (
                metric_norm_plus
                + active.number_density.square()
            ).abs(),
            (
                metric_norm_minus
                + active.carrier_density.square()
            ).abs(),
        )
        opposite_norm_plus = torch.einsum(
            "bi,bij,bj->b",
            opposite.j_plus,
            opposite.metric,
            opposite.j_plus,
        )
        opposite_norm_minus = torch.einsum(
            "bi,bij,bj->b",
            opposite.j_minus,
            opposite.metric,
            opposite.j_minus,
        )
        normalization_error = torch.maximum(
            normalization_error,
            torch.maximum(
                (
                    opposite_norm_plus
                    + opposite.number_density.square()
                ).abs(),
                (
                    opposite_norm_minus
                    + opposite.carrier_density.square()
                ).abs(),
            ),
        )

        characteristic_valid = None
        maximum_characteristic_speed = None
        if audit_characteristics:
            audits = self.characteristic_audits(active)
            opposite_audits = self.characteristic_audits(opposite)
            characteristic_valid = torch.tensor(
                [
                    audit.causal and opposite_audit.causal
                    for audit, opposite_audit in zip(
                        audits, opposite_audits, strict=True
                    )
                ],
                dtype=torch.bool,
                device=z.device,
            )
            maximum_characteristic_speed = torch.tensor(
                [
                    max(
                        audit.maximum_absolute_speed,
                        opposite_audit.maximum_absolute_speed,
                    )
                    for audit, opposite_audit in zip(
                        audits, opposite_audits, strict=True
                    )
                ],
                dtype=z.dtype,
                device=z.device,
            )

        def maybe_detach(value: Tensor) -> Tensor:
            return value.detach() if detach_diagnostics else value

        return result, Theory33StepDiagnostics(
            **diagnostics.__dict__,
            physical_valid=(
                active.physical_valid & opposite.physical_valid
            ).detach(),
            legendre_margin=maybe_detach(
                torch.minimum(
                    active.legendre_margin, opposite.legendre_margin
                )
            ),
            thermodynamic_margin=maybe_detach(
                torch.minimum(
                    active.thermodynamic_margin,
                    opposite.thermodynamic_margin,
                )
            ),
            causal_screen_margin=maybe_detach(
                torch.minimum(
                    active.causal_screen_margin,
                    opposite.causal_screen_margin,
                )
            ),
            timelike_margin=maybe_detach(
                torch.minimum(
                    active.timelike_margin, opposite.timelike_margin
                )
            ),
            metric_signature_margin=maybe_detach(
                torch.minimum(
                    active.metric_signature_margin,
                    opposite.metric_signature_margin,
                )
            ),
            maximum_relative_lorentz_factor=maybe_detach(
                torch.maximum(
                    active.relative_lorentz_factor,
                    opposite.relative_lorentz_factor,
                )
            ),
            current_normalization_error=maybe_detach(
                normalization_error
            ),
            characteristic_valid=(
                None
                if characteristic_valid is None
                else characteristic_valid.detach()
            ),
            maximum_characteristic_speed=(
                None
                if maximum_characteristic_speed is None
                else maximum_characteristic_speed.detach()
            ),
        )
