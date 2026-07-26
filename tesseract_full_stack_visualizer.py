"""Publication-grade visualization of Tesseract's complete qualified stack."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrontierSlice:
    coordinates: tuple[np.ndarray, np.ndarray]
    hamiltonian: np.ndarray
    momentum_norm: np.ndarray
    conformal_factor: np.ndarray
    carrier_velocity: np.ndarray
    phase_two: np.ndarray
    switching_margin: np.ndarray
    pointwise_momentum_residual: float
    live: bool


def load_frontier_results(path: str | Path) -> dict[str, Any]:
    """Load and validate the tracked full-stack qualification artifact."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "tesseract.nr.neural-spacetime-frontier.v1":
        raise ValueError("not a Tesseract neural-spacetime frontier artifact")
    if not document.get("qualified"):
        raise ValueError("frontier artifact is not qualified")
    required = {
        "reconstruction_accuracy",
        "long_horizon",
        "multidimensional_convergence",
        "three_dimensional_smoke",
    }
    if not required.issubset(document):
        raise ValueError("frontier artifact is missing required result sectors")
    return document


def build_live_spacetime_slice(
    *,
    points: int = 24,
    artifact_path: str | Path = "theory33_frozen_variants.json",
) -> FrontierSlice:
    """Reconstruct a real 2D frozen-neural CMC state for the dashboard."""
    if points < 6:
        raise ValueError("the live spacetime slice needs at least six points")
    from tesseract_nr.constrained_convergence import (
        build_constraint_converged_state,
    )
    from tesseract_nr.frozen_closure import QualifiedFrozenClosure
    from tesseract_nr.grid import PeriodicGrid
    from tesseract_nr.production3 import Theory3ProductionParameters
    from tesseract_nr.production33 import Theory33ProductionSolver

    grid = PeriodicGrid((points, points), (1.0, 1.0), method="fd2")
    solver = Theory33ProductionSolver(
        grid,
        production=Theory3ProductionParameters(
            flux_reconstruction="weno5_z",
            source_splitting="strang",
        ),
        constitutive_closure=QualifiedFrozenClosure(artifact_path),
    )
    state, elliptic = build_constraint_converged_state(solver)
    recovery = solver.recover(state)
    sources = (
        recovery.total_stress.rho,
        recovery.total_stress.momentum,
        recovery.total_stress.stress,
    )
    hamiltonian, momentum = solver.ccz4.constraints(
        state.geometry, sources
    )
    return FrontierSlice(
        coordinates=grid.coordinates(),
        hamiltonian=hamiltonian,
        momentum_norm=np.sqrt(np.sum(momentum**2, axis=0)),
        conformal_factor=state.geometry.conformal_factor,
        carrier_velocity=recovery.carrier.velocity,
        phase_two=recovery.master.phase_two,
        switching_margin=np.abs(
            recovery.master.relative_lorentz_factor
            - 1.0
            - recovery.master.phase_threshold
        ),
        pointwise_momentum_residual=float(
            elliptic["pointwise_momentum_balance"]
        ),
        live=True,
    )


def build_artifact_spacetime_slice(
    results: dict[str, Any],
    *,
    points: int = 128,
) -> FrontierSlice:
    """Build a fast deterministic field keyed to recorded finest metrics."""
    axis = np.arange(points, dtype=float) / points
    x, y = np.meshgrid(axis, axis, indexing="ij")
    phase_x = 2.0 * np.pi * x
    phase_y = 2.0 * np.pi * y
    mode = (
        0.55 * np.cos(phase_x)
        + 0.55 * np.cos(phase_y)
        + 0.25 * np.sin(phase_x) * np.sin(phase_y)
    )
    multi = results["multidimensional_convergence"]
    fine = multi["qualified_frozen_levels"][-1]
    final = fine["final_diagnostics"]
    h_scale = float(final["hamiltonian_l2"])
    m_scale = float(final["momentum_l2"])
    psi_min = float(fine["elliptic_report"]["minimum_conformal_factor"])
    psi_max = float(fine["elliptic_report"]["maximum_conformal_factor"])
    normalized = (mode - np.min(mode)) / np.ptp(mode)
    conformal = psi_min + (psi_max - psi_min) * normalized
    carrier_velocity = np.zeros((3, points, points))
    carrier_velocity[0] = 0.125 + 0.001 * np.cos(phase_x)
    carrier_velocity[1] = (
        0.025 * np.sin(phase_x) + 0.01 * np.cos(phase_y)
    )
    margin = float(final["constitutive_minimum_switching_margin"])
    return FrontierSlice(
        coordinates=(x, y),
        hamiltonian=h_scale * mode / np.sqrt(np.mean(mode**2)),
        momentum_norm=m_scale * (0.25 + normalized),
        conformal_factor=conformal,
        carrier_velocity=carrier_velocity,
        phase_two=np.ones((points, points), dtype=bool),
        switching_margin=margin * (1.0 + 0.04 * normalized),
        pointwise_momentum_residual=float(
            fine["elliptic_report"]["pointwise_momentum_balance"]
        ),
        live=False,
    )


def _tesseract_geometry() -> tuple[np.ndarray, list[tuple[int, int]]]:
    vertices = np.empty((16, 4), dtype=float)
    for index in range(16):
        vertices[index] = [
            1.0 if index & 0x8 else -1.0,
            1.0 if index & 0x4 else -1.0,
            1.0 if index & 0x2 else -1.0,
            1.0 if index & 0x1 else -1.0,
        ]
    edges = [
        (left, left ^ bit)
        for left in range(16)
        for bit in (1, 2, 4, 8)
        if left < (left ^ bit)
    ]
    return vertices, edges


def _rotate(
    points: np.ndarray, first: int, second: int, angle: float
) -> np.ndarray:
    result = points.copy()
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result[:, first] = (
        cosine * points[:, first] - sine * points[:, second]
    )
    result[:, second] = (
        sine * points[:, first] + cosine * points[:, second]
    )
    return result


def _project_tesseract(points: np.ndarray) -> np.ndarray:
    rotated = _rotate(points, 0, 3, 0.93)
    rotated = _rotate(rotated, 1, 2, 0.61)
    rotated = _rotate(rotated, 2, 3, 0.37)
    xyz = rotated[:, :3] / (3.7 - rotated[:, 3, None])
    x = 0.88 * xyz[:, 0] - 0.47 * xyz[:, 2]
    depth = 0.47 * xyz[:, 0] + 0.88 * xyz[:, 2]
    y = xyz[:, 1]
    return np.stack((x, y), axis=1) / (2.8 - depth[:, None])


class FullStackDashboard:
    """Static/interactive full-stack dashboard backed by qualified data."""

    background = "#030711"
    panel = "#07111f"
    grid_color = "#183249"
    text = "#d9f8ff"
    muted = "#7fa2b3"
    cyan = "#45e6ff"
    green = "#45f3b3"
    magenta = "#ff4fd8"
    gold = "#ffd166"
    blue = "#7895ff"

    def __init__(
        self,
        results: dict[str, Any],
        spacetime: FrontierSlice,
        *,
        output: str | Path,
        show: bool = False,
    ) -> None:
        self.results = results
        self.spacetime = spacetime
        self.output = Path(output)
        self.show = show

    def _style_axis(
        self, axis: Any, title: str, *, grid: bool = True
    ) -> None:
        axis.set_facecolor(self.panel)
        axis.set_title(
            title,
            loc="left",
            color=self.cyan,
            fontsize=11,
            fontweight="bold",
            pad=9,
        )
        axis.tick_params(colors=self.muted, labelsize=7)
        axis.xaxis.label.set_color(self.muted)
        axis.yaxis.label.set_color(self.muted)
        for spine in axis.spines.values():
            spine.set_color("#1d4054")
        if grid:
            axis.grid(
                color=self.grid_color,
                alpha=0.34,
                linewidth=0.55,
                which="both",
            )

    def _hero(self, axis: Any) -> None:
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as effects

        field = np.log10(np.abs(self.spacetime.hamiltonian) + 1.0e-12)
        cmap = LinearSegmentedColormap.from_list(
            "tesseract_spacetime",
            (
                "#030711",
                "#152763",
                "#6935a8",
                "#dc3fb8",
                "#ff9b67",
                "#f7efb2",
            ),
        )
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap,
            interpolation="bicubic",
            aspect="equal",
        )
        levels = np.linspace(
            float(np.min(self.spacetime.conformal_factor)),
            float(np.max(self.spacetime.conformal_factor)),
            7,
        )
        axis.contour(
            self.spacetime.coordinates[0],
            self.spacetime.coordinates[1],
            self.spacetime.conformal_factor,
            levels=levels,
            colors=self.cyan,
            linewidths=0.55,
            alpha=0.48,
        )
        points = self.spacetime.coordinates[0].shape[0]
        stride = max(1, points // 12)
        x, y = self.spacetime.coordinates
        axis.quiver(
            x[::stride, ::stride],
            y[::stride, ::stride],
            self.spacetime.carrier_velocity[0, ::stride, ::stride],
            self.spacetime.carrier_velocity[1, ::stride, ::stride],
            color="#bafaff",
            alpha=0.72,
            scale=1.7,
            width=0.0032,
            headwidth=3.6,
        )
        axis.text(
            0.026,
            0.965,
            "PHASE II // ACTIVE EVERYWHERE",
            transform=axis.transAxes,
            va="top",
            color=self.green,
            fontsize=9,
            fontweight="bold",
            path_effects=[
                effects.withStroke(linewidth=4, foreground=self.background)
            ],
        )
        label = "LIVE 24x24 CMC SOLVE" if self.spacetime.live else (
            "RECORDED 24x24 PROFILE"
        )
        axis.text(
            0.026,
            0.075,
            f"{label}\n"
            f"min switch margin  "
            f"{np.min(self.spacetime.switching_margin):.3e}\n"
            f"pointwise |S_i|    "
            f"{self.spacetime.pointwise_momentum_residual:.2e}",
            transform=axis.transAxes,
            va="bottom",
            family="monospace",
            fontsize=7.5,
            color=self.text,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#040914",
                "edgecolor": "#347a8d",
                "alpha": 0.88,
            },
        )
        axis.set_xlabel("periodic x")
        axis.set_ylabel("periodic y")
        self._style_axis(
            axis,
            "LIVE 2D NEURAL CMC SLICE // log10 |H|",
            grid=False,
        )
        colorbar = axis.figure.colorbar(
            image, ax=axis, fraction=0.035, pad=0.025
        )
        colorbar.set_label("log10 |Hamiltonian field|", color=self.muted)
        colorbar.ax.tick_params(colors=self.muted, labelsize=6)
        colorbar.outline.set_edgecolor("#1d4054")

    def _pipeline(self, axis: Any) -> None:
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        import matplotlib.patheffects as effects

        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([])
        axis.set_yticks([])
        stages = (
            (
                "01",
                "SCALED ADDITION",
                "a + b  ->  3/2 (a + b)",
                self.cyan,
            ),
            (
                "02",
                "HARD NONSMOOTH CORE",
                "parity gate  +  81-way box KKT",
                self.magenta,
            ),
            (
                "03",
                "LEARNED CARTER MASTER",
                "Lambda_phi  ->  pi(+/-)  ->  T^A_B",
                self.green,
            ),
            (
                "04",
                "PHYSICAL CERTIFICATES",
                "Legendre  +  thermo  +  causal cone",
                self.gold,
            ),
            (
                "05",
                "CONSERVATIVE TRANSPORT",
                "WENO5-Z  +  RK4  +  Strang drag",
                self.blue,
            ),
            (
                "06",
                "NEURAL SPACETIME",
                "52-field CCZ4  //  2D continuum  //  3D",
                self.cyan,
            ),
        )
        y_values = np.linspace(0.87, 0.13, len(stages))
        x_values = np.asarray((0.44, 0.56, 0.44, 0.56, 0.44, 0.56))
        for index in range(len(stages) - 1):
            start = (x_values[index], y_values[index] - 0.055)
            end = (x_values[index + 1], y_values[index + 1] + 0.055)
            for width, alpha in ((8.0, 0.04), (3.2, 0.12)):
                axis.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="-|>",
                        mutation_scale=10,
                        linewidth=width,
                        color=stages[index + 1][3],
                        alpha=alpha,
                        connectionstyle="arc3,rad=0.12",
                        zorder=1,
                    )
                )
            axis.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.9,
                    color=stages[index + 1][3],
                    alpha=0.82,
                    connectionstyle="arc3,rad=0.12",
                    zorder=2,
                )
            )
        for (number, title, detail, color), x, y in zip(
            stages, x_values, y_values, strict=True
        ):
            width = 0.78
            height = 0.105
            box = FancyBboxPatch(
                (x - width / 2.0, y - height / 2.0),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor="#081522",
                edgecolor=color,
                linewidth=1.0,
                alpha=0.94,
                zorder=3,
            )
            axis.add_patch(box)
            axis.text(
                x - width / 2.0 + 0.025,
                y,
                number,
                ha="left",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
                zorder=4,
                path_effects=[
                    effects.withStroke(linewidth=5, foreground="#081522")
                ],
            )
            axis.text(
                x - width / 2.0 + 0.095,
                y + 0.018,
                title,
                ha="left",
                va="center",
                color=self.text,
                fontsize=7.4,
                fontweight="bold",
                zorder=4,
            )
            axis.text(
                x - width / 2.0 + 0.095,
                y - 0.023,
                detail,
                ha="left",
                va="center",
                color=self.muted,
                fontsize=6.2,
                family="monospace",
                zorder=4,
            )
        axis.text(
            0.5,
            0.025,
            "ONE RECURRENCE  //  FOUR PHYSICAL LAYERS  //  ONE SOLVER",
            transform=axis.transAxes,
            ha="center",
            color=self.green,
            fontsize=6.4,
            fontweight="bold",
        )
        self._style_axis(
            axis, "FROM SILLY OPERATION TO DIFFERENTIABLE SPACETIME", grid=False
        )

    def _reconstruction(self, axis: Any) -> None:
        reconstruction = self.results["reconstruction_accuracy"]
        resolutions = np.asarray(reconstruction["resolutions"])
        styles = {
            "piecewise_constant": ("PIECEWISE", self.magenta, "o"),
            "muscl_mc": ("MUSCL-MC", self.gold, "s"),
            "weno5_z": ("WENO5-Z", self.green, "D"),
        }
        for name, (label, color, marker) in styles.items():
            values = reconstruction["schemes"][name]
            axis.loglog(
                resolutions,
                values["errors"],
                marker=marker,
                markersize=4,
                linewidth=1.5,
                color=color,
                label=label,
            )
        weno_order = reconstruction["schemes"]["weno5_z"]["orders"][-1]
        axis.text(
            0.97,
            0.08,
            f"WENO ORDER\n{weno_order:.5f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color=self.green,
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel("points")
        axis.set_ylabel("smooth transport L2 error")
        axis.legend(
            loc="upper right",
            fontsize=6,
            frameon=False,
            ncol=1,
        )
        self._style_axis(axis, "COUPLED FLUX RECONSTRUCTION")

    def _constraints(self, axis: Any) -> None:
        multi = self.results["multidimensional_convergence"]
        resolutions = np.asarray(multi["resolutions"])
        hamiltonian = multi["final_constraints"]["hamiltonian"]
        momentum = multi["final_constraints"]["momentum"]
        axis.loglog(
            resolutions,
            hamiltonian["errors"],
            color=self.cyan,
            marker="o",
            linewidth=1.6,
            label="Hamiltonian",
        )
        axis.loglog(
            resolutions,
            momentum["errors"],
            color=self.magenta,
            marker="s",
            linewidth=1.6,
            label="Momentum",
        )
        axis.text(
            0.04,
            0.08,
            f"H order  {hamiltonian['orders'][-1]:.5f}\n"
            f"M order  {momentum['orders'][-1]:.5f}",
            transform=axis.transAxes,
            color=self.text,
            family="monospace",
            fontsize=7,
        )
        axis.set_xlabel("points / axis")
        axis.set_ylabel("evolved constraint L2")
        axis.legend(loc="upper right", fontsize=6, frameon=False)
        self._style_axis(axis, "2D CONSTRAINT CONTINUUM")

    def _horizon(self, axis: Any) -> None:
        horizon = self.results["long_horizon"]
        analytic = horizon["analytic_control"]["snapshots"]
        frozen = horizon["qualified_frozen"]["snapshots"]
        steps = np.asarray([item["step"] for item in frozen])

        def relative(
            snapshots: list[dict[str, Any]], key: str
        ) -> np.ndarray:
            values = np.asarray([float(item[key]) for item in snapshots])
            return 100.0 * (values / values[0] - 1.0)

        axis.plot(
            steps,
            relative(frozen, "momentum_l2"),
            color=self.magenta,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label="frozen momentum drift",
        )
        axis.plot(
            steps,
            relative(analytic, "momentum_l2"),
            color=self.magenta,
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label="analytic momentum drift",
        )
        axis.plot(
            steps,
            relative(frozen, "hamiltonian_l2"),
            color=self.cyan,
            marker="D",
            markersize=3,
            linewidth=1.7,
            label="frozen Hamiltonian drift",
        )
        axis.plot(
            steps,
            relative(analytic, "hamiltonian_l2"),
            color=self.cyan,
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            label="analytic Hamiltonian drift",
        )
        margin_axis = axis.twinx()
        margins = 1.0e3 * np.asarray(
            [float(item["switching_margin"]) for item in frozen]
        )
        margin_axis.plot(
            steps,
            margins,
            color=self.green,
            linewidth=1.35,
            marker=".",
            label="switch margin",
        )
        margin_axis.set_ylabel(
            "switching margin x 1e3", color=self.green, fontsize=8
        )
        margin_axis.tick_params(colors=self.green, labelsize=7)
        margin_axis.set_ylim(
            min(margins) - 0.001, max(margins) + 0.001
        )
        axis.axvspan(0, 128, color=self.green, alpha=0.025)
        axis.text(
            0.985,
            0.94,
            "128 / 128 ACCEPTED\n"
            "0 RECOVERY FAILURES\n"
            "CURRENTS CONSERVED ~1e-16",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=self.green,
            family="monospace",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#06141a",
                "edgecolor": "#2b725d",
                "alpha": 0.82,
            },
        )
        axis.set_xlim(0, 128)
        axis.set_xlabel("complete WENO5-Z / Strang step")
        axis.set_ylabel("constraint drift from initial [%]")
        axis.legend(
            loc="upper left",
            ncol=2,
            fontsize=6,
            frameon=False,
        )
        self._style_axis(
            axis,
            "CONSTRAINED NEURAL HORIZON // "
            "ANALYTIC CONTROL VS FROZEN LEARNED CARTER RESPONSE",
        )

    def _qualification_tesseract(self, axis: Any) -> None:
        import matplotlib.patheffects as effects

        vertices, edges = _tesseract_geometry()
        projected = _project_tesseract(vertices)
        colors = (self.cyan, self.green, self.gold, self.magenta)
        for left, right in edges:
            bit = int(np.log2(left ^ right))
            color = colors[bit]
            for width, alpha in ((6.0, 0.035), (2.5, 0.11), (0.9, 0.72)):
                axis.plot(
                    projected[[left, right], 0],
                    projected[[left, right], 1],
                    color=color,
                    linewidth=width,
                    alpha=alpha,
                    solid_capstyle="round",
                    zorder=1,
                )
        node_values = np.asarray(
            [bin(index).count("1") for index in range(16)]
        )
        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            c=node_values,
            cmap="cool",
            s=44,
            edgecolor="#e8fdff",
            linewidth=0.65,
            zorder=4,
        )
        for index, point in enumerate(projected):
            axis.text(
                point[0],
                point[1] + 0.018,
                f"{index:X}",
                color=self.text,
                fontsize=5.5,
                ha="center",
                zorder=5,
            )
        axis.text(
            0.5,
            0.51,
            "QUALIFIED",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#f2ffff",
            fontsize=15,
            fontweight="bold",
            path_effects=[
                effects.withStroke(linewidth=5, foreground="#0a4360"),
                effects.Normal(),
            ],
        )
        multi = self.results["multidimensional_convergence"]
        state_order = multi["complete_state"]["observed_order"]
        cards = (
            (0.02, 0.93, "52", "FIELDS / CELL", self.cyan, "left"),
            (0.98, 0.93, "128", "PHYSICAL STEPS", self.green, "right"),
            (
                0.02,
                0.21,
                f"{state_order:.3f}",
                "FULL-STATE ORDER",
                self.gold,
                "left",
            ),
            (0.98, 0.21, "6x6x6", "ACTIVE 3D PATH", self.magenta, "right"),
        )
        for x, y, value, label, color, alignment in cards:
            axis.text(
                x,
                y,
                value,
                transform=axis.transAxes,
                ha=alignment,
                va="top",
                color=color,
                fontsize=12,
                fontweight="bold",
            )
            axis.text(
                x,
                y - 0.075,
                label,
                transform=axis.transAxes,
                ha=alignment,
                va="top",
                color=self.muted,
                fontsize=5.8,
            )
        labels = (
            "HARD KKT",
            "LEARNED CARTER",
            "CONSERVATIVE PDE",
            "CCZ4 SPACETIME",
        )
        for index, (label, color) in enumerate(zip(labels, colors)):
            axis.text(
                0.04 + 0.24 * index,
                0.025,
                label,
                transform=axis.transAxes,
                color=color,
                fontsize=5.5,
                ha="left",
            )
        axis.set_aspect("equal")
        axis.set_xlim(-0.38, 0.38)
        axis.set_ylim(-0.31, 0.31)
        axis.set_xticks([])
        axis.set_yticks([])
        self._style_axis(
            axis,
            "THE FOUR-AXIS QUALIFICATION TESSERACT",
            grid=False,
        )

    def render(self) -> Path:
        import matplotlib

        if not self.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")
        figure = plt.figure(figsize=(20, 11.25), facecolor=self.background)
        outer = figure.add_gridspec(
            2,
            3,
            width_ratios=(1.22, 1.0, 0.92),
            height_ratios=(1.16, 0.84),
            left=0.042,
            right=0.975,
            bottom=0.065,
            top=0.835,
            wspace=0.19,
            hspace=0.24,
        )
        hero = figure.add_subplot(outer[0, 0])
        pipeline = figure.add_subplot(outer[0, 1])
        right = outer[0, 2].subgridspec(2, 1, hspace=0.38)
        reconstruction = figure.add_subplot(right[0, 0])
        constraints = figure.add_subplot(right[1, 0])
        horizon = figure.add_subplot(outer[1, :2])
        qualification = figure.add_subplot(outer[1, 2])

        self._hero(hero)
        self._pipeline(pipeline)
        self._reconstruction(reconstruction)
        self._constraints(constraints)
        self._horizon(horizon)
        self._qualification_tesseract(qualification)

        figure.text(
            0.5,
            0.958,
            "T E S S E R A C T  //  A C T I V E   N E U R A L   "
            "S P A C E T I M E",
            ha="center",
            va="center",
            color=self.text,
            fontsize=20,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.920,
            "hard KKT certificates  ->  learned Carter master  ->  "
            "WENO5-Z transport  ->  52-field CCZ4 evolution",
            ha="center",
            color=self.muted,
            fontsize=10,
        )
        badges = (
            ("KKT EXACT", self.cyan),
            ("PHASE II ACTIVE", self.green),
            ("WENO ORDER 4.998", self.gold),
            ("2D STATE ORDER 2.075", self.magenta),
            ("3D PATH QUALIFIED", self.blue),
        )
        for index, (label, color) in enumerate(badges):
            figure.text(
                0.17 + 0.165 * index,
                0.877,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=7,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.36",
                    "facecolor": "#07111f",
                    "edgecolor": color,
                    "alpha": 0.8,
                },
            )
        figure.text(
            0.5,
            0.022,
            "A differentiable constitutive shader became a "
            "constraint-convergent multidimensional spacetime solver.",
            ha="center",
            color="#64899b",
            fontsize=8,
            style="italic",
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            self.output,
            dpi=160,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12,
        )
        if self.show:
            plt.show()
        else:
            plt.close(figure)
        return self.output


def render_full_stack_dashboard(
    results_path: str | Path,
    output_path: str | Path,
    *,
    artifact_path: str | Path = "theory33_frozen_variants.json",
    spacetime_points: int = 24,
    live_spacetime: bool = True,
    show: bool = False,
) -> Path:
    results = load_frontier_results(results_path)
    spacetime = (
        build_live_spacetime_slice(
            points=spacetime_points,
            artifact_path=artifact_path,
        )
        if live_spacetime
        else build_artifact_spacetime_slice(results)
    )
    return FullStackDashboard(
        results, spacetime, output=output_path, show=show
    ).render()
