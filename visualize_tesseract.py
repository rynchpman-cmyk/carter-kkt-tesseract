#!/usr/bin/env python3
"""Render Tesseract's full stack or interactive Vulkan literal recurrence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

spy: Any | None = None


HERE = Path(__file__).resolve().parent
TAPE_WIDTH = 32
ATTRACTOR = np.float32(0.1019493853295916)
INITIAL_LANES = np.asarray([-0.8, -0.2, 0.2, 0.8], dtype=np.float32)


@dataclass
class Atlas:
    name: str
    initial: np.ndarray
    states: np.ndarray
    active: np.ndarray
    classifier: np.ndarray
    valid: np.ndarray
    curve_z: np.ndarray
    curve_next: np.ndarray
    lanes: np.ndarray
    lane_valid: np.ndarray
    tape: np.ndarray
    diagnostics: np.ndarray


class ShaderModel:
    def __init__(
        self,
        device: Any,
        *,
        name: str,
        module_path: Path,
        model_path: Path,
    ) -> None:
        document = json.loads(model_path.read_text(encoding="utf-8"))
        weights = np.asarray(document["master_weights"], dtype=np.float32)
        self.name = name
        self.device = device
        self.module = spy.Module.load_from_file(device, str(module_path))
        self.weights = spy.Tensor.from_numpy(device, weights)

    def step(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float32)
        z_tensor = spy.Tensor.from_numpy(self.device, values)
        result = spy.Tensor.empty(
            self.device, shape=(values.size,), dtype=self.module.float4
        )
        self.module.fullPhysicalStep(
            z=z_tensor, weights=self.weights, _result=result
        )
        return result.to_numpy()

    def tape(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float32)
        z_tensor = spy.Tensor.from_numpy(self.device, values)
        tape = spy.Tensor.empty(
            self.device,
            shape=(values.size, TAPE_WIDTH),
            dtype=self.module.float,
        )
        self.module.fullPhysicalTapeKernel(
            sample=spy.grid(shape=(values.size,)),
            zValues=z_tensor,
            weights=self.weights,
            hardTape=tape,
        )
        return tape.to_numpy()

    def diagnostics(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float32)
        z_tensor = spy.Tensor.from_numpy(self.device, values)
        output = spy.Tensor.empty(
            self.device,
            shape=(values.size, 16, 4),
            dtype=self.module.float,
        )
        self.module.carterDiagnosticKernel(
            sample=spy.grid(shape=(values.size,)),
            zValues=z_tensor,
            weights=self.weights,
            diagnostics=output,
        )
        return output.to_numpy()

    def rollout(
        self,
        initial: np.ndarray,
        steps: int,
        *,
        state_limit: float = 1.0e4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial = np.asarray(initial, dtype=np.float32)
        count = initial.size
        states = np.full((steps + 1, count), np.nan, dtype=np.float32)
        active = np.full((steps, count), -1, dtype=np.int32)
        classifier = np.full((steps, count), -1, dtype=np.int32)
        valid = np.zeros((steps + 1, count), dtype=bool)
        states[0] = initial
        valid[0] = np.isfinite(initial)
        z = initial.copy()
        alive = valid[0].copy()

        for step in range(steps):
            safe_z = np.where(alive, z, np.float32(0.0))
            result = self.step(safe_z)
            next_z = result[:, 0]
            certificate = result[:, 3] == 1.0
            finite = np.isfinite(result).all(axis=1)
            bounded = np.abs(next_z) <= state_limit
            step_valid = alive & certificate & finite & bounded
            active[step, step_valid] = result[step_valid, 1].astype(
                np.int32
            )
            classifier[step, step_valid] = result[
                step_valid, 2
            ].astype(np.int32)
            states[step + 1, step_valid] = next_z[step_valid]
            valid[step + 1] = step_valid
            z = np.where(step_valid, next_z, np.float32(0.0))
            alive = step_valid
        return states, active, classifier, valid

    def build_atlas(
        self,
        *,
        samples: int,
        steps: int,
        domain: tuple[float, float],
    ) -> Atlas:
        initial = np.linspace(
            domain[0], domain[1], samples, dtype=np.float32
        )
        states, active, classifier, valid = self.rollout(initial, steps)
        curve_z = np.linspace(
            domain[0], domain[1], samples * 2, dtype=np.float32
        )
        curve_result = self.step(curve_z)
        curve_next = curve_result[:, 0]
        curve_next[curve_result[:, 3] != 1.0] = np.nan

        lanes, _active, _classifier, lane_valid = self.rollout(
            INITIAL_LANES, steps
        )
        # Tapes/diagnostics are evaluated for the +0.8 lane at every valid
        # state; invalid baseline tail entries use zero only for safe dispatch
        # and are masked by lane_valid in the renderer.
        display_lane = lanes[:, 3]
        safe_lane = np.where(
            lane_valid[:, 3], display_lane, np.float32(0.0)
        )
        tape = self.tape(safe_lane)
        diagnostics = self.diagnostics(safe_lane)
        tape[~lane_valid[:, 3]] = np.nan
        diagnostics[~lane_valid[:, 3]] = np.nan
        return Atlas(
            name=self.name,
            initial=initial,
            states=states,
            active=active,
            classifier=classifier,
            valid=valid,
            curve_z=curve_z,
            curve_next=curve_next,
            lanes=lanes,
            lane_valid=lane_valid,
            tape=tape,
            diagnostics=diagnostics,
        )


def tesseract_geometry() -> tuple[np.ndarray, list[tuple[int, int]]]:
    vertices = np.empty((16, 4), dtype=np.float64)
    for index in range(16):
        vertices[index] = [
            1.0 if index & 0x8 else -1.0,
            1.0 if index & 0x4 else -1.0,
            1.0 if index & 0x2 else -1.0,
            1.0 if index & 0x1 else -1.0,
        ]
    edges = []
    for left in range(16):
        for bit in (1, 2, 4, 8):
            right = left ^ bit
            if left < right:
                edges.append((left, right))
    return vertices, edges


def rotate_plane(
    points: np.ndarray, first: int, second: int, angle: float
) -> np.ndarray:
    result = points.copy()
    cosine = np.cos(angle)
    sine = np.sin(angle)
    a = points[:, first]
    b = points[:, second]
    result[:, first] = cosine * a - sine * b
    result[:, second] = sine * a + cosine * b
    return result


def project_tesseract(vertices: np.ndarray, phase: float) -> np.ndarray:
    rotated = rotate_plane(vertices, 0, 3, 0.73 * phase)
    rotated = rotate_plane(rotated, 1, 2, 0.47 * phase)
    rotated = rotate_plane(rotated, 2, 3, 0.31 * phase)
    perspective4 = 3.4 - rotated[:, 3]
    xyz = rotated[:, :3] / perspective4[:, None]
    cosine = np.cos(0.35 * phase)
    sine = np.sin(0.35 * phase)
    x = cosine * xyz[:, 0] - sine * xyz[:, 2]
    depth = sine * xyz[:, 0] + cosine * xyz[:, 2]
    y = xyz[:, 1]
    perspective3 = 2.7 - depth
    return 2.0 * np.stack([x / perspective3, y / perspective3], axis=-1)


def cobweb_points(trajectory: np.ndarray, step: int) -> np.ndarray:
    finite = trajectory[: step + 1]
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return np.empty((0, 2))
    points = [(float(finite[0]), float(finite[0]))]
    for index in range(finite.size - 1):
        current = float(finite[index])
        following = float(finite[index + 1])
        points.append((current, following))
        points.append((following, following))
    return np.asarray(points)


class Dashboard:
    def __init__(
        self,
        atlases: dict[str, Atlas],
        *,
        steps: int,
        output: Path,
        show: bool,
    ) -> None:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.widgets import Button, RadioButtons, Slider

        plt.style.use("dark_background")
        self.plt = plt
        self.atlases = atlases
        self.steps = steps
        self.output = output
        self.show = show
        self.name = "Conditioned"
        self.step = min(16, steps)
        self.playing = False
        self.vertices, self.edges = tesseract_geometry()

        self.figure = plt.figure(figsize=(16, 10), facecolor="#050812")
        grid = self.figure.add_gridspec(
            2, 2, left=0.055, right=0.96, bottom=0.13, top=0.91,
            hspace=0.23, wspace=0.18
        )
        self.ax_basin = self.figure.add_subplot(grid[0, 0])
        self.ax_cobweb = self.figure.add_subplot(grid[0, 1])
        self.ax_kkt = self.figure.add_subplot(grid[1, 0])
        self.ax_tesseract = self.figure.add_subplot(grid[1, 1])

        self.figure.suptitle(
            "CARTER / KKT TESSERACT — LITERAL DYNAMICS INSTRUMENT",
            color="#b7f7ff",
            fontsize=17,
            fontweight="bold",
        )
        self.figure.text(
            0.5,
            0.935,
            "Vulkan full-physics recurrence • active-set certificates • "
            "Lyapunov conditioning",
            ha="center",
            color="#6f9aaa",
            fontsize=10,
        )

        radio_axis = self.figure.add_axes(
            [0.055, 0.025, 0.12, 0.07], facecolor="#09111d"
        )
        self.radio = RadioButtons(
            radio_axis, ("Conditioned", "Baseline"), active=0
        )
        self.radio.on_clicked(self._select_model)
        slider_axis = self.figure.add_axes(
            [0.23, 0.052, 0.48, 0.025], facecolor="#09111d"
        )
        self.slider = Slider(
            slider_axis,
            "step",
            0,
            steps,
            valinit=self.step,
            valstep=1,
            color="#39e6b4",
        )
        self.slider.on_changed(self._select_step)
        play_axis = self.figure.add_axes(
            [0.75, 0.035, 0.08, 0.05], facecolor="#09111d"
        )
        self.play_button = Button(
            play_axis, "PLAY", color="#0b2630", hovercolor="#124454"
        )
        self.play_button.on_clicked(self._toggle_play)
        save_axis = self.figure.add_axes(
            [0.85, 0.035, 0.08, 0.05], facecolor="#09111d"
        )
        self.save_button = Button(
            save_axis, "SNAPSHOT", color="#25162f", hovercolor="#48255d"
        )
        self.save_button.on_clicked(self._save)

        state_cmap = plt.colormaps["coolwarm"].copy()
        state_cmap.set_bad("#020306")
        phase_cmap = plt.colormaps["turbo"].copy()
        phase_cmap.set_bad("#020306")
        self.state_cmap = state_cmap
        self.phase_cmap = phase_cmap
        self.vertex_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        self.animation = FuncAnimation(
            self.figure,
            self._animate,
            interval=180,
            cache_frame_data=False,
        )
        self.render()

    @property
    def atlas(self) -> Atlas:
        return self.atlases[self.name]

    def _style_axis(self, axis: Any, title: str) -> None:
        axis.set_facecolor("#050812")
        axis.set_title(title, loc="left", color="#8deaff", fontsize=12)
        axis.tick_params(colors="#76909d", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#203340")
        axis.grid(color="#17303c", alpha=0.28, linewidth=0.6)

    def render(self) -> None:
        atlas = self.atlas
        step = int(self.step)
        for axis in (
            self.ax_basin,
            self.ax_cobweb,
            self.ax_kkt,
            self.ax_tesseract,
        ):
            axis.clear()

        # Panel 1: state/time atlas. arcsinh keeps the explosive baseline
        # visible without sacrificing the conditioned basin structure.
        transformed = np.arcsinh(atlas.states)
        masked = np.ma.masked_invalid(transformed)
        self.ax_basin.imshow(
            masked,
            origin="lower",
            aspect="auto",
            extent=[
                float(atlas.initial[0]),
                float(atlas.initial[-1]),
                0,
                self.steps,
            ],
            cmap=self.state_cmap,
            vmin=-2.5,
            vmax=2.5,
            interpolation="nearest",
        )
        self.ax_basin.axhline(step, color="#f4ed72", linewidth=1.0)
        self.ax_basin.axvline(
            float(ATTRACTOR),
            color="#39e6b4",
            linewidth=0.8,
            linestyle="--",
        )
        self.ax_basin.set_xlabel("initial state z₀")
        self.ax_basin.set_ylabel("literal recurrence step")
        self._style_axis(
            self.ax_basin, f"{atlas.name.upper()} BASIN / TIME ATLAS"
        )

        # Panel 2: cobweb diagram for all four validated lanes.
        self.ax_cobweb.plot(
            atlas.curve_z,
            atlas.curve_next,
            color="#58b8e8",
            linewidth=1.4,
            label="F(z)",
        )
        self.ax_cobweb.plot(
            atlas.curve_z,
            atlas.curve_z,
            color="#7c8992",
            linewidth=0.8,
            linestyle="--",
            label="F(z)=z",
        )
        lane_colors = ("#ff6b6b", "#ffd166", "#39e6b4", "#a78bfa")
        for lane, color, initial in zip(
            atlas.lanes.T, lane_colors, INITIAL_LANES
        ):
            points = cobweb_points(lane, step)
            if points.size:
                self.ax_cobweb.plot(
                    points[:, 0],
                    points[:, 1],
                    color=color,
                    linewidth=1.0,
                    alpha=0.9,
                )
                self.ax_cobweb.scatter(
                    points[-1, 0],
                    points[-1, 1],
                    s=28,
                    color=color,
                    edgecolor="#ffffff",
                    linewidth=0.4,
                    label=f"z₀={initial:g}",
                )
        self.ax_cobweb.scatter(
            [ATTRACTOR],
            [ATTRACTOR],
            s=70,
            marker="*",
            color="#ffffff",
            edgecolor="#39e6b4",
            linewidth=1.0,
            zorder=8,
        )
        self.ax_cobweb.set_xlim(
            float(atlas.curve_z[0]), float(atlas.curve_z[-1])
        )
        self.ax_cobweb.set_ylim(
            float(atlas.curve_z[0]), float(atlas.curve_z[-1])
        )
        self.ax_cobweb.legend(
            fontsize=7, ncol=3, framealpha=0.15, loc="upper left"
        )
        self.ax_cobweb.set_xlabel("z")
        self.ax_cobweb.set_ylabel("F(z)")
        self._style_axis(
            self.ax_cobweb, f"LITERAL MAP / COBWEB — STEP {step}"
        )

        # Panel 3: exact KKT active mask atlas with invalid certificates black.
        phase = atlas.active.astype(np.float32)
        phase[atlas.active < 0] = np.nan
        self.ax_kkt.imshow(
            np.ma.masked_invalid(np.mod(phase, 20.0)),
            origin="lower",
            aspect="auto",
            extent=[
                float(atlas.initial[0]),
                float(atlas.initial[-1]),
                0,
                self.steps,
            ],
            cmap=self.phase_cmap,
            vmin=0,
            vmax=19,
            interpolation="nearest",
        )
        self.ax_kkt.axhline(step, color="#f4ed72", linewidth=1.0)
        self.ax_kkt.set_xlabel("initial state z₀")
        self.ax_kkt.set_ylabel("step")
        valid_fraction = float(np.mean(atlas.valid[-1])) * 100.0
        self._style_axis(
            self.ax_kkt,
            f"KKT ACTIVE-SET PHASE MAP — "
            f"{valid_fraction:.1f}% VALID AT H={self.steps}",
        )

        # Panel 4: rotating projected tesseract driven by the +0.8 shader lane.
        phase_angle = 0.11 * step
        projected = project_tesseract(self.vertices, phase_angle)
        diagnostic = atlas.diagnostics[min(step, self.steps)]
        if np.isfinite(diagnostic).all():
            scalar = diagnostic[:, 0]
            tangent = diagnostic[:, 1]
            vector_norm = diagnostic[:, 2]
            matrix_norm = diagnostic[:, 3]
            color_value = scalar + 0.15 * tangent
            scale = max(float(np.nanpercentile(np.abs(color_value), 90)), 1e-5)
            normalized = np.clip(color_value / scale, -1.0, 1.0)
            size = 35.0 + 22.0 * np.log1p(
                vector_norm + 0.2 * matrix_norm
            )
        else:
            normalized = np.zeros(16)
            size = np.full(16, 35.0)
        for left, right in self.edges:
            self.ax_tesseract.plot(
                projected[[left, right], 0],
                projected[[left, right], 1],
                color="#3c7890",
                linewidth=0.8,
                alpha=0.55,
            )
        self.ax_tesseract.scatter(
            projected[:, 0],
            projected[:, 1],
            c=normalized,
            s=size,
            cmap=self.state_cmap,
            norm=self.vertex_norm,
            edgecolor="#d8fbff",
            linewidth=0.55,
            zorder=5,
        )
        for index, point in enumerate(projected):
            self.ax_tesseract.text(
                point[0],
                point[1] + 0.018,
                f"{index:X}",
                ha="center",
                fontsize=6,
                color="#b9d7df",
            )
        tape = atlas.tape[min(step, self.steps)]
        lane_z = atlas.lanes[min(step, self.steps), 3]
        if np.isfinite(tape).all() and np.isfinite(lane_z):
            info = (
                f"lane z₀=+0.8   z={lane_z:+.7f}\n"
                f"KKT mask={int(tape[18]):02d}   "
                f"classifier=0x{int(tape[19]):02X}   "
                f"parity cell={int(tape[30])}\n"
                f"margins: KKT={min(tape[23], tape[24]):.4g}   "
                f"class={min(tape[25], tape[26]):.4g}   "
                f"parity={tape[31]:.4g}\n"
                f"condition={max(tape[27], tape[28]):.4g}   "
                f"V={(lane_z - ATTRACTOR) ** 2:.3e}"
            )
        else:
            info = "trajectory left the valid physical regime"
        self.ax_tesseract.text(
            0.02,
            0.02,
            info,
            transform=self.ax_tesseract.transAxes,
            fontsize=8,
            color="#84b5c2",
            va="bottom",
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#07121c",
                "edgecolor": "#1d4d5c",
                "alpha": 0.88,
            },
        )
        self.ax_tesseract.set_aspect("equal")
        self.ax_tesseract.set_xlim(-0.58, 0.58)
        self.ax_tesseract.set_ylim(-0.49, 0.49)
        self.ax_tesseract.set_xticks([])
        self.ax_tesseract.set_yticks([])
        self._style_axis(
            self.ax_tesseract,
            "CARTER VERTICES / ROTATING 4D PROJECTION",
        )
        self.figure.canvas.draw_idle()

    def _select_model(self, label: str) -> None:
        self.name = label
        self.render()

    def _select_step(self, value: float) -> None:
        self.step = int(value)
        self.render()

    def _toggle_play(self, _event: Any) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("PAUSE" if self.playing else "PLAY")

    def _animate(self, _frame: int):
        if self.playing:
            next_step = (self.step + 1) % (self.steps + 1)
            self.slider.set_val(next_step)
        return ()

    def _save(self, _event: Any = None) -> None:
        self.figure.savefig(
            self.output, dpi=180, facecolor=self.figure.get_facecolor()
        )
        print(f"saved visualization: {self.output}", flush=True)

    def run(self) -> None:
        self._save()
        if self.show:
            self.plt.show()
        else:
            self.plt.close(self.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full-stack", "literal"),
        default="full-stack",
        help=(
            "render the qualified NR/constitutive stack or the original "
            "interactive Vulkan literal-recurrence instrument"
        ),
    )
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--domain", type=float, nargs=2, default=(-1.0, 1.0))
    parser.add_argument(
        "--output",
        type=Path,
        help="snapshot path; defaults depend on --mode",
    )
    parser.add_argument(
        "--frontier-results",
        type=Path,
        default=HERE / "tesseract_nr_frontier_results.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=HERE / "theory33_frozen_variants.json",
    )
    parser.add_argument("--spacetime-points", type=int, default=24)
    parser.add_argument(
        "--skip-live-spacetime",
        action="store_true",
        help="render the tracked qualified profile without rebuilding CMC data",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="render the snapshot without opening an interactive window",
    )
    return parser.parse_args()


def main() -> int:
    global spy
    args = parse_args()
    if args.mode == "full-stack":
        from tesseract_full_stack_visualizer import (
            render_full_stack_dashboard,
        )

        output = (
            args.output
            if args.output is not None
            else HERE / "tesseract_full_stack_dashboard.png"
        )
        path = render_full_stack_dashboard(
            args.frontier_results,
            output.resolve(),
            artifact_path=args.artifact,
            spacetime_points=args.spacetime_points,
            live_spacetime=not args.skip_live_spacetime,
            show=not args.no_show,
        )
        print(f"saved full-stack visualization: {path}", flush=True)
        return 0

    if args.samples < 32 or args.steps < 1:
        raise SystemExit("--samples must be >=32 and --steps must be positive")
    try:
        import slangpy as slang_runtime
    except ImportError:
        raise SystemExit(
            "literal mode requires slangpy; run through run_visualization.ps1"
        )
    spy = slang_runtime
    output = (
        args.output
        if args.output is not None
        else HERE / "tesseract_conditioning_atlas.png"
    )
    device = spy.create_device(
        type=spy.DeviceType.vulkan,
        include_paths=[HERE],
    )
    definitions = (
        (
            "Baseline",
            HERE / "carter_tesseract_full.slang",
            HERE / "hybrid_model.json",
        ),
        (
            "Conditioned",
            HERE / "carter_tesseract_conditioned.slang",
            HERE / "hybrid_model_conditioned.json",
        ),
    )
    atlases: dict[str, Atlas] = {}
    for name, module_path, model_path in definitions:
        print(f"capturing {name.lower()} Vulkan atlas...", flush=True)
        model = ShaderModel(
            device,
            name=name,
            module_path=module_path,
            model_path=model_path,
        )
        atlases[name] = model.build_atlas(
            samples=args.samples,
            steps=args.steps,
            domain=(args.domain[0], args.domain[1]),
        )
        valid_fraction = float(np.mean(atlases[name].valid[-1]))
        print(
            f"  horizon {args.steps}: {100.0 * valid_fraction:.1f}% "
            "of sampled initial states remain valid",
            flush=True,
        )
    Dashboard(
        atlases,
        steps=args.steps,
        output=output.resolve(),
        show=not args.no_show,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
