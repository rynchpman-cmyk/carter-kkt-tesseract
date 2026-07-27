"""Conservative block-structured AMR building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np

from .grid import Array

State = TypeVar("State")


def minmod(a: Array, b: Array) -> Array:
    same_sign = a * b > 0.0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def conservative_prolong(coarse: Array, spatial_ndim: int, ratio: int = 2) -> Array:
    """Limited piecewise-linear cell-centered prolongation.

    Each group of children averages exactly to its parent.  Constant edge
    extension supplies the physical/coarse-patch boundary closure.
    """
    if ratio != 2 or not 1 <= spatial_ndim <= 3:
        raise ValueError("this reference prolongator supports ratio=2 in 1--3D")
    coarse = np.asarray(coarse, dtype=float)
    leading = coarse.ndim - spatial_ndim
    result = coarse
    for axis in range(spatial_ndim):
        array_axis = leading + axis
        left = np.take(coarse, np.maximum(np.arange(coarse.shape[array_axis]) - 1, 0), axis=array_axis)
        right = np.take(coarse, np.minimum(np.arange(coarse.shape[array_axis]) + 1, coarse.shape[array_axis] - 1), axis=array_axis)
        slope = minmod(coarse - left, right - coarse)
        children = np.stack((coarse - 0.25 * slope, coarse + 0.25 * slope), axis=array_axis + 1)
        new_shape = list(children.shape)
        new_shape[array_axis] *= new_shape.pop(array_axis + 1)
        refined_axis = children.reshape(new_shape)
        if result is coarse:
            result = refined_axis
        else:
            # Prolong this axis on the already-refined result using the parent
            # slope repeated along all previously refined axes.
            repeated_slope = slope
            for prior in range(axis):
                repeated_slope = np.repeat(repeated_slope, 2, axis=leading + prior)
            low = result - 0.25 * repeated_slope
            high = result + 0.25 * repeated_slope
            children = np.stack((low, high), axis=array_axis + 1)
            new_shape = list(children.shape)
            new_shape[array_axis] *= new_shape.pop(array_axis + 1)
            result = children.reshape(new_shape)
    return result


def conservative_restrict(fine: Array, spatial_ndim: int, ratio: int = 2) -> Array:
    """Volume-average fine cell data onto its parent cells."""
    if ratio < 2 or not 1 <= spatial_ndim <= 3:
        raise ValueError("invalid restriction dimension or ratio")
    result = np.asarray(fine, dtype=float)
    leading = result.ndim - spatial_ndim
    for axis in reversed(range(spatial_ndim)):
        array_axis = leading + axis
        n = result.shape[array_axis]
        if n % ratio:
            raise ValueError("fine shape must be divisible by refinement ratio")
        shape = list(result.shape)
        shape[array_axis : array_axis + 1] = [n // ratio, ratio]
        result = result.reshape(shape).mean(axis=array_axis + 1)
    return result


@dataclass
class FluxRegister:
    """Berger--Colella correction register for one refined rectangular patch."""

    coarse_shape: tuple[int, ...]
    spacing: tuple[float, ...]
    bounds: tuple[tuple[int, int], ...]
    components: tuple[int, ...] = ()
    ratio: int = 2

    def __post_init__(self) -> None:
        if len(self.coarse_shape) != len(self.spacing) or len(self.bounds) != len(self.coarse_shape):
            raise ValueError("flux-register dimensionality mismatch")
        self.correction = np.zeros(self.components + self.coarse_shape)

    def clear(self) -> None:
        self.correction.fill(0.0)

    def accumulate(
        self,
        axis: int,
        side: int,
        coarse_flux: Array,
        fine_flux: Array,
        dt: float,
    ) -> None:
        """Accumulate the coarse/fine flux mismatch at one patch face."""
        if side not in {-1, 1} or dt <= 0.0:
            raise ValueError("invalid interface side or time interval")
        tangential_ndim = len(self.coarse_shape) - 1
        averaged_fine = (
            conservative_restrict(fine_flux, tangential_ndim, self.ratio)
            if tangential_ndim
            else np.asarray(fine_flux)
        )
        mismatch = np.asarray(averaged_fine) - np.asarray(coarse_flux)
        slices: list[slice | int] = [slice(None)] * len(self.components)
        for direction, (lower, upper) in enumerate(self.bounds):
            if direction == axis:
                index = lower - 1 if side < 0 else upper
                if not 0 <= index < self.coarse_shape[direction]:
                    return  # refined patch touches a physical boundary
                slices.append(index)
            else:
                slices.append(slice(lower, upper))
        self.correction[tuple(slices)] += side * dt * mismatch / self.spacing[axis]

    def reflux(self, coarse_state: Array) -> Array:
        result = np.asarray(coarse_state, dtype=float).copy()
        result += self.correction
        return result


class BergerColellaStepper(Generic[State]):
    """Time-subcycling orchestration; physics-specific callbacks own RK stages."""

    def __init__(self, refinement_ratio: int = 2) -> None:
        if refinement_ratio < 2:
            raise ValueError("refinement_ratio must be at least two")
        self.refinement_ratio = refinement_ratio

    def advance(
        self,
        coarse: State,
        fine: State,
        time: float,
        dt: float,
        coarse_step: Callable[[State, float, float], State],
        fine_step: Callable[[State, float, float], State],
        synchronize: Callable[[State, State], tuple[State, State]],
    ) -> tuple[State, State]:
        coarse = coarse_step(coarse, time, dt)
        fine_dt = dt / self.refinement_ratio
        for substep in range(self.refinement_ratio):
            fine = fine_step(fine, time + substep * fine_dt, fine_dt)
        return synchronize(coarse, fine)


@dataclass
class Theory3AMRFields:
    """Constraint-carrying Theory 3 fields synchronized across AMR levels."""

    pi_A: Array
    pi_B: Array
    longitudinal_A: Array
    longitudinal_B: Array
    target_charge: Array
    target_current: Array


def project_theory3_gauss(
    fields: Theory3AMRFields,
    grid,
    h: Array,
    parameters=None,
) -> Theory3AMRFields:
    """Project both cell-centered Gauss charges after an AMR transfer."""
    from .theory3 import Theory3System

    system = Theory3System(grid, parameters)
    longitudinal_A, longitudinal_B = system.initial_longitudinal_charges(
        h, fields.pi_A, fields.pi_B
    )
    return Theory3AMRFields(
        fields.pi_A,
        fields.pi_B,
        longitudinal_A,
        longitudinal_B,
        fields.target_charge,
        fields.target_current,
    )


def prolong_theory3_fields(
    coarse: Theory3AMRFields,
    fine_grid,
    fine_h: Array,
    parameters=None,
    ratio: int = 2,
) -> Theory3AMRFields:
    """Conservatively prolong Theory 3 densities and reproject both Gauss laws.

    Reprojection is necessary because the reference discretization is cell
    centered; its derivative does not commute exactly with generic limited
    prolongation.  Constraint-satisfying periodic data have zero integrated
    longitudinal charge, so this projection preserves their global charge.
    """
    ndim = fine_grid.ndim
    transferred = Theory3AMRFields(
        conservative_prolong(coarse.pi_A, ndim, ratio),
        conservative_prolong(coarse.pi_B, ndim, ratio),
        conservative_prolong(coarse.longitudinal_A, ndim, ratio),
        conservative_prolong(coarse.longitudinal_B, ndim, ratio),
        conservative_prolong(coarse.target_charge, ndim, ratio),
        conservative_prolong(coarse.target_current, ndim, ratio),
    )
    return project_theory3_gauss(
        transferred, fine_grid, fine_h, parameters
    )


def restrict_theory3_fields(
    fine: Theory3AMRFields,
    coarse_grid,
    coarse_h: Array,
    parameters=None,
    ratio: int = 2,
) -> Theory3AMRFields:
    """Volume-restrict Theory 3 fields and restore coarse Gauss compatibility."""
    ndim = coarse_grid.ndim
    transferred = Theory3AMRFields(
        conservative_restrict(fine.pi_A, ndim, ratio),
        conservative_restrict(fine.pi_B, ndim, ratio),
        conservative_restrict(fine.longitudinal_A, ndim, ratio),
        conservative_restrict(fine.longitudinal_B, ndim, ratio),
        conservative_restrict(fine.target_charge, ndim, ratio),
        conservative_restrict(fine.target_current, ndim, ratio),
    )
    return project_theory3_gauss(
        transferred, coarse_grid, coarse_h, parameters
    )


class Theory3FluxRegisters:
    """Berger--Colella registers for every Theory 3 conserved charge."""

    _components = {
        "pi_A": (3,),
        "pi_B": (3,),
        "longitudinal_A": (),
        "longitudinal_B": (),
        "target_charge": (),
        "target_current": (3,),
    }

    def __init__(
        self,
        coarse_shape: tuple[int, ...],
        spacing: tuple[float, ...],
        bounds: tuple[tuple[int, int], ...],
        ratio: int = 2,
    ) -> None:
        self.registers = {
            name: FluxRegister(
                coarse_shape, spacing, bounds, components, ratio
            )
            for name, components in self._components.items()
        }

    def clear(self) -> None:
        for register in self.registers.values():
            register.clear()

    def accumulate(
        self,
        name: str,
        axis: int,
        side: int,
        coarse_flux: Array,
        fine_flux: Array,
        dt: float,
    ) -> None:
        if name not in self.registers:
            raise KeyError(f"{name!r} is not a Theory 3 refluxed charge")
        self.registers[name].accumulate(
            axis, side, coarse_flux, fine_flux, dt
        )

    def reflux(self, fields: Theory3AMRFields) -> Theory3AMRFields:
        return Theory3AMRFields(
            self.registers["pi_A"].reflux(fields.pi_A),
            self.registers["pi_B"].reflux(fields.pi_B),
            self.registers["longitudinal_A"].reflux(
                fields.longitudinal_A
            ),
            self.registers["longitudinal_B"].reflux(
                fields.longitudinal_B
            ),
            self.registers["target_charge"].reflux(fields.target_charge),
            self.registers["target_current"].reflux(fields.target_current),
        )

    def reflux_and_project(
        self, fields: Theory3AMRFields, grid, h: Array, parameters=None
    ) -> Theory3AMRFields:
        return project_theory3_gauss(
            self.reflux(fields), grid, h, parameters
        )
