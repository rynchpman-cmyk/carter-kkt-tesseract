"""Periodic Cartesian grids and spatial differential operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PeriodicGrid:
    """A one-, two-, or three-dimensional periodic Cartesian grid.

    Vector and tensor components always occupy leading axes. Spatial grid axes
    are the trailing ``ndim`` axes.
    """

    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    method: str = "spectral"

    def __post_init__(self) -> None:
        shape = tuple(int(n) for n in self.shape)
        lengths = tuple(float(length) for length in self.lengths)
        if not 1 <= len(shape) <= 3 or len(shape) != len(lengths):
            raise ValueError("shape and lengths must describe 1, 2, or 3 axes")
        if any(n < 4 for n in shape) or any(length <= 0 for length in lengths):
            raise ValueError("each grid axis needs at least 4 points and positive length")
        if self.method not in {"spectral", "fd2", "fd4"}:
            raise ValueError("method must be 'spectral', 'fd2', or 'fd4'")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lengths", lengths)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_periodic(self) -> bool:
        return True

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(length / n for length, n in zip(self.lengths, self.shape))

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def volume(self) -> float:
        return float(np.prod(self.lengths))

    def coordinates(self) -> tuple[Array, ...]:
        axes = [np.arange(n, dtype=float) * dx for n, dx in zip(self.shape, self.spacing)]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def zeros(self, components: Iterable[int] = ()) -> Array:
        return np.zeros(tuple(components) + self.shape, dtype=float)

    def _spatial_axis(self, field: Array, axis: int) -> int:
        if not 0 <= axis < self.ndim:
            raise ValueError(f"axis {axis} is inactive for a {self.ndim}D grid")
        if tuple(field.shape[-self.ndim :]) != self.shape:
            raise ValueError(f"field trailing shape {field.shape} does not match {self.shape}")
        return field.ndim - self.ndim + axis

    def derivative(self, field: Array, axis: int) -> Array:
        field = np.asarray(field, dtype=float)
        array_axis = self._spatial_axis(field, axis)
        dx = self.spacing[axis]
        if self.method == "fd2":
            return (
                np.roll(field, -1, axis=array_axis) - np.roll(field, 1, axis=array_axis)
            ) / (2.0 * dx)
        if self.method == "fd4":
            return (
                -np.roll(field, -2, axis=array_axis)
                + 8.0 * np.roll(field, -1, axis=array_axis)
                - 8.0 * np.roll(field, 1, axis=array_axis)
                + np.roll(field, 2, axis=array_axis)
            ) / (12.0 * dx)

        k = 2.0 * np.pi * np.fft.fftfreq(self.shape[axis], d=dx)
        k_shape = [1] * field.ndim
        k_shape[array_axis] = self.shape[axis]
        transformed = np.fft.fft(field, axis=array_axis)
        result = np.fft.ifft(1j * k.reshape(k_shape) * transformed, axis=array_axis)
        return result.real

    def second_derivative(self, field: Array, axis: int) -> Array:
        field = np.asarray(field, dtype=float)
        array_axis = self._spatial_axis(field, axis)
        dx = self.spacing[axis]
        if self.method == "fd2":
            return (
                np.roll(field, -1, axis=array_axis)
                - 2.0 * field
                + np.roll(field, 1, axis=array_axis)
            ) / dx**2
        if self.method == "fd4":
            return (
                -np.roll(field, -2, axis=array_axis)
                + 16.0 * np.roll(field, -1, axis=array_axis)
                - 30.0 * field
                + 16.0 * np.roll(field, 1, axis=array_axis)
                - np.roll(field, 2, axis=array_axis)
            ) / (12.0 * dx**2)

        k = 2.0 * np.pi * np.fft.fftfreq(self.shape[axis], d=dx)
        k_shape = [1] * field.ndim
        k_shape[array_axis] = self.shape[axis]
        transformed = np.fft.fft(field, axis=array_axis)
        result = np.fft.ifft(-(k.reshape(k_shape) ** 2) * transformed, axis=array_axis)
        return result.real

    def gradient(self, scalar: Array) -> Array:
        scalar = np.asarray(scalar, dtype=float)
        result = self.zeros((3,))
        for axis in range(self.ndim):
            result[axis] = self.derivative(scalar, axis)
        return result

    def divergence(self, vector: Array) -> Array:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) + self.shape:
            raise ValueError(f"expected vector shape {(3,) + self.shape}, got {vector.shape}")
        result = self.zeros()
        for axis in range(self.ndim):
            result += self.derivative(vector[axis], axis)
        return result

    def laplacian(self, field: Array) -> Array:
        result = np.zeros_like(field, dtype=float)
        for axis in range(self.ndim):
            result += self.second_derivative(field, axis)
        return result

    def curl(self, vector: Array) -> Array:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) + self.shape:
            raise ValueError(f"expected vector shape {(3,) + self.shape}, got {vector.shape}")

        def d(component: int, axis: int) -> Array:
            return self.derivative(vector[component], axis) if axis < self.ndim else self.zeros()

        return np.stack(
            (d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1))
        )

    def integrate(self, scalar: Array) -> float:
        return float(np.sum(np.asarray(scalar, dtype=float)) * self.cell_volume)

    def shift(self, field: Array, offset: int, axis: int) -> Array:
        """Shift along a spatial axis with periodic wrapping."""
        return np.roll(field, offset, axis=self._spatial_axis(np.asarray(field), axis))


@dataclass(frozen=True)
class CartesianGrid:
    """Cell-centered nonperiodic Cartesian patch.

    The patch stores interior cells only. Differential operators use centered
    stencils in the interior and one-sided stencils at physical edges. ``shift``
    performs constant ghost extrapolation, which boundary-aware flux/RHS layers
    may override with characteristic data.
    """

    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    origin: tuple[float, ...] | None = None
    method: str = "fd4"

    def __post_init__(self) -> None:
        shape = tuple(int(n) for n in self.shape)
        lengths = tuple(float(length) for length in self.lengths)
        if not 1 <= len(shape) <= 3 or len(shape) != len(lengths):
            raise ValueError("shape and lengths must describe 1, 2, or 3 axes")
        if any(n < 6 for n in shape) or any(length <= 0.0 for length in lengths):
            raise ValueError("nonperiodic axes need at least 6 cells and positive length")
        if self.method not in {"fd2", "fd4"}:
            raise ValueError("CartesianGrid supports fd2 or fd4 derivatives")
        origin = (
            tuple(-0.5 * length for length in lengths)
            if self.origin is None
            else tuple(float(value) for value in self.origin)
        )
        if len(origin) != len(shape):
            raise ValueError("origin must match the grid dimension")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "origin", origin)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_periodic(self) -> bool:
        return False

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(length / n for length, n in zip(self.lengths, self.shape))

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def volume(self) -> float:
        return float(np.prod(self.lengths))

    def coordinates(self) -> tuple[Array, ...]:
        axes = [
            start + (np.arange(n, dtype=float) + 0.5) * dx
            for start, n, dx in zip(self.origin, self.shape, self.spacing)
        ]
        return tuple(np.meshgrid(*axes, indexing="ij"))

    def zeros(self, components: Iterable[int] = ()) -> Array:
        return np.zeros(tuple(components) + self.shape, dtype=float)

    def _spatial_axis(self, field: Array, axis: int) -> int:
        if not 0 <= axis < self.ndim:
            raise ValueError(f"axis {axis} is inactive for a {self.ndim}D grid")
        if tuple(field.shape[-self.ndim :]) != self.shape:
            raise ValueError(f"field trailing shape {field.shape} does not match {self.shape}")
        return field.ndim - self.ndim + axis

    def _slice(self, field: Array, array_axis: int, index: int) -> tuple[slice | int, ...]:
        slices: list[slice | int] = [slice(None)] * field.ndim
        slices[array_axis] = index
        return tuple(slices)

    def shift(self, field: Array, offset: int, axis: int) -> Array:
        field = np.asarray(field)
        array_axis = self._spatial_axis(field, axis)
        if offset == 0:
            return field.copy()
        indices = np.arange(self.shape[axis]) - offset
        indices = np.clip(indices, 0, self.shape[axis] - 1)
        return np.take(field, indices, axis=array_axis)

    def derivative(self, field: Array, axis: int) -> Array:
        field = np.asarray(field, dtype=float)
        array_axis = self._spatial_axis(field, axis)
        dx = self.spacing[axis]
        moved = np.moveaxis(field, array_axis, -1)
        result = np.empty_like(moved)
        if self.method == "fd2":
            result[..., 1:-1] = (moved[..., 2:] - moved[..., :-2]) / (2.0 * dx)
            result[..., 0] = (-3.0 * moved[..., 0] + 4.0 * moved[..., 1] - moved[..., 2]) / (2.0 * dx)
            result[..., -1] = (3.0 * moved[..., -1] - 4.0 * moved[..., -2] + moved[..., -3]) / (2.0 * dx)
        else:
            result[..., 2:-2] = (
                -moved[..., 4:] + 8.0 * moved[..., 3:-1] - 8.0 * moved[..., 1:-3] + moved[..., :-4]
            ) / (12.0 * dx)
            result[..., 0] = (
                -25.0 * moved[..., 0]
                + 48.0 * moved[..., 1]
                - 36.0 * moved[..., 2]
                + 16.0 * moved[..., 3]
                - 3.0 * moved[..., 4]
            ) / (12.0 * dx)
            result[..., 1] = (
                -3.0 * moved[..., 0]
                - 10.0 * moved[..., 1]
                + 18.0 * moved[..., 2]
                - 6.0 * moved[..., 3]
                + moved[..., 4]
            ) / (12.0 * dx)
            result[..., -1] = -(
                -25.0 * moved[..., -1]
                + 48.0 * moved[..., -2]
                - 36.0 * moved[..., -3]
                + 16.0 * moved[..., -4]
                - 3.0 * moved[..., -5]
            ) / (12.0 * dx)
            result[..., -2] = -(
                -3.0 * moved[..., -1]
                - 10.0 * moved[..., -2]
                + 18.0 * moved[..., -3]
                - 6.0 * moved[..., -4]
                + moved[..., -5]
            ) / (12.0 * dx)
        return np.moveaxis(result, -1, array_axis)

    def second_derivative(self, field: Array, axis: int) -> Array:
        # Reapplying the stable first-derivative operator keeps boundary closure
        # consistent and is second order at physical edges, fourth inside.
        return self.derivative(self.derivative(field, axis), axis)

    def gradient(self, scalar: Array) -> Array:
        result = self.zeros((3,))
        for axis in range(self.ndim):
            result[axis] = self.derivative(scalar, axis)
        return result

    def divergence(self, vector: Array) -> Array:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) + self.shape:
            raise ValueError(f"expected vector shape {(3,) + self.shape}, got {vector.shape}")
        result = self.zeros()
        for axis in range(self.ndim):
            result += self.derivative(vector[axis], axis)
        return result

    def laplacian(self, field: Array) -> Array:
        result = np.zeros_like(field, dtype=float)
        for axis in range(self.ndim):
            result += self.second_derivative(field, axis)
        return result

    def curl(self, vector: Array) -> Array:
        def d(component: int, axis: int) -> Array:
            return self.derivative(vector[component], axis) if axis < self.ndim else self.zeros()

        return np.stack(
            (d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1))
        )

    def integrate(self, scalar: Array) -> float:
        return float(np.sum(np.asarray(scalar, dtype=float)) * self.cell_volume)


@dataclass
class GhostZonePatch:
    """Explicit ghost-zone storage associated with a Cartesian grid.

    Component axes precede the spatial axes, as elsewhere in the package.  The
    object deliberately owns its storage so MPI exchange, AMR interpolation,
    and physical boundary filling can all target the same representation.
    """

    grid: CartesianGrid
    components: tuple[int, ...] = ()
    ghost_width: int = 3

    def __post_init__(self) -> None:
        self.components = tuple(int(n) for n in self.components)
        self.ghost_width = int(self.ghost_width)
        if self.ghost_width < 1:
            raise ValueError("ghost_width must be positive")
        extended = tuple(n + 2 * self.ghost_width for n in self.grid.shape)
        self.data = np.zeros(self.components + extended, dtype=float)

    @property
    def interior_slices(self) -> tuple[slice, ...]:
        g = self.ghost_width
        return (slice(None),) * len(self.components) + tuple(
            slice(g, g + n) for n in self.grid.shape
        )

    @property
    def interior(self) -> Array:
        return self.data[self.interior_slices]

    def set_interior(self, values: Array) -> None:
        values = np.asarray(values, dtype=float)
        expected = self.components + self.grid.shape
        if values.shape != expected:
            raise ValueError(f"expected interior shape {expected}, got {values.shape}")
        self.interior[...] = values

    def fill_outflow(self) -> None:
        """Fill every physical ghost face by constant normal extrapolation."""
        g = self.ghost_width
        leading = len(self.components)
        for axis, n in enumerate(self.grid.shape):
            array_axis = leading + axis
            lower = [slice(None)] * self.data.ndim
            lower_edge = [slice(None)] * self.data.ndim
            upper = [slice(None)] * self.data.ndim
            upper_edge = [slice(None)] * self.data.ndim
            lower[array_axis] = slice(0, g)
            lower_edge[array_axis] = slice(g, g + 1)
            upper[array_axis] = slice(g + n, g + n + g)
            upper_edge[array_axis] = slice(g + n - 1, g + n)
            self.data[tuple(lower)] = self.data[tuple(lower_edge)]
            self.data[tuple(upper)] = self.data[tuple(upper_edge)]

    def face(self, axis: int, side: int, include_ghosts: bool = False) -> Array:
        """Return a boundary face; ``side`` is -1 (lower) or +1 (upper)."""
        if not 0 <= axis < self.grid.ndim or side not in {-1, 1}:
            raise ValueError("invalid face axis or side")
        g = self.ghost_width
        array_axis = len(self.components) + axis
        index = (g - 1 if side < 0 else g + self.grid.shape[axis]) if include_ghosts else (
            g if side < 0 else g + self.grid.shape[axis] - 1
        )
        slices: list[slice | int] = [slice(None)] * self.data.ndim
        slices[array_axis] = index
        return self.data[tuple(slices)]
