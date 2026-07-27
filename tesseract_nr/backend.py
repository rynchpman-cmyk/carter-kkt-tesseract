"""Optional accelerator and distributed-memory execution interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class ArrayBackend:
    """Small NumPy/CuPy compatibility boundary for performance kernels."""

    def __init__(self, name: str = "numpy") -> None:
        name = name.lower()
        if name == "auto":
            try:
                import cupy as cp
                self.xp = cp
                self.name = "cupy"
            except ImportError:
                self.xp = np
                self.name = "numpy"
        elif name == "numpy":
            self.xp = np
            self.name = name
        elif name == "cupy":
            try:
                import cupy as cp
            except ImportError as error:
                raise RuntimeError("the CuPy backend was requested but cupy is not installed") from error
            self.xp = cp
            self.name = name
        else:
            raise ValueError("backend must be numpy, cupy, or auto")

    @property
    def accelerated(self) -> bool:
        return self.name == "cupy"

    def asarray(self, value: Any, dtype=float):
        return self.xp.asarray(value, dtype=dtype)

    def to_host(self, value: Any) -> np.ndarray:
        return self.xp.asnumpy(value) if self.accelerated else np.asarray(value)

    def synchronize(self) -> None:
        if self.accelerated:
            self.xp.cuda.get_current_stream().synchronize()


def apply_constant_2x2_exponential(
    matrix: Any,
    values: Any,
    timestep: Any,
    backend: ArrayBackend | None = None,
):
    """Apply ``exp(timestep*matrix)`` to a leading two-component array.

    The closed 2x2 formula is branch-safe at a repeated eigenvalue and uses
    only NumPy/CuPy-compatible elementwise kernels.  It avoids a host-side
    eigendecomposition in the local Theory 3 damping hot path.
    """
    backend = backend or ArrayBackend("numpy")
    xp = backend.xp
    matrix = xp.asarray(matrix, dtype=float)
    values = xp.asarray(values, dtype=float)
    timestep = xp.asarray(timestep, dtype=float)
    if matrix.shape != (2, 2) or values.shape[0] != 2:
        raise ValueError("expected a constant 2x2 matrix and two-component values")
    half_trace = 0.5 * (matrix[0, 0] + matrix[1, 1])
    centered_00 = matrix[0, 0] - half_trace
    centered_11 = matrix[1, 1] - half_trace
    delta_sq = centered_00**2 + matrix[0, 1] * matrix[1, 0]
    delta = xp.sqrt(xp.maximum(delta_sq, 0.0))
    argument = delta * timestep
    scale = xp.where(
        xp.abs(delta) > 1.0e-14,
        xp.sinh(argument) / xp.maximum(delta, 1.0e-300),
        timestep,
    )
    common = xp.exp(half_trace * timestep)
    first = common * (
        xp.cosh(argument) * values[0]
        + scale * (centered_00 * values[0] + matrix[0, 1] * values[1])
    )
    second = common * (
        xp.cosh(argument) * values[1]
        + scale * (matrix[1, 0] * values[0] + centered_11 * values[1])
    )
    return xp.stack((first, second), axis=0)


@dataclass(frozen=True)
class SlabPartition:
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    start: int
    stop: int
    axis: int
    rank: int
    size: int


class ParallelContext:
    """MPI context with a complete single-rank fallback."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.comm = None
        if enabled is not False:
            try:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
            except ImportError:
                if enabled:
                    raise RuntimeError("MPI was requested but mpi4py is not installed")
        self.rank = 0 if self.comm is None else self.comm.Get_rank()
        self.size = 1 if self.comm is None else self.comm.Get_size()

    @property
    def distributed(self) -> bool:
        return self.size > 1

    def partition(self, shape: tuple[int, ...], axis: int = 0) -> SlabPartition:
        if not 0 <= axis < len(shape):
            raise ValueError("partition axis is out of range")
        base, remainder = divmod(shape[axis], self.size)
        count = base + (self.rank < remainder)
        start = self.rank * base + min(self.rank, remainder)
        local = list(shape)
        local[axis] = count
        return SlabPartition(shape, tuple(local), start, start + count, axis, self.rank, self.size)

    @staticmethod
    def _pairwise_sum(values: list[np.ndarray]) -> np.ndarray:
        work = [np.asarray(value, dtype=np.float64) for value in values]
        while len(work) > 1:
            work = [
                work[i] + work[i + 1] if i + 1 < len(work) else work[i]
                for i in range(0, len(work), 2)
            ]
        return work[0]

    def deterministic_sum(self, value: Any):
        local = np.asarray(value, dtype=np.float64)
        if self.comm is None:
            result = local.copy()
        else:
            gathered = self.comm.gather(local, root=0)
            result = self._pairwise_sum(gathered) if self.rank == 0 else None
            result = self.comm.bcast(result, root=0)
        return float(result) if result.ndim == 0 else result

    def exchange_halos(
        self,
        data: np.ndarray,
        partition: SlabPartition,
        ghost_width: int,
        component_axes: int = 0,
    ) -> None:
        """Exchange periodic-neighbor slabs in explicit ghost-zone storage.

        Physical end ranks retain their existing exterior ghosts so the
        boundary-condition layer can fill them afterward.
        """
        if ghost_width < 1:
            raise ValueError("ghost_width must be positive")
        if self.comm is None:
            return
        axis = component_axes + partition.axis
        g = ghost_width
        n = partition.local_shape[partition.axis]
        left = self.rank - 1 if self.rank > 0 else None
        right = self.rank + 1 if self.rank + 1 < self.size else None
        lower_send = [slice(None)] * data.ndim
        lower_recv = [slice(None)] * data.ndim
        upper_send = [slice(None)] * data.ndim
        upper_recv = [slice(None)] * data.ndim
        lower_send[axis] = slice(g, 2 * g)
        lower_recv[axis] = slice(0, g)
        upper_send[axis] = slice(g + n - g, g + n)
        upper_recv[axis] = slice(g + n, g + n + g)
        if left is not None:
            receive = np.empty_like(np.ascontiguousarray(data[tuple(lower_recv)]))
            self.comm.Sendrecv(
                np.ascontiguousarray(data[tuple(lower_send)]), dest=left, sendtag=10,
                recvbuf=receive, source=left, recvtag=11,
            )
            data[tuple(lower_recv)] = receive
        if right is not None:
            receive = np.empty_like(np.ascontiguousarray(data[tuple(upper_recv)]))
            self.comm.Sendrecv(
                np.ascontiguousarray(data[tuple(upper_send)]), dest=right, sendtag=11,
                recvbuf=receive, source=right, recvtag=10,
            )
            data[tuple(upper_recv)] = receive


_THEORY3_COMPONENT_AXES = {
    "a": 1,
    "pi_A": 1,
    "b": 1,
    "pi_B": 1,
    "longitudinal_A": 0,
    "longitudinal_B": 0,
    "cleaning_A": 0,
    "cleaning_B": 0,
    "target_charge": 0,
    "target_current": 1,
}


def exchange_theory3_halos(
    context: ParallelContext,
    partition: SlabPartition,
    ghost_width: int,
    fields: Mapping[str, np.ndarray],
) -> None:
    """Exchange every enlarged Theory 3 field through one MPI contract."""
    unknown = set(fields) - set(_THEORY3_COMPONENT_AXES)
    if unknown:
        raise KeyError(f"unknown Theory 3 halo fields: {sorted(unknown)}")
    for name, value in fields.items():
        context.exchange_halos(
            value,
            partition,
            ghost_width,
            component_axes=_THEORY3_COMPONENT_AXES[name],
        )
