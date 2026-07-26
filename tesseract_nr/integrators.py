"""Small dependency-free time integration utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Array = np.ndarray


def rk4_arrays(
    values: Sequence[Array],
    time: float,
    dt: float,
    rhs: Callable[[float, Sequence[Array]], Sequence[Array]],
) -> tuple[Array, ...]:
    """Classical fourth-order Runge--Kutta for a tuple of arrays."""
    k1 = tuple(rhs(time, values))
    y2 = tuple(y + 0.5 * dt * k for y, k in zip(values, k1))
    k2 = tuple(rhs(time + 0.5 * dt, y2))
    y3 = tuple(y + 0.5 * dt * k for y, k in zip(values, k2))
    k3 = tuple(rhs(time + 0.5 * dt, y3))
    y4 = tuple(y + dt * k for y, k in zip(values, k3))
    k4 = tuple(rhs(time + dt, y4))
    return tuple(
        y + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for y, a, b, c, d in zip(values, k1, k2, k3, k4)
    )
