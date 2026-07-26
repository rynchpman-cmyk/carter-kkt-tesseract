"""Theory 2.0 radiative and outflow boundary operators."""

from __future__ import annotations

import numpy as np

from .grid import Array


def sommerfeld_rhs(
    field: Array,
    radial_derivative: Array,
    radius: Array,
    asymptotic_value: Array | float = 0.0,
    characteristic_speed: float = 1.0,
) -> Array:
    """Return the outgoing Sommerfeld time derivative at a boundary."""
    if characteristic_speed <= 0.0:
        raise ValueError("characteristic_speed must be positive")
    safe_radius = np.maximum(np.asarray(radius, dtype=float), 1.0e-15)
    return -characteristic_speed * (
        radial_derivative + (field - asymptotic_value) / safe_radius
    )


def constant_outflow_ghost(interior_edge: Array, ghost_width: int) -> Array:
    """Constant-extrapolation ghost cells for fluid, sigma, or Astar."""
    if ghost_width < 1:
        raise ValueError("ghost_width must be positive")
    return np.repeat(np.expand_dims(interior_edge, axis=-1), ghost_width, axis=-1)


def prevent_inflow(normal_velocity: Array, outward_sign: int) -> Array:
    """Clamp the inward part of a boundary-normal coordinate velocity."""
    if outward_sign not in {-1, 1}:
        raise ValueError("outward_sign must be -1 or +1")
    velocity = np.asarray(normal_velocity, dtype=float).copy()
    inward = outward_sign * velocity < 0.0
    velocity[inward] = 0.0
    return velocity


def zero_incoming_characteristic(outgoing: Array, incoming: Array) -> tuple[Array, Array]:
    """Preserve the outgoing characteristic and remove incoming radiation."""
    return np.asarray(outgoing, dtype=float), np.zeros_like(incoming, dtype=float)


class RadiativeBoundary:
    """Cell-centered radiative boundary projection for isolated Cartesian patches."""

    def __init__(self, grid, center: tuple[float, ...] | None = None) -> None:
        self.grid = grid
        self.center = center or tuple(0.0 for _ in range(grid.ndim))
        if len(self.center) != grid.ndim:
            raise ValueError("boundary center must match the grid dimension")
        coordinates = grid.coordinates()
        radius_sq = np.zeros(grid.shape)
        self.radial_directions = []
        for coordinate, center_value in zip(coordinates, self.center):
            displacement = coordinate - center_value
            self.radial_directions.append(displacement)
            radius_sq += displacement**2
        self.radius = np.sqrt(np.maximum(radius_sq, 1.0e-30))
        self.radial_directions = [value / self.radius for value in self.radial_directions]
        mask = np.zeros(grid.shape, dtype=bool)
        for axis in range(grid.ndim):
            lower = [slice(None)] * grid.ndim
            upper = [slice(None)] * grid.ndim
            lower[axis] = 0
            upper[axis] = -1
            mask[tuple(lower)] = True
            mask[tuple(upper)] = True
        self.mask = mask

    def radiative_rhs(
        self,
        field: Array,
        current_rhs: Array,
        asymptotic_value: Array | float,
        speed: float = 1.0,
    ) -> Array:
        if speed <= 0.0:
            raise ValueError("radiative speed must be positive")
        field = np.asarray(field, dtype=float)
        result = np.array(current_rhs, copy=True)
        leading = field.ndim - self.grid.ndim
        radial_derivative = np.zeros_like(field)
        for axis in range(self.grid.ndim):
            direction = self.radial_directions[axis].reshape(
                (1,) * leading + self.grid.shape
            )
            radial_derivative += direction * self.grid.derivative(field, axis)
        radius = self.radius.reshape((1,) * leading + self.grid.shape)
        target = -speed * (
            radial_derivative + (field - np.asarray(asymptotic_value)) / radius
        )
        mask = self.mask.reshape((1,) * leading + self.grid.shape)
        return np.where(mask, target, result)

    def outflow_rhs(self, current_rhs: Array) -> Array:
        """Extrapolate interior RHS to boundary cells without injecting data."""
        result = np.array(current_rhs, copy=True)
        for axis in range(self.grid.ndim):
            array_axis = result.ndim - self.grid.ndim + axis
            lower = [slice(None)] * result.ndim
            lower_source = [slice(None)] * result.ndim
            upper = [slice(None)] * result.ndim
            upper_source = [slice(None)] * result.ndim
            lower[array_axis] = 0
            lower_source[array_axis] = 1
            upper[array_axis] = -1
            upper_source[array_axis] = -2
            result[tuple(lower)] = result[tuple(lower_source)]
            result[tuple(upper)] = result[tuple(upper_source)]
        return result

    def ccz4_rhs(self, state, rhs: tuple[Array, ...]) -> tuple[Array, ...]:
        identity = self.grid.zeros((3, 3))
        for i in range(3):
            identity[i, i] = 1.0
        asymptotic = (
            identity,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        speeds = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.sqrt(2.0), 1.0, 1.0)
        fields = (
            state.conformal_metric,
            state.conformal_A,
            state.conformal_factor,
            state.trace_K,
            state.theta,
            state.gamma_hat,
            state.lapse,
            state.shift,
            state.shift_driver,
        )
        projected = tuple(
            self.radiative_rhs(field, derivative, infinity, speed)
            for field, derivative, infinity, speed in zip(fields, rhs, asymptotic, speeds)
        )
        # Linearized Z4 constraint characteristic projection.  With outward
        # normal n_i, C_in=Theta+Z_n is the incoming constraint mode in this
        # convention.  Preserve C_out=Theta-Z_n and damp only C_in.  Z_i is
        # reconstructed from GammaHat^i-GammaTilde^i; at principal order its
        # RHS is one half gt_ij dGammaHat^j.
        values = list(projected)
        values[4], values[5] = self.z4_constraint_rhs(
            state, values[4], values[5]
        )
        return tuple(values)

    def z4_constraint_rhs(
        self,
        state,
        theta_rhs: Array,
        gamma_hat_rhs: Array,
        damping: float = 1.0,
    ) -> tuple[Array, Array]:
        """Project the incoming linearized Z4 constraint characteristic."""
        if damping < 0.0:
            raise ValueError("constraint damping must be nonnegative")
        metric = state.conformal_metric
        matrices = np.moveaxis(metric, (0, 1), (-2, -1))
        metric_inv = np.moveaxis(np.linalg.inv(matrices), (-2, -1), (0, 1))
        gamma_tilde = np.zeros_like(state.gamma_hat)
        # This local connection calculation avoids importing the CCZ4 solver
        # and therefore keeps the boundary module dependency one-way.
        for i in range(3):
            # Standard contracted-connection identity for det(gt)=1.
            for j in range(self.grid.ndim):
                gamma_tilde[i] -= self.grid.derivative(metric_inv[i, j], j)
        Z_down = 0.5 * np.einsum(
            "ij...,j...->i...", metric, state.gamma_hat - gamma_tilde
        )
        leading_mask = self.mask
        normal = np.zeros_like(state.gamma_hat)
        for axis in range(self.grid.ndim):
            normal[axis] = self.radial_directions[axis]
        Z_normal = np.einsum("i...,i...->...", normal, Z_down)
        dZ_normal = 0.5 * np.einsum(
            "i...,ij...,j...->...", normal, metric, gamma_hat_rhs
        )
        outgoing_rhs = theta_rhs - dZ_normal
        incoming_rhs = -damping * (state.theta + Z_normal)
        target_theta_rhs = 0.5 * (outgoing_rhs + incoming_rhs)
        target_dZ_normal = 0.5 * (incoming_rhs - outgoing_rhs)
        delta = target_dZ_normal - dZ_normal
        gamma_correction = 2.0 * np.einsum(
            "ij...,j...->i...", metric_inv, normal * delta[None, ...]
        )
        theta_result = np.where(leading_mask, target_theta_rhs, theta_rhs)
        gamma_result = np.where(
            leading_mask[None, ...], gamma_hat_rhs + gamma_correction, gamma_hat_rhs
        )
        return theta_result, gamma_result


def zero_incoming_linear_flux(
    matrix: Array, state: Array, outward_sign: int
) -> Array:
    """Return a boundary flux with incoming modes set to homogeneous data.

    ``matrix`` has shape ``(m,m,...)`` and ``state`` has shape ``(m,...)``.
    A mode is incoming when its coordinate speed dotted with the outward
    normal is negative.
    """
    if outward_sign not in {-1, 1}:
        raise ValueError("outward_sign must be -1 or +1")
    matrix = np.asarray(matrix, dtype=float)
    state = np.asarray(state, dtype=float)
    if matrix.shape[:2] != (state.shape[0], state.shape[0]):
        raise ValueError("characteristic matrix and state dimensions disagree")
    moved_matrix = np.moveaxis(matrix, (0, 1), (-2, -1))
    moved_state = np.moveaxis(state, 0, -1)
    eigenvalues, eigenvectors = np.linalg.eig(moved_matrix)
    inverse = np.linalg.inv(eigenvectors)
    modes = np.einsum("...ij,...j->...i", inverse, moved_state)
    incoming = outward_sign * eigenvalues.real < 0.0
    modes = np.where(incoming, 0.0, modes)
    flux = np.einsum(
        "...ij,...j->...i", eigenvectors, eigenvalues * modes
    ).real
    return np.moveaxis(flux, -1, 0)


class Theory3CharacteristicBoundary(RadiativeBoundary):
    """Outer-boundary operators for the enlarged Theory 3 state."""

    def mixed_proca_rhs(
        self,
        system,
        h: Array,
        state,
        field_rhs: tuple[Array, ...],
    ) -> tuple[Array, ...]:
        """Radiate both Proca pairs while preserving semidiscrete Gauss."""
        (
            da,
            dpi_A,
            db,
            dpi_B,
            dlong_A,
            dlong_B,
            dclean_A,
            dclean_B,
        ) = field_rhs
        new_da = self.radiative_rhs(state.a, da, 0.0, 1.0)
        new_db = self.radiative_rhs(state.b, db, 0.0, 1.0)
        new_dpi_A = self.radiative_rhs(state.pi_A, dpi_A, 0.0, 1.0)
        new_dpi_B = self.radiative_rhs(state.pi_B, dpi_B, 0.0, 1.0)
        sqrt_h = np.sqrt(
            np.maximum(
                np.linalg.det(np.moveaxis(h, (0, 1), (-2, -1))), 1.0e-300
            )
        )
        delta_E_A = -(new_dpi_A - dpi_A) / (
            system.parameters.Z_A * sqrt_h
        )[None, ...]
        delta_E_B = -(new_dpi_B - dpi_B) / (
            system.parameters.Z_B * sqrt_h
        )[None, ...]
        # The correction reaches the first interior stencil point as required
        # by the one-sided divergence operator.
        new_dlong_A = dlong_A - self.grid.divergence(
            sqrt_h[None, ...] * delta_E_A
        )
        new_dlong_B = dlong_B - self.grid.divergence(
            sqrt_h[None, ...] * delta_E_B
        )
        new_dclean_A = self.radiative_rhs(
            state.cleaning_A, dclean_A, 0.0, 1.0
        )
        new_dclean_B = self.radiative_rhs(
            state.cleaning_B, dclean_B, 0.0, 1.0
        )
        return (
            new_da,
            new_dpi_A,
            new_db,
            new_dpi_B,
            new_dlong_A,
            new_dlong_B,
            new_dclean_A,
            new_dclean_B,
        )

    def target_boundary_flux(
        self,
        matrix: Array,
        charge: Array,
        normal_current: Array,
        axis: int,
        side: int,
    ) -> tuple[Array, Array]:
        array_axis = charge.ndim - self.grid.ndim + axis
        index = 0 if side < 0 else -1
        face = [slice(None)] * charge.ndim
        face[array_axis] = index
        matrix_face = [slice(None), slice(None)] + face
        state_face = np.stack(
            (charge[tuple(face)], normal_current[tuple(face)]), axis=0
        )
        flux = zero_incoming_linear_flux(
            matrix[tuple(matrix_face)], state_face, side
        )
        return flux[0], flux[1]
