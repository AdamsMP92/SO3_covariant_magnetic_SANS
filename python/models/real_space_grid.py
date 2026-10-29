"""Cartesian point coordinates inside a sphere centered at the origin."""

import numpy as np

def real_space_grid(x, y, z, R):
    """Return x, y, z as equally sized 1D arrays with x**2+y**2+z**2 <= R**2.

    Inputs x, y, z are finite, strictly increasing one-dimensional axes.
    Their Cartesian product uses indexing="ij" before the sphere cut.
    Surface points are included within floating-point roundoff.
    """
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive.")
    axes = [np.asarray(axis, dtype=float) for axis in (x, y, z)]
    for name, axis in zip("xyz", axes):
        if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} must be a finite, nonempty one-dimensional axis.")
        if np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} must be strictly increasing.")

    x_grid, y_grid, z_grid = np.meshgrid(*axes, indexing="ij")
    radius_squared = (x_grid / R)**2 + (y_grid / R)**2 + (z_grid / R)**2
    inside = radius_squared <= 1 + 8 * np.finfo(float).eps
    return {"x": x_grid[inside], "y": y_grid[inside], "z": z_grid[inside]}


def regular_real_space_grid(R=1.0, grid_points=21):
    """Sample [-R, R] with grid_points points per axis, then apply the sphere cut."""
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive.")
    if not isinstance(grid_points, (int, np.integer)) or grid_points < 3:
        raise ValueError("grid_points must be an integer of at least 3.")
    axis = np.linspace(-R, R, grid_points)
    return real_space_grid(axis, axis, axis, R)
