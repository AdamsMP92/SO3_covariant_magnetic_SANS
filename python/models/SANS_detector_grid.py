"""Detector coordinates: incident beam along x, theta measured from +z to +y."""

import numpy as np


def detector_grid(q, theta):
    """Return polar and Cartesian detector arrays with shape (n_theta, n_q).

    q and theta are strictly increasing one-dimensional axes, theta in
    radians. q must be positive because the transverse projection is used
    downstream. Coordinates are qz=q*cos(theta), qy=q*sin(theta), qx=0.
    """
    q, theta = np.asarray(q, dtype=float), np.asarray(theta, dtype=float)
    for name, axis in (("q", q), ("theta", theta)):
        if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} must be a finite, nonempty one-dimensional axis.")
        if np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} must be strictly increasing.")
    if np.any(q <= 0):
        raise ValueError("Detector q values must be positive; q=0 has no projection direction.")
    q_grid, theta_grid = np.meshgrid(q, theta, indexing="xy")
    qz, qy = q_grid*np.cos(theta_grid), q_grid*np.sin(theta_grid)
    return {
        "q_axis": q.copy(), "theta_axis": theta.copy(),
        "q": q_grid, "theta": theta_grid, "qz": qz, "qy": qy,
        "q_vectors": np.stack([np.zeros_like(qz), qy, qz], axis=-1),
    }
