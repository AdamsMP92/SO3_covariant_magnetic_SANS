"""Transverse projection of an averaged Gamma tensor and its spectral form."""

import numpy as np

from .gamma_tensor_avg import gamma_spectra


def transverse_projector(q):
    """Return P_perp = I - q_hat q_hat.T for nonzero q of shape (..., 3)."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1:] != (3,):
        raise ValueError("q must have shape (..., 3).")
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q_norm)) or np.any(q_norm == 0):
        raise ValueError("Transverse projection requires finite, nonzero q vectors.")
    q_hat = q / q_norm
    return np.eye(3) - q_hat[..., :, None] * q_hat[..., None, :]


def G_tensor_avg(q, Gamma):
    """Project a supplied averaged Gamma tensor: G = P_perp @ Gamma @ P_perp.

    Gamma has shape (..., 3, 3) and may come from any orientation density.
    Its leading dimensions broadcast with those of q. Gamma is used as
    supplied; no orientational average is recomputed here.
    """
    projector = transverse_projector(q)
    Gamma = np.asarray(Gamma)
    if Gamma.shape[-2:] != (3, 3):
        raise ValueError("Gamma must have shape (..., 3, 3).")
    return projector @ Gamma @ projector


def G_tensor_avg_from_spectra(q, gamma_perp, gamma_cross):
    """Construct G = Gamma_perp P_perp - 1j Gamma_cross [q_hat]_cross.

    This spectral form assumes a joint SO(3)-covariant averaged tensor.
    Only the transverse and chiral spectra are needed; Gamma_parallel
    is removed by the projection. Spectra broadcast with q's leading axes.
    """
    q = np.asarray(q, dtype=float)
    projector = transverse_projector(q)
    q_hat = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z = np.moveaxis(q_hat, -1, 0)
    zero = np.zeros_like(x)
    cross = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(q.shape[:-1] + (3, 3))
    gamma_perp = np.asarray(gamma_perp)
    gamma_cross = np.asarray(gamma_cross)
    return (
        gamma_perp[..., None, None] * projector
        - 1j * gamma_cross[..., None, None] * cross
    )


def G_tensor_avg_analytical(q, m0, m1, R):
    """Construct G from the analytical affine spectra for psi = 1 and |q| > 0."""
    q = np.asarray(q, dtype=float)
    gamma_perp, _, gamma_cross = gamma_spectra(
        np.linalg.norm(q, axis=-1), m0, m1, R
    )
    return G_tensor_avg_from_spectra(q, gamma_perp, gamma_cross)
