"""Magnetic basis cross sections from G or its transverse and chiral spectra."""

import numpy as np

from .gamma_tensor_avg import gamma_spectra


def _unit_vector(vector, name):
    vector = np.asarray(vector, dtype=float)
    if vector.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3).")
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm == 0):
        raise ValueError(f"{name} must contain finite, nonzero vectors.")
    return vector / norm


def basic_cross_sections(G, P=(0.0, 0.0, 1.0), K=1.0):
    """Return I_M, I_P and I_chi by contracting a supplied projected G tensor.

    G has shape (..., 3, 3) and may originate from any orientation density.
    P is the spin-quantization direction and is normalized here; its leading
    dimensions broadcast with G. K multiplies the supplied G normalization;
    the default K=1 retains the normalized model scale.
    """
    G = np.asarray(G)
    if G.shape[-2:] != (3, 3):
        raise ValueError("G must have shape (..., 3, 3).")
    P = _unit_vector(P, "P")
    K = np.asarray(K, dtype=float)

    I_M = K * np.trace(G, axis1=-2, axis2=-1).real
    I_P = K * np.einsum("...i,...ij,...j->...", P, G, P).real
    chiral_vector = np.stack([
        G[..., 1, 2] - G[..., 2, 1],
        G[..., 2, 0] - G[..., 0, 2],
        G[..., 0, 1] - G[..., 1, 0],
    ], axis=-1)
    I_chi = K * (-1j * np.sum(P * chiral_vector, axis=-1)).real
    I_M, I_P, I_chi = np.broadcast_arrays(I_M, I_P, I_chi)
    return {"I_M": I_M, "I_P": I_P, "I_chi": I_chi}


def basic_cross_sections_from_spectra(
    q, gamma_perp, gamma_cross, P=(0.0, 0.0, 1.0), K=1.0
):
    """Return the basis cross sections for a joint SO(3)-covariant tensor.

    For mu = P_hat dot q_hat: I_M = 2 K Gamma_perp,
    I_P = K Gamma_perp (1-mu**2), I_chi = 2 K Gamma_cross mu.
    Gamma_parallel is removed by the transverse projection. q is nonzero
    with shape (..., 3); spectra, P and K broadcast over the leading axes.
    """
    q_hat = _unit_vector(q, "q")
    P = _unit_vector(P, "P")
    mu = np.sum(P * q_hat, axis=-1)
    mu, gamma_perp, gamma_cross, K = np.broadcast_arrays(
        mu, np.asarray(gamma_perp), np.asarray(gamma_cross), np.asarray(K)
    )
    return {
        "I_M": 2 * K * gamma_perp,
        "I_P": K * gamma_perp * (1 - mu**2),
        "I_chi": 2 * K * gamma_cross * mu,
    }


def basic_cross_sections_analytical(q, m0, m1, R, P=(0.0, 0.0, 1.0), K=1.0):
    """Use the analytical affine spectra for psi=1 to compute the basis channels."""
    q = np.asarray(q, dtype=float)
    gamma_perp, _, gamma_cross = gamma_spectra(
        np.linalg.norm(q, axis=-1), m0, m1, R
    )
    return basic_cross_sections_from_spectra(q, gamma_perp, gamma_cross, P, K)
