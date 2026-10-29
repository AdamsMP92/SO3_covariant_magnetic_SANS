"""Uniform analytical spectra and psi-weighted Sobol tensor averages."""

import numpy as np
from scipy.stats import qmc

from .gamma_tensor import gamma_tensor
from .local_affine_model import F, F_u
from .global_affine_model_avg import axis_moments_analytical


def is_axial_vortex(m0, m1):
    """Whether the coefficients exactly match the paper's body-z linear vortex."""
    m0, m1 = np.asarray(m0), np.asarray(m1)
    if m0.shape != (3,) or m1.shape != (3, 3):
        return False
    kappa = m1[1, 0]
    expected = np.array([[0, -kappa, 0], [kappa, 0, 0], [0, 0, 0]])
    return np.array_equal(m0[:2], np.zeros(2)) and np.array_equal(m1, expected)


def gamma_tensor_avg_vortex_analytical(q, m0, m1, R, *, distribution="uniform"):
    """Use the paper's T <a a.T> T.H factorization for an axial linear vortex.

    This also covers compensated axes, whose tensor is generally not jointly
    SO(3)-covariant and therefore is not represented by three radial spectra.
    """
    if not is_axial_vortex(m0, m1):
        raise ValueError("The axis-moment formula requires an axial linear vortex.")
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1)
    direction = np.divide(q, norm[..., None], out=np.zeros_like(q), where=norm[..., None] != 0)
    x, y, z = np.moveaxis(direction, -1, 0)
    zero = np.zeros_like(x)
    cross = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1)
    cross = cross.reshape(q.shape[:-1] + (3, 3))
    T = (
        np.asarray(m0)[2] * F(norm*R)[..., None, None] * np.eye(3)
        - 1j * np.asarray(m1)[1, 0] * R * F_u(norm*R)[..., None, None] * cross
    )
    _, second = axis_moments_analytical(distribution)
    return T @ second @ np.swapaxes(T.conj(), -1, -2)


def gamma_spectra(q, m0, m1, R):
    """Return (Gamma_perp, Gamma_parallel, Gamma_cross) for radial q >= 0.

    q can be a scalar or an array of magnitudes; m0 and m1 have shapes
    (3,) and (3, 3). The normalization follows the local Fourier model.
    """
    q = np.asarray(q, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    if np.any(q < 0):
        raise ValueError("q must contain nonnegative magnitudes.")

    zeta = np.trace(m1) / 3
    S = (m1 + m1.T) / 2 - zeta * np.eye(3)
    omega = np.array([
        m1[2, 1] - m1[1, 2],
        m1[0, 2] - m1[2, 0],
        m1[1, 0] - m1[0, 1],
    ]) / 2
    s_squared = np.sum(S**2)
    omega_squared = omega @ omega
    f = F(q * R)
    f_q = R * F_u(q * R)
    constant = (m0 @ m0) * f**2 / 3

    gamma_perp = constant + (s_squared / 10 + omega_squared / 3) * f_q**2
    gamma_parallel = constant + (zeta**2 + 2 * s_squared / 15) * f_q**2
    gamma_cross = (2 / 3) * (m0 @ omega) * f * f_q
    return gamma_perp, gamma_parallel, gamma_cross


def gamma_tensor_avg_analytical(q, m0, m1, R):
    """Reconstruct the full Haar-averaged tensor for q of shape (..., 3).

    Gamma = Gamma_perp P_perp + Gamma_parallel P_parallel
            - 1j Gamma_cross [q_hat]_cross.
    At q = 0 the continuous limit is |m0|**2 I / 3.
    """
    q = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q, axis=-1)
    gamma_perp, gamma_parallel, gamma_cross = gamma_spectra(q_norm, m0, m1, R)
    # At q = 0 this is only a placeholder; the directional terms vanish.
    q_hat = np.divide(
        q, q_norm[..., None], out=np.zeros_like(q), where=q_norm[..., None] != 0
    )
    longitudinal = q_hat[..., :, None] * q_hat[..., None, :]
    x, y, z = np.moveaxis(q_hat, -1, 0)
    zero = np.zeros_like(x)
    cross = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(q.shape[:-1] + (3, 3))
    return (
        gamma_perp[..., None, None] * np.eye(3)
        + (gamma_parallel - gamma_perp)[..., None, None] * longitudinal
        - 1j * gamma_cross[..., None, None] * cross
    )


def gamma_tensor_avg_sobol(q, m0, m1, R, *, psi=None, power=14, seed=0, batch_size=1024):
    """Average gamma_tensor with 2**power scrambled Sobol orientations.

    q has shape (..., 3); m0 and m1 are fixed, with shapes (3,) and (3, 3).
    alpha = 2*pi*u0, cos(beta) = 1-2*u1, gamma = 2*pi*u2 sample the
    normalized Haar measure. psi(alpha, beta, gamma) is a nonnegative
    density with integral psi dQ = 1; None means the uniform density 1.
    The integral is estimated as mean(psi * Gamma), retaining the supplied
    density normalization.
    seed makes the scramble reproducible. Orientations are evaluated in
    blocks of batch_size, processing one q vector at a time.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[-1:] != (3,):
        raise ValueError("q must have shape (..., 3).")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    samples = qmc.Sobol(d=3, scramble=True, rng=seed).random_base2(power)
    alpha = 2 * np.pi * samples[:, 0]
    beta = np.arccos(1 - 2 * samples[:, 1])
    gamma = 2 * np.pi * samples[:, 2]
    n_samples = len(samples)
    weights = np.ones(n_samples) if psi is None else np.asarray(psi(alpha, beta, gamma))
    if np.iscomplexobj(weights):
        raise ValueError("psi must be real.")
    weights = np.broadcast_to(np.asarray(weights, dtype=float), (n_samples,))
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("psi must be finite and nonnegative at the sampled angles.")
    if not np.any(weights > 0):
        raise ValueError("psi has zero sampled mass; check the density or increase power.")
    result = np.zeros(q.shape[:-1] + (3, 3), dtype=complex)
    flat_result = result.reshape(-1, 3, 3)

    for index, q_vector in enumerate(q.reshape(-1, 3)):
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            tensors = gamma_tensor(
                q_vector, m0, m1, R,
                alpha[start:stop], beta[start:stop], gamma[start:stop],
            )
            flat_result[index] += (weights[start:stop, None, None] * tensors).sum(axis=0)
    return result / n_samples
