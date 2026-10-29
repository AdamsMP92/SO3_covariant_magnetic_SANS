"""One-point field means and volume-averaged magnetization in real space.

Analytical cases use the paper's beta-dependent Haar densities: uniform,
hemisphere (2 for cos(beta) >= 0, otherwise 0), P4 (1 + epsilon P4(cos(beta))),
and compensated (3 cos(beta)**2). alpha and gamma are uniform in all cases.
The P4 first and second axis moments are independent of epsilon.
"""

import numpy as np
from scipy.stats import qmc

from .global_affine_model import affine_magnetization


def axis_moments_analytical(distribution):
    """Return the first two vortex-axis moments of a named paper density."""
    distribution = distribution.lower()
    first = np.zeros(3)
    if distribution == "hemisphere":
        first[2] = 0.5
    elif distribution not in ("uniform", "p4", "compensated"):
        raise ValueError("distribution must be uniform, hemisphere, P4 or compensated.")
    second = (
        np.diag([1 / 5, 1 / 5, 3 / 5])
        if distribution == "compensated" else np.eye(3) / 3
    )
    return first, second


def _sobol_data(dimensions, power, seed, psi):
    samples = qmc.Sobol(d=dimensions, scramble=True, rng=seed).random_base2(power)
    alpha = 2 * np.pi * samples[:, 0]
    beta = np.arccos(1 - 2 * samples[:, 1])
    gamma = 2 * np.pi * samples[:, 2]
    weights = np.ones(len(samples)) if psi is None else np.asarray(psi(alpha, beta, gamma))
    if np.iscomplexobj(weights):
        raise ValueError("psi must be real.")
    weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(samples),))
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("psi must be finite and nonnegative at the sampled angles.")
    if not np.any(weights > 0):
        raise ValueError("psi has zero sampled mass; check the density or increase power.")
    return samples, alpha, beta, gamma, weights


def affine_magnetization_avg_analytical(x, m0, m1, R, *, distribution="uniform"):
    """Return the orientation-averaged interior field at x of shape (..., 3).

    Use |x| < R for the sphere-supported model. The beta-dependent densities
    listed above permit integration of the body spin gamma first, followed
    by the first two axis moments. For uniform and P4 this gives
    (trace(m1)/3) x. For a linear vortex, hemisphere gives
    m0_z e_z/2 + omega_z (e_z cross x)/2; uniform and P4 give zero.
    """
    x = np.asarray(x, dtype=float)
    if x.shape[-1:] != (3,):
        raise ValueError("x must have shape (..., 3).")
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    first, second = axis_moments_analytical(distribution)
    # Uniform body spin retains m0_z, the two diagonal weights, and omega_z.
    transverse = (m1[0, 0] + m1[1, 1]) / 2
    omega_z = (m1[1, 0] - m1[0, 1]) / 2
    return (
        m0[2] * first
        + transverse * x
        + (m1[2, 2] - transverse) * (second @ x[..., None])[..., 0]
        + omega_z * np.cross(first, x)
    )


def average_magnetization_analytical(m0, m1, R, *, distribution="uniform"):
    """Return the orientation- and sphere-volume-averaged magnetization.

    The linear spatial term integrates to zero on the centered sphere.
    The result is 0 for uniform, P4 and compensated, and m0_z e_z/2 for
    hemisphere. Multiply by 4*pi*R**3/3 to obtain the mean magnetic moment.
    """
    return affine_magnetization_avg_analytical(
        np.zeros(3), m0, m1, R, distribution=distribution
    )


def affine_magnetization_avg_sobol(
    x, m0, m1, R, *, psi=None, power=14, seed=0, batch_size=1024
):
    """Average the global interior field over orientations at fixed positions x.

    x has shape (..., 3); use |x| < R for the sphere-supported model.
    psi is a normalized Haar density, evaluated on broadcastable angle
    arrays. None means psi=1. The estimator is mean(psi * M(x; Q)).
    """
    x = np.asarray(x, dtype=float)
    if x.shape[-1:] != (3,):
        raise ValueError("x must have shape (..., 3).")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    samples, alpha, beta, gamma, weights = _sobol_data(3, power, seed, psi)
    n_samples = len(samples)
    result = np.zeros(x.shape, dtype=float)
    flat_result = result.reshape(-1, 3)

    for index, position in enumerate(x.reshape(-1, 3)):
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            field = affine_magnetization(
                position, m0, m1, R,
                alpha[start:stop], beta[start:stop], gamma[start:stop],
            )
            flat_result[index] += (weights[start:stop, None] * field).sum(axis=0)
    return result / n_samples


def average_magnetization_sobol(
    m0, m1, R, *, psi=None, power=16, seed=0, batch_size=1024
):
    """Estimate (1/V_s) integral_sphere integral_SO(3) M(x; Q) psi(Q) dQ dx.

    A six-dimensional Sobol sequence samples three Haar orientation
    coordinates and three spatial coordinates. Radius R*u**(1/3) and a
    uniform spatial direction give points uniformly distributed by volume.
    The estimator is mean(psi * M); psi must already be Haar-normalized.
    Multiply the result by V_s=4*pi*R**3/3 for the mean magnetic moment.
    """
    R = float(R)
    if not np.isfinite(R) or R <= 0:
        raise ValueError("R must be finite and positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    samples, alpha, beta, gamma, weights = _sobol_data(6, power, seed, psi)
    result = np.zeros(3)

    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        spatial = samples[start:stop, 3:]
        radius = R * np.cbrt(spatial[:, 0])
        cosine = 1 - 2 * spatial[:, 1]
        azimuth = 2 * np.pi * spatial[:, 2]
        sine = np.sqrt(1 - cosine**2)
        positions = radius[:, None] * np.stack([
            sine * np.cos(azimuth), sine * np.sin(azimuth), cosine,
        ], axis=-1)
        field = affine_magnetization(
            positions, m0, m1, R,
            alpha[start:stop], beta[start:stop], gamma[start:stop],
        )
        result += (weights[start:stop, None] * field).sum(axis=0)
    return result / len(samples)
