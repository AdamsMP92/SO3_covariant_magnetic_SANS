import numpy as np

from . import local_affine_model as local
from .rotation_matrix import zyzMat


def affine_magnetization(x, m0, m1, R, alpha, beta, gamma):
    """Global field Q M0(Q.T x), with x of shape (..., 3) and angles in radians."""
    Q = zyzMat(alpha, beta, gamma)
    x_local = (np.swapaxes(Q, -1, -2) @ np.asarray(x)[..., None])[..., 0]
    m_local = local.affine_magnetization(x_local, m0, m1, R)
    return (Q @ m_local[..., None])[..., 0]


def affine_magnetization_fourier(q, m0, m1, R, alpha, beta, gamma):
    """Global Fourier amplitude Q M0_tilde(Q.T q), retaining local normalization."""
    Q = zyzMat(alpha, beta, gamma)
    q_local = (np.swapaxes(Q, -1, -2) @ np.asarray(q)[..., None])[..., 0]
    m_local = local.affine_magnetization_fourier(q_local, m0, m1, R)
    return (Q @ m_local[..., None])[..., 0]
