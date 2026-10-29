from .global_affine_model import affine_magnetization_fourier


def gamma_tensor(q, m0, m1, R, alpha, beta, gamma):
    """Unaveraged tensor Gamma_ij = M_i M_j*, with q of shape (..., 3).

    Angles are in radians and broadcast with q as in the global model.
    The tensor retains the squared normalization of the local Fourier model.
    """
    m = affine_magnetization_fourier(q, m0, m1, R, alpha, beta, gamma)
    return m[..., :, None] * m.conj()[..., None, :]
