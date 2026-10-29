import numpy as np

def zyzMat(alpha, beta, gamma):
    """Return Rz(alpha) @ Ry(beta) @ Rz(gamma), with angles in radians.

    Broadcast angle arrays to a common shape; the result has shape (..., 3, 3).
    """
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    return np.stack(
        [
            ca * cb * cg - sa * sg,
            -ca * cb * sg - sa * cg,
            ca * sb,
            sa * cb * cg + ca * sg,
            -sa * cb * sg + ca * cg,
            sa * sb,
            -sb * cg,
            sb * sg,
            cb,
        ],
        axis=-1,
    ).reshape(alpha.shape + (3, 3))
