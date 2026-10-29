import numpy as np

def affine_magnetization(x, m0, m1, R):
    """Local affine field for positions x of shape (..., 3)."""
    return m0 + (m1 @ np.asarray(x)[..., None])[..., 0]


def F(u):
    """Normalized sphere form factor 3(sin(u) - u cos(u))/u**3; u = |q|R."""
    u = np.asarray(u, dtype=float)
    result = np.empty_like(u)
    small = np.abs(u) < 0.1

    # Taylor expansion avoids cancellation near u = 0 and gives F(0) = 1.
    v = u[small]
    result[small] = 1 - v**2 / 10 + v**4 / 280 - v**6 / 15120 + v**8 / 1330560
    v = u[~small]
    result[~small] = 3 * (np.sin(v) - v * np.cos(v)) / v**3
    return result


def F_u(u):
    """Derivative dF/du, with F_u(0) = 0."""
    u = np.asarray(u, dtype=float)
    result = np.empty_like(u)
    small = np.abs(u) < 0.1

    v = u[small]
    result[small] = -v / 5 + v**3 / 70 - v**5 / 2520 + v**7 / 166320
    v = u[~small]
    result[~small] = 3 * ((v**2 - 3) * np.sin(v) + 3 * v * np.cos(v)) / v**4
    return result


def affine_magnetization_fourier(q, m0, m1, R):
    """Volume-normalized Fourier amplitude for q of shape (..., 3); R is the radius.

    Multiply by (4*pi*R**3/3)/(2*pi)**1.5 for the unitary Fourier convention.
    """
    q = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    qR = q_norm * R
    q_hat = np.divide(q, q_norm, out=np.zeros_like(q), where=q_norm != 0)
    m1_q_hat = (m1 @ q_hat[..., None])[..., 0]
    return m0 * F(qR) + 1j * m1_q_hat * R * F_u(qR)


