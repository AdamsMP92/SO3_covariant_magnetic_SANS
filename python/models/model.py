import numpy as np


def uniform_psi(alpha, beta, gamma):
    """Uniform orientation density relative to the normalized Haar measure."""
    return np.ones(np.broadcast(alpha, beta, gamma).shape)


class OrientationDistribution:
    """Named, Haar-normalized densities used in the paper."""

    def __init__(self, name, *, epsilon=0.9):
        self.name = name.lower()
        if self.name not in ("uniform", "hemisphere", "p4", "compensated"):
            raise ValueError("Unknown orientation distribution.")
        self.epsilon = float(epsilon)
        if self.name == "p4" and (not np.isfinite(self.epsilon) or abs(self.epsilon) > 1):
            raise ValueError("The paper's P4 family requires |epsilon| <= 1.")

    def __call__(self, alpha, beta, gamma):
        alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
        z = np.cos(beta)
        if self.name == "hemisphere":
            return 2.0 * (z >= 0)
        if self.name == "p4":
            return 1 + self.epsilon * (35*z**4 - 30*z**2 + 3) / 8
        if self.name == "compensated":
            return 3 * z**2
        return np.ones(alpha.shape)


class Model:
    """Affine parameters and an orientation density psi(alpha, beta, gamma).

    R is the sphere radius. psi must accept broadcastable angle arrays in
    radians and be nonnegative and normalized with respect to Haar measure:
    integral psi dQ = 1. The default psi is the uniform density 1.
    A string selects a named OrientationDistribution; custom callables remain
    supported. For a chosen P4 epsilon, pass OrientationDistribution("P4", epsilon=...).
    """

    def __init__(self, m0, m1, R=1.0, psi=None):
        if np.iscomplexobj(m0) or np.iscomplexobj(m1):
            raise ValueError("m0 and m1 must be real.")
        self.m0 = np.array(m0, dtype=float, copy=True)
        self.m1 = np.array(m1, dtype=float, copy=True)
        self.R = float(R)
        self.psi = uniform_psi if psi is None else (
            OrientationDistribution(psi) if isinstance(psi, str) else psi
        )

        if self.m0.shape != (3,) or self.m1.shape != (3, 3):
            raise ValueError("m0 and m1 must have shapes (3,) and (3, 3).")
        if not np.all(np.isfinite(self.m0)) or not np.all(np.isfinite(self.m1)):
            raise ValueError("m0 and m1 must be finite.")
        if not np.isfinite(self.R) or self.R <= 0:
            raise ValueError("R must be finite and positive.")
        if not callable(self.psi):
            raise TypeError("psi must be callable as psi(alpha, beta, gamma).")
