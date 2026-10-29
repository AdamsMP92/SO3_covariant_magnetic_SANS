"""Analytical magnetization models used by the Letter figures.

The functions and model classes in this module contain the real-space fields
and spherical form factor. Correlation spectra, detector geometry, and SANS
channel contractions are kept in ``magnetic_SANS_models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def legendre_p4(value: npt.ArrayLike) -> FloatArray:
    """Return the fourth Legendre polynomial with ``P4(1) = 1``."""
    # Preserve the input dtype because PyVista stores its surface coordinates
    # as float32; keeping that dtype reproduces the reference scalar rendering.
    value_array = np.asarray(value)
    return (35.0 * value_array**4 - 30.0 * value_array**2 + 3.0) / 8.0


def sphere_form_factor(qr: npt.ArrayLike) -> FloatArray:
    """Return ``3 [sin(qR) - qR cos(qR)] / (qR)^3`` at all ``qR``."""
    qr_array = np.asarray(qr, dtype=float)
    result = np.empty_like(qr_array)
    small = np.abs(qr_array) < 1.0e-2
    value = qr_array[small]
    result[small] = (
        1.0 - value**2 / 10.0 + value**4 / 280.0 - value**6 / 15120.0
    )
    value = qr_array[~small]
    result[~small] = 3.0 * (np.sin(value) - value * np.cos(value)) / value**3
    return result


def sphere_form_factor_derivative(qr: npt.ArrayLike) -> FloatArray:
    """Return the derivative of the normalized sphere form factor by ``qR``."""
    qr_array = np.asarray(qr, dtype=float)
    result = np.empty_like(qr_array)
    small = np.abs(qr_array) < 1.0e-2
    value = qr_array[small]
    result[small] = -value / 5.0 + value**3 / 70.0 - value**5 / 2520.0
    value = qr_array[~small]
    result[~small] = (
        3.0
        * ((value**2 - 3.0) * np.sin(value) + 3.0 * value * np.cos(value))
        / value**4
    )
    return result


def cross_matrix(vector: npt.ArrayLike) -> FloatArray:
    """Return the skew-symmetric matrix representing a cross product."""
    x_component, y_component, z_component = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z_component, y_component],
            [z_component, 0.0, -x_component],
            [-y_component, x_component, 0.0],
        ]
    )


@dataclass(frozen=True)
class AffineMagnetizationModel:
    """General affine magnetization field on a spherical support.

    The field inside the sphere is ``M_0(r') = m_0 + m_1 r'``. The matrix
    ``m_1`` retains its physical inverse-length factor; the radius is applied
    to the derivative of the form factor in the SANS forward model.
    """

    m0: npt.ArrayLike
    m1: npt.ArrayLike
    radius: float = 1.0

    def __post_init__(self) -> None:
        m0 = np.array(self.m0, dtype=float, copy=True)
        m1 = np.array(self.m1, dtype=float, copy=True)
        if m0.shape != (3,):
            raise ValueError(f"m0 must have shape (3,), got {m0.shape}")
        if m1.shape != (3, 3):
            raise ValueError(f"m1 must have shape (3, 3), got {m1.shape}")
        if not np.all(np.isfinite(m0)) or not np.all(np.isfinite(m1)):
            raise ValueError("m0 and m1 must contain only finite values")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        m0.setflags(write=False)
        m1.setflags(write=False)
        object.__setattr__(self, "m0", m0)
        object.__setattr__(self, "m1", m1)

    @property
    def zeta(self) -> float:
        """Return the scalar part of the first-order matrix."""
        return float(np.trace(self.m1) / 3.0)

    @property
    def symmetric_traceless(self) -> FloatArray:
        """Return the symmetric-traceless part of the first-order matrix."""
        return 0.5 * (self.m1 + self.m1.T) - self.zeta * np.eye(3)

    @property
    def omega(self) -> FloatArray:
        """Return the axial vector of the antisymmetric matrix part."""
        antisymmetric = 0.5 * (self.m1 - self.m1.T)
        return np.array(
            [
                antisymmetric[2, 1],
                antisymmetric[0, 2],
                antisymmetric[1, 0],
            ]
        )

    def reconstructed_first_order_matrix(self) -> FloatArray:
        """Reconstruct ``m1`` from its scalar, symmetric and axial parts."""
        return (
            self.zeta * np.eye(3)
            + self.symmetric_traceless
            + cross_matrix(self.omega)
        )

    def evaluate(self, positions: npt.ArrayLike) -> FloatArray:
        """Evaluate the affine field at one or more body-frame positions."""
        position_array = np.asarray(positions, dtype=float)
        if position_array.ndim == 0 or position_array.shape[-1] != 3:
            raise ValueError("positions must have a final dimension of length 3")
        return self.m0 + position_array @ self.m1.T

    def normalized_fourier_amplitude(
        self,
        q_vectors: npt.ArrayLike,
    ) -> ComplexArray:
        """Evaluate the normalized body-frame Fourier amplitude.

        The spherical shape transform is normalized to one at ``q=0``. With
        the unitary Fourier convention of the Letter, the omitted common
        factor is the unitary transform of the sphere indicator at the
        origin. This common factor cancels from every normalized quantity in
        the repository.
        """
        q_array = np.asarray(q_vectors, dtype=float)
        if q_array.ndim == 0 or q_array.shape[-1] != 3:
            raise ValueError("q_vectors must have a final dimension of length 3")
        q_norm = np.linalg.norm(q_array, axis=-1)
        q_r = q_norm * self.radius
        q_hat = np.zeros_like(q_array)
        np.divide(
            q_array,
            q_norm[..., None],
            out=q_hat,
            where=q_norm[..., None] > 0.0,
        )
        form_factor = sphere_form_factor(q_r)
        radius_weighted_derivative = (
            self.radius * sphere_form_factor_derivative(q_r)
        )
        first_order_direction = q_hat @ self.m1.T
        return (
            form_factor[..., None] * self.m0
            + 1j * radius_weighted_derivative[..., None] * first_order_direction
        )

    def sample_field(
        self,
        grid_points: int = 26,
        support_fraction: float = 0.9,
    ) -> tuple[FloatArray, ...]:
        """Sample the affine field on a Cartesian grid inside the sphere."""
        coordinates = np.linspace(-self.radius, self.radius, grid_points)
        x_coord, y_coord, z_coord = np.meshgrid(
            coordinates,
            coordinates,
            coordinates,
            indexing="ij",
        )
        inside = (
            x_coord**2 + y_coord**2 + z_coord**2
            <= (support_fraction * self.radius) ** 2
        )
        x_coord = x_coord[inside]
        y_coord = y_coord[inside]
        z_coord = z_coord[inside]
        magnetization = self.evaluate(np.column_stack((x_coord, y_coord, z_coord)))
        magnitude = np.linalg.norm(magnetization, axis=1)
        return (
            x_coord,
            y_coord,
            z_coord,
            magnetization[:, 0],
            magnetization[:, 1],
            magnetization[:, 2],
            magnitude,
        )


def affine_linear_vortex(
    *,
    radius: float = 1.0,
    m0: float = 1.0,
    m1_r_over_m0: float = 7.0,
) -> AffineMagnetizationModel:
    """Parametrize the general affine model as a linear vortex."""
    axis = np.array([0.0, 0.0, 1.0])
    circulation = m1_r_over_m0 * m0 / radius
    return AffineMagnetizationModel(
        m0=m0 * axis,
        m1=circulation * cross_matrix(axis),
        radius=radius,
    )


def affine_vortex_pair(
    *,
    radius: float = 1.0,
    m: float = 1.0,
    kappa_r_over_m: float = 7.0,
    alpha: float = np.pi / 6.0,
) -> tuple[AffineMagnetizationModel, AffineMagnetizationModel]:
    """Parametrize the displaced circular and centered elliptic affine fields."""
    kappa = kappa_r_over_m * m / radius
    cosine = np.cos(alpha)
    sine = np.sin(alpha)
    shear = np.sqrt(5.0 / 3.0) * kappa * sine
    circular = AffineMagnetizationModel(
        m0=m * np.array([sine, 0.0, cosine]),
        m1=kappa * cross_matrix([0.0, 0.0, 1.0]),
        radius=radius,
    )
    elliptic = AffineMagnetizationModel(
        m0=m * np.array([0.0, 0.0, 1.0]),
        m1=np.array(
            [
                [0.0, -kappa * cosine + shear, 0.0],
                [kappa * cosine + shear, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        radius=radius,
    )
    return circular, elliptic
