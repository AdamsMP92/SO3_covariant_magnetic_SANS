"""Magnetic SANS mappings used by the Letter figures.

This module maps the fields supplied by ``magnetization_models.py`` to their
analytical correlation spectra, detector geometry, and magnetic SANS channel
contractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from magnetization_models import (
    AffineMagnetizationModel,
    sphere_form_factor,
    sphere_form_factor_derivative,
)


FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class MagneticChannelData:
    """Purely magnetic polarization channels obtained from ``G``."""

    I_plus_minus: FloatArray
    I_minus_plus: FloatArray
    I_plus_plus: FloatArray
    I_minus_minus: FloatArray
    I_plus: FloatArray
    I_minus: FloatArray
    I_unpolarized: FloatArray
    I_sf: FloatArray
    I_chi: FloatArray


class TransverseChiralSpectra(Protocol):
    """Protocol for a model supplying the spectra visible to magnetic SANS."""

    def transverse_and_chiral_spectra(
        self,
        qr: npt.ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``Gamma_perp`` and ``Gamma_cross`` at ``qR``."""


@dataclass(frozen=True)
class AffineMagneticSANSModel:
    """Analytical powder-averaged SANS model of a general affine field."""

    magnetization_model: AffineMagnetizationModel

    def correlation_spectra(
        self,
        qr: npt.ArrayLike,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``Gamma_perp``, ``Gamma_parallel`` and ``Gamma_cross``."""
        form_factor = sphere_form_factor(qr)
        radius_weighted_derivative = (
            self.magnetization_model.radius * sphere_form_factor_derivative(qr)
        )
        m0 = self.magnetization_model.m0
        zeta = self.magnetization_model.zeta
        symmetric = self.magnetization_model.symmetric_traceless
        omega = self.magnetization_model.omega
        m0_squared = float(m0 @ m0)
        symmetric_norm_squared = float(np.sum(symmetric**2))
        omega_squared = float(omega @ omega)

        gamma_perpendicular = (
            m0_squared * form_factor**2 / 3.0
            + (symmetric_norm_squared / 10.0 + omega_squared / 3.0)
            * radius_weighted_derivative**2
        )
        gamma_parallel = (
            m0_squared * form_factor**2 / 3.0
            + (zeta**2 + 2.0 * symmetric_norm_squared / 15.0)
            * radius_weighted_derivative**2
        )
        gamma_cross = (
            2.0
            * float(m0 @ omega)
            * form_factor
            * radius_weighted_derivative
            / 3.0
        )
        return gamma_perpendicular, gamma_parallel, gamma_cross

    def transverse_and_chiral_spectra(
        self,
        qr: npt.ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the two spectra retained by magnetic SANS."""
        gamma_perpendicular, _, gamma_cross = self.correlation_spectra(qr)
        return gamma_perpendicular, gamma_cross


def transverse_projector(q_vector: npt.ArrayLike) -> FloatArray:
    """Return the projector perpendicular to a nonzero scattering vector."""
    q_array = np.asarray(q_vector, dtype=float)
    if q_array.shape != (3,):
        raise ValueError(f"q_vector must have shape (3,), got {q_array.shape}")
    q_norm = float(np.linalg.norm(q_array))
    if q_norm == 0.0:
        raise ValueError("the transverse projector is undefined at q=0")
    q_hat = q_array / q_norm
    return np.eye(3) - np.outer(q_hat, q_hat)


def halpern_johnson_projection(
    correlation_tensor: npt.ArrayLike,
    q_vector: npt.ArrayLike,
) -> ComplexArray:
    """Apply ``Pi_perp Gamma Pi_perp`` to a magnetic correlation tensor."""
    tensor = np.asarray(correlation_tensor, dtype=complex)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("correlation_tensor must end in dimensions (3, 3)")
    projector = transverse_projector(q_vector)
    return np.einsum("ij,...jk,kl->...il", projector, tensor, projector)


def magnetic_channel_contractions(
    projected_tensor: npt.ArrayLike,
    polarization: npt.ArrayLike,
    *,
    prefactor: float = 1.0,
) -> MagneticChannelData:
    """Contract ``G`` into the purely magnetic SANS channels.

    Nuclear and spin-incoherent terms are intentionally absent. The returned
    quantities use the channel convention of Eq. (8) in the Letter.
    """
    tensor = np.asarray(projected_tensor, dtype=complex)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("projected_tensor must end in dimensions (3, 3)")
    polarization_array = np.asarray(polarization, dtype=float)
    if polarization_array.shape != (3,):
        raise ValueError("polarization must have shape (3,)")
    polarization_norm = float(np.linalg.norm(polarization_array))
    if polarization_norm == 0.0:
        raise ValueError("polarization must be nonzero")
    p_hat = polarization_array / polarization_norm
    levi_civita = np.zeros((3, 3, 3))
    levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1.0
    levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1.0

    trace = np.trace(tensor, axis1=-2, axis2=-1)
    non_spin_flip = np.einsum("i,...ij,j->...", p_hat, tensor, p_hat)
    spin_flip = trace - non_spin_flip
    chiral = -1j * np.einsum(
        "i,ijk,...jk->...",
        p_hat,
        levi_civita,
        tensor,
    )

    def real_part(value: npt.ArrayLike) -> FloatArray:
        value_array = np.asarray(value)
        if np.max(np.abs(np.imag(value_array)), initial=0.0) > 1.0e-11:
            raise ValueError(
                "a channel contraction has a non-negligible imaginary part"
            )
        return prefactor * np.real(value_array)

    spin_flip_real = real_part(spin_flip)
    chiral_real = real_part(chiral)
    non_spin_flip_real = real_part(non_spin_flip)
    trace_real = real_part(trace)
    return MagneticChannelData(
        I_plus_minus=spin_flip_real + chiral_real,
        I_minus_plus=spin_flip_real - chiral_real,
        I_plus_plus=non_spin_flip_real,
        I_minus_minus=non_spin_flip_real,
        I_plus=trace_real + chiral_real,
        I_minus=trace_real - chiral_real,
        I_unpolarized=trace_real,
        I_sf=spin_flip_real,
        I_chi=chiral_real,
    )


def perpendicular_channel_maps(
    gamma_perpendicular: npt.ArrayLike,
    gamma_cross: npt.ArrayLike,
    theta: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return the SF mean and chiral half-difference in perpendicular geometry."""
    gamma_perpendicular_array = np.asarray(gamma_perpendicular, dtype=float)
    gamma_cross_array = np.asarray(gamma_cross, dtype=float)
    theta_array = np.asarray(theta, dtype=float)
    spin_flip = gamma_perpendicular_array * (1.0 + np.cos(theta_array) ** 2)
    chiral = 2.0 * gamma_cross_array * np.cos(theta_array)
    return spin_flip, chiral


def perpendicular_detector_grid(
    model: TransverseChiralSpectra,
    *,
    q_r_max: float = 5.0,
    radial_bins: int = 360,
    angular_bins: int = 540,
    normalize: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return detector edges and the SF and chiral maps of a model."""
    q_edges = np.linspace(0.0, q_r_max, radial_bins + 1)
    theta_edges = np.linspace(-np.pi, np.pi, angular_bins + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    q_r, theta = np.meshgrid(q_centers, theta_centers, indexing="xy")
    gamma_perpendicular, gamma_cross = model.transverse_and_chiral_spectra(q_r)
    spin_flip, chiral = perpendicular_channel_maps(
        gamma_perpendicular,
        gamma_cross,
        theta,
    )
    if normalize:
        spin_flip = spin_flip / np.nanmax(spin_flip)
        chiral = chiral / np.nanmax(np.abs(chiral))

    q_r_edges, theta_grid_edges = np.meshgrid(
        q_edges,
        theta_edges,
        indexing="xy",
    )
    q_z_r = q_r_edges * np.cos(theta_grid_edges)
    q_y_r = q_r_edges * np.sin(theta_grid_edges)
    return q_z_r, q_y_r, spin_flip, chiral
