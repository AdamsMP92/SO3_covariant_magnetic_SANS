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
