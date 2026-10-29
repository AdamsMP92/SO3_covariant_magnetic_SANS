"""Numerical data for the two magnetic-SANS constructions.

The data products use dimensionless model parameters, ``qR``, normalized
correlation spectra, and normalized detector maps; no physical SI scaling is
implied.  Renderers and MaMMoS exports consume the same physics-level data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from magnetic_SANS_models import (
    AffineMagneticSANSModel,
    perpendicular_channel_maps,
)
from magnetization_models import (
    AffineMagnetizationModel,
    affine_linear_vortex,
    affine_vortex_pair,
    legendre_p4,
)
from model_validation import (
    validate_affine_vortex_pair,
    validate_linear_vortex_ensembles,
)


FloatArray = npt.NDArray[np.float64]

RADIUS = 1.0
M0 = 1.0
M1_R_OVER_M0 = 7.0
EPSILON = 0.9
Q_R_MAX = 5.0
SPECTRAL_SAMPLES = 601
RADIAL_BINS = 360
ANGULAR_BINS = 540
FIELD_GRID_POINTS = 26
FIELD_SUPPORT_FRACTION = 0.9

AFFINE_PAIR_M = 1.0
AFFINE_PAIR_KAPPA_R_OVER_M = 7.0
AFFINE_PAIR_ALPHA = np.pi / 6.0


@dataclass(frozen=True)
class RealSpaceFieldData:
    """Sampled body-frame magnetization field on the spherical support."""

    positions: FloatArray
    magnetization: FloatArray
    magnitude: FloatArray
    axial_fraction: FloatArray
    grid_points: int
    support_fraction: float


@dataclass(frozen=True)
class CorrelationSpectraData:
    """Three dimensionless correlation spectra on a common ``qR`` grid."""

    q_r: FloatArray
    gamma_perpendicular: FloatArray
    gamma_parallel: FloatArray
    gamma_cross: FloatArray


@dataclass(frozen=True)
class DetectorData:
    """Raw and linearly normalized magnetic SANS maps in perpendicular geometry."""

    q_r_edges: FloatArray
    theta_edges: FloatArray
    q_r_centers: FloatArray
    theta_centers: FloatArray
    q_z_r_edges: FloatArray
    q_y_r_edges: FloatArray
    spin_flip_raw: FloatArray
    chiral_raw: FloatArray
    spin_flip_display: FloatArray
    chiral_display: FloatArray
    spin_flip_scale: float
    chiral_scale: float


@dataclass(frozen=True)
class OrientationDistributionData:
    """Axisymmetric orientation density and its first two axis moments."""

    name: str
    formula: str
    z: FloatArray
    density: FloatArray
    first_moment: FloatArray
    second_moment: FloatArray


@dataclass(frozen=True)
class LinearVortexOrientationData:
    """Linear-vortex data with three observationally equivalent ensembles."""

    model: AffineMagnetizationModel
    field: RealSpaceFieldData
    orientations: tuple[OrientationDistributionData, ...]
    spectra: CorrelationSpectraData
    scattering: DetectorData
    validation: dict[str, object]


@dataclass(frozen=True)
class AffinePairData:
    """Numerical data for one member of the affine vortex pair."""

    name: str
    model: AffineMagnetizationModel
    field: RealSpaceFieldData
    spectra: CorrelationSpectraData
    scattering: DetectorData


@dataclass(frozen=True)
class AffineProjectionData:
    """Affine pair differing only in the longitudinal correlation sector."""

    circular: AffinePairData
    elliptic: AffinePairData
    validation: dict[str, object]


def sample_real_space_field(model: AffineMagnetizationModel) -> RealSpaceFieldData:
    """Sample the affine field on a masked Cartesian grid."""
    x_coord, y_coord, z_coord, m_x, m_y, m_z, magnitude = model.sample_field(
        grid_points=FIELD_GRID_POINTS,
        support_fraction=FIELD_SUPPORT_FRACTION,
    )
    return RealSpaceFieldData(
        positions=np.column_stack((x_coord, y_coord, z_coord)),
        magnetization=np.column_stack((m_x, m_y, m_z)),
        magnitude=magnitude,
        axial_fraction=m_z / magnitude,
        grid_points=FIELD_GRID_POINTS,
        support_fraction=FIELD_SUPPORT_FRACTION,
    )


def correlation_spectra_data(
    model: AffineMagneticSANSModel,
) -> CorrelationSpectraData:
    """Evaluate all three spectra on the common ``qR`` grid."""
    q_r = np.linspace(0.0, Q_R_MAX, SPECTRAL_SAMPLES)
    gamma_perpendicular, gamma_parallel, gamma_cross = model.correlation_spectra(q_r)
    return CorrelationSpectraData(
        q_r=q_r,
        gamma_perpendicular=gamma_perpendicular,
        gamma_parallel=gamma_parallel,
        gamma_cross=gamma_cross,
    )


def detector_data(model: AffineMagneticSANSModel) -> DetectorData:
    """Evaluate raw channels and their deterministic linear normalizations."""
    q_r_edges = np.linspace(0.0, Q_R_MAX, RADIAL_BINS + 1)
    theta_edges = np.linspace(-np.pi, np.pi, ANGULAR_BINS + 1)
    q_r_centers = 0.5 * (q_r_edges[:-1] + q_r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    q_r_grid, theta_grid = np.meshgrid(
        q_r_centers,
        theta_centers,
        indexing="xy",
    )
    gamma_perpendicular, gamma_cross = model.transverse_and_chiral_spectra(
        q_r_grid
    )
    spin_flip_raw, chiral_raw = perpendicular_channel_maps(
        gamma_perpendicular,
        gamma_cross,
        theta_grid,
    )
    spin_flip_scale = float(np.nanmax(spin_flip_raw))
    chiral_scale = float(np.nanmax(np.abs(chiral_raw)))

    q_r_edge_grid, theta_edge_grid = np.meshgrid(
        q_r_edges,
        theta_edges,
        indexing="xy",
    )
    q_z_r_edges = q_r_edge_grid * np.cos(theta_edge_grid)
    q_y_r_edges = q_r_edge_grid * np.sin(theta_edge_grid)
    return DetectorData(
        q_r_edges=q_r_edges,
        theta_edges=theta_edges,
        q_r_centers=q_r_centers,
        theta_centers=theta_centers,
        q_z_r_edges=q_z_r_edges,
        q_y_r_edges=q_y_r_edges,
        spin_flip_raw=spin_flip_raw,
        chiral_raw=chiral_raw,
        spin_flip_display=spin_flip_raw / spin_flip_scale,
        chiral_display=chiral_raw / chiral_scale,
        spin_flip_scale=spin_flip_scale,
        chiral_scale=chiral_scale,
    )


def orientation_density(name: str, z: npt.ArrayLike) -> FloatArray:
    """Evaluate one of the three axisymmetric densities relative to Haar measure."""
    z_array = np.asarray(z)
    if name == "uniform":
        return np.ones_like(z_array)
    if name == "hemisphere":
        return 2.0 * (z_array >= 0.0).astype(float)
    if name == "P4":
        return 1.0 + EPSILON * legendre_p4(z_array)
    raise ValueError(f"unknown orientation distribution: {name!r}")


def _orientation_data(
    validation: dict[str, object],
) -> tuple[OrientationDistributionData, ...]:
    z = np.linspace(-1.0, 1.0, 1001)
    formulas = {
        "uniform": "psi(R) = 1",
        "hemisphere": "psi_H(R) = 2 1_{a_z >= 0}",
        "P4": "psi_epsilon(R) = 1 + epsilon P4(a_z)",
    }
    return tuple(
        OrientationDistributionData(
            name=name,
            formula=formulas[name],
            z=z.copy(),
            density=orientation_density(name, z),
            first_moment=np.asarray(validation[name]["first_moment"], dtype=float),
            second_moment=np.asarray(validation[name]["second_moment"], dtype=float),
        )
        for name in ("uniform", "hemisphere", "P4")
    )


@lru_cache(maxsize=1)
def build_linear_vortex_orientation_data() -> LinearVortexOrientationData:
    """Build and cache the linear-vortex orientation data product."""
    model = affine_linear_vortex(
        radius=RADIUS,
        m0=M0,
        m1_r_over_m0=M1_R_OVER_M0,
    )
    sans_model = AffineMagneticSANSModel(model)
    validation = validate_linear_vortex_ensembles(sans_model, epsilon=EPSILON)
    return LinearVortexOrientationData(
        model=model,
        field=sample_real_space_field(model),
        orientations=_orientation_data(validation),
        spectra=correlation_spectra_data(sans_model),
        scattering=detector_data(sans_model),
        validation=validation,
    )


def _affine_pair_data(
    name: str,
    model: AffineMagnetizationModel,
) -> AffinePairData:
    sans_model = AffineMagneticSANSModel(model)
    return AffinePairData(
        name=name,
        model=model,
        field=sample_real_space_field(model),
        spectra=correlation_spectra_data(sans_model),
        scattering=detector_data(sans_model),
    )


@lru_cache(maxsize=1)
def build_affine_projection_data() -> AffineProjectionData:
    """Build and cache the affine longitudinal-null-space data product."""
    models = affine_vortex_pair(
        radius=RADIUS,
        m=AFFINE_PAIR_M,
        kappa_r_over_m=AFFINE_PAIR_KAPPA_R_OVER_M,
        alpha=AFFINE_PAIR_ALPHA,
    )
    sans_models = tuple(AffineMagneticSANSModel(model) for model in models)
    validation = validate_affine_vortex_pair(sans_models, q_r_max=Q_R_MAX)
    circular = _affine_pair_data("displaced_circular_vortex", models[0])
    elliptic = _affine_pair_data("centered_elliptic_vortex", models[1])
    np.testing.assert_allclose(
        circular.spectra.gamma_perpendicular,
        elliptic.spectra.gamma_perpendicular,
    )
    np.testing.assert_allclose(
        circular.spectra.gamma_cross,
        elliptic.spectra.gamma_cross,
    )
    np.testing.assert_allclose(
        circular.scattering.spin_flip_raw,
        elliptic.scattering.spin_flip_raw,
    )
    np.testing.assert_allclose(
        circular.scattering.chiral_raw,
        elliptic.scattering.chiral_raw,
    )
    return AffineProjectionData(
        circular=circular,
        elliptic=elliptic,
        validation=validation,
    )
