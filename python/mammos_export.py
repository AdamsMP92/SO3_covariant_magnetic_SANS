"""MaMMoS serialization of the numerical data underlying the Letter figures."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import mammos_entity as me

from figure_data import (
    AffinePairCaseData,
    CorrelationSpectraData,
    DetectorData,
    Figure1Data,
    Figure2Data,
    OrientationDistributionData,
    RealSpaceFieldData,
    build_figure_1_data,
    build_figure_2_data,
)
from magnetization_models import AffineMagnetizationModel


SCHEMA_VERSION = "1.1"


def _quantity(
    ontology_label: str,
    value: object,
    description: str,
) -> me.Entity:
    """Return a dimensionless MaMMoS quantity with explicit semantics."""
    return me.Entity(
        ontology_label,
        value=value,
        description=description,
    )


def _git_metadata(repository: Path) -> tuple[str, bool]:
    """Return the current commit and whether the repository has local changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable", True


def _collection_from_mapping(
    values: Mapping[str, object],
    *,
    description: str,
) -> me.EntityCollection:
    """Convert a diagnostics mapping recursively to an EntityCollection."""
    collection = me.EntityCollection(description=description)
    for name, value in values.items():
        if isinstance(value, Mapping):
            collection[name] = _collection_from_mapping(
                value,
                description=f"Nested diagnostics for {name}.",
            )
        else:
            collection[name] = value
    return collection


def _model_collection(model: AffineMagnetizationModel) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Dimensionless affine body-frame model M_0(r') = m_0 + m_1 r' "
            "on a spherical support. The values reproduce the Letter figure and "
            "do not carry an SI magnetization or length scale."
        ),
        radius=_quantity(
            "ISQDimensionlessQuantity",
            model.radius,
            "Sphere radius in the internal normalized length convention.",
        ),
        m0=_quantity(
            "ISQDimensionlessQuantity",
            model.m0,
            (
                "Constant term m_0 of the affine magnetization field in "
                "normalized model amplitude."
            ),
        ),
        m1=_quantity(
            "ISQDimensionlessQuantity",
            model.m1,
            (
                "First-order matrix m_1 of the affine magnetization field in the "
                "internal normalized coordinate and amplitude convention."
            ),
        ),
        zeta=_quantity(
            "ISQDimensionlessQuantity",
            model.zeta,
            "Scalar part zeta = Tr(m_1)/3 of the first-order matrix.",
        ),
        symmetric_traceless=_quantity(
            "ISQDimensionlessQuantity",
            model.symmetric_traceless,
            "Symmetric-traceless part S of the first-order matrix m_1.",
        ),
        omega=_quantity(
            "ISQDimensionlessQuantity",
            model.omega,
            "Axial vector omega of the antisymmetric part [omega]_cross of m_1.",
        ),
    )


def _field_collection(field: RealSpaceFieldData) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Body-frame sample passed to the paperfig 3D vector-field panel. "
            "Positions are expressed in units of R and magnetization in the "
            "dimensionless model amplitude."
        ),
        positions_over_R=_quantity(
            "RatioQuantity",
            field.positions,
            "Sample coordinates r'/R in Cartesian body-frame order x', y', z'.",
        ),
        magnetization=_quantity(
            "ISQDimensionlessQuantity",
            field.magnetization,
            "Sampled body-frame magnetization vectors in normalized model amplitude.",
        ),
        magnitude=_quantity(
            "ISQDimensionlessQuantity",
            field.magnitude,
            "Magnitude of each sampled normalized magnetization vector.",
        ),
        Mz_over_magnitude=_quantity(
            "RatioQuantity",
            field.axial_fraction,
            "Axial component M_z divided by the local magnetization magnitude.",
        ),
        cartesian_grid_points_per_axis=_quantity(
            "NumberOfElements",
            field.grid_points,
            (
                "Number of Cartesian sampling points along each coordinate axis "
                "before masking."
            ),
        ),
        sampled_support_fraction_of_R=_quantity(
            "Fraction",
            field.support_fraction,
            (
                "Fraction of the sphere radius retained in the sampled "
                "vector-field support."
            ),
        ),
    )


def _spectra_collection(
    spectra: CorrelationSpectraData,
) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Dimensionless correlation spectra plotted as Gamma/Gamma_0. The "
            "tensor convention is Gamma_perp Pi_perp + Gamma_parallel "
            "Pi_parallel - i Gamma_cross [q_hat]_cross."
        ),
        qR=_quantity(
            "ISQDimensionlessQuantity",
            spectra.q_r,
            "Dimensionless scattering-vector magnitude qR.",
        ),
        Gamma_perp=_quantity(
            "RatioQuantity",
            spectra.gamma_perpendicular,
            "Transverse correlation spectrum Gamma_perp normalized by Gamma_0.",
        ),
        Gamma_parallel=_quantity(
            "RatioQuantity",
            spectra.gamma_parallel,
            "Longitudinal correlation spectrum Gamma_parallel normalized by Gamma_0.",
        ),
        Gamma_cross=_quantity(
            "RatioQuantity",
            spectra.gamma_cross,
            "Signed chiral correlation spectrum Gamma_cross normalized by Gamma_0.",
        ),
    )


def _scattering_collection(scattering: DetectorData) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Perpendicular-geometry magnetic SANS channel maps. Raw arrays are "
            "dimensionless model outputs; display arrays are divided by the stored "
            "linear normalization factors and are the arrays passed to the figures."
        ),
        qR_edges=_quantity(
            "ISQDimensionlessQuantity",
            scattering.q_r_edges,
            "Bin edges of the dimensionless radial detector coordinate qR.",
        ),
        theta_edges_radian=_quantity(
            "Angle",
            scattering.theta_edges,
            "Bin edges of the detector azimuth theta; numerical values are in radians.",
        ),
        qR_centers=_quantity(
            "ISQDimensionlessQuantity",
            scattering.q_r_centers,
            "Bin centers of the dimensionless radial detector coordinate qR.",
        ),
        theta_centers_radian=_quantity(
            "Angle",
            scattering.theta_centers,
            (
                "Bin centers of the detector azimuth theta; numerical values are "
                "in radians."
            ),
        ),
        qzR_edges=_quantity(
            "ISQDimensionlessQuantity",
            scattering.q_z_r_edges,
            "Detector-grid edges of the dimensionless coordinate q_z R.",
        ),
        qyR_edges=_quantity(
            "ISQDimensionlessQuantity",
            scattering.q_y_r_edges,
            "Detector-grid edges of the dimensionless coordinate q_y R.",
        ),
        I_sf_raw=_quantity(
            "RatioQuantity",
            scattering.spin_flip_raw,
            (
                "Analytical spin-flip mean I_sf in the normalized correlation-spectrum "
                "scale, before display normalization."
            ),
        ),
        I_chi_raw=_quantity(
            "RatioQuantity",
            scattering.chiral_raw,
            (
                "Analytical chiral half-difference I_chi in the normalized "
                "correlation-spectrum scale, before display normalization."
            ),
        ),
        I_sf_display=_quantity(
            "RatioQuantity",
            scattering.spin_flip_display,
            "Spin-flip mean divided by I_sf_display_scale for the figure color map.",
        ),
        I_chi_display=_quantity(
            "RatioQuantity",
            scattering.chiral_display,
            (
                "Chiral half-difference divided by I_chi_display_scale for the "
                "figure color map."
            ),
        ),
        I_sf_display_scale=_quantity(
            "RatioQuantity",
            scattering.spin_flip_scale,
            "Maximum of I_sf_raw used for linear display normalization.",
        ),
        I_chi_display_scale=_quantity(
            "RatioQuantity",
            scattering.chiral_scale,
            (
                "Maximum absolute value of I_chi_raw used for linear display "
                "normalization."
            ),
        ),
    )


def _orientation_collection(
    orientation: OrientationDistributionData,
) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Axisymmetric orientation density relative to normalized Haar measure. "
            "The sampled density is stored as a function of a_z."
        ),
        formula=orientation.formula,
        a_z=_quantity(
            "ISQDimensionlessQuantity",
            orientation.z,
            "Axial coordinate a_z = e_z dot a_hat of the vortex-axis direction.",
        ),
        density=_quantity(
            "RatioQuantity",
            orientation.density,
            "Orientation probability density psi relative to normalized Haar measure.",
        ),
        first_axis_moment=_quantity(
            "ISQDimensionlessQuantity",
            orientation.first_moment,
            "First moment <a_hat>_psi of the vortex-axis direction.",
        ),
        second_axis_moment=_quantity(
            "ISQDimensionlessQuantity",
            orientation.second_moment,
            "Second moment <a_hat tensor a_hat>_psi of the vortex-axis direction.",
        ),
    )


def _figure_1_collection(data: Figure1Data) -> me.EntityCollection:
    orientations = me.EntityCollection(
        description="Orientation distributions used in the three Figure 1 paths."
    )
    for orientation in data.orientations:
        orientations[orientation.name] = _orientation_collection(orientation)
    return me.EntityCollection(
        description=(
            "Figure 1: one linear-vortex texture with three orientation "
            "distributions and identical averaged correlation tensors and magnetic "
            "SANS channel maps."
        ),
        model=_model_collection(data.model),
        real_space_field=_field_collection(data.field),
        orientation_distributions=orientations,
        correlation_spectra=_spectra_collection(data.spectra),
        sans_cross_sections=_scattering_collection(data.scattering),
        validation=_collection_from_mapping(
            data.validation,
            description="Numerical checks of the three orientation ensembles.",
        ),
    )


def _affine_case_collection(case: AffinePairCaseData) -> me.EntityCollection:
    return me.EntityCollection(
        description=f"Figure 2 data for {case.name.replace('_', ' ')}.",
        model=_model_collection(case.model),
        real_space_field=_field_collection(case.field),
        correlation_spectra=_spectra_collection(case.spectra),
        sans_cross_sections=_scattering_collection(case.scattering),
    )


def _figure_2_collection(data: Figure2Data) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Figure 2: two affine textures with different longitudinal correlation "
            "spectra and identical projected magnetic SANS channel maps for psi=1."
        ),
        orientation_distribution=me.EntityCollection(
            description="Uniform orientation distribution relative to Haar measure.",
            formula="psi(R) = 1",
        ),
        displaced_circular_vortex=_affine_case_collection(data.circular),
        centered_elliptic_vortex=_affine_case_collection(data.elliptic),
        validation=_collection_from_mapping(
            data.validation,
            description="Numerical checks of the affine vortex pair.",
        ),
    )


def build_mammos_collection(
    *,
    generated_at: str | None = None,
) -> me.EntityCollection:
    """Build the complete nested MaMMoS data product for both figures."""
    repository = Path(__file__).resolve().parents[1]
    commit, dirty = _git_metadata(repository)
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()
    metadata = me.EntityCollection(
        description="Provenance and conventions for this data product.",
        schema_version=SCHEMA_VERSION,
        generated_at_utc=generated_at,
        git_commit=commit,
        git_worktree_dirty=dirty,
        mammos_entity_version=me.__version__,
        numpy_version=importlib.metadata.version("numpy"),
        data_generator="python/export_figure_data.py",
        source_module="python/figure_data.py",
        coordinate_order="Cartesian x, y, z",
        reciprocal_coordinate_order="q_x, q_y, q_z; detector plane q_x = 0",
        fourier_convention=(
            "Normalized spherical form factor F(qR) and derivative dF/d(qR); "
            "first-order terms use R dF/d(qR)."
        ),
        normalization=(
            "Dimensionless Letter data. Raw SANS maps retain the analytical model "
            "scale; display maps are linearly normalized by their stored maxima."
        ),
        semantic_scope=(
            "Dimensionless numerical quantities carry generic MaMMoS ontology "
            "labels and explicit descriptions. No SI-specific magnetization, "
            "wave-vector, or cross-section labels are assigned because the figures "
            "do not define the corresponding physical scales."
        ),
    )
    return me.EntityCollection(
        description=(
            "Numerical data underlying Figures 1 and 2 of the Letter on "
            "SO(3)-covariant powder-averaged magnetic SANS."
        ),
        metadata=metadata,
        figure_1=_figure_1_collection(build_figure_1_data()),
        figure_2=_figure_2_collection(build_figure_2_data()),
    )


def export_mammos_hdf5(
    output: str | os.PathLike[str],
    *,
    generated_at: str | None = None,
) -> Path:
    """Atomically write the complete figure data product to HDF5."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collection = build_mammos_collection(generated_at=generated_at)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        collection.to_hdf5(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
