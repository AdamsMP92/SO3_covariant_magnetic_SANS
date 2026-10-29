"""MaMMoS serialization of the two magnetic-SANS data products."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import mammos_entity as me
import numpy as np

from data_products import (
    AffineProjectionData,
    AffinePairData,
    CorrelationSpectraData,
    DetectorData,
    LinearVortexOrientationData,
    OrientationDistributionData,
    RealSpaceFieldData,
    build_affine_projection_data,
    build_linear_vortex_orientation_data,
)
from magnetization_models import AffineMagnetizationModel


SCHEMA_VERSION = "2.0"


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
            "on a spherical support. The values do not carry an SI "
            "magnetization or length scale."
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
            "Sampled body-frame magnetization field. Positions are expressed "
            "in units of R and magnetization in the dimensionless model amplitude."
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
            "Dimensionless correlation spectra computed from the stored "
            "magnetization model and orientation assumptions and normalized as "
            "Gamma/Gamma_0. They parameterize Gamma_perp Pi_perp + "
            "Gamma_parallel Pi_parallel - i Gamma_cross [q_hat]_cross; the full "
            "tensor field is not stored."
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
            "Perpendicular-geometry magnetic SANS channel maps computed "
            "directly from the stored q-dependent correlation spectra in the "
            "dimensionless correlation-spectrum scale."
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
                "Analytical spin-flip mean I_sf computed from Gamma_perp and "
                "the detector angle in the normalized correlation-spectrum scale."
            ),
        ),
        I_chi_raw=_quantity(
            "RatioQuantity",
            scattering.chiral_raw,
            (
                "Analytical chiral half-difference I_chi computed from "
                "Gamma_cross and the detector angle in the normalized "
                "correlation-spectrum scale."
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


def _linear_vortex_orientation_collection(
    data: LinearVortexOrientationData,
    *,
    generated_at: str,
) -> me.EntityCollection:
    orientations = me.EntityCollection(
        description=(
            "Three orientation distributions with the same second axis moment "
            "and the same averaged two-point correlation tensor."
        )
    )
    for orientation in data.orientations:
        orientations[orientation.name] = _orientation_collection(orientation)
    return me.EntityCollection(
        description=(
            "Linear-vortex data product of orientation-distribution "
            "non-identifiability: three distinct distributions produce the same "
            "averaged correlation tensor and magnetic SANS cross sections."
        ),
        metadata=_metadata_collection(
            generated_at=generated_at,
            data_product="data1",
        ),
        model=_model_collection(data.model),
        real_space_field=_field_collection(data.field),
        orientation_distributions=orientations,
        correlation_spectra=_spectra_collection(data.spectra),
        magnetic_SANS_cross_sections=_scattering_collection(data.scattering),
        validation=_collection_from_mapping(
            data.validation,
            description="Numerical checks of the three orientation ensembles.",
        ),
    )


def _affine_model_collection(data: AffinePairData) -> me.EntityCollection:
    return me.EntityCollection(
        description=f"Data for the {data.name.replace('_', ' ')} model.",
        model=_model_collection(data.model),
        real_space_field=_field_collection(data.field),
        correlation_spectra=_spectra_collection(data.spectra),
        magnetic_SANS_cross_sections=_scattering_collection(data.scattering),
    )


def _affine_projection_collection(
    data: AffineProjectionData,
    *,
    generated_at: str,
) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Affine-model data product of the Halpern-Johnson null space: two "
            "textures have different longitudinal correlation spectra but "
            "identical magnetic SANS cross sections for psi(R)=1."
        ),
        metadata=_metadata_collection(
            generated_at=generated_at,
            data_product="data2",
        ),
        orientation_distribution=me.EntityCollection(
            description="Uniform orientation distribution relative to Haar measure.",
            formula="psi(R) = 1",
        ),
        displaced_circular_vortex=_affine_model_collection(data.circular),
        centered_elliptic_vortex=_affine_model_collection(data.elliptic),
        validation=_collection_from_mapping(
            data.validation,
            description="Numerical checks of the affine vortex pair.",
        ),
    )


def _generation_time(generated_at: str | None) -> str:
    if generated_at is None:
        return datetime.now(timezone.utc).isoformat()
    return generated_at


def _metadata_collection(
    *,
    generated_at: str,
    data_product: str,
) -> me.EntityCollection:
    """Return common provenance and numerical conventions."""
    repository = Path(__file__).resolve().parents[1]
    commit, dirty = _git_metadata(repository)
    return me.EntityCollection(
        description="Provenance and conventions for this data product.",
        schema_version=SCHEMA_VERSION,
        data_product=data_product,
        generated_at_utc=generated_at,
        git_commit=commit,
        git_worktree_dirty=dirty,
        mammos_entity_version=me.__version__,
        numpy_version=importlib.metadata.version("numpy"),
        data_generator="python/export_data.py",
        source_modules="python/data_products.py",
        coordinate_order="Cartesian x, y, z",
        reciprocal_coordinate_order="q_x, q_y, q_z; detector plane q_x = 0",
        fourier_convention=(
            "Normalized spherical form factor F(qR) and derivative dF/d(qR); "
            "first-order terms use R dF/d(qR)."
        ),
        normalization=(
            "Dimensionless data. Magnetic SANS maps retain the analytical model "
            "scale; no rendering normalization is stored."
        ),
        semantic_scope=(
            "Dimensionless numerical quantities carry generic MaMMoS ontology "
            "labels and explicit descriptions. No SI-specific magnetization, "
            "scattering-vector, or cross-section labels are assigned because "
            "physical scales are not specified by the present dimensionless models."
        ),
    )


def build_linear_vortex_orientation_collection(
    *,
    generated_at: str | None = None,
) -> me.EntityCollection:
    """Build the orientation-distribution non-identifiability data product."""
    timestamp = _generation_time(generated_at)
    return _linear_vortex_orientation_collection(
        build_linear_vortex_orientation_data(),
        generated_at=timestamp,
    )


def build_affine_projection_collection(
    *,
    generated_at: str | None = None,
) -> me.EntityCollection:
    """Build the longitudinal-null-space non-identifiability data product."""
    timestamp = _generation_time(generated_at)
    return _affine_projection_collection(
        build_affine_projection_data(),
        generated_at=timestamp,
    )


def _write_hdf5(
    collection: me.EntityCollection,
    output: str | os.PathLike[str],
) -> Path:
    """Atomically write one MaMMoS data-product collection."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def export_data_products(
    output_directory: str | os.PathLike[str],
    *,
    generated_at: str | None = None,
) -> tuple[Path, Path]:
    """Write the two independent data products to separate HDF5 files."""
    output_directory = Path(output_directory)
    timestamp = _generation_time(generated_at)
    data1_path = _write_hdf5(
        build_linear_vortex_orientation_collection(generated_at=timestamp),
        output_directory / "data1.h5",
    )
    data2_path = _write_hdf5(
        build_affine_projection_collection(generated_at=timestamp),
        output_directory / "data2.h5",
    )
    return data1_path, data2_path
