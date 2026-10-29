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


SCHEMA_VERSION = "1.0"


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
        radius=model.radius,
        m0=model.m0,
        m1=model.m1,
        zeta=model.zeta,
        symmetric_traceless=model.symmetric_traceless,
        omega=model.omega,
    )


def _field_collection(field: RealSpaceFieldData) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Body-frame sample passed to the paperfig 3D vector-field panel. "
            "Positions are expressed in units of R and magnetization in the "
            "dimensionless model amplitude."
        ),
        positions_over_R=field.positions,
        magnetization=field.magnetization,
        magnitude=field.magnitude,
        Mz_over_magnitude=field.axial_fraction,
        cartesian_grid_points_per_axis=field.grid_points,
        sampled_support_fraction_of_R=field.support_fraction,
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
        qR=spectra.q_r,
        Gamma_perp=spectra.gamma_perpendicular,
        Gamma_parallel=spectra.gamma_parallel,
        Gamma_cross=spectra.gamma_cross,
    )


def _scattering_collection(scattering: DetectorData) -> me.EntityCollection:
    return me.EntityCollection(
        description=(
            "Perpendicular-geometry magnetic SANS channel maps. Raw arrays are "
            "dimensionless model outputs; display arrays are divided by the stored "
            "linear normalization factors and are the arrays passed to the figures."
        ),
        qR_edges=scattering.q_r_edges,
        theta_edges_radian=scattering.theta_edges,
        qR_centers=scattering.q_r_centers,
        theta_centers_radian=scattering.theta_centers,
        qzR_edges=scattering.q_z_r_edges,
        qyR_edges=scattering.q_y_r_edges,
        I_sf_raw=scattering.spin_flip_raw,
        I_chi_raw=scattering.chiral_raw,
        I_sf_display=scattering.spin_flip_display,
        I_chi_display=scattering.chiral_display,
        I_sf_display_scale=scattering.spin_flip_scale,
        I_chi_display_scale=scattering.chiral_scale,
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
        a_z=orientation.z,
        density=orientation.density,
        first_axis_moment=orientation.first_moment,
        second_axis_moment=orientation.second_moment,
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
            "No SI ontology labels are assigned to qR, normalized Gamma spectra, "
            "or normalized SANS maps because the figures do not define an SI scale."
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
