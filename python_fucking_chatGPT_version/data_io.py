"""Load the two MaMMoS data products for analysis and rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mammos_entity as me
import numpy as np

from data_products import (
    AffinePairData,
    AffineProjectionData,
    CorrelationSpectraData,
    DetectorData,
    LinearVortexOrientationData,
    OrientationDistributionData,
    RealSpaceFieldData,
)
from magnetization_models import AffineMagnetizationModel


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = REPOSITORY / "output" / "data"
DATA1_PATH = DEFAULT_DATA_DIRECTORY / "data1.h5"
DATA2_PATH = DEFAULT_DATA_DIRECTORY / "data2.h5"


def _value(collection: me.EntityCollection, name: str) -> Any:
    item = collection[name]
    return item.value if isinstance(item, me.Entity) else item


def _plain(item: Any) -> Any:
    if isinstance(item, me.Entity):
        return _plain(item.value)
    if isinstance(item, me.EntityCollection):
        return {name: _plain(value) for name, value in item}
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, np.generic):
        return item.item()
    return item


def _model(collection: me.EntityCollection) -> AffineMagnetizationModel:
    return AffineMagnetizationModel(
        radius=float(_value(collection, "radius")),
        m0=np.asarray(_value(collection, "m0"), dtype=float),
        m1=np.asarray(_value(collection, "m1"), dtype=float),
    )


def _field(collection: me.EntityCollection) -> RealSpaceFieldData:
    return RealSpaceFieldData(
        positions=np.asarray(_value(collection, "positions_over_R"), dtype=float),
        magnetization=np.asarray(_value(collection, "magnetization"), dtype=float),
        magnitude=np.asarray(_value(collection, "magnitude"), dtype=float),
        axial_fraction=np.asarray(
            _value(collection, "Mz_over_magnitude"), dtype=float
        ),
        grid_points=int(_value(collection, "cartesian_grid_points_per_axis")),
        support_fraction=float(
            _value(collection, "sampled_support_fraction_of_R")
        ),
    )


def _spectra(collection: me.EntityCollection) -> CorrelationSpectraData:
    return CorrelationSpectraData(
        q_r=np.asarray(_value(collection, "qR"), dtype=float),
        gamma_perpendicular=np.asarray(_value(collection, "Gamma_perp"), dtype=float),
        gamma_parallel=np.asarray(_value(collection, "Gamma_parallel"), dtype=float),
        gamma_cross=np.asarray(_value(collection, "Gamma_cross"), dtype=float),
    )


def _scattering(collection: me.EntityCollection) -> DetectorData:
    spin_flip_raw = np.asarray(_value(collection, "I_sf_raw"), dtype=float)
    chiral_raw = np.asarray(_value(collection, "I_chi_raw"), dtype=float)
    spin_flip_scale = float(np.nanmax(spin_flip_raw))
    chiral_scale = float(np.nanmax(np.abs(chiral_raw)))
    return DetectorData(
        q_r_edges=np.asarray(_value(collection, "qR_edges"), dtype=float),
        theta_edges=np.asarray(_value(collection, "theta_edges_radian"), dtype=float),
        q_r_centers=np.asarray(_value(collection, "qR_centers"), dtype=float),
        theta_centers=np.asarray(
            _value(collection, "theta_centers_radian"), dtype=float
        ),
        q_z_r_edges=np.asarray(_value(collection, "qzR_edges"), dtype=float),
        q_y_r_edges=np.asarray(_value(collection, "qyR_edges"), dtype=float),
        spin_flip_raw=spin_flip_raw,
        chiral_raw=chiral_raw,
        spin_flip_display=spin_flip_raw / spin_flip_scale,
        chiral_display=chiral_raw / chiral_scale,
        spin_flip_scale=spin_flip_scale,
        chiral_scale=chiral_scale,
    )


def _orientation(
    name: str,
    collection: me.EntityCollection,
) -> OrientationDistributionData:
    return OrientationDistributionData(
        name=name,
        formula=str(collection["formula"]),
        z=np.asarray(_value(collection, "a_z"), dtype=float),
        density=np.asarray(_value(collection, "density"), dtype=float),
        first_moment=np.asarray(_value(collection, "first_axis_moment"), dtype=float),
        second_moment=np.asarray(
            _value(collection, "second_axis_moment"), dtype=float
        ),
    )


def _affine_pair_member(
    name: str,
    collection: me.EntityCollection,
) -> AffinePairData:
    return AffinePairData(
        name=name,
        model=_model(collection["model"]),
        field=_field(collection["real_space_field"]),
        spectra=_spectra(collection["correlation_spectra"]),
        scattering=_scattering(collection["magnetic_SANS_cross_sections"]),
    )


def _load(path: str | Path) -> me.EntityCollection:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}. Run `pixi run data` first."
        )
    return me.from_hdf5(path)


def load_linear_vortex_orientation_data(
    path: str | Path = DATA1_PATH,
) -> LinearVortexOrientationData:
    """Load the linear-vortex orientation data from ``data1.h5``."""
    collection = _load(path)
    orientations = collection["orientation_distributions"]
    return LinearVortexOrientationData(
        model=_model(collection["model"]),
        field=_field(collection["real_space_field"]),
        orientations=tuple(
            _orientation(name, orientations[name])
            for name in ("uniform", "hemisphere", "P4")
        ),
        spectra=_spectra(collection["correlation_spectra"]),
        scattering=_scattering(collection["magnetic_SANS_cross_sections"]),
        validation=_plain(collection["validation"]),
    )


def load_affine_projection_data(
    path: str | Path = DATA2_PATH,
) -> AffineProjectionData:
    """Load the affine projection data from ``data2.h5``."""
    collection = _load(path)
    return AffineProjectionData(
        circular=_affine_pair_member(
            "displaced_circular_vortex",
            collection["displaced_circular_vortex"],
        ),
        elliptic=_affine_pair_member(
            "centered_elliptic_vortex",
            collection["centered_elliptic_vortex"],
        ),
        validation=_plain(collection["validation"]),
    )
