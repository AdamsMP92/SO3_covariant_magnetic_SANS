"""Tests for the numerical data and their MaMMoS HDF5 products."""

from __future__ import annotations

import h5py
import mammos_entity as me
import numpy as np

from data_products import (
    build_affine_projection_data,
    build_linear_vortex_orientation_data,
)
from data_io import (
    load_affine_projection_data,
    load_linear_vortex_orientation_data,
)
from magnetic_SANS_models import (
    AffineMagneticSANSModel,
    perpendicular_detector_grid,
)
from mammos_export import SCHEMA_VERSION, export_data_products


def test_linear_vortex_data_contains_model_outputs() -> None:
    data = build_linear_vortex_orientation_data()

    np.testing.assert_allclose(
        data.field.magnetization,
        data.model.evaluate(data.field.positions),
    )
    np.testing.assert_allclose(
        data.field.magnitude,
        np.linalg.norm(data.field.magnetization, axis=1),
    )
    np.testing.assert_allclose(
        data.scattering.spin_flip_display,
        data.scattering.spin_flip_raw / data.scattering.spin_flip_scale,
    )
    np.testing.assert_allclose(
        data.scattering.chiral_display,
        data.scattering.chiral_raw / data.scattering.chiral_scale,
    )
    q_z_r, q_y_r, spin_flip, chiral = perpendicular_detector_grid(
        AffineMagneticSANSModel(data.model)
    )
    np.testing.assert_array_equal(q_z_r, data.scattering.q_z_r_edges)
    np.testing.assert_array_equal(q_y_r, data.scattering.q_y_r_edges)
    np.testing.assert_array_equal(spin_flip, data.scattering.spin_flip_display)
    np.testing.assert_array_equal(chiral, data.scattering.chiral_display)

    orientations = {entry.name: entry for entry in data.orientations}
    np.testing.assert_allclose(
        orientations["uniform"].second_moment,
        np.eye(3) / 3.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        orientations["hemisphere"].first_moment,
        [0.0, 0.0, 0.5],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        orientations["P4"].first_moment,
        np.zeros(3),
        atol=1.0e-14,
    )


def test_affine_projection_data_has_equal_observables_and_distinct_longitudinal_spectra() -> None:
    data = build_affine_projection_data()

    np.testing.assert_allclose(
        data.circular.spectra.gamma_perpendicular,
        data.elliptic.spectra.gamma_perpendicular,
    )
    np.testing.assert_allclose(
        data.circular.spectra.gamma_cross,
        data.elliptic.spectra.gamma_cross,
    )
    assert np.max(
        np.abs(
            data.circular.spectra.gamma_parallel
            - data.elliptic.spectra.gamma_parallel
        )
    ) > 0.1
    np.testing.assert_allclose(
        data.circular.scattering.spin_flip_raw,
        data.elliptic.scattering.spin_flip_raw,
    )
    np.testing.assert_allclose(
        data.circular.scattering.chiral_raw,
        data.elliptic.scattering.chiral_raw,
    )


def _assert_no_figure_language(hdf5_file: h5py.File) -> None:
    def check(name: str, item: h5py.Group | h5py.Dataset) -> None:
        assert "figure" not in name.lower()
        for key, value in item.attrs.items():
            assert "figure" not in str(key).lower()
            if isinstance(value, (str, bytes)):
                text = value.decode() if isinstance(value, bytes) else value
                assert "figure" not in text.lower()

    hdf5_file.visititems(check)


def test_two_hdf5_products_roundtrip_without_rendering_language(tmp_path) -> None:
    data1_path, data2_path = export_data_products(
        tmp_path,
        generated_at="2026-10-29T00:00:00+00:00",
    )

    assert data1_path.name == "data1.h5"
    assert data2_path.name == "data2.h5"
    assert sorted(path.name for path in tmp_path.glob("*.h5")) == sorted(
        (data1_path.name, data2_path.name)
    )

    orientation = me.from_hdf5(data1_path)
    projection = me.from_hdf5(data2_path)
    for product in (orientation, projection):
        assert product["metadata"]["schema_version"] == SCHEMA_VERSION
        assert (
            product["metadata"]["generated_at_utc"]
            == "2026-10-29T00:00:00+00:00"
        )
    assert orientation["metadata"]["data_product"] == "data1"
    assert projection["metadata"]["data_product"] == "data2"

    assert "forward_chain" not in orientation
    assert "displaced_circular_vortex" in projection
    assert "centered_elliptic_vortex" in projection
    assert "magnetic_SANS_cross_sections" in orientation
    assert "I_sf_display" not in orientation["magnetic_SANS_cross_sections"]
    assert "I_chi_display" not in orientation["magnetic_SANS_cross_sections"]

    expected_orientation = build_linear_vortex_orientation_data()
    np.testing.assert_array_equal(
        orientation["real_space_field"]["positions_over_R"].value,
        expected_orientation.field.positions,
    )
    np.testing.assert_array_equal(
        orientation["correlation_spectra"]["Gamma_perp"].value,
        expected_orientation.spectra.gamma_perpendicular,
    )
    np.testing.assert_array_equal(
        orientation["magnetic_SANS_cross_sections"]["I_sf_raw"].value,
        expected_orientation.scattering.spin_flip_raw,
    )

    expected_projection = build_affine_projection_data()
    np.testing.assert_array_equal(
        projection["displaced_circular_vortex"]["correlation_spectra"]
        ["Gamma_parallel"].value,
        expected_projection.circular.spectra.gamma_parallel,
    )
    np.testing.assert_array_equal(
        projection["centered_elliptic_vortex"]["correlation_spectra"]
        ["Gamma_parallel"].value,
        expected_projection.elliptic.spectra.gamma_parallel,
    )

    with h5py.File(data1_path, "r") as hdf5_file:
        _assert_no_figure_language(hdf5_file)
        assert hdf5_file.attrs["mammos_entity_version"] == me.__version__
        assert "forward_chain" not in hdf5_file
    with h5py.File(data2_path, "r") as hdf5_file:
        _assert_no_figure_language(hdf5_file)

    loaded_orientation = load_linear_vortex_orientation_data(data1_path)
    loaded_projection = load_affine_projection_data(data2_path)
    np.testing.assert_array_equal(
        loaded_orientation.scattering.spin_flip_display,
        expected_orientation.scattering.spin_flip_display,
    )
    np.testing.assert_array_equal(
        loaded_projection.circular.scattering.chiral_display,
        expected_projection.circular.scattering.chiral_display,
    )
    assert not list(tmp_path.glob(".*.tmp"))
