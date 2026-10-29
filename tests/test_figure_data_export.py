"""Tests for the shared figure data and MaMMoS HDF5 product."""

from __future__ import annotations

import h5py
import mammos_entity as me
import numpy as np

from figure_data import build_figure_1_data, build_figure_2_data
from magnetic_SANS_models import (
    AffineMagneticSANSModel,
    perpendicular_detector_grid,
)
from mammos_export import SCHEMA_VERSION, export_mammos_hdf5


def test_figure_1_data_are_the_model_outputs_passed_to_the_panels() -> None:
    data = build_figure_1_data()

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
    assert np.max(data.scattering.spin_flip_display) == 1.0
    assert np.max(np.abs(data.scattering.chiral_display)) == 1.0
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


def test_figure_2_stores_equal_observables_and_distinct_longitudinal_spectra() -> None:
    data = build_figure_2_data()

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


def test_mammos_hdf5_roundtrip_preserves_schema_and_numerical_data(tmp_path) -> None:
    output = tmp_path / "SO3_covariant_magnetic_SANS.h5"
    export_mammos_hdf5(
        output,
        generated_at="2026-10-29T00:00:00+00:00",
    )

    loaded = me.from_hdf5(output)
    assert loaded["metadata"]["schema_version"] == SCHEMA_VERSION
    assert loaded["metadata"]["generated_at_utc"] == "2026-10-29T00:00:00+00:00"
    assert "figure_1" in loaded
    assert "figure_2" in loaded

    figure_1 = build_figure_1_data()
    stored_figure_1 = loaded["figure_1"]
    np.testing.assert_array_equal(
        stored_figure_1["real_space_field"]["positions_over_R"].value,
        figure_1.field.positions,
    )
    np.testing.assert_array_equal(
        stored_figure_1["correlation_spectra"]["Gamma_perp"].value,
        figure_1.spectra.gamma_perpendicular,
    )
    np.testing.assert_array_equal(
        stored_figure_1["sans_cross_sections"]["I_sf_display"].value,
        figure_1.scattering.spin_flip_display,
    )

    figure_2 = build_figure_2_data()
    stored_figure_2 = loaded["figure_2"]
    np.testing.assert_array_equal(
        stored_figure_2["displaced_circular_vortex"]["correlation_spectra"]
        ["Gamma_parallel"].value,
        figure_2.circular.spectra.gamma_parallel,
    )
    np.testing.assert_array_equal(
        stored_figure_2["centered_elliptic_vortex"]["correlation_spectra"]
        ["Gamma_parallel"].value,
        figure_2.elliptic.spectra.gamma_parallel,
    )

    gamma_perp = stored_figure_1["correlation_spectra"]["Gamma_perp"]
    assert gamma_perp.ontology_label == "RatioQuantity"
    assert gamma_perp.unit == ""
    assert gamma_perp.description == (
        "Transverse correlation spectrum Gamma_perp normalized by Gamma_0."
    )

    detector_angle = stored_figure_1["sans_cross_sections"][
        "theta_centers_radian"
    ]
    assert detector_angle.ontology_label == "Angle"
    assert detector_angle.unit == ""
    assert "radians" in detector_angle.description

    grid_points = stored_figure_1["real_space_field"][
        "cartesian_grid_points_per_axis"
    ]
    assert grid_points.ontology_label == "NumberOfElements"
    assert grid_points.description

    with h5py.File(output, "r") as hdf5_file:
        assert hdf5_file.attrs["mammos_entity_version"] == me.__version__
        assert hdf5_file["metadata"].attrs["description"] == (
            "Provenance and conventions for this data product."
        )
        gamma_dataset = hdf5_file[
            "figure_1/correlation_spectra/Gamma_perp"
        ]
        assert gamma_dataset.attrs["ontology_label"] == "RatioQuantity"
        assert gamma_dataset.attrs["description"] == (
            "Transverse correlation spectrum Gamma_perp normalized by Gamma_0."
        )
        assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
