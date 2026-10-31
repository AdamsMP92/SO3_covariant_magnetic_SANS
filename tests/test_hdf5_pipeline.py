"""Exercise real MaMMoS files, branch selection and sequential postprocessing."""

import h5py
import mammos_entity as me
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.stats import qmc

from python.generate_data.gen_data import gen_data
from python.generate_data.initialize_data import initialize_hdf5
from python.generate_data.load_model import load_model
from python.generate_data.add_magnetization_data import add_magnetization_data
from python.generate_data.add_average_gamma_tensor_data import add_average_gamma_tensor_data
from python.generate_data.add_average_G_tensor_data import add_average_G_tensor_data
from python.generate_data.add_SANS_cross_sections_data import add_SANS_cross_sections_data
from python.models import Model, OrientationDistribution


def custom_density(alpha, beta, gamma):
    """Importable normalized density without a registered analytical branch."""
    return 1 + 0.3 * np.cos(beta)


def read_tensor(branch):
    return np.stack([
        np.stack([branch[first + second].value for second in "xyz"], axis=-1)
        for first in "xyz"
    ], axis=-2)


@pytest.mark.parametrize("distribution", ["uniform", "hemisphere", "p4", "compensated", "custom"])
def test_complete_pipeline_roundtrip(tmp_path, monkeypatch, vortex_model, distribution):
    model = vortex_model
    model.psi = custom_density if distribution == "custom" else OrientationDistribution(distribution)
    filename = tmp_path / "model.h5"
    q = np.array([0.4, 1.2])
    theta = np.array([-np.pi, 0.0, np.pi / 2])

    # The volume mean must use the exact spatial integral and 3D orientation
    # sampling, not the retained six-dimensional reference benchmark.
    dimensions = []
    original_sobol = qmc.Sobol

    def record_sobol(*args, **kwargs):
        dimensions.append(kwargs["d"])
        return original_sobol(*args, **kwargs)

    monkeypatch.setattr(qmc, "Sobol", record_sobol)
    assert gen_data(model, filename, q, theta, grid_points=3, power=13, seed=0, batch_size=256) == filename
    assert dimensions == [3, 3, 3]

    data = me.from_hdf5(filename)
    assert {name for name, _ in data} == {
        "model", "real_space", "detector", "average_magnetization",
        "average_gamma_tensor", "average_G_tensor", "SANS_cross_sections",
    }
    assert_array_equal(data["model"]["m0"].value, model.m0)
    assert_array_equal(data["model"]["m1"].value, model.m1)
    assert data["model"]["R"].value == model.R
    assert data["model"]["psi"]["name"] == distribution
    real = data["real_space"]
    assert {name for name, _ in real} == {"x", "y", "z", "mx", "my", "mz"}
    assert all(item.value.shape == (7,) for _, item in real)
    assert_allclose(real["mx"].value, -7 * real["y"].value)
    assert_allclose(real["my"].value, 7 * real["x"].value)
    assert_allclose(real["mz"].value, 1)
    assert real["mx"].description
    assert real["mx"].q.unit.is_equivalent(1)
    assert_array_equal(data["detector"]["q_axis"].value, q)
    assert_array_equal(data["detector"]["theta_axis"].value, theta)

    expected_methods = {"qmc"} if distribution == "custom" else {"qmc", "analytical"}
    for group in ("average_gamma_tensor", "average_G_tensor", "average_magnetization"):
        assert {name for name, _ in data[group]} == expected_methods
    direction = np.stack([
        np.zeros((3, 2)), data["detector"]["qy"].value, data["detector"]["qz"].value,
    ], axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    for method in expected_methods:
        for group in ("average_gamma_tensor", "average_G_tensor"):
            tensor = read_tensor(data[group][method])
            assert tensor.shape == (3, 2, 3, 3)
            assert np.iscomplexobj(tensor)
            assert_allclose(tensor, tensor.conj().swapaxes(-1, -2), atol=1e-14)
            assert np.linalg.eigvalsh(tensor).min() >= -1e-12
        G = read_tensor(data["average_G_tensor"][method])
        assert_allclose((G @ direction[..., None])[..., 0], 0, atol=1e-14)
        mean = data["average_magnetization"][method]
        for component in ("mx", "my", "mz"):
            assert mean[component].value.shape == (7,)
        assert mean["volume_mean"].value.shape == (3,)
        scattering = data["SANS_cross_sections"][method]
        for family in ("basic", "derived"):
            for _, channel in scattering[family]:
                assert channel.value.shape == (3, 2)
                assert np.isfinite(channel.value).all()
        assert_allclose(scattering["basic"]["I_M"].value, np.trace(G, axis1=-2, axis2=-1).real)
        assert_allclose(scattering["derived"]["I_unp"].value, scattering["basic"]["I_M"].value)

    if distribution != "custom":
        for group in ("average_gamma_tensor", "average_G_tensor"):
            assert_allclose(read_tensor(data[group]["qmc"]), read_tensor(data[group]["analytical"]), atol=5e-3, rtol=0)
        expected_mean = [0, 0, 0.5] if distribution == "hemisphere" else [0, 0, 0]
        assert_allclose(data["average_magnetization"]["analytical"]["volume_mean"].value, expected_mean)
        assert_allclose(data["average_magnetization"]["qmc"]["volume_mean"].value, expected_mean, atol=1e-3)
        analytical = data["average_gamma_tensor"]["analytical"]
        assert ("spectra" in analytical) == (distribution != "compensated")
        if "spectra" in analytical:
            assert {name for name, _ in analytical["spectra"]} == {"perpendicular", "parallel", "cross"}
            assert all(item.value.shape == (2,) for _, item in analytical["spectra"])

    with h5py.File(filename) as handle:
        loaded = load_model(handle)
        assert_allclose(loaded.psi(0.2, 0.7, -0.4), model.psi(0.2, 0.7, -0.4))
        assert handle["real_space"].attrs["grid_points_per_axis"] == 3
        assert handle["average_gamma_tensor/qmc"].attrs["power"] == 13
        assert handle["average_gamma_tensor/qmc"].attrs["seed"] == 0


def test_p4_parameter_survives_model_serialization(tmp_path, affine_model):
    affine_model.psi = OrientationDistribution("p4", epsilon=-0.6)
    filename = tmp_path / "p4.h5"
    initialize_hdf5(affine_model, filename)
    with h5py.File(filename) as handle:
        loaded = load_model(handle)
    assert loaded.psi.name == "p4"
    assert loaded.psi.epsilon == -0.6


def test_existing_file_and_real_space_are_not_overwritten(tmp_path, vortex_model):
    filename = tmp_path / "existing.h5"
    initialize_hdf5(vortex_model, filename)
    add_magnetization_data(filename, grid_points=3)
    original = filename.read_bytes()
    with pytest.raises(FileExistsError):
        initialize_hdf5(vortex_model, filename)
    with pytest.raises(ValueError, match="already contains real_space"):
        add_magnetization_data(filename, grid_points=5)
    assert filename.read_bytes() == original


def test_postprocessing_consumes_the_stored_tensors(tmp_path, vortex_model):
    filename = tmp_path / "stages.h5"
    initialize_hdf5(vortex_model, filename)
    theta = np.array([0.0, np.pi / 3])
    add_average_gamma_tensor_data(filename, [0.5], theta, power=4, seed=0, batch_size=8)
    with h5py.File(filename, "r+") as handle:
        del handle["average_gamma_tensor/analytical"]
        for name in handle["average_gamma_tensor/qmc"]:
            handle[f"average_gamma_tensor/qmc/{name}"][...] = {"xx": 4, "yy": 1}.get(name, 0)
    projected = add_average_G_tensor_data(filename)
    assert_allclose(projected["qmc"]["xx"].value, 4)
    assert_allclose(projected["qmc"]["zz"].value[:, 0], (np.sin(theta) * np.cos(theta))**2, atol=1e-15)
    with h5py.File(filename, "r+") as handle:
        del handle["average_gamma_tensor"]
        for name in handle["average_G_tensor/qmc"]:
            handle[f"average_G_tensor/qmc/{name}"][...] = {"xx": 4, "yy": 2, "zz": 1}.get(name, 0)
    scattering = add_SANS_cross_sections_data(filename)
    assert_allclose(scattering["qmc"]["derived"]["I_sf"].value, 6)
    assert_allclose(scattering["qmc"]["derived"]["I_unp"].value, 7)


def test_non_importable_density_fails_before_creating_file(tmp_path, vortex_model):
    filename = tmp_path / "lambda.h5"
    vortex_model.psi = lambda alpha, beta, gamma: 1.0
    with pytest.raises(ValueError, match="cannot be imported"):
        initialize_hdf5(vortex_model, filename)
    assert not filename.exists()
