"""Check geometry and Fourier conventions against independent references."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.spatial.transform import Rotation

from python.models import Model, zyzMat
from python.models import global_affine_model as global_model
from python.models import local_affine_model as local
from python.models.SANS_detector_grid import detector_grid
from python.models.real_space_grid import regular_real_space_grid


def test_zyz_rotation_matches_scipy_with_broadcasting():
    angles = np.broadcast_arrays(
        np.array([0.0, 0.7])[:, None], np.array([0.0, 0.4, np.pi]), -0.3,
    )
    expected = Rotation.from_euler("ZYZ", np.stack(angles, axis=-1).reshape(-1, 3))
    actual = zyzMat(*angles)
    assert actual.shape == (2, 3, 3, 3)
    assert_allclose(actual, expected.as_matrix().reshape(actual.shape), atol=5e-16)
    assert_allclose(np.linalg.det(actual), 1.0, atol=1e-15)


@pytest.mark.parametrize("u", [0.0, 1e-7, 0.099, 0.101, 1.0, 4.0, 12.5])
def test_sphere_form_factor_against_cross_section_integral(u):
    # A sphere slice normal to q has area pi*R**2*(1-t**2).
    t, weights = leggauss(80)
    integrand = 0.75 * (1 - t**2) * np.exp(-1j * u * t)
    assert_allclose(local.F(u), weights @ integrand, atol=2e-11, rtol=1e-11)
    assert_allclose(local.F_u(u), weights @ (-1j * t * integrand), atol=2e-11, rtol=1e-11)


@pytest.mark.parametrize("R", [0.7, 2.1])
def test_fourier_amplitude_against_direct_volume_integral(affine_model, R):
    radial_nodes, radial_weights = leggauss(12)
    radii = R * (radial_nodes + 1) / 2
    mu, mu_weights = leggauss(12)
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    radius, cosine, azimuth = np.meshgrid(radii, mu, phi, indexing="ij")
    positions = np.stack([
        radius * np.sqrt(1 - cosine**2) * np.cos(azimuth),
        radius * np.sqrt(1 - cosine**2) * np.sin(azimuth),
        radius * cosine,
    ], axis=-1).reshape(-1, 3)
    weights = np.broadcast_to(
        (3 / (2 * R**3)) * (R / 2 * radial_weights * radii**2)[:, None, None]
        * mu_weights[None, :, None] / len(phi), radius.shape,
    ).ravel()
    q = np.array([[0.0, 0.0, 0.0], [0.2, -0.3, 0.7], [1.1, 0.6, -0.4]])
    field = affine_model.m0 + positions @ affine_model.m1.T
    assert_allclose(local.affine_magnetization(positions, affine_model.m0, affine_model.m1, R), field)
    expected = np.einsum("n,kn,ni->ki", weights, np.exp(-1j * q @ positions.T), field)
    actual = local.affine_magnetization_fourier(q, affine_model.m0, affine_model.m1, R)
    assert_allclose(actual, expected, atol=2e-13)
    assert_allclose(local.affine_magnetization_fourier(-q, affine_model.m0, affine_model.m1, R), actual.conj())


def test_global_field_rotates_coordinates_and_vectors(affine_model):
    angles = (0.3, 1.1, -0.7)
    rotation = Rotation.from_euler("ZYZ", angles).as_matrix()
    points = np.array([[0.1, -0.2, 0.4], [-0.3, 0.5, 0.2]])
    m0 = rotation @ affine_model.m0
    m1 = rotation @ affine_model.m1 @ rotation.T
    expected = m0 + points @ m1.T
    actual = global_model.affine_magnetization(
        points, affine_model.m0, affine_model.m1, affine_model.R, *angles,
    )
    assert_allclose(actual, expected, atol=1e-15)
    assert_allclose(
        global_model.affine_magnetization_fourier(
            points, affine_model.m0, affine_model.m1, affine_model.R, *angles,
        ),
        local.affine_magnetization_fourier(points, m0, m1, affine_model.R),
        atol=2e-14,
    )


def test_real_space_grid_is_only_geometry_and_includes_sphere_surface():
    grid = regular_real_space_grid(R=2.0, grid_points=3)
    assert set(grid) == {"x", "y", "z"}
    points = np.column_stack([grid[axis] for axis in "xyz"])
    expected = {(0.0, 0.0, 0.0)} | {tuple(sign * 2 * row) for row in np.eye(3) for sign in (-1, 1)}
    assert set(map(tuple, points)) == expected
    assert all(values.ndim == 1 for values in grid.values())


def test_detector_grid_has_one_coordinate_per_intensity_point():
    q = np.array([0.3, 1.7])
    theta = np.array([-np.pi / 2, 0.0, np.pi / 2])
    grid = detector_grid(q, theta)
    assert grid["q_vectors"].shape == (3, 2, 3)
    assert_array_equal(grid["q_vectors"][..., 0], 0)
    assert_allclose(np.linalg.norm(grid["q_vectors"], axis=-1), np.broadcast_to(q, (3, 2)))
    assert_allclose(grid["q_vectors"][:, 1], [[0, -1.7, 0], [0, 0, 1.7], [0, 1.7, 0]], atol=2e-16)
    assert not any("edges" in name for name in grid)


@pytest.mark.parametrize("q,theta", [([0.0, 1.0], [0.0]), ([1.0, 0.5], [0.0]), ([1.0], [0.0, 0.0])])
def test_detector_grid_rejects_undefined_or_unordered_coordinates(q, theta):
    with pytest.raises(ValueError):
        detector_grid(q, theta)


@pytest.mark.parametrize("R", [0.0, -1.0, np.inf, np.nan])
def test_model_rejects_invalid_sphere_radius(R):
    with pytest.raises(ValueError):
        Model(np.zeros(3), np.zeros((3, 3)), R=R)
