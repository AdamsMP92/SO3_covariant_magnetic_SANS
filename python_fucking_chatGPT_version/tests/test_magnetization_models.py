"""Tests for the analytical real-space magnetization models."""

from __future__ import annotations

import numpy as np
import pytest

from magnetization_models import (
    AffineMagnetizationModel,
    affine_linear_vortex,
    affine_vortex_pair,
    cross_matrix,
    legendre_p4,
    sphere_form_factor,
    sphere_form_factor_derivative,
)


def test_sphere_form_factor_parity_and_origin() -> None:
    qr = np.linspace(0.0, 8.0, 81)

    np.testing.assert_allclose(
        sphere_form_factor(-qr),
        sphere_form_factor(qr),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        sphere_form_factor_derivative(-qr),
        -sphere_form_factor_derivative(qr),
        atol=2.0e-14,
    )
    assert sphere_form_factor(np.array(0.0)) == pytest.approx(1.0)
    assert sphere_form_factor_derivative(np.array(0.0)) == pytest.approx(0.0)


def test_sphere_form_factor_derivative_matches_finite_difference() -> None:
    qr = np.array([0.2, 0.8, 1.7, 3.1, 5.6])
    step = 1.0e-6
    numerical_derivative = (
        sphere_form_factor(qr + step) - sphere_form_factor(qr - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        sphere_form_factor_derivative(qr),
        numerical_derivative,
        rtol=5.0e-8,
        atol=2.0e-10,
    )


def test_legendre_p4_values_and_orthogonality() -> None:
    nodes, weights = np.polynomial.legendre.leggauss(16)

    assert legendre_p4(np.array(1.0)) == pytest.approx(1.0)
    assert legendre_p4(np.array(-1.0)) == pytest.approx(1.0)
    for degree in range(4):
        lower_polynomial = np.polynomial.legendre.Legendre.basis(degree)(nodes)
        integral = np.sum(weights * legendre_p4(nodes) * lower_polynomial)
        assert integral == pytest.approx(0.0, abs=2.0e-14)


def test_cross_matrix_is_antisymmetric_and_represents_cross_product() -> None:
    vector = np.array([0.4, -1.2, 2.3])
    operand = np.array([-0.7, 0.5, 1.1])
    matrix = cross_matrix(vector)

    np.testing.assert_allclose(matrix.T, -matrix, atol=0.0)
    np.testing.assert_allclose(matrix @ operand, np.cross(vector, operand))


def test_linear_vortex_samples_its_defining_field() -> None:
    radius = 1.7
    m0 = 2.1
    ratio = 4.3
    model = affine_linear_vortex(
        radius=radius,
        m0=m0,
        m1_r_over_m0=ratio,
    )
    x_coord, y_coord, z_coord, m_x, m_y, m_z, magnitude = model.sample_field(
        grid_points=9,
        support_fraction=0.8,
    )

    circulation = ratio * m0 / radius
    np.testing.assert_allclose(m_x, -circulation * y_coord)
    np.testing.assert_allclose(m_y, circulation * x_coord)
    np.testing.assert_allclose(m_z, m0)
    np.testing.assert_allclose(magnitude, np.sqrt(m_x**2 + m_y**2 + m_z**2))
    assert np.max(x_coord**2 + y_coord**2 + z_coord**2) <= (
        0.8 * model.radius
    ) ** 2 + 1.0e-14


@pytest.mark.parametrize("label", ["d", "e"])
def test_affine_pair_samples_the_field_defined_by_its_parameters(label: str) -> None:
    models = affine_vortex_pair()
    model = models[0 if label == "d" else 1]
    x_coord, y_coord, z_coord, m_x, m_y, m_z, _ = model.sample_field(grid_points=8)
    positions = np.column_stack((x_coord, y_coord, z_coord))
    expected = model.m0 + positions @ model.m1.T

    np.testing.assert_allclose(np.column_stack((m_x, m_y, m_z)), expected)


@pytest.mark.parametrize(
    ("m0", "m1", "message"),
    [
        ([1.0, 2.0], np.eye(3), "m0 must have shape"),
        ([1.0, 2.0, 3.0], np.eye(2), "m1 must have shape"),
    ],
)
def test_general_affine_model_validates_array_shapes(m0, m1, message) -> None:
    with pytest.raises(ValueError, match=message):
        AffineMagnetizationModel(m0=m0, m1=m1)
