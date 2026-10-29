"""Tests of the general affine model and its analytical SANS solution."""

from __future__ import annotations

import numpy as np

from magnetic_SANS_models import (
    AffineMagneticSANSModel,
)
from magnetization_models import (
    AffineMagnetizationModel,
    affine_linear_vortex,
    affine_vortex_pair,
    cross_matrix,
    sphere_form_factor,
    sphere_form_factor_derivative,
)


def _rotation_quadrature() -> tuple[np.ndarray, np.ndarray]:
    z_coord, z_weights = np.polynomial.legendre.leggauss(8)
    alpha, z_grid, gamma = np.meshgrid(
        2.0 * np.pi * np.arange(8) / 8.0,
        z_coord,
        2.0 * np.pi * np.arange(16) / 16.0,
        indexing="ij",
    )
    weights = np.broadcast_to(
        z_weights[None, :, None] / (2.0 * 8 * 16),
        alpha.shape,
    ).ravel()
    cosine_alpha, sine_alpha = np.cos(alpha).ravel(), np.sin(alpha).ravel()
    cosine_gamma, sine_gamma = np.cos(gamma).ravel(), np.sin(gamma).ravel()
    z_grid = z_grid.ravel()
    sine_beta = np.sqrt(1.0 - z_grid**2)
    rotations = np.stack(
        (
            cosine_alpha * z_grid * cosine_gamma - sine_alpha * sine_gamma,
            -cosine_alpha * z_grid * sine_gamma - sine_alpha * cosine_gamma,
            cosine_alpha * sine_beta,
            sine_alpha * z_grid * cosine_gamma + cosine_alpha * sine_gamma,
            -sine_alpha * z_grid * sine_gamma + cosine_alpha * cosine_gamma,
            sine_alpha * sine_beta,
            -sine_beta * cosine_gamma,
            sine_beta * sine_gamma,
            z_grid,
        ),
        axis=-1,
    ).reshape(-1, 3, 3)
    return rotations, weights


def test_general_affine_decomposition_round_trip() -> None:
    model = AffineMagnetizationModel(
        m0=[0.7, -1.1, 2.3],
        m1=[
            [0.4, -1.7, 0.2],
            [0.8, -0.3, 2.1],
            [-1.4, 0.5, 1.2],
        ],
        radius=2.4,
    )

    np.testing.assert_allclose(model.reconstructed_first_order_matrix(), model.m1)
    np.testing.assert_allclose(
        model.symmetric_traceless,
        model.symmetric_traceless.T,
    )
    assert abs(np.trace(model.symmetric_traceless)) < 1.0e-14
    np.testing.assert_allclose(
        cross_matrix(model.omega),
        0.5 * (model.m1 - model.m1.T),
    )


def test_general_linear_vortex_matches_closed_special_solution() -> None:
    radius = 1.8
    m0 = 2.3
    ratio = 4.7
    qr = np.linspace(0.0, 7.0, 151)
    general = AffineMagneticSANSModel(
        affine_linear_vortex(radius=radius, m0=m0, m1_r_over_m0=ratio)
    )

    general_spectra = general.correlation_spectra(qr)
    form_factor = sphere_form_factor(qr)
    derivative = sphere_form_factor_derivative(qr)
    expected = (
        m0**2 * (form_factor**2 + ratio**2 * derivative**2) / 3.0,
        m0**2 * form_factor**2 / 3.0,
        2.0 * m0**2 * ratio * form_factor * derivative / 3.0,
    )
    for general_spectrum, expected_spectrum in zip(general_spectra, expected):
        np.testing.assert_allclose(general_spectrum, expected_spectrum)


def test_general_affine_pair_matches_closed_special_solution() -> None:
    radius = 1.9
    m = 1.6
    ratio = 5.2
    alpha = 0.41
    qr = np.linspace(0.0, 6.0, 121)
    circular, elliptic = affine_vortex_pair(
        radius=radius,
        m=m,
        kappa_r_over_m=ratio,
        alpha=alpha,
    )
    circular_spectra = AffineMagneticSANSModel(circular).correlation_spectra(qr)
    elliptic_spectra = AffineMagneticSANSModel(elliptic).correlation_spectra(qr)
    form_factor = sphere_form_factor(qr)
    derivative = sphere_form_factor_derivative(qr)
    transverse = m**2 * (form_factor**2 + ratio**2 * derivative**2) / 3.0
    parallel_d = m**2 * form_factor**2 / 3.0
    parallel_e = (
        parallel_d
        + 4.0 * m**2 * ratio**2 * np.sin(alpha) ** 2 * derivative**2 / 9.0
    )
    chiral = (
        2.0
        * m**2
        * ratio
        * np.cos(alpha)
        * form_factor
        * derivative
        / 3.0
    )

    np.testing.assert_allclose(circular_spectra[0], transverse)
    np.testing.assert_allclose(circular_spectra[1], parallel_d)
    np.testing.assert_allclose(circular_spectra[2], chiral)
    np.testing.assert_allclose(elliptic_spectra[0], transverse)
    np.testing.assert_allclose(elliptic_spectra[1], parallel_e)
    np.testing.assert_allclose(elliptic_spectra[2], chiral)


def test_general_spectra_match_direct_so3_average() -> None:
    model = AffineMagnetizationModel(
        m0=[0.8, -0.35, 1.2],
        m1=[
            [0.4, -1.1, 0.3],
            [0.7, -0.2, 0.5],
            [-0.6, 0.9, 0.15],
        ],
        radius=1.7,
    )
    sans_model = AffineMagneticSANSModel(model)
    rotations, weights = _rotation_quadrature()

    for q_vector in (
        np.array([0.0, 0.0, 0.6]),
        np.array([0.4, -1.1, 2.0]),
        np.array([-1.3, 0.7, 2.4]),
    ):
        qr = float(np.linalg.norm(q_vector))
        q_hat = q_vector / qr
        form_factor = sphere_form_factor(np.array(qr))
        radius_weighted_derivative = (
            model.radius * sphere_form_factor_derivative(np.array(qr))
        )
        rotated_m0 = np.einsum("nij,j->ni", rotations, model.m0)
        rotated_m1 = np.einsum(
            "nai,ij,nbj->nab",
            rotations,
            model.m1,
            rotations,
        )
        amplitude = (
            form_factor * rotated_m0
            + 1j
            * radius_weighted_derivative
            * np.einsum("nij,j->ni", rotated_m1, q_hat)
        )
        numerical_tensor = np.einsum(
            "n,ni,nj->ij",
            weights,
            amplitude,
            amplitude.conj(),
        )

        gamma_perpendicular, gamma_parallel, gamma_cross = (
            sans_model.correlation_spectra(np.array(qr))
        )
        longitudinal = np.outer(q_hat, q_hat)
        analytical_tensor = (
            gamma_perpendicular * (np.eye(3) - longitudinal)
            + gamma_parallel * longitudinal
            - 1j * gamma_cross * cross_matrix(q_hat)
        )
        np.testing.assert_allclose(
            numerical_tensor,
            analytical_tensor,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
