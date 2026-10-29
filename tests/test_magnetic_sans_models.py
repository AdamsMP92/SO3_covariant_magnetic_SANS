"""Tests for correlation tensors and magnetic SANS channel mappings."""

from __future__ import annotations

import numpy as np

from magnetic_SANS_models import (
    AffineMagneticSANSModel,
    perpendicular_channel_maps,
)
from magnetization_models import (
    affine_linear_vortex,
    affine_vortex_pair,
    cross_matrix,
)
from model_validation import (
    validate_affine_vortex_pair,
    validate_linear_vortex_ensembles,
)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    generator = cross_matrix(axis)
    return (
        np.eye(3)
        + np.sin(angle) * generator
        + (1.0 - np.cos(angle)) * generator @ generator
    )


def _correlation_tensor(
    gamma_perpendicular: float,
    gamma_parallel: float,
    gamma_cross: float,
    q_hat: np.ndarray,
) -> np.ndarray:
    longitudinal = np.outer(q_hat, q_hat)
    return (
        gamma_perpendicular * (np.eye(3) - longitudinal)
        + gamma_parallel * longitudinal
        - 1j * gamma_cross * cross_matrix(q_hat)
    )


def test_linear_vortex_spectra_are_positive_and_have_correct_q_zero_limit() -> None:
    model = AffineMagneticSANSModel(affine_linear_vortex(m1_r_over_m0=7.0))
    qr = np.linspace(0.0, 8.0, 801)
    gamma_perpendicular, gamma_parallel, gamma_cross = model.correlation_spectra(qr)

    np.testing.assert_allclose(
        [gamma_perpendicular[0], gamma_parallel[0], gamma_cross[0]],
        [1.0 / 3.0, 1.0 / 3.0, 0.0],
        atol=1.0e-15,
    )
    assert np.min(gamma_parallel) >= 0.0
    assert np.min(gamma_perpendicular - np.abs(gamma_cross)) >= -2.0e-15


def test_covariant_tensor_is_hermitian_and_jointly_so3_covariant() -> None:
    model = AffineMagneticSANSModel(affine_linear_vortex())
    q_vector = np.array([0.7, -1.2, 2.4])
    q_norm = np.linalg.norm(q_vector)
    q_hat = q_vector / q_norm
    spectra = tuple(float(value) for value in model.correlation_spectra(q_norm))
    tensor = _correlation_tensor(*spectra, q_hat)

    rotation = _rotation_matrix(np.array([1.0, 2.0, -0.5]), 0.73)
    rotated_q_hat = rotation @ q_hat
    rotated_tensor = _correlation_tensor(*spectra, rotated_q_hat)

    np.testing.assert_allclose(tensor, tensor.conj().T, atol=2.0e-15)
    np.testing.assert_allclose(
        rotated_tensor,
        rotation @ tensor @ rotation.T,
        atol=2.0e-15,
    )


def test_halpern_johnson_projection_removes_longitudinal_spectrum() -> None:
    q_hat = np.array([0.3, -0.4, np.sqrt(0.75)])
    q_hat = q_hat / np.linalg.norm(q_hat)
    transverse = np.eye(3) - np.outer(q_hat, q_hat)
    first = _correlation_tensor(1.4, 0.2, -0.3, q_hat)
    second = _correlation_tensor(1.4, 8.7, -0.3, q_hat)

    np.testing.assert_allclose(
        transverse @ first @ transverse,
        transverse @ second @ transverse,
        atol=2.0e-15,
    )


def test_perpendicular_channels_have_expected_angular_symmetries() -> None:
    theta = np.linspace(-np.pi, np.pi, 41)
    spin_flip, chiral = perpendicular_channel_maps(1.7, -0.4, theta)
    reflected_spin_flip, reflected_chiral = perpendicular_channel_maps(
        1.7,
        -0.4,
        -theta,
    )
    opposite_qz_spin_flip, opposite_qz_chiral = perpendicular_channel_maps(
        1.7,
        -0.4,
        np.pi - theta,
    )

    np.testing.assert_allclose(spin_flip, reflected_spin_flip)
    np.testing.assert_allclose(chiral, reflected_chiral)
    np.testing.assert_allclose(spin_flip, opposite_qz_spin_flip)
    np.testing.assert_allclose(chiral, -opposite_qz_chiral, atol=3.0e-16)


def test_linear_vortex_orientation_ensembles_share_the_second_moment() -> None:
    report = validate_linear_vortex_ensembles(
        AffineMagneticSANSModel(affine_linear_vortex()),
        epsilon=0.9,
    )

    np.testing.assert_allclose(report["uniform"]["first_moment"], np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(
        report["hemisphere"]["first_moment"],
        [0.0, 0.0, 0.5],
        atol=1.0e-14,
    )
    np.testing.assert_allclose(report["P4"]["first_moment"], np.zeros(3), atol=1e-14)
    assert report["full_tensor_max_ensemble_difference"] < 2.0e-14


def test_affine_pair_has_same_visible_spectra_but_different_longitudinal_spectra() -> None:
    magnetization_models = affine_vortex_pair(alpha=np.pi / 6.0)
    models = tuple(AffineMagneticSANSModel(model) for model in magnetization_models)
    qr = np.linspace(0.1, 5.0, 301)
    gamma_perpendicular, parallel_d, gamma_cross = models[0].correlation_spectra(qr)
    gamma_perpendicular_e, parallel_e, gamma_cross_e = models[1].correlation_spectra(qr)

    assert np.max(np.abs(parallel_e - parallel_d)) > 0.1
    np.testing.assert_allclose(gamma_perpendicular_e, gamma_perpendicular)
    np.testing.assert_allclose(gamma_cross_e, gamma_cross)
    visible_perpendicular, visible_cross = models[0].transverse_and_chiral_spectra(qr)
    np.testing.assert_allclose(visible_perpendicular, gamma_perpendicular)
    np.testing.assert_allclose(visible_cross, gamma_cross)
    report = validate_affine_vortex_pair(models)
    assert report["same_transverse_and_chiral_spectra"] is True
