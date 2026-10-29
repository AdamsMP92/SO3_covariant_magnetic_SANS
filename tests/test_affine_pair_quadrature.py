"""Independent SO(3)-quadrature test of the affine vortex pair."""

from __future__ import annotations

import numpy as np

from magnetic_SANS_models import AffineMagneticSANSModel
from magnetization_models import (
    affine_vortex_pair,
    cross_matrix,
    sphere_form_factor,
    sphere_form_factor_derivative,
)


def _rotation_quadrature() -> tuple[np.ndarray, np.ndarray]:
    z_coord, z_weights = np.polynomial.legendre.leggauss(12)
    alpha, z_grid, gamma = np.meshgrid(
        2.0 * np.pi * np.arange(12) / 12.0,
        z_coord,
        2.0 * np.pi * np.arange(24) / 24.0,
        indexing="ij",
    )
    weights = np.broadcast_to(
        z_weights[None, :, None] / (2.0 * 12 * 24),
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
    orthogonality_error = rotations @ rotations.transpose(0, 2, 1) - np.eye(3)
    assert np.max(np.abs(orthogonality_error)) < 1.0e-14
    assert np.max(np.abs(np.linalg.det(rotations) - 1.0)) < 1.0e-14
    assert abs(weights.sum() - 1.0) < 1.0e-14
    return rotations, weights


def test_full_so3_average_reproduces_spectra_projection_and_channels() -> None:
    rotations, weights = _rotation_quadrature()
    maximum_errors = {
        "tensor": 0.0,
        "longitudinal_difference": 0.0,
        "projected_tensor": 0.0,
        "channels": 0.0,
        "first_moment": 0.0,
    }
    realization_difference = 0.0

    for angle in (0.0, np.pi / 9.0, np.pi / 6.0):
        magnetization_models = affine_vortex_pair(alpha=angle)
        sans_models = tuple(
            AffineMagneticSANSModel(model) for model in magnetization_models
        )
        fields = tuple((model.m0, model.m1) for model in magnetization_models)
        for q_vector in (
            (0.0, 0.0, 0.3),
            (0.0, 2.0, 1.0),
            (1.0, -2.0, 2.5),
            (0.0, 0.0, 4.8),
            (0.0, 3.0, 0.0),
        ):
            q_vector_array = np.asarray(q_vector, dtype=float)
            qr = float(np.linalg.norm(q_vector_array))
            q_hat = q_vector_array / qr
            form_factor = sphere_form_factor(np.array(qr))
            derivative = sphere_form_factor_derivative(np.array(qr))
            longitudinal = np.outer(q_hat, q_hat)
            transverse = np.eye(3) - longitudinal
            gamma_perpendicular, parallel_d, gamma_cross = (
                sans_models[0].correlation_spectra(np.array(qr))
            )
            gamma_perpendicular_e, parallel_e, gamma_cross_e = (
                sans_models[1].correlation_spectra(np.array(qr))
            )
            np.testing.assert_allclose(gamma_perpendicular_e, gamma_perpendicular)
            np.testing.assert_allclose(gamma_cross_e, gamma_cross)
            parallel_spectra = (parallel_d, parallel_e)
            tensors = []
            projected_tensors = []
            channels = []
            single_orientation_amplitudes = []

            for index, (constant, first_order) in enumerate(fields):
                rotated_constant = np.einsum("nij,j->ni", rotations, constant)
                rotated_first_order = np.einsum(
                    "nai,ij,nbj->nab",
                    rotations,
                    first_order,
                    rotations,
                )
                amplitude = form_factor * rotated_constant + 1j * derivative * np.einsum(
                    "nij,j->ni",
                    rotated_first_order,
                    q_hat,
                )
                tensor = np.einsum(
                    "n,ni,nj->ij",
                    weights,
                    amplitude,
                    amplitude.conj(),
                )
                expected = (
                    gamma_perpendicular * transverse
                    + parallel_spectra[index] * longitudinal
                    - 1j * gamma_cross * cross_matrix(q_hat)
                )
                maximum_errors["tensor"] = max(
                    maximum_errors["tensor"],
                    float(np.max(np.abs(tensor - expected))),
                )
                maximum_errors["first_moment"] = max(
                    maximum_errors["first_moment"],
                    float(
                        np.max(
                            np.abs(np.einsum("n,ni->i", weights, amplitude))
                        )
                    ),
                )
                projected = transverse @ tensor @ transverse
                spin_flip = float(np.real(projected[0, 0] + projected[1, 1]))
                chiral = float(
                    np.real(-1j * (projected[0, 1] - projected[1, 0]))
                )
                non_spin_flip = float(np.real(projected[2, 2]))
                channels.append(
                    np.array(
                        [
                            spin_flip + chiral,
                            spin_flip - chiral,
                            non_spin_flip,
                            non_spin_flip,
                            spin_flip + chiral + non_spin_flip,
                            spin_flip - chiral + non_spin_flip,
                            spin_flip + non_spin_flip,
                        ]
                    )
                )
                tensors.append(tensor)
                projected_tensors.append(projected)
                single_orientation_amplitudes.append(
                    transverse
                    @ (form_factor * constant + 1j * derivative * first_order @ q_hat)
                )

            longitudinal_difference = (parallel_e - parallel_d) * longitudinal
            maximum_errors["longitudinal_difference"] = max(
                maximum_errors["longitudinal_difference"],
                float(
                    np.max(
                        np.abs(tensors[1] - tensors[0] - longitudinal_difference)
                    )
                ),
            )
            maximum_errors["projected_tensor"] = max(
                maximum_errors["projected_tensor"],
                float(np.max(np.abs(projected_tensors[1] - projected_tensors[0]))),
            )
            maximum_errors["channels"] = max(
                maximum_errors["channels"],
                float(np.max(np.abs(channels[1] - channels[0]))),
            )
            realization_difference = max(
                realization_difference,
                float(
                    np.linalg.norm(
                        single_orientation_amplitudes[1]
                        - single_orientation_amplitudes[0]
                    )
                ),
            )

    assert max(maximum_errors.values()) < 1.0e-11, maximum_errors
    assert realization_difference > 0.1
