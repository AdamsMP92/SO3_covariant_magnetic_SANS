"""Numerical diagnostics for the model constructions used in the figures."""

from __future__ import annotations

import numpy as np

from magnetic_SANS_models import AffineMagneticSANSModel
from magnetization_models import (
    cross_matrix,
    legendre_p4,
    sphere_form_factor,
    sphere_form_factor_derivative,
)


def validate_linear_vortex_ensembles(
    model: AffineMagneticSANSModel,
    *,
    epsilon: float = 0.9,
) -> dict[str, object]:
    """Validate the three orientation ensembles used in Figure 1."""
    magnetization_model = model.magnetization_model
    m0_amplitude = float(np.linalg.norm(magnetization_model.m0))
    if m0_amplitude == 0.0:
        raise ValueError("the linear-vortex validation requires nonzero m0")
    body_axis = magnetization_model.m0 / m0_amplitude
    circulation = float(magnetization_model.omega @ body_axis)
    assert np.max(
        np.abs(magnetization_model.omega - circulation * body_axis)
    ) < 1.0e-14
    assert abs(magnetization_model.zeta) < 1.0e-14
    assert np.max(np.abs(magnetization_model.symmetric_traceless)) < 1.0e-14
    z_nodes, z_weights = np.polynomial.legendre.leggauss(32)
    phi = 2.0 * np.pi * np.arange(64) / 64.0
    report: dict[str, object] = {}
    moments = {}
    for name in ("uniform", "hemisphere", "P4"):
        z_coord = (z_nodes + 1.0) / 2.0 if name == "hemisphere" else z_nodes
        z_grid, phi_grid = np.meshgrid(z_coord, phi, indexing="ij")
        radius = np.sqrt(1.0 - z_grid**2)
        axes = np.stack(
            (
                radius * np.cos(phi_grid),
                radius * np.sin(phi_grid),
                z_grid,
            ),
            axis=-1,
        ).reshape(-1, 3)
        weights = np.broadcast_to(
            z_weights[:, None] / (2.0 * len(phi)),
            z_grid.shape,
        ).ravel().copy()
        if name == "P4":
            weights *= 1.0 + epsilon * legendre_p4(axes[:, 2])
        mass = float(weights.sum())
        first = np.einsum("n,ni->i", weights, axes)
        second = np.einsum("n,ni,nj->ij", weights, axes, axes)
        second_error = float(np.max(np.abs(second - np.eye(3) / 3.0)))
        assert abs(mass - 1.0) < 1.0e-12
        expected_first = np.array(
            [0.0, 0.0, 0.5 if name == "hemisphere" else 0.0]
        )
        assert np.max(np.abs(first - expected_first)) < 1.0e-12
        assert second_error < 1.0e-12
        moments[name] = (axes, weights, first, second)
        report[name] = {
            "normalization": mass,
            "first_moment": first.tolist(),
            "second_moment": second.tolist(),
            "second_moment_max_error": second_error,
        }

    tensor_error = 0.0
    analytic_error = 0.0
    chiral_error = 0.0
    chiral_ensemble_error = 0.0
    chiral_values = {}
    wavevectors = (
        (0.0, 0.0, 0.2),
        (0.0, 0.0, 1.0),
        (0.0, 2.0, 1.0),
        (1.0, -2.0, 2.5),
        (0.0, 3.0, 4.0),
        (0.0, 2.0, -1.0),
        (0.0, -2.0, 1.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 4.0),
        (0.0, 0.0, 4.8),
    )
    for q_vector in wavevectors:
        q_vector_array = np.asarray(q_vector, dtype=float)
        qr = float(np.linalg.norm(q_vector_array))
        q_hat = q_vector_array / qr
        form_factor = sphere_form_factor(np.array(qr))
        radius_weighted_derivative = (
            magnetization_model.radius
            * sphere_form_factor_derivative(np.array(qr))
        )
        tensors = {}
        for name, (axes, weights, _, _) in moments.items():
            amplitude = (
                m0_amplitude * form_factor * axes
                + 1j
                * circulation
                * radius_weighted_derivative
                * np.cross(axes, q_hat)
            )
            tensors[name] = np.einsum(
                "n,ni,nj->ij",
                weights,
                amplitude,
                amplitude.conj(),
            )
        for name in ("hemisphere", "P4"):
            tensor_error = max(
                tensor_error,
                float(np.max(np.abs(tensors["uniform"] - tensors[name]))),
            )

        gamma_perpendicular, gamma_parallel, gamma_cross = (
            model.correlation_spectra(np.array(qr))
        )
        longitudinal = np.outer(q_hat, q_hat)
        cross = cross_matrix(q_hat)
        analytic = (
            gamma_perpendicular * (np.eye(3) - longitudinal)
            + gamma_parallel * longitudinal
            - 1j * gamma_cross * cross
        )
        analytic_error = max(
            analytic_error,
            float(np.max(np.abs(tensors["uniform"] - analytic))),
        )
        transverse = np.eye(3) - longitudinal
        chiral = {}
        for name, tensor in tensors.items():
            projected = transverse @ tensor @ transverse
            chiral[name] = float(
                np.real(-1j * (projected[0, 1] - projected[1, 0]))
            )
            chiral_error = max(
                chiral_error,
                abs(chiral[name] - 2.0 * gamma_cross * q_hat[2]),
            )
        chiral_ensemble_error = max(
            chiral_ensemble_error,
            *(
                abs(chiral[name] - chiral["uniform"])
                for name in ("hemisphere", "P4")
            ),
        )
        chiral_values[tuple(q_vector_array)] = chiral["uniform"]

    assert tensor_error < 1.0e-12
    assert analytic_error < 1.0e-12
    assert chiral_error < 1.0e-12
    assert chiral_ensemble_error < 1.0e-12
    assert (
        abs(
            chiral_values[(0.0, 2.0, 1.0)]
            + chiral_values[(0.0, 2.0, -1.0)]
        )
        < 1.0e-12
    )
    assert (
        abs(
            chiral_values[(0.0, 2.0, 1.0)]
            - chiral_values[(0.0, -2.0, 1.0)]
        )
        < 1.0e-12
    )
    assert abs(chiral_values[(0.0, 2.0, 0.0)]) < 1.0e-12
    assert chiral_values[(0.0, 0.0, 4.0)] < 0.0 < chiral_values[(0.0, 0.0, 4.8)]

    report["full_tensor_max_ensemble_difference"] = tensor_error
    report["full_tensor_max_analytic_error"] = analytic_error
    report["chiral_max_ensemble_difference"] = chiral_ensemble_error
    report["chiral_max_analytic_error"] = chiral_error
    report["chiral_qz_symmetry"] = "odd"
    report["chiral_qy_symmetry"] = "even"
    report["chiral_radial_sign_check"] = {
        "qR=4.0, theta=0": chiral_values[(0.0, 0.0, 4.0)],
        "qR=4.8, theta=0": chiral_values[(0.0, 0.0, 4.8)],
    }
    report["epsilon"] = epsilon
    report["density_exact_minimum"] = 1.0 - epsilon * 3.0 / 7.0
    report["density_exact_maximum"] = 1.0 + epsilon
    report["m0"] = m0_amplitude
    report["m1R_over_m0"] = (
        circulation * magnetization_model.radius / m0_amplitude
    )
    return report


def validate_affine_vortex_pair(
    models: tuple[AffineMagneticSANSModel, AffineMagneticSANSModel],
    *,
    q_r_max: float = 5.0,
) -> dict[str, object]:
    """Validate the affine invariants that determine the plotted spectra."""
    circular_sans, elliptic_sans = models
    circular = circular_sans.magnetization_model
    elliptic = elliptic_sans.magnetization_model
    assert np.isclose(circular.radius, elliptic.radius)
    assert np.isclose(circular.m0 @ circular.m0, elliptic.m0 @ elliptic.m0)
    assert abs(circular.zeta) < 1.0e-14
    assert abs(elliptic.zeta) < 1.0e-14
    assert np.max(np.abs(circular.symmetric_traceless)) < 1.0e-14
    m = float(np.linalg.norm(circular.m0))
    kappa = float(np.linalg.norm(circular.omega))
    cosine = float(circular.m0 @ circular.omega / (m * kappa))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    sine = float(np.sqrt(max(0.0, 1.0 - cosine**2)))
    alpha = float(np.arctan2(sine, cosine))
    shear = float(elliptic.symmetric_traceless[0, 1])
    assert np.isclose(
        circular.omega @ circular.omega / 3.0,
        elliptic.omega @ elliptic.omega / 3.0
        + np.sum(elliptic.symmetric_traceless**2) / 10.0,
    )
    assert np.isclose(circular.m0 @ circular.omega, elliptic.m0 @ elliptic.omega)
    assert np.allclose(
        circular.m1,
        cross_matrix(circular.omega),
    )
    assert np.isclose(
        elliptic.m1[0, 1],
        -kappa * cosine + shear,
    )
    assert 0.0 < shear < kappa * cosine
    assert m * sine / kappa < circular.radius

    qr = np.linspace(0.0, q_r_max, 601)
    gamma_perpendicular, gamma_parallel_d, gamma_cross = (
        circular_sans.correlation_spectra(qr)
    )
    gamma_perpendicular_e, gamma_parallel_e, gamma_cross_e = (
        elliptic_sans.correlation_spectra(qr)
    )
    np.testing.assert_allclose(gamma_perpendicular_e, gamma_perpendicular)
    np.testing.assert_allclose(gamma_cross_e, gamma_cross)
    assert np.min(gamma_perpendicular - np.abs(gamma_cross)) > -1.0e-12
    assert np.min(gamma_parallel_d) >= 0.0
    assert np.min(gamma_parallel_e - gamma_parallel_d) >= 0.0
    return {
        "alpha_degrees": float(np.degrees(alpha)),
        "kappaR_over_m": kappa * circular.radius / m,
        "core_displacement_over_R": m * sine / (kappa * circular.radius),
        "elliptic_first_order_coefficients_over_kappa": [
            cosine - shear / kappa,
            cosine + shear / kappa,
        ],
        "same_transverse_and_chiral_spectra": True,
        "parallel_difference_prefactor_over_kappa_squared": float(
            4.0 * sine**2 / 9.0
        ),
    }
