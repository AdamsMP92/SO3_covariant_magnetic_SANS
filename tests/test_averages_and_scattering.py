"""Haar quadrature, physical constraints and the paper's two case studies."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from python.models import Model, OrientationDistribution
from python.models.gamma_tensor import gamma_tensor
from python.models.gamma_tensor_avg import (
    gamma_spectra, gamma_tensor_avg_analytical, gamma_tensor_avg_sobol,
    gamma_tensor_avg_vortex_analytical,
)
from python.models.global_affine_model_avg import (
    affine_magnetization_avg_analytical, affine_magnetization_avg_sobol,
    average_magnetization_analytical,
)
from python.models.G_tensor_avg import G_tensor_avg, G_tensor_avg_from_spectra, transverse_projector
from python.models.local_affine_model import F_u
from python.models.SANS_cross_sections_basic import basic_cross_sections, basic_cross_sections_from_spectra
from python.models.SANS_cross_sections_derived import derived_cross_sections


Q = np.array([[0.0, 0.0, 0.7], [0.6, 0.8, 0.0], [-0.4, 0.3, 1.2]])


def test_analytical_gamma_against_deterministic_haar_quadrature(affine_model):
    mu, weights = leggauss(10)
    azimuth = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    alpha, beta, gamma = np.meshgrid(azimuth, np.arccos(mu), azimuth, indexing="ij")
    weights = np.broadcast_to(weights[None, :, None] / (2 * len(azimuth)**2), alpha.shape).ravel()
    tensors = gamma_tensor(
        Q[None, ...], affine_model.m0, affine_model.m1, affine_model.R,
        alpha.ravel()[:, None], beta.ravel()[:, None], gamma.ravel()[:, None],
    )
    expected = np.einsum("n,nqij->qij", weights, tensors)
    actual = gamma_tensor_avg_analytical(Q, affine_model.m0, affine_model.m1, affine_model.R)
    assert_allclose(actual, expected, atol=2e-13)


@pytest.mark.parametrize("distribution", ["uniform", "hemisphere", "p4", "compensated"])
def test_sobol_vortex_averages_match_analytical_reference(vortex_model, distribution):
    psi = OrientationDistribution(distribution)
    settings = dict(psi=psi, power=15, seed=0, batch_size=1024)
    m = vortex_model
    actual = gamma_tensor_avg_sobol(Q, m.m0, m.m1, m.R, **settings)
    expected = gamma_tensor_avg_vortex_analytical(Q, m.m0, m.m1, m.R, distribution=distribution)
    # Fixed scrambled Sobol sequence: numerical integration is not exact arithmetic.
    assert_allclose(actual, expected, atol=5e-4, rtol=0)
    positions = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
    actual_mean = affine_magnetization_avg_sobol(positions, m.m0, m.m1, m.R, **settings)
    expected_mean = affine_magnetization_avg_analytical(positions, m.m0, m.m1, m.R, distribution=distribution)
    assert_allclose(actual_mean, expected_mean, atol=5e-4, rtol=0)


def test_sobol_result_is_independent_of_batch_partition(affine_model):
    m = affine_model
    first = gamma_tensor_avg_sobol(Q[:1], m.m0, m.m1, m.R, power=9, seed=2, batch_size=37)
    second = gamma_tensor_avg_sobol(Q[:1], m.m0, m.m1, m.R, power=9, seed=2, batch_size=128)
    assert_allclose(first, second, atol=1e-15)


def test_averaged_tensor_is_covariant_hermitian_and_positive(affine_model):
    m = affine_model
    tensor = gamma_tensor_avg_analytical(Q, m.m0, m.m1, m.R)
    assert_allclose(tensor, tensor.conj().swapaxes(-1, -2), atol=1e-15)
    assert np.linalg.eigvalsh(tensor).min() >= -1e-14
    rotation = Rotation.from_rotvec([0.3, -0.6, 0.2]).as_matrix()
    rotated = gamma_tensor_avg_analytical(Q @ rotation.T, m.m0, m.m1, m.R)
    assert_allclose(rotated, rotation @ tensor @ rotation.T, atol=2e-13)
    zero = gamma_tensor_avg_analytical(np.zeros(3), m.m0, m.m1, m.R)
    assert_allclose(zero, np.dot(m.m0, m.m0) * np.eye(3) / 3)


def test_projection_removes_arbitrary_longitudinal_power(affine_model):
    m = affine_model
    gamma = gamma_tensor_avg_analytical(Q, m.m0, m.m1, m.R)
    direction = Q / np.linalg.norm(Q, axis=-1, keepdims=True)
    longitudinal = direction[..., :, None] * direction[..., None, :]
    projected = G_tensor_avg(Q, gamma)
    assert_allclose(G_tensor_avg(Q, gamma + 19 * longitudinal), projected, atol=5e-15)
    assert_allclose((projected @ direction[..., None])[..., 0], 0, atol=1e-15)
    assert_allclose(projected, projected.conj().swapaxes(-1, -2), atol=1e-15)
    assert np.linalg.eigvalsh(projected).min() >= -1e-14
    perp, _, cross = gamma_spectra(np.linalg.norm(Q, axis=-1), m.m0, m.m1, m.R)
    assert_allclose(projected, G_tensor_avg_from_spectra(Q, perp, cross), atol=1e-15)


@pytest.mark.parametrize("q", [[0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]])
def test_projection_rejects_undefined_directions(q):
    with pytest.raises(ValueError, match="finite, nonzero"):
        transverse_projector(q)


def test_spin_channels_match_pauli_matrix_transition_amplitudes():
    # A coherent projected amplitude, P=e_z. Check the spin signs independently.
    amplitude = np.array([1.0 + 0.4j, -0.3 + 0.8j, 0.7 - 0.2j])
    G = np.outer(amplitude, amplitude.conj())
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    nuclear = 0.4 + 0.1j
    operator = nuclear * np.eye(2) + np.einsum("i,ijk->jk", amplitude, sigma)
    basic = basic_cross_sections(G)
    channels = derived_cross_sections(
        basic, I_N=abs(nuclear)**2, I_NM=2 * np.real(nuclear.conjugate() * amplitude[2]),
    )
    for name, outgoing, incoming in (("I_pp", 0, 0), ("I_mm", 1, 1), ("I_pm", 1, 0), ("I_mp", 0, 1)):
        assert_allclose(channels[name], abs(operator[outgoing, incoming])**2, atol=1e-15)
    assert_allclose(channels["I_unp"], np.sum(abs(operator)**2) / 2)


def test_tensor_and_spectral_cross_sections_obey_detector_angular_law(affine_model):
    m = affine_model
    theta = np.linspace(-np.pi, np.pi, 9)
    q = np.column_stack([np.zeros_like(theta), np.sin(theta), np.cos(theta)])
    perp, _, cross = gamma_spectra(1.0, m.m0, m.m1, m.R)
    G = G_tensor_avg(q, gamma_tensor_avg_analytical(q, m.m0, m.m1, m.R))
    basic = basic_cross_sections(G)
    spectral = basic_cross_sections_from_spectra(q, perp, cross)
    for name in basic:
        assert_allclose(basic[name], spectral[name], atol=2e-14)
    channels = derived_cross_sections(basic)
    assert_allclose(channels["I_sf"], perp * (1 + np.cos(theta)**2), atol=2e-14)
    assert_allclose(basic["I_chi"], 2 * cross * np.cos(theta), atol=2e-14)
    for name in ("I_pp", "I_mm", "I_pm", "I_mp", "I_plus", "I_minus", "I_unp"):
        assert np.min(channels[name]) >= -1e-14


def test_figure1_equal_correlations_do_not_imply_equal_magnetization(vortex_model):
    m = vortex_model
    tensors = [gamma_tensor_avg_vortex_analytical(Q, m.m0, m.m1, m.R, distribution=name) for name in ("uniform", "hemisphere", "p4")]
    assert_allclose(tensors[0], tensors[1])
    assert_allclose(tensors[0], tensors[2])
    means = [average_magnetization_analytical(m.m0, m.m1, m.R, distribution=name) for name in ("uniform", "hemisphere", "p4")]
    assert_allclose(means, [[0, 0, 0], [0, 0, 0.5], [0, 0, 0]])


def test_figure2_distinct_longitudinal_spectra_give_identical_scattering():
    alpha, kappa = np.pi / 6, 7.0
    a, b = kappa * (np.cos(alpha) + np.array([-1, 1]) * np.sqrt(5 / 3) * np.sin(alpha))
    displaced = Model([np.sin(alpha), 0, np.cos(alpha)], [[0, -kappa, 0], [kappa, 0, 0], [0, 0, 0]])
    elliptic = Model([0, 0, 1], [[0, -a, 0], [b, 0, 0], [0, 0, 0]])
    radial = np.linalg.norm(Q, axis=-1)
    d = gamma_spectra(radial, displaced.m0, displaced.m1, displaced.R)
    e = gamma_spectra(radial, elliptic.m0, elliptic.m1, elliptic.R)
    assert_allclose(d[0], e[0], atol=1e-14)
    assert_allclose(d[2], e[2], atol=1e-14)
    difference = 4 * kappa**2 * np.sin(alpha)**2 * F_u(radial)**2 / 9
    assert np.max(difference) > 0
    assert_allclose(e[1] - d[1], difference, atol=1e-14)
    for name, values in basic_cross_sections_from_spectra(Q, d[0], d[2]).items():
        assert_allclose(values, basic_cross_sections_from_spectra(Q, e[0], e[2])[name], atol=1e-14)
