"""Small deterministic models; all generated files live in pytest's tmp_path."""

import pytest

from python.models import Model


@pytest.fixture
def affine_model():
    return Model(
        m0=[0.3, -0.5, 0.8],
        m1=[[0.4, -0.7, 0.2], [0.9, -0.1, 0.3], [-0.2, 0.5, 0.6]],
        R=1.3,
        psi="uniform",
    )


@pytest.fixture
def vortex_model():
    return Model(
        m0=[0.0, 0.0, 1.0],
        m1=[[0.0, -7.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        R=1.0,
        psi="uniform",
    )
