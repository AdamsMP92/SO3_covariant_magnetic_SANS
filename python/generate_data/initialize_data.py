"""Stepwise export of affine model data to MaMMoS HDF5."""

import h5py
import mammos_entity as me

from ..models.model import OrientationDistribution, uniform_psi
from .load_model import load_psi


def initialize_hdf5(model, filename):
    """Create a new HDF5 file containing model information; return its collection.

    Numerical parameters use the model's dimensionless convention. Named psi
    distributions are stored by name and parameters. Custom densities must
    be importable Python functions and are recorded by module and qualified
    name; their code is provided by that module. An existing filename raises
    FileExistsError.
    """
    psi = me.EntityCollection(
        "Orientation density relative to the normalized SO(3) Haar measure."
    )
    if isinstance(model.psi, OrientationDistribution):
        psi.name = model.psi.name
        if model.psi.name == "p4":
            psi.epsilon = me.Entity(
                "ISQDimensionlessQuantity", model.psi.epsilon,
                description="Coefficient in psi = 1 + epsilon * P4(cos(beta)).",
            )
    elif model.psi is uniform_psi:
        psi.name = "uniform"
    else:
        function = model.psi if hasattr(model.psi, "__qualname__") else type(model.psi)
        psi.name = "custom"
        psi.module = function.__module__
        psi.qualname = function.__qualname__
        if load_psi(psi) is not model.psi:
            raise ValueError("Custom psi must resolve to the original importable function.")

    parameters = me.EntityCollection(
        "Local affine model M(x) = m0 + m1 @ x in dimensionless coordinates.",
        m0=me.Entity(
            "ISQDimensionlessQuantity", model.m0,
            description="Constant local magnetization vector; shape (3,).",
        ),
        m1=me.Entity(
            "ISQDimensionlessQuantity", model.m1,
            description="Local magnetization gradient matrix; shape (3, 3).",
        ),
        R=me.Entity(
            "ISQDimensionlessQuantity", model.R,
            description="Sphere radius in dimensionless model coordinates.",
        ),
        psi=psi,
    )
    data = me.EntityCollection("Affine magnetization data.", model=parameters)
    with h5py.File(filename, "x") as handle:
        data.to_hdf5(handle)
    return data
