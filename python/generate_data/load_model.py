"""Read the affine model and its orientation density from stored model data."""

from importlib import import_module
from inspect import isfunction
from operator import attrgetter

import mammos_entity as me

from ..models.model import Model, OrientationDistribution


def load_psi(data):
    """Reconstruct a named density or resolve a stored importable function."""
    name = data["name"]
    if name != "custom":
        if name == "p4":
            return OrientationDistribution(name, epsilon=data["epsilon"].value)
        return OrientationDistribution(name)

    if data["module"] == "__main__":
        raise ValueError("Custom psi must be defined in an importable Python module.")
    try:
        psi = attrgetter(data["qualname"])(import_module(data["module"]))
    except (ImportError, AttributeError) as error:
        raise ValueError("The stored custom psi function cannot be imported.") from error
    if not isfunction(psi):
        raise ValueError("Custom psi must be an importable function; callable instance state is not stored.")
    return psi


def load_model(handle):
    """Load Model from the model group of an open HDF5 file."""
    parameters = me.from_hdf5(handle["model"])
    return Model(
        parameters["m0"].value,
        parameters["m1"].value,
        R=parameters["R"].value,
        psi=load_psi(parameters["psi"]),
    )
