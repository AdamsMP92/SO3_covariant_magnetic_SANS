"""Add detector coordinates and orientation-averaged Gamma tensors to HDF5."""

import h5py
import mammos_entity as me
import numpy as np

from ..models.SANS_detector_grid import detector_grid
from ..models.gamma_tensor_avg import (
    gamma_spectra,
    gamma_tensor_avg_analytical,
    gamma_tensor_avg_sobol,
    gamma_tensor_avg_vortex_analytical,
    is_axial_vortex,
)
from ..models.model import OrientationDistribution
from .load_model import load_model


def add_average_gamma_tensor_data(
    filename, q, theta, *, power=14, seed=0, batch_size=1024,
):
    """Read the stored model and add detector and average_gamma_tensor data.

    q and theta are increasing 1D axes, with q > 0 and theta in radians.
    Store all nine complex components as (n_theta, n_q) arrays under qmc.
    Supported analytical cases receive the same components; covariant cases
    also receive perpendicular, parallel and cross spectra on detector/q_axis.
    An existing detector must match the supplied points. Return the Gamma
    EntityCollection. The orientation density comes entirely from model/psi.
    """
    for name, value, minimum in (
        ("power", power, 0), ("seed", seed, 0), ("batch_size", batch_size, 1),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    grid = detector_grid(q, theta)

    with h5py.File(filename, "r+") as handle:
        if "average_gamma_tensor" in handle:
            raise ValueError("The file already contains average_gamma_tensor data.")
        if "detector" in handle:
            for name in ("q", "theta"):
                stored = me.from_hdf5(handle[f"detector/{name}"]).value
                if not np.array_equal(stored, grid[name]):
                    raise ValueError("The stored detector does not match the supplied q and theta axes.")

        model = load_model(handle)
        m0, m1, R = model.m0, model.m1, model.R
        distribution = model.psi.name if isinstance(model.psi, OrientationDistribution) else "custom"
        tensors = {
            "qmc": gamma_tensor_avg_sobol(
                grid["q_vectors"], m0, m1, R,
                psi=model.psi, power=power, seed=seed, batch_size=batch_size,
            ),
        }
        axial_vortex = is_axial_vortex(m0, m1)
        if distribution == "uniform":
            tensors["analytical"] = gamma_tensor_avg_analytical(grid["q_vectors"], m0, m1, R)
        elif distribution != "custom" and axial_vortex:
            tensors["analytical"] = gamma_tensor_avg_vortex_analytical(
                grid["q_vectors"], m0, m1, R, distribution=distribution,
            )

        data = me.EntityCollection("Orientation-averaged Gamma tensors on the stored detector points.")
        for method, tensor in tensors.items():
            branch = me.EntityCollection(f"{method} Gamma components; shape (n_theta, n_q).")
            for i, first in enumerate("xyz"):
                for j, second in enumerate("xyz"):
                    component = first + second
                    branch[component] = me.Entity(
                        "ISQDimensionlessQuantity", tensor[..., i, j],
                        description=f"Averaged complex Gamma_{component} on detector/q and detector/theta.",
                    )
            data[method] = branch

        if distribution == "uniform" or (axial_vortex and distribution in ("hemisphere", "p4")):
            spectra = me.EntityCollection("Three covariant Gamma spectra on detector/q_axis.")
            for name, values in zip(
                ("perpendicular", "parallel", "cross"),
                gamma_spectra(grid["q_axis"], m0, m1, R),
            ):
                spectra[name] = me.Entity(
                    "ISQDimensionlessQuantity", values,
                    description=f"Analytical Gamma_{name} spectrum on detector/q_axis.",
                )
            data["analytical"].spectra = spectra

        if "detector" not in handle:
            coordinates = me.EntityCollection("Detector points: qx=0, qy=q*sin(theta), qz=q*cos(theta).")
            for name in ("q_axis", "theta_axis", "q", "theta", "qy", "qz"):
                angle = name.startswith("theta")
                coordinates[name] = me.Entity(
                    "Angle" if angle else "ISQDimensionlessQuantity", grid[name],
                    description=(
                        f"{name} in radians, measured from +z towards +y."
                        if angle else f"Detector {name} in dimensionless model coordinates."
                    ),
                )
            coordinates.to_hdf5(handle, name="detector")
        group = data.to_hdf5(handle, name="average_gamma_tensor")
        group["qmc"].attrs.update(power=power, seed=seed, batch_size=batch_size)
    return data
