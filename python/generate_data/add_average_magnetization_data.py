"""Add orientation- and volume-averaged magnetization to a MaMMoS HDF5 file."""

import h5py
import mammos_entity as me
import numpy as np

from ..models.global_affine_model_avg import (
    affine_magnetization_avg_analytical,
    affine_magnetization_avg_sobol,
    average_magnetization_analytical,
)
from ..models.model import OrientationDistribution
from .load_model import load_model


def add_average_magnetization_data(
    filename, *, power=14, seed=0, batch_size=1024,
):
    """Read model and real_space coordinates and add average_magnetization.

    qmc/mx,my,mz follow the stored coordinate order. qmc/volume_mean uses
    the exact spatial integral of the affine field and 3D orientation QMC
    with 2**power samples and a nonnegative integer seed. Named densities also receive
    analytical fields and volume means. The orientation density is loaded
    from model/psi. Return the written EntityCollection.
    """
    for name, value, minimum in (
        ("power", power, 0), ("seed", seed, 0), ("batch_size", batch_size, 1),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    with h5py.File(filename, "r+") as handle:
        if "average_magnetization" in handle:
            raise ValueError("The file already contains average_magnetization data.")
        model = load_model(handle)
        m0, m1, R = model.m0, model.m1, model.R
        positions = np.column_stack([
            me.from_hdf5(handle[f"real_space/{axis}"]).value for axis in "xyz"
        ])

        distribution = model.psi.name if isinstance(model.psi, OrientationDistribution) else "custom"
        settings = dict(psi=model.psi, power=power, seed=seed, batch_size=batch_size)
        fields = {
            "qmc": affine_magnetization_avg_sobol(positions, m0, m1, R, **settings),
        }
        volume_means = {
            "qmc": affine_magnetization_avg_sobol(np.zeros(3), m0, m1, R, **settings),
        }
        if distribution != "custom":
            fields["analytical"] = affine_magnetization_avg_analytical(
                positions, m0, m1, R, distribution=distribution,
            )
            volume_means["analytical"] = average_magnetization_analytical(
                m0, m1, R, distribution=distribution,
            )

        data = me.EntityCollection(
            "Orientation-averaged global fields at /real_space coordinates and sphere-volume means."
        )
        for method, field in fields.items():
            branch = me.EntityCollection(f"{method} magnetization averages.")
            for index, axis in enumerate("xyz"):
                branch[f"m{axis}"] = me.Entity(
                    "ISQDimensionlessQuantity", field[:, index],
                    description=f"Orientation-averaged global m{axis} at /real_space positions.",
                )
            branch.volume_mean = me.Entity(
                "ISQDimensionlessQuantity", volume_means[method],
                description="Orientation- and sphere-volume-averaged magnetization (x, y, z); exact spatial integral.",
            )
            data[method] = branch
        group = data.to_hdf5(handle, name="average_magnetization")
        group["qmc"].attrs.update(power=power, seed=seed, batch_size=batch_size)
    return data
