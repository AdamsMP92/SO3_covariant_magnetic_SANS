"""Add local real-space magnetization data to an existing MaMMoS HDF5 file."""

import h5py
import mammos_entity as me
import numpy as np

from ..models.local_affine_model import affine_magnetization
from ..models.real_space_grid import regular_real_space_grid


def add_magnetization_data(filename, grid_points=21):
    """Read m0, m1, R from the file and add x, y, z, mx, my, mz under real_space.

    grid_points is the number of Cartesian points per axis before the sphere
    cut. Values use the stored model's dimensionless convention. Return the
    written EntityCollection. The file must exist and real_space must be new.
    """
    with h5py.File(filename, "r+") as handle:
        if "real_space" in handle:
            raise ValueError("The file already contains real_space data.")
        parameters = me.from_hdf5(handle["model"])
        R = parameters["R"].value
        grid = regular_real_space_grid(R=R, grid_points=grid_points)
        positions = np.column_stack([grid[axis] for axis in "xyz"])
        magnetization = affine_magnetization(
            positions, parameters["m0"].value, parameters["m1"].value, R,
        )

        data = me.EntityCollection("Local affine magnetization on a spherical Cartesian grid.")
        for axis in "xyz":
            data[axis] = me.Entity(
                "ISQDimensionlessQuantity", grid[axis],
                description=f"Local {axis} coordinate after the sphere cut.",
            )
        for index, axis in enumerate("xyz"):
            data[f"m{axis}"] = me.Entity(
                "ISQDimensionlessQuantity", magnetization[:, index],
                description=f"Local magnetization component m{axis} at the stored positions.",
            )

        group = data.to_hdf5(handle, name="real_space")
        group.attrs["grid_points_per_axis"] = grid_points
    return data
