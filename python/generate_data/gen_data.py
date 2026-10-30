"""Generate affine-model data and SANS cross sections in one HDF5 file."""

from .initialize_data import initialize_hdf5
from .add_magnetization_data import add_magnetization_data
from .add_average_magnetization_data import add_average_magnetization_data
from .add_average_gamma_tensor_data import add_average_gamma_tensor_data
from .add_average_G_tensor_data import add_average_G_tensor_data
from .add_SANS_cross_sections_data import add_SANS_cross_sections_data


def gen_data(
    model, filename, q, theta, *, grid_points=21, power=14, seed=0, batch_size=1024,
):
    """Run the available data-generation steps in order and return filename.

    q and theta specify the detector axes; grid_points sets the number of
    real-space points per axis before the sphere cut. Both QMC steps use
    the supplied power, seed and batch_size. The output file must be new.
    """
    initialize_hdf5(model, filename)
    add_magnetization_data(filename, grid_points=grid_points)
    add_average_magnetization_data(
        filename, power=power, seed=seed, batch_size=batch_size,
    )
    add_average_gamma_tensor_data(
        filename, q, theta, power=power, seed=seed, batch_size=batch_size,
    )
    add_average_G_tensor_data(filename)
    add_SANS_cross_sections_data(filename)
    return filename
