"""Add G tensors projected from stored Gamma data to a MaMMoS HDF5 file."""

import h5py
import mammos_entity as me
import numpy as np

from ..models.G_tensor_avg import G_tensor_avg, G_tensor_avg_from_spectra


def add_average_G_tensor_data(filename):
    """Read detector and average_gamma_tensor and add average_G_tensor.

    Project the stored QMC Gamma tensor. For analytical data, use the two
    transverse/chiral spectra when available, otherwise project analytical
    Gamma. Each branch stores nine complex (n_theta, n_q) components.
    Return the written EntityCollection.
    """
    with h5py.File(filename, "r+") as handle:
        if "average_G_tensor" in handle:
            raise ValueError("The file already contains average_G_tensor data.")
        qy = me.from_hdf5(handle["detector/qy"]).value
        qz = me.from_hdf5(handle["detector/qz"]).value
        q_vectors = np.stack([np.zeros_like(qy), qy, qz], axis=-1)
        gamma = me.from_hdf5(handle["average_gamma_tensor"])

        data = me.EntityCollection("Transversely projected G tensors on the stored detector points.")
        methods = ("qmc", "analytical") if "analytical" in gamma else ("qmc",)
        for method in methods:
            source = gamma[method]
            if method == "analytical" and "spectra" in source:
                tensor = G_tensor_avg_from_spectra(
                    q_vectors,
                    source["spectra"]["perpendicular"].value,
                    source["spectra"]["cross"].value,
                )
                description = "Analytical G from stored perpendicular and cross Gamma spectra."
            else:
                Gamma = np.stack([
                    np.stack([source[first + second].value for second in "xyz"], axis=-1)
                    for first in "xyz"
                ], axis=-2)
                tensor = G_tensor_avg(q_vectors, Gamma)
                description = f"G from transverse projection of the stored {method} Gamma tensor."

            branch = me.EntityCollection(description)
            for i, first in enumerate("xyz"):
                for j, second in enumerate("xyz"):
                    component = first + second
                    branch[component] = me.Entity(
                        "ISQDimensionlessQuantity", tensor[..., i, j],
                        description=f"Averaged complex G_{component} on detector/q and detector/theta.",
                    )
            data[method] = branch

        data.to_hdf5(handle, name="average_G_tensor")
    return data
