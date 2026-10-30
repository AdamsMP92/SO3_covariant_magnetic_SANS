"""Add normalized magnetic SANS cross sections from stored G data to HDF5."""

import h5py
import mammos_entity as me
import numpy as np

from ..models.SANS_cross_sections_basic import (
    basic_cross_sections,
    basic_cross_sections_from_spectra,
)
from ..models.SANS_cross_sections_derived import derived_cross_sections


def add_SANS_cross_sections_data(filename):
    """Read average_G_tensor and add SANS_cross_sections.

    Use P=e_z, K=1 and I_N=I_NM=0, matching the existing magnetic-model
    defaults; record these inputs under parameters. Each available QMC or
    analytical branch stores basic and derived (n_theta, n_q) intensities.
    The analytical branch uses stored perpendicular/cross Gamma spectra
    when available, otherwise its stored G tensor. Return the collection.
    """
    with h5py.File(filename, "r+") as handle:
        if "SANS_cross_sections" in handle:
            raise ValueError("The file already contains SANS_cross_sections data.")
        G = me.from_hdf5(handle["average_G_tensor"])

        parameters = me.EntityCollection("Normalized magnetic SANS parameters.")
        for name, value, description in (
            ("P", [0.0, 0.0, 1.0], "Spin-quantization direction in global coordinates."),
            ("K", 1.0, "Intensity prefactor on the normalized model scale."),
            ("I_N", 0.0, "Coherent nuclear intensity on the normalized model scale."),
            ("I_NM", 0.0, "Signed nuclear-magnetic interference on the normalized model scale."),
        ):
            parameters[name] = me.Entity(
                "ISQDimensionlessQuantity", value, description=description,
            )
        P, K = parameters["P"].value, parameters["K"].value
        data = me.EntityCollection("SANS cross sections on the stored detector points.", parameters=parameters)

        methods = ("qmc", "analytical") if "analytical" in G else ("qmc",)
        for method in methods:
            if method == "analytical" and "average_gamma_tensor/analytical/spectra" in handle:
                spectra = me.from_hdf5(handle["average_gamma_tensor/analytical/spectra"])
                qy = me.from_hdf5(handle["detector/qy"]).value
                qz = me.from_hdf5(handle["detector/qz"]).value
                q_vectors = np.stack([np.zeros_like(qy), qy, qz], axis=-1)
                basic = basic_cross_sections_from_spectra(
                    q_vectors, spectra["perpendicular"].value, spectra["cross"].value, P=P, K=K,
                )
                description = "Analytical cross sections from stored perpendicular and cross Gamma spectra."
            else:
                source = G[method]
                tensor = np.stack([
                    np.stack([source[first + second].value for second in "xyz"], axis=-1)
                    for first in "xyz"
                ], axis=-2)
                basic = basic_cross_sections(tensor, P=P, K=K)
                description = f"Cross sections from the stored {method} G tensor."

            derived = derived_cross_sections(
                basic, I_N=parameters["I_N"].value, I_NM=parameters["I_NM"].value,
            )
            branch = me.EntityCollection(description)
            for family, intensities in (("basic", basic), ("derived", derived)):
                channels = me.EntityCollection(f"{family} magnetic SANS channels.")
                for name, values in intensities.items():
                    channels[name] = me.Entity(
                        "ISQDimensionlessQuantity", values,
                        description=f"{name} on detector/q and detector/theta in normalized model intensity units.",
                    )
                branch[family] = channels
            data[method] = branch

        data.to_hdf5(handle, name="SANS_cross_sections")
    return data
