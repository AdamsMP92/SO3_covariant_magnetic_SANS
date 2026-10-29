"""Construct ideal POLARIS, SANSPOL and unpolarized channels from the basis."""

import numpy as np


def derived_cross_sections(basic, *, I_N=0.0, I_NM=0.0):
    """Return derived cross sections from basic['I_M'], ['I_P'] and ['I_chi'].

    I_N is the coherent nuclear cross section and I_NM the signed nuclear-
    magnetic interference contribution, supplied in the same intensity
    scale as the basis. Both default to zero for purely magnetic scattering.
    Spin-incoherent scattering is omitted. The first sign in pp, pm, mp, mm
    denotes the incident spin; the second denotes the analyzed outgoing spin.
    """
    I_M, I_P, I_chi, I_N, I_NM = np.broadcast_arrays(
        np.asarray(basic["I_M"]),
        np.asarray(basic["I_P"]),
        np.asarray(basic["I_chi"]),
        np.asarray(I_N),
        np.asarray(I_NM),
    )
    I_sf = I_M - I_P
    I_pp = I_N + I_P + I_NM
    I_mm = I_N + I_P - I_NM
    I_pm = I_sf + I_chi
    I_mp = I_sf - I_chi
    I_plus = I_pp + I_pm
    I_minus = I_mm + I_mp
    I_unp = (I_plus + I_minus) / 2
    return {
        "I_sf": I_sf,
        "I_pp": I_pp,
        "I_mm": I_mm,
        "I_pm": I_pm,
        "I_mp": I_mp,
        "I_plus": I_plus,
        "I_minus": I_minus,
        "I_unp": I_unp,
    }
