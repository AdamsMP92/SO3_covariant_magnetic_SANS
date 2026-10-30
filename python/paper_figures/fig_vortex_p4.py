#!/usr/bin/env python3
"""Figure 1 from the analytical data in the three case-study HDF5 files.

The dense 3D vortex stays left. Each of the three horizontal paths then
contains an orientation sphere, a spectrum panel, and SF and chiral maps.
A shared branch and short arrows connect the panels. All spectrum panels
have identical data, normalization, and axes.
All text uses the unmodified paperfig font sizes. The scalar surfaces show
psi=1, a uniform northern hemisphere, and psi=1+0.9*P4(a_z), on a shared
0--2 color scale. All three second axis moments agree, whereas the
hemispherical first moment is nonzero.

Run from the repository root with the project's environment::
    python -m python.paper_figures.fig_vortex_p4
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "so3-paperfig-cache"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VTK_SMP_MAX_THREADS", "2")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pyvista as pv
import mammos_entity as me


import paperfig as pf
import paperfig.panel_3d as paperfig_panel_3d

from ..generate_data.load_model import load_psi
from ..models.model import OrientationDistribution


DENSITY_MIN = 0.0
DENSITY_MAX = 2.0
FIGURE_WIDTH_CM = 18.0
FIGURE_HEIGHT_CM = 12.55
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_CM / 2.54
Q_R_MAX = 5.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILENAMES = tuple(f"case_study1_{index}.h5" for index in (1, 2, 3))


def load_figure_data(data_dir):
    """Read the three generated MaMMoS collections in the paper's row order."""
    data = [me.from_hdf5(Path(data_dir) / name) for name in DATA_FILENAMES]
    for case, expected in zip(data, ("uniform", "hemisphere", "p4")):
        if case["model"]["psi"]["name"] != expected:
            raise ValueError(f"Figure 1 expects the {expected} orientation density in this row.")
        for name in ("m0", "m1", "R"):
            if not np.array_equal(case["model"][name].value, data[0]["model"][name].value):
                raise ValueError("Figure 1 requires the same local texture in all three cases.")
        for name in ("q", "theta"):
            if not np.array_equal(case["detector"][name].value, data[0]["detector"][name].value):
                raise ValueError("Figure 1 requires matching detector points for all three cases.")
    return data


def draw_orientation_sphere(
    figure,
    model=None,
    *,
    pos_cm,
    size_cm=(2.9, 2.9),
) -> None:
    """Use the original sphere/camera, with a continuous scalar surface."""
    orientation = OrientationDistribution("uniform") if model is None else load_psi(model["psi"])

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling()

    if orientation.name == "hemisphere":
        # Separate meshes preserve the exact discontinuity at the equator.
        upper_value = float(orientation(0.0, np.arccos(0.5), 0.0))
        lower_value = float(orientation(0.0, np.arccos(-0.5), 0.0))
        surfaces = [(pv.Sphere(theta_resolution=128, phi_resolution=65,
                              radius=1.0, start_phi=0, end_phi=90), upper_value),
                    (pv.Sphere(theta_resolution=128, phi_resolution=65,
                               radius=1.0, start_phi=90, end_phi=180), lower_value)]
    else:
        surfaces = [(pv.Sphere(radius=1.0, theta_resolution=128,
                               phi_resolution=129), None)]
    for sphere, value in surfaces:
        unit_points = sphere.points / np.linalg.norm(sphere.points, axis=1)[:, None]
        sphere["psi"] = (
            np.full(sphere.n_points, value)
            if value is not None
            else orientation(0.0, np.arccos(np.clip(unit_points[:, 2], -1.0, 1.0)), 0.0)
        )
        plotter.add_mesh(
            sphere, scalars="psi", cmap="viridis",
            clim=(DENSITY_MIN, DENSITY_MAX), opacity=1.0,
            smooth_shading=True, lighting=False,
            show_edges=False, show_scalar_bar=False,
        )

    azimuth = np.linspace(0.0, 2.0 * np.pi, 361)
    equator = np.column_stack(
        (1.02 * np.cos(azimuth), 1.02 * np.sin(azimuth), np.zeros_like(azimuth))
    )
    meridian = np.column_stack(
        (np.cos(azimuth), np.zeros_like(azimuth), np.sin(azimuth))
    )
    equator_ring = pv.Spline(equator, 361).tube(radius=0.009, n_sides=12)
    plotter.add_mesh(
        equator_ring,
        color="#4c5459",
        smooth_shading=True,
        opacity=0.90,
    )
    plotter.add_mesh(
        pv.Spline(meridian, 361),
        color="#9da5aa",
        line_width=1.0,
        opacity=0.45,
    )

    z_axis = pv.Arrow(
        start=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        tip_length=0.12,
        tip_radius=0.045,
        shaft_radius=0.010,
        scale=1.40,
    )
    plotter.add_mesh(z_axis, color="#303030", smooth_shading=True)

    plotter.camera_position = [
        (3.0, -5.0, 2.1),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 1.50

    image = plotter.screenshot(return_img=True)
    plotter.clear()
    plotter.close()

    axis, _ = pf.plot2D_panel_core(
        figure, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.flipud(image),
        pos_cm=pos_cm, size_cm=size_cm, xlabel="", ylabel="",
        xticks=(), yticks=(),
    )
    axis.axis("off")
    pf.add_label_cm(
        figure,
        r"$\mathbf e_z$",
        pos_cm[0] + 0.57 * size_cm[0],
        pos_cm[1] + 0.94 * size_cm[1],
        ha="left",
        va="center",
        clip_on=False,
    )


def draw_detector_map(
    figure,
    q_z_r: np.ndarray,
    q_y_r: np.ndarray,
    intensity: np.ndarray,
    *,
    pos_cm,
    size_cm=(2.55, 2.55),
    chiral: bool = False,
    xlabel="",
    ylabel="",
):
    """Draw the SF mean or the signed chiral half-difference."""
    color_map = mpl.colormaps["RdBu_r" if chiral else "magma"].copy()
    color_map.set_bad("white")
    shading = "flat"
    if q_z_r.shape == intensity.shape:
        # Close the periodic angle axis while retaining the stored point coordinates.
        q_z_r = np.concatenate([q_z_r, q_z_r[:1]], axis=0)
        q_y_r = np.concatenate([q_y_r, q_y_r[:1]], axis=0)
        intensity = np.concatenate([intensity, intensity[:1]], axis=0)
        shading = "gouraud"
    axis, mesh = pf.plot2D_pcolormesh_panel_core(
        figure, q_z_r, q_y_r, intensity,
        pos_cm=pos_cm, size_cm=size_cm, shading=shading,
        cmap=color_map, vmin=-1.0 if chiral else 0.0, vmax=1.0,
        aspect="equal", xlim=(-Q_R_MAX, Q_R_MAX), ylim=(-Q_R_MAX, Q_R_MAX),
        xticks=(-5, 0, 5), yticks=(-5, 0, 5), xlabel=xlabel, ylabel=ylabel,
    )
    mesh.set_rasterized(True)
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            Q_R_MAX,
            facecolor="none",
            edgecolor="0.20",
            linewidth=0.55,
        )
    )
    return axis


def draw_linear_vortex(
    figure: mpl.figure.Figure,
    field,
    *,
    display_radius=None,
) -> mpl.axes.Axes:
    """Render the common 3D vortex texture with paperfig's fast glyph path."""
    positions = np.column_stack([field[axis].value for axis in "xyz"])
    magnetization = np.column_stack([field[f"m{axis}"].value for axis in "xyz"])
    if display_radius is not None:
        inside = np.linalg.norm(positions, axis=1) < display_radius
        positions, magnetization = positions[inside], magnetization[inside]
    norm = np.linalg.norm(magnetization, axis=1)
    axial_fraction = np.divide(magnetization[:, 2], norm, out=np.zeros_like(norm), where=norm != 0)
    original_reference_axes = paperfig_panel_3d.add_reference_axes
    paperfig_panel_3d.add_reference_axes = lambda *args, **kwargs: None
    try:
        axis, _ = pf.quiver3_advanced_panel_fast(
            figure,
            positions[:, 0], positions[:, 1], positions[:, 2],
            magnetization[:, 0], magnetization[:, 1], magnetization[:, 2],
            axial_fraction,
            Cmin=0.0,
            Cmax=1.0,
            cmap="RdBu",
            head_length=0.34,
            stick_radius=0.055,
            head_radius=0.13,
            arrow_scale=0.20,
            subsample=1,
            view="custom",
            cam_pos=(3.0, -3.0, 2.15),
            focal_point=(0.0, 0.0, 0.0),
            up_direction=(0.0, 0.0, 1.0),
            axes_width_cm=3.15,
            axes_pos_x_cm=0.02,
            axes_pos_y_cm=4.725,
            margin_cm=0.16,
            crop_cm=(0.12, 0.12, 0.12, 0.12),
            dpi=600,
        )
    finally:
        paperfig_panel_3d.add_reference_axes = original_reference_axes
    return axis


def add_panel_label(figure, letter, left_cm, top_cm, *, outside=False):
    """Place a panel label using paperfig's unchanged text defaults."""
    return pf.add_label_cm(
        figure, f"({letter})",
        left_cm - 0.65 if outside else left_cm + 0.10,
        top_cm - 0.25 if outside else top_cm - 0.10,
        ha="left", va="top", zorder=20,
    )


def make_figure(data=None) -> mpl.figure.Figure:
    """Vortex -> three rows of sphere -> spectra -> SF/chiral, using paperfig.

    Font sizes are deliberately untouched. Shared labels and a branching
    connector make room for the extra column within the same figure width.
    """
    if data is None:
        data = load_figure_data(PROJECT_ROOT / "output" / "data")
    figure = pf.create_paper_figure(
        width_cm=FIGURE_WIDTH_CM,
        height_cm=FIGURE_HEIGHT_CM,
        use_latex=True,
        fontserif="Computer Modern Roman",
    )
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
    # Preserve the original glyph sampling; the HDF5 grid covers the full sphere.
    draw_linear_vortex(
        figure, data[0]["real_space"],
        display_radius=0.9 * float(data[0]["model"]["R"].value),
    )
    add_panel_label(figure, "a", 0.02, 7.875)

    def label(text, x, y, **kwargs):
        return pf.add_label_cm(figure, text, x, y, centered=True, **kwargs)

    def arrow(start, end):
        patch = FancyArrowPatch(
            (start[0] / FIGURE_WIDTH_CM, start[1] / FIGURE_HEIGHT_CM),
            (end[0] / FIGURE_WIDTH_CM, end[1] / FIGURE_HEIGHT_CM),
            transform=figure.transFigure, arrowstyle="-|>",
            mutation_scale=8.5, linewidth=0.8, color="0.28",
        )
        figure.add_artist(patch)

    label("linear-vortex\ntexture", 1.6, 8.4, linespacing=1.1)
    label(r"$\Gamma/\Gamma_0$", 8.7, 12.15)
    label("identical cross sections", 14.6, 12.13)
    label(r"$I_{{\rm sf},1}^{\perp}=I_{{\rm sf},H}^{\perp}=I_{{\rm sf},\epsilon}^{\perp}$",
          12.925, 11.58)
    label(r"$I_{\chi,1}^{\perp}=I_{\chi,H}^{\perp}=I_{\chi,\epsilon}^{\perp}$",
          16.275, 11.58)

    # The texture feeds a common branch; each row then reads left to right.
    pf.add_line_cm(figure, 3.05, 6.3, 3.35, 6.3, linewidth=0.8, color="0.28")
    pf.add_line_cm(figure, 3.35, 2.9, 3.35, 9.7, linewidth=0.8, color="0.28")

    sf_max = max(np.max(case["SANS_cross_sections"]["analytical"]["derived"]["I_sf"].value) for case in data)
    chi_max = max(np.max(np.abs(case["SANS_cross_sections"]["analytical"]["basic"]["I_chi"].value)) for case in data)
    options = pf.global_options
    for case, (distribution, center_y, title, letters) in zip(data, (
        ("uniform", 9.7, r"$\psi(\mathbf R)=1,\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\mathbf0$", "bcde"),
        ("hemisphere", 6.3, r"$\psi_H(\mathbf R),\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\frac{m_0\mathbf e_z}{2}$", "fghi"),
        ("P4", 2.9, r"$\psi_\epsilon(\mathbf R),\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\mathbf0$", "jklm"),
    )):
        draw_orientation_sphere(figure, case["model"], pos_cm=(3.85, center_y - 1.45))
        label(title, 5.3, center_y + 1.6)

        radius = case["model"]["R"].value
        gamma0 = np.dot(case["model"]["m0"].value, case["model"]["m0"].value)
        spectra = case["average_gamma_tensor"]["analytical"]["spectra"]
        qr = case["detector"]["q_axis"].value * radius
        curves = [
            {"x": qr, "y": spectra["perpendicular"].value / gamma0, "color": options.colors[1],
             "linewidth": options.linewidth, "label": r"$\Gamma_\perp$"},
            {"x": qr, "y": spectra["parallel"].value / gamma0, "color": options.colors[0],
             "linewidth": options.linewidth, "linestyle": "--", "label": r"$\Gamma_\parallel$"},
            {"x": qr, "y": spectra["cross"].value / gamma0, "color": options.colors[2],
             "linewidth": options.linewidth, "linestyle": "-.", "label": r"$\Gamma_\times$"},
        ]

        panel = pf.plotLinLin_panel_core(
            figure, curves=curves,
            pos_cm=(7.35, center_y - 1.35), size_cm=(2.7, 2.7),
            xlabel=r"$qR$" if distribution == "P4" else "", ylabel="",
            xlim=(0, Q_R_MAX), ylim=(-1.2, 2.4),
            xticks=(0, 2, 4), yticks=(-1, 0, 1, 2),
        )
        if distribution == "uniform":
            panel.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1),
                         frameon=False, ncol=3, borderaxespad=0.0,
                         handlelength=1.1, handletextpad=0.4, columnspacing=0.8)
        else:
            panel.get_legend().remove()

        q_z_r = case["detector"]["qz"].value * radius
        q_y_r = case["detector"]["qy"].value * radius
        scattering = case["SANS_cross_sections"]["analytical"]
        intensity = scattering["derived"]["I_sf"].value / sf_max
        chiral = scattering["basic"]["I_chi"].value / chi_max
        detector = draw_detector_map(
            figure, q_z_r, q_y_r, intensity,
            pos_cm=(11.65, center_y - 1.275), ylabel=r"$q_yR$",
            xlabel=r"$q_zR$" if distribution == "P4" else "",
        )
        chiral_detector = draw_detector_map(
            figure, q_z_r, q_y_r, chiral, chiral=True,
            pos_cm=(15.0, center_y - 1.275),
            xlabel=r"$q_zR$" if distribution == "P4" else "",
        )

        arrow((3.35, center_y), (4.05, center_y))
        arrow((6.5, center_y), (6.9, center_y))
        arrow((10.15, center_y), (10.65, center_y))

        for letter, left, top, outside in zip(
            letters, (3.85, 7.35, 11.65, 15.0), (1.45, 1.35, 1.275, 1.275),
            (False, False, True, True),
        ):
            add_panel_label(figure, letter, left, center_y + top, outside=outside)

    # Group all three equal cross sections explicitly, as in the prior figure.
    for start, end in (
        ((17.7, 10.975), (17.9, 10.975)),
        ((17.9, 10.975), (17.9, 1.625)),
        ((17.9, 1.625), (17.7, 1.625)),
    ):
        pf.add_line_cm(figure, *start, *end, linewidth=0.6, color="0.28")

    pf.add_colorbar_cm(
        figure, pos_cm=(0.5, 4.1), size_cm=(2.2, 0.16),
        vmin=0.0, vmax=1.0, cmap="RdBu",
        ticks=(0, 0.5, 1), orientation="horizontal",
    )
    label(r"$M_{z,0}/|\mathbf M_0|$", 1.6, 3.35)

    pf.add_colorbar_cm(
        figure, pos_cm=(4.2, 1.3), size_cm=(2.2, 0.16),
        vmin=DENSITY_MIN, vmax=DENSITY_MAX, cmap="viridis",
        ticks=(0, 1, 2), orientation="horizontal",
    )
    label(r"$\psi(\mathbf R)$", 5.3, 0.55)

    # One colorbar per detector column, shared by all three ensemble rows.
    for axis, left, limits, ticks, title, title_x in (
        (detector, 11.65, (0.0, 1.0), (0, 0.5, 1), r"$I_{\rm sf}^{\perp}$", 11.0),
        (chiral_detector, 15.0, (-1.0, 1.0), (-1, 0, 1), r"$I_\chi^{\perp}$", 14.6),
    ):
        mesh = axis.collections[0]
        colorbar = pf.add_colorbar_cm(
            figure, pos_cm=(left, 0.6), size_cm=(2.55, 0.14),
            vmin=limits[0], vmax=limits[1], cmap=mesh.cmap,
            ticks=ticks, orientation="horizontal",
        )
        colorbar.update_normal(mesh)
        colorbar.set_ticks(ticks)
        colorbar.minorticks_off()
        label(title, title_x, 0.67)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "output" / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_figure_data(args.data_dir)
    report = {
        "data_files": [str((args.data_dir / name).resolve()) for name in DATA_FILENAMES],
        "source": "analytical",
        "volume_means": [case["average_magnetization"]["analytical"]["volume_mean"].value.tolist() for case in data],
    }
    report["paperfig_module"] = str(pf.__file__)
    report["figure_size_inches"] = [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN]
    report["spin_flip_display"] = "magma, linear, common scale [0, 1]"
    report["texture_color"] = "M_{z,0}/|M_0|, RdBu, linear scale [0, 1]"
    report["chiral_display"] = "RdBu_r, linear, common symmetric scale [-1, 1]"
    output_stem = args.output_dir / "fig1"
    figure = make_figure(data)
    figure.savefig(output_stem.with_suffix(".png"), dpi=600)
    plt.close(figure)
    output_stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
