#!/usr/bin/env python3
"""Extend the original paperfig composition to three orientation ensembles.

The dense 3D vortex stays left. Each of the three horizontal paths then
contains an orientation sphere, a spectrum panel, and SF and chiral maps.
A shared branch and short arrows connect the panels. All spectrum panels
have identical data, normalization, and axes.
All text uses the unmodified paperfig font sizes. The scalar surfaces show
psi=1, a uniform northern hemisphere, and psi=1+0.9*P4(a_z), on a shared
0--2 color scale. All three second axis moments agree, whereas the
hemispherical first moment is nonzero.

Run with the project's environment containing paperfig/PyVista::
    python python/fig_vortex_p4.py --output-dir /tmp/so3-vortex-p4
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "so3-paperfig-cache"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VTK_SMP_MAX_THREADS", "2")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pyvista as pv


import paperfig as pf
import paperfig.panel_3d as paperfig_panel_3d

from data_products import (
    LinearVortexOrientationData,
    OrientationDistributionData,
    Q_R_MAX,
    RealSpaceFieldData,
)
from data_io import load_linear_vortex_orientation_data


DENSITY_MIN = 0.0
DENSITY_MAX = 2.0
FIGURE_WIDTH_CM = 18.0
FIGURE_HEIGHT_CM = 12.55
FIGURE_WIDTH_IN = FIGURE_WIDTH_CM / 2.54
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_CM / 2.54


def draw_orientation_sphere(
    axis: mpl.axes.Axes,
    distribution: str,
    orientation: OrientationDistributionData | None = None,
) -> None:
    """Use the original sphere/camera, with a continuous scalar surface."""

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling()

    if distribution == "hemisphere":
        # Separate meshes preserve the exact discontinuity at the equator.
        upper_value = (
            2.0
            if orientation is None
            else float(np.interp(0.5, orientation.z, orientation.density))
        )
        lower_value = (
            0.0
            if orientation is None
            else float(np.interp(-0.5, orientation.z, orientation.density))
        )
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
            else (
                np.ones(sphere.n_points)
                if orientation is None
                else np.interp(
                    unit_points[:, 2], orientation.z, orientation.density
                )
            )
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

    axis.imshow(image)
    axis.axis("off")
    axis.text(
        0.57,
        0.94,
        r"$\mathbf e_z$",
        transform=axis.transAxes,
        ha="left",
        va="center",
        clip_on=False,
    )


def draw_detector_map(
    axis: mpl.axes.Axes,
    q_z_r: np.ndarray,
    q_y_r: np.ndarray,
    intensity: np.ndarray,
    *,
    chiral: bool = False,
) -> None:
    """Draw the SF mean or the signed chiral half-difference."""
    color_map = mpl.colormaps["RdBu_r" if chiral else "magma"].copy()
    color_map.set_bad("white")
    norm = mpl.colors.Normalize(vmin=-1.0 if chiral else 0.0, vmax=1.0)
    axis.pcolormesh(
        q_z_r,
        q_y_r,
        intensity,
        shading="flat",
        cmap=color_map,
        norm=norm,
        rasterized=True,
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            Q_R_MAX,
            facecolor="none",
            edgecolor="0.20",
            linewidth=0.55,
        )
    )
    axis.set(
        aspect="equal",
        xlim=(-Q_R_MAX, Q_R_MAX),
        ylim=(-Q_R_MAX, Q_R_MAX),
        xticks=(-5, 0, 5),
        yticks=(-5, 0, 5),
    )
    axis.tick_params(direction="out", length=1.8, width=0.5, pad=1.0)


def draw_linear_vortex(
    figure: mpl.figure.Figure,
    field: RealSpaceFieldData,
) -> mpl.axes.Axes:
    """Render the common 3D vortex texture with paperfig's fast glyph path."""
    original_reference_axes = paperfig_panel_3d.add_reference_axes
    paperfig_panel_3d.add_reference_axes = lambda *args, **kwargs: None
    try:
        axis, _ = pf.quiver3_advanced_panel_fast(
            figure,
            field.positions[:, 0],
            field.positions[:, 1],
            field.positions[:, 2],
            field.magnetization[:, 0],
            field.magnetization[:, 1],
            field.magnetization[:, 2],
            field.axial_fraction,
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


def make_figure(
    data: LinearVortexOrientationData | None = None,
) -> mpl.figure.Figure:
    """Vortex -> three rows of sphere -> spectra -> SF/chiral, using paperfig.

    Font sizes are deliberately untouched. Shared labels and a branching
    connector make room for the extra column within the same figure width.
    """
    if data is None:
        data = load_linear_vortex_orientation_data()
    figure = pf.create_paper_figure(
        width_cm=FIGURE_WIDTH_CM,
        height_cm=FIGURE_HEIGHT_CM,
        use_latex=True,
        fontserif="Computer Modern Roman",
    )
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
    draw_linear_vortex(figure, data.field)
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

    scattering = data.scattering
    spectra = data.spectra
    q_z_r = scattering.q_z_r_edges
    q_y_r = scattering.q_y_r_edges
    intensity = scattering.spin_flip_display
    chiral = scattering.chiral_display
    qr = spectra.q_r
    transverse = spectra.gamma_perpendicular
    longitudinal = spectra.gamma_parallel
    cross = spectra.gamma_cross
    options = pf.global_options
    curves = [
        {"x": qr, "y": transverse, "color": options.colors[1],
         "linewidth": options.linewidth, "label": r"$\Gamma_\perp$"},
        {"x": qr, "y": longitudinal, "color": options.colors[0],
         "linewidth": options.linewidth, "linestyle": "--",
         "label": r"$\Gamma_\parallel$"},
        {"x": qr, "y": cross, "color": options.colors[2],
         "linewidth": options.linewidth, "linestyle": "-.",
         "label": r"$\Gamma_\times$"},
    ]
    orientations = {item.name: item for item in data.orientations}
    for distribution, center_y, title, letters in (
        ("uniform", 9.7, r"$\psi(\mathbf R)=1,\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\mathbf0$", "bcde"),
        ("hemisphere", 6.3, r"$\psi_H(\mathbf R),\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\frac{m_0\mathbf e_z}{2}$", "fghi"),
        ("P4", 2.9, r"$\psi_\epsilon(\mathbf R),\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\mathbf0$", "jklm"),
    ):
        sphere = pf.add_axes_cm(figure, 3.85, center_y - 1.45, 2.9, 2.9)
        draw_orientation_sphere(sphere, distribution, orientations[distribution])
        label(title, 5.3, center_y + 1.6)

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

        detector = pf.add_axes_cm(figure, 11.65, center_y - 1.275, 2.55, 2.55)
        draw_detector_map(detector, q_z_r, q_y_r, intensity)
        detector.set_ylabel(r"$q_yR$", labelpad=0.5)
        if distribution == "P4":
            detector.set_xlabel(r"$q_zR$", labelpad=0.5)

        chiral_detector = pf.add_axes_cm(figure, 15.0, center_y - 1.275, 2.55, 2.55)
        draw_detector_map(chiral_detector, q_z_r, q_y_r, chiral, chiral=True)
        if distribution == "P4":
            chiral_detector.set_xlabel(r"$q_zR$", labelpad=0.5)

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
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "so3-vortex-p4")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_linear_vortex_orientation_data()
    report = dict(data.validation)
    report["paperfig_module"] = str(pf.__file__)
    report["figure_size_inches"] = [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN]
    report["spin_flip_display"] = "magma, linear, common scale [0, 1]"
    report["texture_color"] = "M_{z,0}/|M_0|, RdBu, linear scale [0, 1]"
    report["chiral_display"] = "RdBu_r, linear, common symmetric scale [-1, 1]"
    output_stem = args.output_dir / "fig1"
    figure = make_figure(data)
    # Preserve the rendering pipeline, keeping its PDF intermediate temporary.
    with tempfile.TemporaryDirectory(prefix="so3-figure-export-") as directory:
        intermediate = Path(directory) / output_stem.with_suffix(".pdf").name
        figure.savefig(intermediate, dpi=900)
        plt.close(figure)
        subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile",
                        str(intermediate), str(output_stem)], check=True)
    output_stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
