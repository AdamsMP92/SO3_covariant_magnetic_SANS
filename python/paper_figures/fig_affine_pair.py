#!/usr/bin/env python3
"""Two affine vortex textures with identical Haar-averaged magnetic SANS.

Reuse the paperfig panels, cameras, fonts and detector conventions of Fig. 1.
No font sizes are overridden. The two rows differ in the reference texture
and longitudinal spectrum; the orientation law and both detector maps agree.
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import paper_figures.fig_vortex_p4 as base
from data_products import AffinePairData, AffineProjectionData
from data_io import load_affine_projection_data

mpl = base.mpl
plt = base.plt
pf = base.pf

WIDTH_CM = 18.0
HEIGHT_CM = 9.0


def draw_texture(figure, data: AffinePairData, center_y):
    field = data.field
    original = base.paperfig_panel_3d.add_reference_axes
    base.paperfig_panel_3d.add_reference_axes = lambda *args, **kwargs: None
    try:
        pf.quiver3_advanced_panel_fast(
            figure,
            field.positions[:, 0], field.positions[:, 1], field.positions[:, 2],
            field.magnetization[:, 0], field.magnetization[:, 1],
            field.magnetization[:, 2], field.axial_fraction,
            Cmin=0.0, Cmax=1.0,
            cmap="RdBu", head_length=0.34, stick_radius=0.055,
            head_radius=0.13, arrow_scale=0.20, subsample=1,
            view="custom", cam_pos=(3.0, -3.0, 2.15),
            focal_point=(0.0, 0.0, 0.0), up_direction=(0.0, 0.0, 1.0),
            axes_width_cm=3.15, axes_pos_x_cm=0.02,
            axes_pos_y_cm=center_y - 1.575,
            margin_cm=0.16, crop_cm=(0.12, 0.12, 0.12, 0.12), dpi=600,
        )
    finally:
        base.paperfig_panel_3d.add_reference_axes = original


def make_figure(data: AffineProjectionData | None = None):
    if data is None:
        data = load_affine_projection_data()
    figure = pf.create_paper_figure(
        width_cm=WIDTH_CM, height_cm=HEIGHT_CM, use_latex=True,
        fontserif="Computer Modern Roman",
    )
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"

    def label(text, x, y, **kwargs):
        return pf.add_label_cm(figure, text, x, y, centered=True, **kwargs)

    def arrow(x1, x2, y):
        figure.add_artist(base.FancyArrowPatch(
            (x1 / WIDTH_CM, y / HEIGHT_CM), (x2 / WIDTH_CM, y / HEIGHT_CM),
            transform=figure.transFigure, arrowstyle="-|>",
            mutation_scale=8.5, linewidth=0.8, color="0.28",
        ))

    label(r"$\Gamma/\Gamma_0$", 8.7, 8.6)
    label("identical cross sections", 14.6, 8.6)
    label(r"$I_{{\rm sf},d}^{\perp}=I_{{\rm sf},e}^{\perp}$", 12.925, 8.05)
    label(r"$I_{\chi,d}^{\perp}=I_{\chi,e}^{\perp}$", 16.275, 8.05)

    options = pf.global_options
    for data, model, cy, title, letters in (
        (data.circular, "d", 6.4, "displaced circular\nvortex", "abcde"),
        (data.elliptic, "e", 2.8, "centered elliptic\nvortex", "fghij"),
    ):
        draw_texture(figure, data, cy)
        label(title, 1.6, cy + 1.7, linespacing=1.1)
        base.draw_orientation_sphere(figure, pos_cm=(3.85, cy - 1.45))
        label(r"$\psi(\mathbf R)=1,\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\mathbf0$",
              5.3, cy + 1.6)

        spectra = data.spectra
        scattering = data.scattering
        qr = spectra.q_r
        curves = [
            {"x": qr, "y": spectra.gamma_perpendicular, "color": options.colors[1],
             "linewidth": options.linewidth, "label": r"$\Gamma_\perp$"},
            {"x": qr, "y": spectra.gamma_parallel, "color": options.colors[0],
             "linewidth": options.linewidth, "linestyle": "--",
             "label": r"$\Gamma_\parallel$"},
            {"x": qr, "y": spectra.gamma_cross, "color": options.colors[2],
             "linewidth": options.linewidth, "linestyle": "-.",
             "label": r"$\Gamma_\times$"},
        ]
        panel = pf.plotLinLin_panel_core(
            figure, curves=curves, pos_cm=(7.35, cy - 1.35), size_cm=(2.7, 2.7),
            xlabel=r"$qR$" if model == "e" else "", ylabel="",
            xlim=(0, base.Q_R_MAX), ylim=(-1.2, 2.4),
            xticks=(0, 2, 4), yticks=(-1, 0, 1, 2),
        )
        if model == "d":
            panel.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                         frameon=False, ncol=3, borderaxespad=0.0,
                         handlelength=1.1, handletextpad=0.4, columnspacing=0.8)
        else:
            panel.get_legend().remove()

        sf_axis = base.draw_detector_map(
            figure,
            scattering.q_z_r_edges,
            scattering.q_y_r_edges,
            scattering.spin_flip_display,
            pos_cm=(11.65, cy - 1.275),
        )
        sf_axis.set_ylabel(r"$q_yR$", labelpad=0.5)
        chi_axis = base.draw_detector_map(
            figure,
            scattering.q_z_r_edges,
            scattering.q_y_r_edges,
            scattering.chiral_display,
            pos_cm=(15.0, cy - 1.275),
            chiral=True,
        )
        if model == "e":
            sf_axis.set_xlabel(r"$q_zR$", labelpad=0.5)
            chi_axis.set_xlabel(r"$q_zR$", labelpad=0.5)
        arrow(3.2, 4.05, cy)
        arrow(6.5, 6.9, cy)
        arrow(10.15, 10.65, cy)

        for letter, left, top, outside in zip(
            letters, (0.02, 3.85, 7.35, 11.65, 15.0),
            (1.575, 1.45, 1.35, 1.275, 1.275),
            (False, False, False, True, True),
        ):
            base.add_panel_label(figure, letter, left, cy + top, outside=outside)

    for start, end in (
        ((17.7, 7.675), (17.9, 7.675)),
        ((17.9, 7.675), (17.9, 1.525)),
        ((17.9, 1.525), (17.7, 1.525)),
    ):
        pf.add_line_cm(figure, *start, *end, linewidth=0.6, color="0.28")
    pf.add_colorbar_cm(
        figure, pos_cm=(0.5, 0.5), size_cm=(2.2, 0.14),
        vmin=0.0, vmax=1.0, cmap="RdBu", ticks=(0, 0.5, 1), orientation="horizontal",
    )
    label(r"$M_{z,0}/|\mathbf M_0|$", 1.6, 0.88)

    # Share each detector map's actual colormap and linear normalization
    # across the two texture rows.
    for axis, left, limits, ticks, title, title_x in (
        (sf_axis, 11.65, (0.0, 1.0), (0, 0.5, 1), r"$I_{\rm sf}^{\perp}$", 11.0),
        (chi_axis, 15.0, (-1.0, 1.0), (-1, 0, 1), r"$I_\chi^{\perp}$", 14.6),
    ):
        mesh = axis.collections[0]
        colorbar = pf.add_colorbar_cm(
            figure, pos_cm=(left, 0.5), size_cm=(2.55, 0.14),
            vmin=limits[0], vmax=limits[1], cmap=mesh.cmap,
            ticks=ticks, orientation="horizontal",
        )
        colorbar.update_normal(mesh)
        colorbar.set_ticks(ticks)
        colorbar.minorticks_off()
        label(title, title_x, 0.57)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "so3-affine-pair")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_affine_projection_data()
    report = dict(data.validation)
    report["paperfig_module"] = str(pf.__file__)
    report["figure_size_cm"] = [WIDTH_CM, HEIGHT_CM]
    report["PNG_dpi"] = 600
    report["texture_color"] = "M_{z,0}/|M_0|, RdBu, linear scale [0, 1]"
    output = args.output_dir / "fig2"
    figure = make_figure(data)
    # Preserve the rendering pipeline, keeping its PDF intermediate temporary.
    with tempfile.TemporaryDirectory(prefix="so3-figure-export-") as directory:
        intermediate = Path(directory) / output.with_suffix(".pdf").name
        figure.savefig(intermediate, dpi=900)
        plt.close(figure)
        subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile",
                        str(intermediate), str(output)], check=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
