#!/usr/bin/env python3
"""Two affine vortex textures with identical Haar-averaged magnetic SANS.

Reuse the paperfig panels, cameras, fonts and detector conventions of Fig. 1.
No font sizes are overridden. The two rows differ in the reference texture
and longitudinal spectrum; the orientation law and both detector maps agree.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import mammos_entity as me

from . import fig_vortex_p4 as base

mpl = base.mpl
plt = base.plt
pf = base.pf

WIDTH_CM = 18.0
HEIGHT_CM = 9.0
DATA_FILENAMES = ("case_study2_1.h5", "case_study2_2.h5")


def load_figure_data(data_dir):
    """Read the displaced circular and centered elliptic cases from MaMMoS."""
    data = [me.from_hdf5(Path(data_dir) / name) for name in DATA_FILENAMES]
    reference = data[0]
    for case in data:
        if case["model"]["psi"]["name"] != "uniform":
            raise ValueError("Figure 2 requires uniform orientation densities.")
        if case["model"]["R"].value != reference["model"]["R"].value:
            raise ValueError("Figure 2 requires matching sphere radii.")
        for name in ("q", "theta"):
            if not np.array_equal(case["detector"][name].value, reference["detector"][name].value):
                raise ValueError("Figure 2 requires matching detector points.")
        for family, name in (("derived", "I_sf"), ("basic", "I_chi")):
            if not np.allclose(
                case["SANS_cross_sections"]["analytical"][family][name].value,
                reference["SANS_cross_sections"]["analytical"][family][name].value,
                rtol=1e-12, atol=1e-14,
            ):
                raise ValueError(f"Figure 2 requires identical analytical {name} maps.")
    return data


def make_figure(data=None):
    if data is None:
        data = load_figure_data(base.PROJECT_ROOT / "output" / "data")
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
    sf_max = max(np.max(case["SANS_cross_sections"]["analytical"]["derived"]["I_sf"].value) for case in data)
    chi_max = max(np.max(np.abs(case["SANS_cross_sections"]["analytical"]["basic"]["I_chi"].value)) for case in data)
    for case, (model, cy, title, letters) in zip(data, (
        ("d", 6.4, "displaced circular\nvortex", "abcde"),
        ("e", 2.8, "centered elliptic\nvortex", "fghij"),
    )):
        radius = float(case["model"]["R"].value)
        base.draw_linear_vortex(
            figure, case["real_space"], display_radius=0.9 * radius,
            pos_cm=(0.02, cy - 1.575),
        )
        label(title, 1.6, cy + 1.7, linespacing=1.1)
        base.draw_orientation_sphere(figure, case["model"], pos_cm=(3.85, cy - 1.45))
        label(r"$\psi(\mathbf R)=1,\ \langle\mathbf M\rangle_{\psi,\mathbf r}=\mathbf0$",
              5.3, cy + 1.6)

        spectra = case["average_gamma_tensor"]["analytical"]["spectra"]
        scattering = case["SANS_cross_sections"]["analytical"]
        qr = case["detector"]["q_axis"].value * radius
        gamma0 = np.dot(case["model"]["m0"].value, case["model"]["m0"].value)
        curves = [
            {"x": qr, "y": spectra["perpendicular"].value / gamma0, "color": options.colors[1],
             "linewidth": options.linewidth, "label": r"$\Gamma_\perp$"},
            {"x": qr, "y": spectra["parallel"].value / gamma0, "color": options.colors[0],
             "linewidth": options.linewidth, "linestyle": "--",
             "label": r"$\Gamma_\parallel$"},
            {"x": qr, "y": spectra["cross"].value / gamma0, "color": options.colors[2],
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

        q_z_r = case["detector"]["qz"].value * radius
        q_y_r = case["detector"]["qy"].value * radius
        sf_axis = base.draw_detector_map(
            figure,
            q_z_r, q_y_r, scattering["derived"]["I_sf"].value / sf_max,
            pos_cm=(11.65, cy - 1.275),
        )
        sf_axis.set_ylabel(r"$q_yR$", labelpad=0.5)
        chi_axis = base.draw_detector_map(
            figure,
            q_z_r, q_y_r, scattering["basic"]["I_chi"].value / chi_max,
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
    parser.add_argument("--data-dir", type=Path, default=base.PROJECT_ROOT / "output" / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=base.PROJECT_ROOT / "output")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_figure_data(args.data_dir)
    report = {
        "data_files": [str((args.data_dir / name).resolve()) for name in DATA_FILENAMES],
        "source": "analytical",
        "volume_means": [case["average_magnetization"]["analytical"]["volume_mean"].value.tolist() for case in data],
    }
    report["paperfig_module"] = str(pf.__file__)
    report["figure_size_cm"] = [WIDTH_CM, HEIGHT_CM]
    report["PNG_dpi"] = 600
    report["texture_color"] = "M_{z,0}/|M_0|, RdBu, linear scale [0, 1]"
    report["spin_flip_display"] = "magma, linear, common scale [0, 1]"
    report["chiral_display"] = "RdBu_r, linear, common symmetric scale [-1, 1]"
    output = args.output_dir / "fig2"
    figure = make_figure(data)
    figure.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
