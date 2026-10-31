"""Follow both case studies through HDF5 into paperfig's 2D panels.

The 3D renderers are skipped here so CI does not need an OpenGL context.
"""

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from python.case_studies import case_study1, case_study2
from python.generate_data import gen_data
from python.paper_figures import fig_vortex_p4, fig_affine_pair


@pytest.fixture(params=[(case_study1, fig_vortex_p4), (case_study2, fig_affine_pair)])
def figure_data(request, tmp_path, monkeypatch):
    case_study, figure_module = request.param
    # Use the actual case-study models with small grids and a short Sobol run.
    monkeypatch.setattr(case_study, "__file__", str(tmp_path / "python/case_studies/case.py"))

    def generate_small(model, filename, q, theta, **settings):
        return gen_data(
            model, filename, q[::16], theta[::12],
            grid_points=3, power=4, seed=0, batch_size=8,
        )

    monkeypatch.setattr(case_study, "gen_data", generate_small)
    assert case_study.main() == 0
    data_dir = tmp_path / "output/data"
    data = figure_module.load_figure_data(data_dir)
    return figure_module, data_dir, data


def test_figures_use_analytical_hdf5_data_and_close_detector_circle(figure_data, monkeypatch):
    module, _, data = figure_data
    monkeypatch.setattr(fig_vortex_p4, "draw_linear_vortex", lambda *args, **kwargs: None)
    monkeypatch.setattr(fig_vortex_p4, "draw_orientation_sphere", lambda *args, **kwargs: None)

    with matplotlib.rc_context():
        figure = module.make_figure(data)
        try:
            spectrum_axes = [axis for axis in figure.axes if len(axis.lines) == 3]
            detector_axes = [axis for axis in figure.axes if axis.patches]
            assert len(spectrum_axes) == len(data)
            assert len(detector_axes) == 2 * len(data)
            sf_max = max(np.max(case["SANS_cross_sections"]["analytical"]["derived"]["I_sf"].value) for case in data)
            chi_max = max(np.max(np.abs(case["SANS_cross_sections"]["analytical"]["basic"]["I_chi"].value)) for case in data)

            for index, case in enumerate(data):
                gamma0 = np.dot(case["model"]["m0"].value, case["model"]["m0"].value)
                spectra = case["average_gamma_tensor"]["analytical"]["spectra"]
                for line, name in zip(spectrum_axes[index].lines, ("perpendicular", "parallel", "cross")):
                    assert_allclose(line.get_ydata(), spectra[name].value / gamma0)

                for offset, family, name, scale in (
                    (0, "derived", "I_sf", sf_max), (1, "basic", "I_chi", chi_max),
                ):
                    mesh = detector_axes[2 * index + offset].collections[0]
                    expected = case["SANS_cross_sections"]["analytical"][family][name].value / scale
                    values = mesh.get_array()
                    assert_allclose(values[:-1], expected)
                    assert_array_equal(values[-1], values[0])
                    coordinates = mesh.get_coordinates()
                    assert_array_equal(coordinates[-1], coordinates[0])
                    assert_allclose(coordinates[:-1, :, 0], case["detector"]["qz"].value * case["model"]["R"].value)
                    assert_allclose(coordinates[:-1, :, 1], case["detector"]["qy"].value * case["model"]["R"].value)
        finally:
            module.plt.close(figure)


@pytest.mark.parametrize("figure_data", [(case_study2, fig_affine_pair)], indirect=True)
def test_figure2_rejects_unequal_cross_sections(figure_data):
    module, data_dir, _ = figure_data
    with h5py.File(data_dir / module.DATA_FILENAMES[1], "r+") as handle:
        channel = handle["SANS_cross_sections/analytical/derived/I_sf"]
        channel[...] = 1.1 * channel[...]
    with pytest.raises(ValueError, match="identical analytical I_sf"):
        module.load_figure_data(data_dir)
