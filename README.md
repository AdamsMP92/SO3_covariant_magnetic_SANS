# SO(3)-covariant magnetic SANS

This repository contains the Python sources used to generate the two figures
in the main Letter on SO(3)-covariant powder-averaged magnetic SANS.

The Pixi environment pins the `paperfig` revision used by the figure scripts
and provides `mammos-entity`, NumPy, Matplotlib, PyVista, and Poppler. A
working LaTeX installation must be available on `PATH` for Matplotlib's TeX
rendering. Generated PNG files and numerical validation reports are written
to `output/`.

```sh
pixi run check
pixi run test
pixi run data
pixi run fig1
pixi run fig2
```

The figure scripts are:

- `python/fig_vortex_p4.py`
- `python/fig_affine_pair.py`

The `tests/` directory contains the automated model checks. These cover the
sphere form factor and its derivative, the general affine decomposition, the
real-space fields, tensor and detector symmetries, joint SO(3) covariance,
the longitudinal null space of the Halpern--Johnson projection, and direct
SO(3) quadrature of both a general affine field and the affine pair. The
affine-pair test carries the numerical content of
`python/check_affine_pair.py` into the Pytest suite while retaining that
script as a standalone diagnostic. GitHub Actions runs the same suite through
`pixi run test`.

The implementation separates the analytical magnetization models from the
magnetic SANS mappings:

- `python/magnetization_models.py` contains the general affine field
  `AffineMagnetizationModel`, its unique scalar/symmetric-traceless/
  antisymmetric decomposition, and parameter functions for the linear vortex
  and affine vortex pair.
- `python/magnetic_SANS_models.py` contains the single general
  `AffineMagneticSANSModel`. It evaluates the three analytical correlation
  spectra with the first-order terms proportional to `[R F'(qR)]^2` and maps
  them to the magnetic SANS channels.
- `python/model_validation.py` contains the numerical quadratures and
  diagnostics for the two non-identifiability constructions.
- `python/figure_data.py` is the common numerical source for the figure
  renderers and data export. It contains the sampled vector fields, all three
  correlation spectra, raw SANS channel maps, their display-normalized forms,
  orientation densities, moments, and validation results.
- `python/mammos_export.py` maps these data to nested
  `mammos_entity.EntityCollection` objects. `python/export_figure_data.py`
  writes the resulting HDF5 data product atomically.
- `python/fig_vortex_p4.py` and `python/fig_affine_pair.py` contain only the
  figure composition and rendering logic.

The figure tasks are intentionally separate because each performs an
off-screen three-dimensional render. `fig1` generates the linear-vortex
orientation-ensemble comparison, while `fig2` generates the affine vortex
pair with identical projected magnetic scattering.

`pixi run data` writes
`output/data/SO3_covariant_magnetic_SANS.h5`. The file contains provenance,
model parameters, sampled body-frame vector fields, orientation distributions,
the three correlation spectra, raw and displayed SANS maps, and numerical
validation reports for both figures. The current paper figures are
dimensionless. Accordingly, the export records their conventions and
normalizations explicitly and does not assign misleading SI ontology labels to
`qR`, normalized correlation spectra, or normalized SANS maps.
