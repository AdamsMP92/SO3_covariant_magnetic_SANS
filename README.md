# SO(3)-covariant magnetic SANS

This repository contains the analytical models, validation tests, numerical
figure data, and rendering sources for the two figures in the main Letter on
SO(3)-covariant powder-averaged magnetic SANS.

> **MaMMoS data product.** The numerical content of both figures is exported
> as one structured `mammos-entity` HDF5 file. It records the magnetization
> models, sampled vector fields, orientation distributions, correlation
> spectra, magnetic SANS channels, validation results, and provenance in a
> common machine-readable hierarchy. The file is therefore a reproducible
> representation of the model-to-observable workflow, rather than an
> auxiliary dump of plotted arrays.

The Pixi environment pins the `paperfig` revision used by the figure scripts
and provides `mammos-entity`, NumPy, Matplotlib, PyVista, and Poppler. A
working LaTeX installation must be available on `PATH` for Matplotlib's TeX
rendering. Generated PNG files and numerical validation reports are written
to `output/`; the MaMMoS data product is written to `output/data/`.

```sh
pixi run check
pixi run test
pixi run data
pixi run fig1
pixi run fig2
```

The commands have the following roles:

- `pixi run check` evaluates the standalone affine-pair diagnostic.
- `pixi run test` runs the Pytest model and data-product checks.
- `pixi run data` generates the MaMMoS HDF5 data product.
- `pixi run fig1` and `pixi run fig2` render the paper figures.

The figure renderers are:

- `python/fig_vortex_p4.py`
- `python/fig_affine_pair.py`

## MaMMoS representation of the forward workflow

The central numerical interface is `python/figure_data.py`. Both the figure
renderers and `python/mammos_export.py` consume the same data objects, so the
archived arrays cannot silently diverge from the quantities passed to the
figure panels. The resulting hierarchy follows the scientific organization of
the two constructions:

```text
SO3_covariant_magnetic_SANS.h5
├── metadata
├── figure_1
│   ├── model
│   ├── real_space_field
│   ├── orientation_distributions
│   ├── correlation_spectra
│   ├── sans_cross_sections
│   └── validation
└── figure_2
    ├── orientation_distribution
    ├── displaced_circular_vortex
    │   ├── model
    │   ├── real_space_field
    │   ├── correlation_spectra
    │   └── sans_cross_sections
    ├── centered_elliptic_vortex
    │   ├── model
    │   ├── real_space_field
    │   ├── correlation_spectra
    │   └── sans_cross_sections
    └── validation
```

Using nested `mammos_entity.EntityCollection` objects has three practical
advantages here:

1. **One data contract across the workflow.** Analytical models, validation,
   visualization, and later inverse-analysis tools can exchange the same
   structured object instead of reconstructing conventions from individual
   plotting scripts.
2. **Physics and provenance remain together.** Model parameters, coordinate
   and Fourier conventions, code version, normalization choices, and numerical
   results are stored in the same artifact. Each primary numerical quantity is
   a `mammos_entity.Entity` with an ontology label and a quantity-specific
   description; the surrounding groups document its role in the forward
   workflow.
3. **Raw observables remain distinguishable from visualization data.** The
   unscaled magnetic SANS maps and the linearly normalized arrays used for the
   color plots are stored separately, together with their normalization
   factors.

Generate and load the data product with:

```sh
pixi run data
```

```python
from pathlib import Path

import mammos_entity as me

data = me.from_hdf5(
    Path("output/data/SO3_covariant_magnetic_SANS.h5")
)
gamma_perp = data["figure_1"]["correlation_spectra"]["Gamma_perp"]
spin_flip = data["figure_1"]["sans_cross_sections"]["I_sf_raw"]

print(gamma_perp.ontology_label)  # RatioQuantity
print(gamma_perp.description)
gamma_perp_values = gamma_perp.value
spin_flip_values = spin_flip.value
```

The export is written atomically: a temporary HDF5 file is completed first
and then moved to its final path. Its metadata record the schema version,
generation time, Git commit and worktree state, package versions, coordinate
ordering, Fourier convention, normalization, and the script that generated
the artifact.

The current Letter figures use dimensionless quantities. The export assigns
generic ontology concepts such as `ISQDimensionlessQuantity`, `RatioQuantity`,
`Angle`, `Fraction`, and `NumberOfElements`, together with descriptions that
state the normalization and scientific role of each quantity. It deliberately
does not label `qR`, normalized correlation spectra, or normalized SANS maps as
physical wave vectors, magnetization fields, or volumetric cross sections.
Those more specific concepts and their units can be used when a future data
product supplies the corresponding physical scales.

## Models, validation, and figures

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
`output/data/SO3_covariant_magnetic_SANS.h5`. Generated artifacts under
`output/` are excluded from version control and can be reproduced from the
pinned environment and repository sources.
