# SO(3)-covariant magnetic SANS

This repository contains the analytical models, validation tests, numerical
data, and rendering sources for the Letter on SO(3)-covariant
powder-averaged magnetic SANS.

The data organization follows the physics rather than the paper layout. Two
independent `mammos-entity` HDF5 files represent the two data products:

- `data1.h5` contains one linear-vortex
  texture, three orientation distributions, their moments, the common
  correlation spectra, and the resulting magnetic SANS cross sections.
- `data2.h5` contains two affine textures with equal
  transverse and chiral spectra, different longitudinal spectra, and identical
  magnetic SANS cross sections after the Halpern--Johnson projection.

Neither file contains figure groups or rendering-normalized arrays. The figure
scripts load the data-product files and construct their linear color
normalizations from the stored raw cross sections. This keeps the physical
data independent of a particular visualization.

The Pixi environment pins the `paperfig` revision and provides
`mammos-entity`, NumPy, Matplotlib, PyVista, Poppler, Pytest, and JupyterLab. A
working LaTeX installation must be available on `PATH` for Matplotlib's TeX
rendering.

```sh
pixi run check
pixi run test
pixi run data
pixi run kernel
pixi run notebook
pixi run fig1
pixi run fig2
```

- `pixi run check` evaluates the standalone affine-pair diagnostic.
- `pixi run test` runs the model and data-product tests.
- `pixi run data` writes both MaMMoS data-product files to `output/data/`.
- `pixi run kernel` registers the repository's Pixi environment in the user
  Jupyter kernel directory under the project-specific name used by the
  notebook and VS Code.
- `pixi run notebook` registers that kernel and opens the MaMMoS-aware
  inspection notebook.
- `pixi run fig1` and `pixi run fig2` first refresh the data products and then
  render the paper figures.

## VS Code

Open this repository itself as the VS Code workspace after creating the Pixi
environment with `pixi install`, and run `pixi run kernel` once. This writes a
small project-specific kernelspec to the user Jupyter directory so that VS Code
can discover the Pixi interpreter. The checked-in
`.vscode/settings.json` selects `.pixi/envs/default/bin/python` for Python
scripts, Pytest, and notebook kernel discovery, and adds `python/` to the
analysis import path. The notebook metadata references the unique
`so3_covariant_magnetic_sans` kernel instead of the ambiguous generic
`python3` kernel. VS Code will also recommend the official Python and Jupyter
extensions when they are missing. If the notebook was already open while the
kernel was registered, reload the VS Code window once.

## MaMMoS data products

`data1.h5` has this structure:

```text
data1.h5
├── metadata
├── model
├── real_space_field
├── orientation_distributions
│   ├── uniform
│   ├── hemisphere
│   └── P4
├── correlation_spectra
├── magnetic_SANS_cross_sections
└── validation
```

`data2.h5` has this structure:

```text
data2.h5
├── metadata
├── orientation_distribution
├── displaced_circular_vortex
│   ├── model
│   ├── real_space_field
│   ├── correlation_spectra
│   └── magnetic_SANS_cross_sections
├── centered_elliptic_vortex
│   ├── model
│   ├── real_space_field
│   ├── correlation_spectra
│   └── magnetic_SANS_cross_sections
└── validation
```

Every primary numerical quantity is a `mammos_entity.Entity` with an ontology
label and a quantity-specific description. The files also record the schema
version, generation time, Git commit and worktree state, package versions,
coordinate ordering, Fourier convention, and normalization. They are written
atomically: a temporary HDF5 file is completed before it replaces the final
path.

The current models are dimensionless. Generic ontology concepts such as
`ISQDimensionlessQuantity`, `RatioQuantity`, `Angle`, `Fraction`, and
`NumberOfElements` are therefore paired with explicit descriptions. The data
are not labeled as SI magnetization fields, scattering vectors, or volumetric
cross sections because the corresponding physical scales are not specified.

Generate and inspect the products with:

```python
from pathlib import Path

import mammos_entity as me

data1 = me.from_hdf5(
    Path("output/data/data1.h5")
)
data2 = me.from_hdf5(
    Path("output/data/data2.h5")
)

gamma_perp = data1["correlation_spectra"]["Gamma_perp"]
spin_flip = data1["magnetic_SANS_cross_sections"]["I_sf_raw"]
gamma_parallel_d = data2["displaced_circular_vortex"][
    "correlation_spectra"
]["Gamma_parallel"]

print(gamma_perp.ontology_label)
print(gamma_perp.description)
```

The notebook at `notebooks/inspect_data.ipynb` loads a product through
`mammos_entity` and inspects the stored model, correlation spectra, and
magnetic SANS cross sections.

## Code organization

- `python/magnetization_models.py` defines the general affine field, its
  scalar/symmetric-traceless/antisymmetric decomposition, and the linear-vortex
  and affine-pair parameterizations.
- `python/magnetic_SANS_models.py` evaluates the three analytical correlation
  spectra, the Halpern--Johnson projection, and magnetic channel contractions.
- `python/model_validation.py` contains numerical quadratures and diagnostics
  for both non-identifiability constructions.
- `python/data_products.py` builds the physics-level data objects.
- `python/mammos_export.py` maps them to two independent nested MaMMoS
  collections; `python/export_data.py` writes the HDF5 files.
- `python/data_io.py` reconstructs the typed data-product objects from the
  HDF5 products. Rendering normalizations are computed here from the raw maps.
- `python/fig_vortex_p4.py` and `python/fig_affine_pair.py` contain the paper
  composition and rendering logic and consume the HDF5 products.

The tests cover the sphere form factor and derivative, the general affine
decomposition, real-space fields, tensor and detector symmetries, direct
SO(3) quadrature, the two non-identifiability constructions, HDF5 roundtrips, and the
absence of figure-specific language from the stored data products.

Generated artifacts under `output/` are excluded from version control and can
be reproduced from the pinned environment and repository sources.
