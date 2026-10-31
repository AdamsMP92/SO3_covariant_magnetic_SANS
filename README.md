# SO(3)-covariant magnetic SANS

A forward pipeline for a sphere-supported affine magnetization model:

```text
Model(m0, m1, R, psi)
  -> global Fourier magnetization
  -> orientation-averaged Gamma
  -> transverse projection G
  -> basic and derived SANS cross sections
```

Each generated MaMMoS HDF5 file contains the model, its local real-space field,
the averaged magnetization, and the forward chain on a common detector grid.
QMC provides the numerical branch. Known cases also receive an analytical
branch, including the three radial Gamma spectra where applicable.

## Run the case studies and figures

Use the pinned environment and run commands from the repository root:

```sh
pixi install
pixi run test
pixi run data
pixi run fig1
pixi run fig2
```

`data` runs the two case studies sequentially. `data1` and `data2` generate
them individually:

- Case study 1: the same axial linear vortex with uniform, hemisphere and P4
  orientations, saved as `output/data/case_study1_1.h5` through `_3.h5`.
- Case study 2: a displaced circular and a centered elliptic vortex with
  uniform orientations, saved as `output/data/case_study2_1.h5` and `_2.h5`.

Generation requires new filenames and refuses to overwrite existing HDF5 files.
Move existing products aside before regenerating them, or use the Python API
with another filename. Figure commands read the existing HDF5 files and use
their analytical branch; they do not trigger data generation. They save
`output/fig1.png` and `output/fig2.png`, with accompanying JSON reports.
All paper-figure panels use `paperfig`.

The figure modules accept `--data-dir` and `--output-dir`, for example:

```sh
pixi run fig2 --data-dir output/data --output-dir output
pixi run notebook
```

The notebook reads the file selected by its `path` variable. `pixi run check`
is an alias for the test suite.

## Python API and incremental export

```python
from pathlib import Path
import numpy as np
import mammos_entity as me

from python.models import Model, OrientationDistribution
from python.generate_data import gen_data

model = Model(
    m0=[0, 0, 1],
    m1=[[0, -7, 0], [7, 0, 0], [0, 0, 0]],
    R=1,
    psi=OrientationDistribution("p4", epsilon=0.9),
)
filename = Path("output/data/p4_example.h5")
filename.parent.mkdir(parents=True, exist_ok=True)
q = np.linspace(0.05, 5, 8) / model.R
theta = np.linspace(-np.pi, np.pi, 12, endpoint=False)
gen_data(model, filename, q, theta, grid_points=9, power=12, seed=0)

data = me.from_hdf5(filename)
gamma_xy = data["average_gamma_tensor"]["qmc"]["xy"].value
spin_flip = data["SANS_cross_sections"]["analytical"]["derived"]["I_sf"].value
```

`gen_data` returns the filename. It calls these functions in order, each from
the corresponding module in `python.generate_data`:

1. `initialize_hdf5(model, filename)` creates the file and stores the model.
2. `add_magnetization_data(filename, grid_points=...)` stores coordinates and
   the local magnetization.
3. `add_average_magnetization_data(filename, power=..., seed=..., batch_size=...)`
   stores the averaged global field and volume mean.
4. `add_average_gamma_tensor_data(filename, q, theta, power=..., seed=..., batch_size=...)`
   stores the detector grid and averaged Gamma.
5. `add_average_G_tensor_data(filename)` projects the stored Gamma.
6. `add_SANS_cross_sections_data(filename)` evaluates the stored G, or the
   analytical spectra when available.

Each step reads its model and existing results from the HDF5 file and refuses
to replace a group that already exists. Writes are incremental: if a later
step fails, earlier groups remain available for inspection or continuation.
The orientation density is specified once in the model.

## Model and analytical cases

`Model` stores real `m0` of shape `(3,)`, real `m1` of shape `(3, 3)`, a positive
sphere radius `R`, and a callable `psi(alpha, beta, gamma)`.

Named densities are `"uniform"` (the default), `"hemisphere"`, `"p4"` and
`"compensated"`, relative to the normalized SO(3) Haar measure. P4 defaults to
epsilon=0.9; use `OrientationDistribution("p4", epsilon=...)` to choose another
coefficient with `abs(epsilon) <= 1`.

Custom densities must be importable Python functions, already normalized to
`integral psi dQ = 1`. The file stores their module and qualified name; that
module must remain available to reload the model. Their code and sampled
orientation weights are not embedded. Custom densities use QMC and are not
automatically assigned an analytical case.

Analytical first moments exist for all four named densities. Analytical Gamma,
G and SANS channels are available for:

- Any affine model with the uniform density.
- An axial linear vortex with hemisphere, P4 or compensated density.

The compensated vortex has an anisotropic analytical tensor without three
radial spectra. Uniform affine models and hemisphere/P4 axial vortices also
store `perpendicular`, `parallel` and `cross` spectra. Only the perpendicular
and cross spectra enter G and the purely magnetic SANS cross sections.
Unsupported analytical branches are omitted.

The spatial volume integral of an affine field over the centered sphere is
exact. Routine generation therefore uses only a 3D Sobol orientation average
for the QMC volume mean. The independent 6D orientation/volume integration in
`python/models/global_affine_model_avg.py` is retained as a reference benchmark
and is not run or exported by the generation pipeline.

## Saved data

```text
model/
  m0, m1, R
  psi/                       name and parameters, or custom module/qualname
real_space/
  x, y, z, mx, my, mz         local body-frame field after the sphere cut
average_magnetization/
  qmc/                       mx, my, mz at the stored positions; volume_mean
  analytical/                same quantities, where available
detector/
  q_axis, theta_axis          one-dimensional point axes
  q, theta, qy, qz            two-dimensional point arrays
average_gamma_tensor/
  qmc/                       all nine complex components xx through zz
  analytical/                same components, where available
    spectra/                 perpendicular, parallel, cross, where applicable
average_G_tensor/
  qmc/                       all nine complex components
  analytical/                same components, where available
SANS_cross_sections/
  parameters/                P, K, I_N, I_NM
  qmc/
    basic/                   I_M, I_P, I_chi
    derived/                 I_sf, I_pp, I_mm, I_pm, I_mp, I_plus, I_minus, I_unp
  analytical/                basic and derived channels, where available
```

Numerical entities retain their ontology label, unit and description. QMC
settings `power`, `seed` and `batch_size` are HDF5 attributes of the respective
QMC groups. The sample count is `2**power`.

Detector arrays and tensor components have shape `(n_theta, n_q)`. The beam
is along x, with `qx=0`, `qy=q*sin(theta)` and `qz=q*cos(theta)`. Theta is in
radians from +z towards +y. Detector q values are positive because the
transverse projection needs a direction. There are no bin-edge arrays.
The figure code repeats the first angular row to close the periodic circle
for display; the stored point grid remains unchanged.

`real_space_grid(x, y, z, R)` constructs coordinates only. Its regular variant
uses `np.linspace(-R, R, grid_points)` along each axis and keeps points with
`x*x+y*y+z*z <= R*R`, including surface points within floating-point roundoff.
`add_magnetization_data` evaluates the local field at those positions and
stores the six one-dimensional arrays shown above.

Data use dimensionless coordinates and coefficients, with `F(0)=1`. To obtain
the unitary Fourier amplitude, multiply the stored normalized amplitude by
`V_s/(2*pi)**1.5`; Gamma and G acquire the square of that factor. Generation
currently fixes `P=e_z`, `K=1` and `I_N=I_NM=0`, recorded under
`SANS_cross_sections/parameters`. Display color normalizations are applied by
the figures.

## Source layout

- `python/models/`: local/global models, rotation, averages, projection,
  cross sections and coordinate grids.
- `python/generate_data/`: incremental MaMMoS HDF5 generation.
- `python/case_studies/`: the model parameters and sampling for the two figures.
- `python/paper_figures/`: paperfig figures reading the analytical HDF5 branch.
- `tests/`: numerical/analytical comparisons, independent reference calculations,
  HDF5 roundtrips and figure-data integration.

Generated files under `output/` and the Pixi environment are excluded from Git.
