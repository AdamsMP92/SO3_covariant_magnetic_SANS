# SO(3)-covariant magnetic SANS

A small forward pipeline for a sphere-supported affine magnetization model:

```text
Model(m0, m1, R, psi)
  -> global Fourier magnetization
  -> QMC-averaged Gamma
  -> transverse projection G
  -> magnetic basis cross sections
  -> POLARIS, SANSPOL and unpolarized channels
  -> MaMMoS EntityCollection / HDF5
```

The primary data are numerical QMC results. Selected known cases also include
an independent analytical branch on the same detector and real-space grids.

## Build and export data

Use the pinned Pixi environment (`pixi install`). A small complete example is:

```sh
pixi run data --model examples/linear_vortex.json --output output/data/linear_vortex_example.h5 --radial-bins 3 --angular-bins 4 --power 12
pixi run test
```

The JSON example specifies the paper's axial linear vortex with a uniform
northern-hemisphere axis distribution. Export preserves existing files unless
`--overwrite` is supplied. Increase detector resolution and Sobol power for the
intended accuracy; sample counts and observed analytical discrepancies are saved.

The Python API, with `python/` on the import path, is:

```python
import mammos_entity as me
from models import Model, OrientationDistribution
from models.SANS_detector_grid import regular_detector_grid
from models.build_data import build_data

model = Model(
    m0=[0, 0, 1],
    m1=[[0, -7, 0], [7, 0, 0], [0, 0, 0]],
    R=1,
    psi=OrientationDistribution("P4", epsilon=0.9),
)
grid = regular_detector_grid(q_max=5, radial_bins=8, angular_bins=12)
data = build_data(model, grid, power=14, seed=0, output="output/data/p4.h5")

restored = me.from_hdf5("output/data/p4.h5")
gamma_xy = restored["qmc"]["Gamma"]["xy"].value
spin_flip = restored["qmc"]["cross_sections"]["derived"]["I_sf"].value
```

`build_data` returns the collection even when exporting. `export_data(data, path)`
can export a collection later. HDF5 writing uses a temporary file followed by an
atomic replacement.

## Model and analytical selection

`Model` stores real `m0` of shape `(3,)`, real `m1` of shape `(3, 3)`, a positive
sphere radius `R`, and a callable `psi(alpha, beta, gamma)`.

The default density is uniform. Named densities can be selected by passing
`psi="uniform"`, `"hemisphere"`, `"P4"` or `"compensated"`. All are relative to
the normalized SO(3) Haar measure. P4 defaults to epsilon=0.9; use
`OrientationDistribution("P4", epsilon=...)` for another coefficient in the
paper's range `abs(epsilon) <= 1`.

Custom callable densities are supported by QMC and must already satisfy
`integral psi dQ = 1`; they are not numerically renormalized. Their sampled values,
normalization estimate and available callable source are saved. Custom callables
are not automatically classified as an analytical case.

Analytical first moments are available for all four named densities. Analytical
Gamma, G and SANS channels are selected for:

- Any affine model with the uniform density.
- An axial linear vortex with hemisphere, P4 or compensated density, using the
  paper's axis-moment factorization.

The compensated vortex has an anisotropic tensor. Its analytical tensor is
stored without labeling it by three radial spectra. For uniform and the
observationally equivalent hemisphere/P4 vortices, all three radial spectra are
stored; G and the cross sections use only the transverse and chiral spectra.
Other nonuniform affine models receive analytical means and QMC tensors.
Availability flags in `analytical` specify which comparisons exist.

## Saved data

```text
metadata       schema, QMC settings, versions, Git state, source snapshot and hashes
model          m0, m1, R, spin axis P, prefactor K, nuclear/interference inputs
  orientation  density, exact sampled Euler angles, sampled mass and axis moments
detector       q/theta axes and 2D maps, qz/qy maps, Cartesian q vectors, optional edges
real_space     x, y, z, mx, my, mz: local body-frame point arrays after sphere cut
qmc
  Gamma        xx, xy, xz, yx, yy, yz, zx, zy, zz (native complex entities)
  G            all nine native complex components
  cross_sections
    basic      I_M, I_P, I_chi
    derived    I_sf, I_pp, I_mm, I_pm, I_mp, I_plus, I_minus, I_unp
  magnetization
    field_mean       orientation QMC at the interior positions
    volume_mean      exact spatial integral, followed by 3D orientation QMC
    volume_mean_6d   independent joint orientation/volume QMC control
analytical     available exact means, tensors, spectra, channels and discrepancies
```

Every detector component has shape `(n_theta, n_q)`. The beam is along x,
`qx=0`, `qy=q*sin(theta)` and `qz=q*cos(theta)`. Theta is measured in radians
from +z towards +y. Regular grids use bin centers at positive q and also store
bin edges; the transverse projection is undefined at q=0.

The data use dimensionless model coordinates and coefficients, `F(0)=1`, and the
normalized Fourier amplitudes used by the local model. Unit-specific SI labels
are not assigned. The unitary Fourier amplitude is obtained by multiplying by
`V_s/(2*pi)**1.5`; Gamma and G require its square. The supplied `K` multiplies
the stored normalization; its default is 1. Nuclear intensity `I_N` and signed
nuclear-magnetic interference `I_NM` are independent optional inputs, both zero
by default. No rendering color normalization is stored.

Local real-space data are generated by `real_space_grid(m0, m1, R, grid_points=5)`.
The grid uses Cartesian cell centers in `[-R,R]`, keeps only `x*x+y*y+z*z < R*R`,
and evaluates the unrotated local affine field. Its result contains exactly the
six one-dimensional arrays `x`, `y`, `z`, `mx`, `my`, `mz`. Set
`build_data(..., real_space_grid_points=9)` or CLI `--real-space-grid-points 9`
to change the resolution. Optional `real_space_points` receive the same sphere
cut. Pointwise mean fields use the order defined by these stored coordinates.

## Source layout

`python/models/` contains the local/global models, rotation matrix, first and
second moments, projection, basis/derived channels, detector grid and data
builder. The tests exercise the actual QMC chain, analytical eligibility and
lossless complex HDF5 roundtrips. `python/paper_figures/` contains the retained
figure scripts; their rendering integration is separate from this data builder.

Generated files under `output/` and the Pixi environment are excluded from Git.
