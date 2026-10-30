"""Figure 2: displaced circular and centered elliptic vortices with psi=1.

Use alpha=pi/6 and kappa*R/m=7 from the main-text figure caption.
Run from the repository root: python -m python.case_studies.case_study2
"""

from pathlib import Path

import numpy as np

from ..models import Model
from ..generate_data.gen_data import gen_data


def main():
    R = 1.0
    m = 1.0
    alpha = np.pi / 6
    kappa = 7 * m / R
    a = kappa * (np.cos(alpha) - np.sqrt(5 / 3) * np.sin(alpha))
    b = kappa * (np.cos(alpha) + np.sqrt(5 / 3) * np.sin(alpha))

    # Displaced circular vortex, first row of Figure 2.
    model1 = Model(
        m0=[m * np.sin(alpha), 0.0, m * np.cos(alpha)],
        m1=[[0.0, -kappa, 0.0], [kappa, 0.0, 0.0], [0.0, 0.0, 0.0]],
        R=R,
        psi="uniform",
    )

    # Centered elliptic vortex, second row of Figure 2.
    model2 = Model(
        m0=[0.0, 0.0, m],
        m1=[[0.0, -a, 0.0], [b, 0.0, 0.0], [0.0, 0.0, 0.0]],
        R=R,
        psi="uniform",
    )

    q = np.linspace(0.05, 5.0, 64) / R
    theta = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    grid_points = 26

    output_dir = Path(__file__).resolve().parents[2] / "output" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_data(
        model1, output_dir / "case_study2_1.h5", q, theta,
        grid_points=grid_points, power=14, seed=0, batch_size=1024,
    )
    gen_data(
        model2, output_dir / "case_study2_2.h5", q, theta,
        grid_points=grid_points, power=14, seed=0, batch_size=1024,
    )
    return 0


if __name__ == "__main__":
    main()
