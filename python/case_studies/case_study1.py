"""Figure 1: linear vortex with uniform, hemisphere and P4 orientations.

Run from the repository root: python -m python.case_studies.case_study1
"""

from pathlib import Path

import numpy as np

from ..models import Model
from ..generate_data import gen_data


def main():

    model1 = Model(
        m0=[0.0, 0.0, 1.0],
        m1=[[0.0, -7.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        R=1.0,
        psi="uniform",
    )

    model2 = Model(
            m0=[0.0, 0.0, 1.0],
            m1=[[0.0, -7.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            R=1.0,
            psi="hemisphere",
    )

    model3 = Model(
            m0=[0.0, 0.0, 1.0],
            m1=[[0.0, -7.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            R=1.0,
            psi="p4",
    )

    q = np.linspace(0.05, 5.0, 64) / model1.R
    theta = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    grid_points = 26  # Original vortex sampling: spacing 2R / 25 = 0.08R.


    output1 = Path(__file__).resolve().parents[2] / "output" / "data" / "case_study1_1.h5"
    output1.parent.mkdir(parents=True, exist_ok=True)
    output2 = Path(__file__).resolve().parents[2] / "output" / "data" / "case_study1_2.h5"
    output2.parent.mkdir(parents=True, exist_ok=True)
    output3 = Path(__file__).resolve().parents[2] / "output" / "data" / "case_study1_3.h5"
    output3.parent.mkdir(parents=True, exist_ok=True)


    gen_data(
        model1, output1, q, theta,
        grid_points=grid_points, power=14, seed=0, batch_size=1024,
    )

    gen_data(
            model2, output2, q, theta,
            grid_points=grid_points, power=14, seed=0, batch_size=1024,
    )

    gen_data(
            model3, output3, q, theta,
            grid_points=grid_points, power=14, seed=0, batch_size=1024,
    )


    return 0


if __name__ == "__main__":
    main()
