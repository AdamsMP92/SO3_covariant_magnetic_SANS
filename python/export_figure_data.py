#!/usr/bin/env python3
"""Export all numerical data underlying the Letter figures as MaMMoS HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

from mammos_export import export_mammos_hdf5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/data/SO3_covariant_magnetic_SANS.h5"),
    )
    args = parser.parse_args()
    output = export_mammos_hdf5(args.output)
    print(output)


if __name__ == "__main__":
    main()
