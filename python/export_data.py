#!/usr/bin/env python3
"""Export the two magnetic-SANS data products as separate MaMMoS HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

from mammos_export import export_data_products


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/data"),
    )
    args = parser.parse_args()
    for output in export_data_products(args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
