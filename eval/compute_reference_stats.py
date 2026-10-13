# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Compute reference FID stats from a cached real feature directory.

Expected input:

real_cache_dir/
└── real_features.npy

Output by default:

real_cache_dir/
└── reference_stats.npz

This script is intentionally separate from evaluation. The evaluation script only
loads precomputed stats and compares generated images against them.
"""

from __future__ import annotations

import argparse

from metrics import save_reference_stats_from_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FID reference stats from cached real features."
    )
    parser.add_argument(
        "--real_cache_dir",
        required=True,
        help="Directory containing cached real features, typically real_features.npy.",
    )
    parser.add_argument(
        "--feature_filename",
        default="real_features.npy",
        help="Cached feature filename inside real_cache_dir.",
    )
    parser.add_argument(
        "--output_npz",
        default=None,
        help="Optional output .npz path. Defaults to <real_cache_dir>/reference_stats.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_reference_stats_from_cache(
        real_cache_dir=args.real_cache_dir,
        output_npz=args.output_npz,
        feature_filename=args.feature_filename,
    )


if __name__ == "__main__":
    main()
