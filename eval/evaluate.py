# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Evaluate generated PixelRush images against precomputed reference stats.

By default, this script loads the provided repository stats file:

    eval/reference_stats.npz

The evaluation script never computes reference statistics; it only loads stats
and compares them to generated images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from metrics import evaluate_directory

DEFAULT_REF_STATS = Path(__file__).resolve().parent / "reference_stats.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and IS for generated images.")
    parser.add_argument(
        "--image_dir", required=True, help="Directory or text file of generated images."
    )
    parser.add_argument(
        "--ref_stats",
        default=str(DEFAULT_REF_STATS),
        help="Reference stats .npz path. Defaults to eval/reference_stats.npz.",
    )
    parser.add_argument(
        "--output_json", default=None, help="Optional path to save metric results JSON."
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--inception_splits", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_directory(
        image_dir=args.image_dir,
        ref_stats_npz=args.ref_stats,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        inception_splits=args.inception_splits,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
