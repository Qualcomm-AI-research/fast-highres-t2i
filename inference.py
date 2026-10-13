# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Minimal PixelRush inference script."""

from __future__ import annotations

import argparse
import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src import PixelRushSDTurbo, PixelRushSDXLTurbo

SD_TURBO_MODEL_ID = "stabilityai/sd-turbo"
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_TURBO_MODEL_ID = "stabilityai/sdxl-turbo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PixelRush on one image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument(
        "--model",
        choices=["sd-turbo", "sdxl-turbo"],
        default="sdxl-turbo",
        help="Backbone to use.",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Total upscale factor. Must be 1 or a power of two.",
    )
    parser.add_argument(
        "--output",
        default="output/pixelrush.png",
        help="Output image path.",
    )
    return parser.parse_args()


def suppress_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")


def validate_upscale(upscale: int) -> int:
    if upscale < 1:
        raise ValueError("--upscale must be >= 1.")
    if upscale > 1 and upscale & (upscale - 1) != 0:
        raise ValueError("--upscale must be a power of two, e.g. 2, 4, or 8.")
    return upscale


def load_image(path: str | Path, device: str | torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.array(image).copy()
    tensor = torch.from_numpy(array).float() / 255.0
    tensor = (2 * tensor - 1.0).clamp(-1, 1)
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return tensor.to(device)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = (tensor / 2 + 0.5).clamp(0, 1)
    array = (image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
    array = np.transpose(array, (1, 2, 0))
    Image.fromarray(array).save(path)


def build_pipeline(model: str) -> Any:
    if model == "sd-turbo":
        return PixelRushSDTurbo(
            model_id=SD_TURBO_MODEL_ID,
        )

    return PixelRushSDXLTurbo(
        model_id=SDXL_BASE_MODEL_ID,
        turbo_model_id=SDXL_TURBO_MODEL_ID,
    )


def filter_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    valid_keys = set(inspect.signature(function).parameters)
    return {key: value for key, value in kwargs.items() if key in valid_keys}


def run_pixelrush(
    pipeline: Any,
    image: torch.Tensor,
    prompt: str,
    upscale: int,
) -> torch.Tensor:
    if upscale == 1:
        return image

    kwargs = filter_kwargs(
        pipeline.pixelrush_one_step,
        {
            "image": image,
            "prompt": prompt,
            "upsample": 2,
            "seed": 42,
            "ddim_inv_to_step": 249,
            "use_temporal_scaling": True,
            "temperature": 0.7,
            "user_define": 2.0,
        },
    )

    output = image
    current_upscale = 2
    while current_upscale <= upscale:
        kwargs["image"] = output
        output = pipeline.pixelrush_one_step(**kwargs)
        current_upscale *= 2

    return output


def main() -> None:
    suppress_warnings()
    args = parse_args()
    upscale = validate_upscale(args.upscale)

    pipeline = build_pipeline(args.model)
    device = getattr(pipeline, "device", "cuda" if torch.cuda.is_available() else "cpu")

    image = load_image(args.image, device)
    output = run_pixelrush(pipeline, image, args.prompt, upscale)

    if hasattr(pipeline, "_save_image"):
        pipeline._save_image(output, args.output)
    else:
        save_image(output, args.output)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
