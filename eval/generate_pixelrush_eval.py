# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generate 1K, 2K, and 4K PixelRush evaluation images from prompts.

This script is intended for paper evaluation runs:

1. Read prompts from a text file, one prompt per line.
2. Generate up to 1000 base 1K images with SDXL base.
3. Upscale 1K -> 2K -> 4K with PixelRush.
4. Save images in a metric-friendly directory layout:

   output_dir/
   ├── image_1k/
   ├── image_2k/
   ├── image_4k/
   └── eval_config.json

For multi-GPU machines, run this script once per GPU with `--rank` and
`--world_size`, or launch it through `eval/run_eval_workflow.sh`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import PixelRushSDTurbo, PixelRushSDXLTurbo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PixelRush evaluation images.")
    parser.add_argument("--prompt_file", required=True, help="Text file with one prompt per line.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for image_1k/image_2k/image_4k outputs.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Maximum number of prompts/images to process.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "base", "upscale"],
        default="all",
        help="Run full workflow, only 1K base generation, or only PixelRush upscaling.",
    )
    parser.add_argument(
        "--model",
        choices=["sd-turbo", "sdxl-turbo"],
        default="sdxl-turbo",
        help="PixelRush backbone used for 2K/4K upscaling.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda:0. Defaults to cuda:<rank> if available.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Shard rank for multi-process/multi-GPU generation.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of shards for multi-process/multi-GPU generation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="SDXL base generation steps for 1K images.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="SDXL base generation guidance scale.",
    )
    parser.add_argument("--ddim_inv_to_step", type=int, default=249)
    parser.add_argument("--use_temporal_scaling", action="store_true", default=True)
    parser.add_argument("--no_temporal_scaling", action="store_false", dest="use_temporal_scaling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--user_define", type=float, default=2.0)
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate images that already exist."
    )
    return parser.parse_args()


def suppress_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")


def read_prompts(prompt_file: str | Path, limit: int) -> list[str]:
    prompts = [
        line.strip()
        for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in: {prompt_file}")
    return prompts[:limit]


def sharded_indices(num_items: int, rank: int, world_size: int) -> list[int]:
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got rank={rank}")
    return list(range(rank, num_items, world_size))


def image_to_tensor(image_path: str | Path, device: str | torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.array(image).copy()
    tensor = torch.from_numpy(array).float() / 255.0
    tensor = (2 * tensor - 1.0).clamp(-1, 1)
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return tensor.to(device)


def save_tensor_image(image_tensor: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_array = (image_tensor.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    Image.fromarray(image_array).save(output_path)


def save_with_pipeline(pipeline: Any, image_tensor: torch.Tensor, output_path: str | Path) -> None:
    if hasattr(pipeline, "_save_image"):
        pipeline._save_image(image_tensor, str(output_path))
    else:
        save_tensor_image(image_tensor, output_path)


def build_pixelrush_pipeline(model: str, device: str) -> Any:
    if model == "sd-turbo":
        return PixelRushSDTurbo(
            model_id="stabilityai/sd-turbo",
            device=device,
        )

    return PixelRushSDXLTurbo(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        turbo_model_id="stabilityai/sdxl-turbo",
        device=device,
    )


def filter_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = set(inspect.signature(function).parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def build_pixelrush_kwargs(
    args: argparse.Namespace, image: torch.Tensor, prompt: str
) -> dict[str, Any]:
    return {
        "image": image,
        "prompt": prompt,
        "seed": args.seed,
        "ddim_inv_to_step": args.ddim_inv_to_step,
        "upsample": 2,
        "use_temporal_scaling": args.use_temporal_scaling,
        "temperature": args.temperature,
        "user_define": args.user_define,
    }


def generate_base_1k(
    prompts: list[str],
    indices: list[int],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> None:
    image_1k_dir = output_dir / "image_1k"
    image_1k_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed + args.rank)

    for index in tqdm(indices, desc=f"Rank {args.rank} generating 1K"):
        output_path = image_1k_dir / f"image_{index + 1:04d}.png"
        if output_path.exists() and not args.overwrite:
            continue

        image = pipe(
            prompt=prompts[index],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]
        image.save(output_path)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def upscale_pixelrush(
    prompts: list[str],
    indices: list[int],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> None:
    image_1k_dir = output_dir / "image_1k"
    image_2k_dir = output_dir / "image_2k"
    image_4k_dir = output_dir / "image_4k"
    image_2k_dir.mkdir(parents=True, exist_ok=True)
    image_4k_dir.mkdir(parents=True, exist_ok=True)

    if not image_1k_dir.exists():
        raise FileNotFoundError(f"Missing 1K image directory: {image_1k_dir}")

    pipeline = build_pixelrush_pipeline(args.model, device=device)

    for index in tqdm(indices, desc=f"Rank {args.rank} PixelRush 2K/4K"):
        prompt = prompts[index]
        image_1k_path = image_1k_dir / f"image_{index + 1:04d}.png"
        image_2k_path = image_2k_dir / f"image_{index + 1:04d}.png"
        image_4k_path = image_4k_dir / f"image_{index + 1:04d}.png"

        if not image_1k_path.exists():
            print(f"Skipping missing input: {image_1k_path}")
            continue

        if (not image_2k_path.exists()) or args.overwrite:
            image_1k = image_to_tensor(image_1k_path, device)
            kwargs = filter_kwargs(
                pipeline.pixelrush_one_step,
                build_pixelrush_kwargs(args, image_1k, prompt),
            )
            image_2k = pipeline.pixelrush_one_step(**kwargs)
            save_with_pipeline(pipeline, image_2k, image_2k_path)
        else:
            image_2k = image_to_tensor(image_2k_path, device)

        if (not image_4k_path.exists()) or args.overwrite:
            image_2k = image_to_tensor(image_2k_path, device)
            kwargs = filter_kwargs(
                pipeline.pixelrush_one_step,
                build_pixelrush_kwargs(args, image_2k, prompt),
            )
            image_4k = pipeline.pixelrush_one_step(**kwargs)
            save_with_pipeline(pipeline, image_4k, image_4k_path)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_config(args: argparse.Namespace, output_dir: Path, num_prompts: int) -> None:
    config = vars(args).copy()
    config["num_loaded_prompts"] = num_prompts
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_config_path = output_dir / f"eval_config_rank{args.rank}.json"
    rank_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.rank == 0:
        global_config = config.copy()
        global_config["device"] = "per-rank"
        global_config["rank"] = "all"
        (output_dir / "eval_config.json").write_text(
            json.dumps(global_config, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    suppress_warnings()
    args = parse_args()

    if args.device is None:
        args.device = f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)

    output_dir = Path(args.output_dir)
    prompts = read_prompts(args.prompt_file, args.num_images)
    indices = sharded_indices(len(prompts), args.rank, args.world_size)
    save_config(args, output_dir, num_prompts=len(prompts))

    if args.stage in {"all", "base"}:
        generate_base_1k(prompts, indices, output_dir, args, device=args.device)

    if args.stage in {"all", "upscale"}:
        upscale_pixelrush(prompts, indices, output_dir, args, device=args.device)

    print("Done.")


if __name__ == "__main__":
    main()
