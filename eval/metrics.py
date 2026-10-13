# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""FID and Inception Score utilities for PixelRush evaluation.

The FID implementation expects reference statistics saved as an `.npz` file with
one of these key pairs:

- `mu`, `sigma`
- `mean`, `cov`

Use `compute_reference_stats(...)` from this file to create compatible stats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(path: str | Path) -> list[Path]:
    """Return sorted image paths from a directory or a text file with one path per line."""
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".txt":
        image_paths = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    elif path.is_dir():
        image_paths = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        image_paths = [path]
    else:
        raise FileNotFoundError(f"No images found at: {path}")

    if not image_paths:
        raise ValueError(f"No supported images found at: {path}")

    return image_paths


class ImageDataset(Dataset[torch.Tensor]):
    """Image dataset for Inception-based metrics."""

    def __init__(self, image_paths: Iterable[str | Path], image_size: int = 299) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=Inception_V3_Weights.IMAGENET1K_V1.transforms().mean,
                    std=Inception_V3_Weights.IMAGENET1K_V1.transforms().std,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


class InceptionFeatureExtractor(nn.Module):
    """Torchvision Inception-v3 wrapper returning pool features and logits."""

    def __init__(self) -> None:
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.model.Conv2d_1a_3x3(images)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.fc(self.model.dropout(features))
        return features, logits


def extract_inception_outputs(
    image_dir: str | Path,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Inception pool features and class probabilities for images."""
    image_paths = list_images(image_dir)
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(device)
    model = InceptionFeatureExtractor().to(device)

    all_features: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(dataloader, desc=f"Extracting Inception features from {image_dir}"):
        batch = batch.to(device, non_blocking=True)
        features, logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        all_features.append(features.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_probs, axis=0)


def compute_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance for FID."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def save_reference_stats(
    image_dir: str | Path,
    output_npz: str | Path,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    num_workers: int = 4,
) -> None:
    """Compute and save reference Inception stats for future FID evaluation."""
    features, _ = extract_inception_outputs(
        image_dir=image_dir,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    mu, sigma = compute_feature_stats(features)

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, mu=mu, sigma=sigma, num_images=len(features))
    print(f"Saved reference stats to: {output_npz}")


def load_reference_stats(stats_npz: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FID reference stats from an `.npz` file."""
    data = np.load(stats_npz)

    if "mu" in data and "sigma" in data:
        return data["mu"], data["sigma"]
    if "mean" in data and "cov" in data:
        return data["mean"], data["cov"]

    raise KeyError(
        f"{stats_npz} must contain either keys ('mu', 'sigma') or ('mean', 'cov'). "
        f"Found keys: {list(data.keys())}"
    )


def reference_stats_path(real_cache_dir: str | Path) -> Path:
    """Return the canonical reference stats path for a real-image cache directory."""
    return Path(real_cache_dir) / "reference_stats.npz"


def save_reference_stats_from_cache(
    real_cache_dir: str | Path,
    output_npz: str | Path | None = None,
    feature_filename: str = "real_features.npy",
) -> Path:
    """Compute FID reference stats from cached real-image features.

    This is the expected publish workflow when a precomputed `real_cache_dir`
    is provided. The cache directory should contain `real_features.npy`, as in
    the earlier evaluation code. The output defaults to:

    `<real_cache_dir>/reference_stats.npz`
    """
    real_cache_dir = Path(real_cache_dir)
    features_path = real_cache_dir / feature_filename
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing cached real features: {features_path}. "
            "Expected real_cache_dir to contain real_features.npy."
        )

    features = np.load(features_path)
    mu, sigma = compute_feature_stats(features)

    output_path = (
        Path(output_npz) if output_npz is not None else reference_stats_path(real_cache_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        mu=mu,
        sigma=sigma,
        num_images=len(features),
        source_features=str(features_path),
    )
    print(f"Saved reference stats to: {output_path}")
    return output_path


def calculate_fid(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Compute Fréchet Inception Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm(
            (sigma1 + np.eye(sigma1.shape[0]) * eps) @ (sigma2 + np.eye(sigma2.shape[0]) * eps)
        )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)


def calculate_inception_score(probs: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """Compute Inception Score mean and standard deviation."""
    if len(probs) < splits:
        splits = max(1, len(probs))

    split_scores = []
    for split in np.array_split(probs, splits):
        py = np.mean(split, axis=0, keepdims=True)
        kl = split * (np.log(split + 1e-16) - np.log(py + 1e-16))
        split_scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def evaluate_directory(
    image_dir: str | Path,
    ref_stats_npz: str | Path,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    num_workers: int = 4,
    inception_splits: int = 10,
    output_json: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Compute FID against precomputed reference stats and IS for generated images."""
    ref_stats_npz = Path(ref_stats_npz)
    if not ref_stats_npz.exists():
        raise FileNotFoundError(
            f"Reference stats not found: {ref_stats_npz}. "
            "Provide a valid stats .npz file with `--ref_stats`."
        )

    ref_mu, ref_sigma = load_reference_stats(ref_stats_npz)

    features, probs = extract_inception_outputs(
        image_dir=image_dir,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    gen_mu, gen_sigma = compute_feature_stats(features)

    fid = calculate_fid(ref_mu, ref_sigma, gen_mu, gen_sigma)
    is_mean, is_std = calculate_inception_score(probs, splits=inception_splits)

    results: dict[str, float | int | str] = {
        "image_dir": str(image_dir),
        "ref_stats_npz": str(ref_stats_npz),
        "num_images": int(len(features)),
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
    }

    print(json.dumps(results, indent=2))

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved evaluation results to: {output_json}")

    return results


def force_memory_cleanup() -> None:
    """Release CUDA cache after metric evaluation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_memory_usage(label: str = "Memory") -> None:
    """Print CUDA memory usage when CUDA is available."""
    if not torch.cuda.is_available():
        print(f"{label}: CUDA is not available.")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{label}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
