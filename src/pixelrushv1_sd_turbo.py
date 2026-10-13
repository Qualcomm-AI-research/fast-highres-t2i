# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import math
import os
import random
import time
from functools import wraps
from typing import Callable, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from tqdm import tqdm


def profile_time(func_name: str = None) -> Callable:
    """
    Decorator to profile execution time of functions.

    Args:
        func_name: Optional custom name for the function in profiling output
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"[PROFILING] {name}: {execution_time:.2f} seconds")
            return result

        return wrapper

    return decorator


def set_seed(seed: int) -> None:
    """
    Set all random seeds for deterministic results.

    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch's CPU and CUDA random generators
    - CUDNN deterministic mode

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predicted_original_from_noise(
    latent: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Predict x_clean from intermediate timestep x_t."""
    alpha = alphas_cumprod[timestep]
    beta = 1 - alpha
    predicted_original_sample = (latent - beta.sqrt() * noise_pred) / alpha.sqrt()
    return predicted_original_sample


def noise_from_predicted_original(
    latent: torch.Tensor,
    predicted_original: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Extract noise from intermediate timestep x_t."""
    alpha = alphas_cumprod[timestep]
    beta = 1 - alpha
    noise = (latent - alpha.sqrt() * predicted_original) / beta.sqrt()
    return noise


def temporal_scaling_noise_estimate(
    noise_pred,
    timestep,
    alphas_cumprod,
    temperature: float = 1.0,
    user_define: float = 1.0,
):
    """Apply temporal scaling to noise prediction."""
    alpha = alphas_cumprod[timestep]
    sigma = (1 - alpha**2).sqrt()
    snr_ratio = (alpha / sigma) ** 2
    rescale_factor = (snr_ratio * user_define**2 + 1) / (
        snr_ratio * user_define**2 / temperature + 1
    )
    noise_pred_rescaled = noise_pred * rescale_factor
    return noise_pred_rescaled


@torch.no_grad()
def optim_tiled_encode(
    vae: nn.Module,
    x: torch.Tensor,
    sample_size: int,
    tile_size: int,
    overlap_factor: float,
) -> torch.Tensor:
    """Optimization version of tiled VAE encoding, we process tiles in batch-fashion."""
    overlap_size = int(sample_size * (1 - overlap_factor))
    blend_extent = int(tile_size * overlap_factor)
    row_limit = tile_size - blend_extent

    acp_batch_size = 1
    # Split the image into 512x512 tiles and encode them separately.
    rows = []
    batch_tiles = []
    batch_i = []
    batch_j = []
    for i in range(0, x.shape[2], overlap_size):
        row = []
        for j in range(0, x.shape[3], overlap_size):
            tile = x[:, :, i : i + sample_size, j : j + sample_size]
            if tile.shape[-2] == sample_size and tile.shape[-1] == sample_size:
                batch_tiles.append(tile)
                batch_i.append(i // overlap_size)
                batch_j.append(j // overlap_size)
                row.append(-1)
            else:
                tile = vae.encoder(tile)
                if vae.config.use_quant_conv:
                    tile = vae.quant_conv(tile)
                row.append(tile)
        rows.append(row)

    # Handle the same shape tile in batch for faster encode
    batch_tiles = [
        batch_tiles[i : i + acp_batch_size] for i in range(0, len(batch_tiles), acp_batch_size)
    ]
    batch_tiles_result = []
    for b_tile in batch_tiles:
        b_tile = torch.cat(b_tile, dim=0)
        b_tile = vae.encoder(b_tile)
        if vae.config.use_quant_conv:
            b_tile = vae.quant_conv(b_tile)
        batch_tiles_result.append(b_tile)
    batch_tiles_result = torch.cat(batch_tiles_result, dim=0)
    for i, j, tile in zip(batch_i, batch_j, batch_tiles_result):
        rows[i][j] = tile.unsqueeze(0)

    # Combine the result
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    enc = torch.cat(result_rows, dim=2)
    return DiagonalGaussianDistribution(enc)


@torch.no_grad()
def optim_tiled_decode(
    vae: nn.Module,
    z: torch.Tensor,
    sample_size: int,
    tile_size: int,
    overlap_factor: float,
) -> torch.Tensor:
    """Optimization version of tiled VAE decoding, we process tiles in batch-fashion."""
    overlap_size = int(tile_size * (1 - overlap_factor))
    blend_extent = int(sample_size * overlap_factor)
    row_limit = sample_size - blend_extent

    # Split z into overlapping 64x64 tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.

    rows = []
    batch_tiles = []
    batch_i = []
    batch_j = []
    acp_batch_size = 4
    for i in range(0, z.shape[2], overlap_size):
        row = []
        for j in range(0, z.shape[3], overlap_size):
            tile = z[:, :, i : i + tile_size, j : j + tile_size]
            if tile.shape[-2] == tile_size and tile.shape[-1] == tile_size:
                batch_tiles.append(tile)
                batch_i.append(i // overlap_size)
                batch_j.append(j // overlap_size)
                row.append(-1)
            else:
                if vae.config.use_post_quant_conv:
                    tile = vae.post_quant_conv(tile)
                decoded = vae.decoder(tile)
                row.append(decoded)
        rows.append(row)

    # Handle the same shape tile in batch for faster encode
    batch_tiles = [
        batch_tiles[i : i + acp_batch_size] for i in range(0, len(batch_tiles), acp_batch_size)
    ]
    batch_tiles_result = []
    for b_tile in batch_tiles:
        b_tile = torch.cat(b_tile, dim=0)
        if vae.config.use_post_quant_conv:
            b_tile = vae.post_quant_conv(b_tile)
        b_tile = vae.decoder(b_tile)
        batch_tiles_result.append(b_tile)
    batch_tiles_result = torch.cat(batch_tiles_result, dim=0)
    for i, j, tile in zip(batch_i, batch_j, batch_tiles_result):
        rows[i][j] = tile.unsqueeze(0)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=3))

    dec = torch.cat(result_rows, dim=2)
    return dec


class PixelRushSDTurbo:

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        model_revision: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        is_cuda = str(self.device).startswith("cuda")
        self.torch_dtype = torch_dtype or (torch.float16 if is_cuda else torch.float32)

        self._load_components()

        self.ddim_scheduler = DDIMScheduler.from_config(
            StableDiffusionPipeline.from_pretrained(
                model_id,
                revision=self.model_revision,
            ).scheduler.config
        )
        self.alphas_cumprod = self.ddim_scheduler.alphas_cumprod
        self.use_tiling = False

        print(f"SD Pipeline initialized on device: {self.device}")

    def _load_components(self):
        """Load individual components from the Stable Diffusion pipeline."""
        print("Loading SD pipeline components...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            # local_files_only=True,
        )

        self.tokenizer = pipeline.tokenizer
        # Many SD checkpoints expect text encoder in fp32; we follow the SDXL counterpart and keep fp16 on CUDA
        self.text_encoder = pipeline.text_encoder.to(self.device, self.torch_dtype)

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            revision=self.model_revision,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            # local_files_only=True,
        ).to(self.device)

        # UNet
        self.unet = pipeline.unet.to(self.device)

        del pipeline
        print("All SD components loaded successfully!")

    def encode_prompt(
        self, prompt: str, negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to embeddings using SD's single text encoder.
        Returns the conditional and unconditional embeddings.
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            ).last_hidden_state
            negative_text_embeddings = self.text_encoder(
                negative_text_inputs.input_ids.to(self.device)
            ).last_hidden_state

        return text_embeddings, negative_text_embeddings

    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: PIL Image or tensor

        Returns:
            Latent representation
        """
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        image = image.to(self.device, dtype=torch.float32)
        if self.use_tiling:
            latents = optim_tiled_encode(
                self.vae,
                image,
                self.vae.tile_sample_min_size,
                self.vae.tile_latent_min_size,
                overlap_factor=0.125,
            ).sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            with torch.no_grad():
                latents = self.vae.encode(image).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

        return latents.to(self.torch_dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to image space.

        Args:
            latents: Latent representation

        Returns:
            Decoded image tensor
        """
        latents = latents / self.vae.config.scaling_factor
        if self.use_tiling:
            image = optim_tiled_decode(
                self.vae,
                latents.to(torch.float32),
                self.vae.tile_sample_min_size,
                self.vae.tile_latent_min_size,
                overlap_factor=0.125 / 4,
            )
        else:
            with torch.no_grad():
                image = self.vae.decode(latents.to(torch.float32)).sample

        return image

    def ddim_inversion(
        self,
        image: Union[Image.Image, torch.Tensor],
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform DDIM inversion to find the initial noise.
        """
        print("Performing DDIM inversion...")
        if latents is None:
            latents = self.encode_image(image)
        text_embeddings, _ = self.encode_prompt(prompt)

        self.ddim_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddim_scheduler.timesteps

        for t in reversed(timesteps):
            timestep = torch.full((latents.shape[0],), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = self.unet(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample
            latents = self.ddim_scheduler.step(
                noise_pred,
                t,
                latents,
                eta=0.0,
            ).prev_sample
        print("DDIM inversion completed!")
        return latents

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate image using the custom SD pipeline.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)

        text_embeddings, negative_text_embeddings = self.encode_prompt(prompt, negative_prompt)
        encoder_hidden_states = torch.cat([negative_text_embeddings, text_embeddings], dim=0)

        self.ddim_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddim_scheduler.timesteps

        latents = (
            torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
                dtype=self.torch_dtype,
            )
            if latents is None
            else latents
        )

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.ddim_scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample

        image = self.decode_latents(latents)
        if output_path:
            self._save_image(image, output_path)
        return image

    def _save_image(self, image_tensor: torch.Tensor, output_path: str):
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))
        pil_image = Image.fromarray(image_np)
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        pil_image.save(output_path)
        print(f"Image saved to: {output_path}")

    @torch.no_grad()
    @profile_time("PixelRush OneStep Total")
    def pixelrush_one_step(
        self,
        image: torch.Tensor,
        prompt: str,
        seed: Optional[int] = None,
        ddim_inv_to_step: int = 249,
        use_temporal_scaling: bool = False,
        temperature: float = 0.7,
        user_define: float = 2.0,
        upsample: int = 2,
    ) -> torch.Tensor:
        """
        Refine a high-resolution image based on (source) image
        Args:
            image: Source image
            prompt: Generated prompt
            seed: Controls the random generation process
            ddim_inv_to_step: Intermediate timestep
            use_temporal_scaling: Using noise injection
            temperature, userdefine: Hyperparameter of noise injection
            upsample: Scale of upsampling

        Returns:
            Decoded high-resolution image tensor with resolution: upsample x source image resolution
        """
        # Set seed for deterministic results
        if seed is not None:
            set_seed(seed)
        # Utils function for temporal scaling
        if use_temporal_scaling:
            rescale_fn = lambda x, t: temporal_scaling_noise_estimate(
                x,
                t,
                self.alphas_cumprod,
                temperature=temperature,
                user_define=user_define,
            )
        else:
            rescale_fn = lambda x, t: x

        # 1. RGB upsampling
        tiling_threshold = 4096 // 8  # 512 in latent space
        if (
            image.shape[2] * 2 // 8 >= tiling_threshold
            or image.shape[3] * 2 // 8 >= tiling_threshold
        ):
            self.use_tiling = False
            self.vae.enable_tiling()
        init_high_resolution = F.interpolate(image, scale_factor=upsample, mode="bicubic")
        latent_high_resolution = self.encode_image(init_high_resolution)

        # 2. Get DDIM-Inv Intermediate Latent
        # Patchify the latent high resolution to small patches
        anchor_mean, anchor_std = (
            latent_high_resolution.mean(),
            latent_high_resolution.std(),
        )
        views, num_blocks_height, num_blocks_width = self.get_views(
            height=latent_high_resolution.shape[2],
            width=latent_high_resolution.shape[3],
            stride=32,
            window_size=64,
        )
        patch_wise = []
        for h_start, h_end, w_start, w_end in views:
            patch_wise.append(latent_high_resolution[:, :, h_start:h_end, w_start:w_end])
        patch_wise = torch.cat(patch_wise, dim=0)
        repeat_times = patch_wise.shape[0]

        # Prepare text embedding for sd-turbo
        text_embeddings, _ = self.encode_prompt(prompt)
        text_embeddings_ddim_inv = einops.repeat(
            text_embeddings, "b seq d -> (repeat b) seq d", repeat=repeat_times
        )

        ddim_inverted_patch_wise = [patch_wise.clone()]
        forward_patch_wise = [patch_wise.clone()]

        acp_batch_size = 8
        batch_patch_wise = [
            patch_wise[i : i + acp_batch_size] for i in range(0, len(patch_wise), acp_batch_size)
        ]

        # Prepare timestep
        self.ddim_scheduler.set_timesteps(4)
        inv_local_list = []
        forward_local_list = []

        for b_patch_wise in tqdm(batch_patch_wise, desc="DDIM inversion"):
            timestep = torch.full(
                (b_patch_wise.shape[0],),
                ddim_inv_to_step,
                device=self.device,
                dtype=torch.long,
            )
            # Init noise for forward process
            noise = torch.randn_like(b_patch_wise)
            fw_patch_wise = self.ddim_scheduler.add_noise(b_patch_wise, noise, timestep)
            forward_local_list.append(fw_patch_wise)
            # Predict noise for ddim-inv process
            with torch.no_grad():
                noise_pred = self.unet(
                    b_patch_wise,
                    timestep,
                    encoder_hidden_states=text_embeddings_ddim_inv[0 : b_patch_wise.shape[0]],
                ).sample
            current_t = 0
            next_t = ddim_inv_to_step
            alpha_t = self.alphas_cumprod[current_t]
            alpha_t_next = self.alphas_cumprod[next_t]
            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            b_patch_wise = (b_patch_wise - (1 - alpha_t).sqrt() * noise_pred) * (
                alpha_t_next.sqrt() / alpha_t.sqrt()
            ) + (1 - alpha_t_next).sqrt() * noise_pred
            inv_local_list.append(b_patch_wise)

        ddim_inverted_patch_wise.append(torch.cat(inv_local_list, dim=0))
        forward_patch_wise.append(torch.cat(forward_local_list, dim=0))

        # 3. Predict clean patches
        forward_blend_weight = 0.2
        patch_wise = (
            forward_blend_weight * forward_patch_wise[-1]
            + (1 - forward_blend_weight) * ddim_inverted_patch_wise[-1]
        )
        batch_patch_wise = [
            patch_wise[i : i + acp_batch_size] for i in range(0, len(patch_wise), acp_batch_size)
        ]
        new_patch_wise = torch.zeros_like(patch_wise)
        for i, b_patch_wise in enumerate(tqdm(batch_patch_wise, desc="DDIM loop")):
            latent_model_input = self.ddim_scheduler.scale_model_input(
                b_patch_wise, ddim_inv_to_step
            )

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    ddim_inv_to_step,
                    encoder_hidden_states=text_embeddings_ddim_inv[0 : b_patch_wise.shape[0]],
                ).sample

            noise_pred = rescale_fn(noise_pred, ddim_inv_to_step)
            b_patch_wise = self.ddim_scheduler.step(
                noise_pred, ddim_inv_to_step, b_patch_wise
            ).prev_sample
            new_patch_wise[i * acp_batch_size : i * acp_batch_size + b_patch_wise.shape[0]] = (
                b_patch_wise
            )

        patch_wise = new_patch_wise
        latent_high_resolution = self.blend_patch_wise(
            patch_wise,
            latent_high_resolution,
            num_blocks_height=num_blocks_height,
            num_blocks_width=num_blocks_width,
        )

        latent_high_resolution = (
            latent_high_resolution - latent_high_resolution.mean()
        ) / latent_high_resolution.std()
        latent_high_resolution = latent_high_resolution * anchor_std + anchor_mean
        image_high_resolution = self.decode_latents(latent_high_resolution)
        return image_high_resolution

    def blend_patch_wise(
        self,
        patch_wise: torch.Tensor,  # (B x n_patch) x C x H x W
        latent_high_resolution: torch.Tensor,  # B x C x H x W
        num_blocks_height: int = 1,
        num_blocks_width: int = 1,
    ) -> torch.Tensor:
        """Combined overlap patches into a single image."""
        if num_blocks_height == num_blocks_width:
            B = latent_high_resolution.shape[0]
            assert B == 1
            sample_size = patch_wise[0].shape[2]
            result = torch.zeros_like(latent_high_resolution)
            n_views = patch_wise.shape[0] // B
            num_each_row = int(math.sqrt(n_views)) + 1
            rows = []

            overlap_size = int(sample_size * 0.5)
            blend_extent = sample_size - overlap_size
            row_limit = sample_size - blend_extent
            for i in range(num_each_row):
                row = []
                if i < num_each_row - 1:
                    for j in range(num_each_row):
                        if j == num_each_row - 1:
                            row.append(
                                patch_wise[
                                    i * (num_each_row - 1) + j - 1, :, :, -blend_extent:
                                ].unsqueeze(0)
                            )
                        else:
                            row.append(patch_wise[i * (num_each_row - 1) + j].unsqueeze(0))
                else:
                    for j in range(num_each_row):
                        if j == num_each_row - 1:
                            row.append(
                                patch_wise[
                                    (i - 1) * (num_each_row - 1) + j - 1,
                                    :,
                                    -blend_extent:,
                                    -blend_extent:,
                                ].unsqueeze(0)
                            )
                        else:
                            row.append(
                                patch_wise[
                                    (i - 1) * (num_each_row - 1) + j,
                                    :,
                                    -blend_extent:,
                                    :,
                                ].unsqueeze(0)
                            )

                rows.append(row)
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))
            result = torch.cat(result_rows, dim=2)
            return result
        else:
            B, C, H, W = latent_high_resolution.shape
            sample_size = patch_wise[0].shape[2]
            assert B == 1
            result = torch.zeros_like(latent_high_resolution)
            n_views = patch_wise.shape[0] // B
            rows = []

            overlap_size = int(sample_size * 0.5)
            blend_extent = sample_size - overlap_size
            row_limit = sample_size - blend_extent

            num_blocks_height += 1
            num_blocks_width += 1
            for i in range(num_blocks_height):
                row = []
                if i < num_blocks_height - 1:
                    for j in range(num_blocks_width):
                        if j == num_blocks_width - 1:
                            row.append(
                                patch_wise[
                                    i * (num_blocks_width - 1) + j - 1,
                                    :,
                                    :,
                                    -blend_extent:,
                                ].unsqueeze(0)
                            )
                        else:
                            row.append(patch_wise[i * (num_blocks_width - 1) + j].unsqueeze(0))
                else:
                    for j in range(num_blocks_width):
                        if j == num_blocks_width - 1:
                            row.append(
                                patch_wise[
                                    (i - 1) * (num_blocks_width - 1) + j - 1,
                                    :,
                                    -blend_extent:,
                                    -blend_extent:,
                                ].unsqueeze(0)
                            )
                        else:
                            row.append(
                                patch_wise[
                                    (i - 1) * (num_blocks_width - 1) + j,
                                    :,
                                    -blend_extent:,
                                    :,
                                ].unsqueeze(0)
                            )
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent)
                    result_row.append(tile[:, :, :row_limit, :row_limit])
                result_rows.append(torch.cat(result_row, dim=3))
            result = torch.cat(result_rows, dim=2)
            return result

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend tiles in vertical direction."""
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, y, :
            ] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend tiles in horizontal direction."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    def get_profiling_info(self) -> dict:
        return {
            "method": "pixelrush",
            "components": ["DDIM Inversion", "DDIM Loop", "Wavelet Processing"],
            "note": "Use @profile_time decorator for detailed timing",
        }

    def get_views(
        self,
        height: int,
        width: int,
        stride: int = 64,
        window_size: int = 128,
        random_jitter: bool = False,
    ) -> Tuple[List[int], int, int]:
        """Split image into overlapping tiles."""
        num_blocks_height = (
            int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
        )
        num_blocks_width = (
            int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        )
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size

            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter:
                jitter_range = (window_size - stride) // 4
                w_jitter = 0
                h_jitter = 0
                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, jitter_range)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range)
                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, jitter_range)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range)
                h_start += h_jitter + jitter_range
                h_end += h_jitter + jitter_range
                w_start += w_jitter + jitter_range
                w_end += w_jitter + jitter_range

            views.append((h_start, h_end, w_start, w_end))
        return views, num_blocks_height, num_blocks_width
