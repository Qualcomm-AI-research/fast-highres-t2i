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
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline
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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_time_ids(
    original_size: List[int],
    crops_coords_top_left: List[int],
    target_size: List[int],
    dtype=torch.float16,
) -> torch.Tensor:
    """
    original_size: (height, width) of original image
    crops_coords_top_left: (top, left) crop coordinates
    target_size: (height, width) of target output
    """
    time_ids = torch.tensor(
        [
            original_size[0],  # original height
            original_size[1],  # original width
            crops_coords_top_left[0],  # crop top
            crops_coords_top_left[1],  # crop left
            target_size[0],  # target height
            target_size[1],  # target width
        ],
        dtype=dtype,
    )

    return time_ids.unsqueeze(0)  # Add batch dimension -> [1, 6]


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
    noise_pred: torch.Tensor,
    timestep: int,
    alphas_cumprod: torch.Tensor,
    temperature: float = 1.0,
    user_define: float = 1.0,
) -> torch.Tensor:
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


class PixelRushSDXLTurbo:
    """
    Custom Stable Diffusion XL Pipeline with DDIM inversion and discrete wavelet sym4 methods.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL default
        model_revision: Optional[str] = None,
        turbo_model_id: str = "stabilityai/sdxl-turbo",  # SDXL-Turbo model
        turbo_model_revision: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the custom pipeline with individual components for SDXL.
        """
        self.model_id = model_id
        self.model_revision = model_revision
        self.turbo_model_id = turbo_model_id
        self.turbo_model_revision = turbo_model_revision
        self.vae_id = "madebyollin/sdxl-vae-fp16-fix"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        is_cuda = str(self.device).startswith("cuda")
        self.torch_dtype = torch_dtype or (torch.float16 if is_cuda else torch.float32)

        # Load individual components
        self._load_components()

        # Initialize DDIM scheduler
        self.ddim_scheduler = DDIMScheduler.from_config(
            StableDiffusionXLPipeline.from_pretrained(
                turbo_model_id,
                revision=self.turbo_model_revision,
            ).scheduler.config,
        )
        self.alphas_cumprod = self.ddim_scheduler.alphas_cumprod

        self.time_ids = get_time_ids((512, 512), (0, 0), (512, 512)).to(self.device)

        self.use_tiling = False
        print(f"SDXL Pipeline initialized on device: {self.device}")

    def _load_components(self):
        """Load individual components from the SDXL pipeline."""
        print("Loading SDXL pipeline components...")
        # SDXL uses two text encoders and two tokenizers
        # 1. OpenCLIP (main)
        # First load the complete pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            torch_dtype=self.torch_dtype,
        )

        # Load SDXL-Turbo pipeline
        turbo_pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.turbo_model_id,
            revision=self.turbo_model_revision,
            torch_dtype=self.torch_dtype,
        )

        # Assign components from the pipeline
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder.to(self.device, self.torch_dtype)
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_2 = pipeline.text_encoder_2.to(self.device, self.torch_dtype)

        # Keep VAE loading the same
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            revision=self.model_revision,
            variant=None,
        ).to(self.device)

        # Assign UNets from both pipelines
        self.unet = pipeline.unet.to(self.device)  # SDXL UNet
        self.unet_turbo = turbo_pipeline.unet.to(self.device)  # SDXL-Turbo UNet

        self.scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config,
        )

        # Clean up pipelines to free memory
        del pipeline
        del turbo_pipeline
        print("All SDXL components loaded successfully!")

    def encode_prompt(
        self, prompt: str, negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to embeddings using both SDXL text encoders.
        Returns concatenated embeddings for both encoders.
        """
        # Tokenize prompts for both encoders
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
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
        negative_text_inputs_2 = self.tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # Encode to embeddings
        text_embed_list = []
        negative_embed_list = []
        with torch.no_grad():

            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
            text_embeddings_2 = self.text_encoder_2(
                text_inputs_2.input_ids.to(self.device),
                output_hidden_states=True,
            )

            text_embed_list = [
                text_embeddings.hidden_states[-2],
                text_embeddings_2.hidden_states[-2],
            ]

            negative_text_embeddings = self.text_encoder(
                negative_text_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
            negative_text_embeddings_2 = self.text_encoder_2(
                negative_text_inputs_2.input_ids.to(self.device),
                output_hidden_states=True,
            )

            negative_embed_list = [
                negative_text_embeddings.hidden_states[-2],
                negative_text_embeddings_2.hidden_states[-2],
            ]

        return (torch.cat(text_embed_list, dim=-1), text_embeddings_2[0]), (
            torch.cat(negative_embed_list, dim=-1),
            negative_text_embeddings_2[0],
        )

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
                overlap_factor=0.125,
            )
        else:
            with torch.no_grad():
                image = self.vae.decode(latents.to(torch.float32)).sample

        return image

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        """
        Generate image using the custom pipeline.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            output_path: Path to save image

        Returns:
            Generated image tensor
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(seed)

        # Encode prompts
        (text_embeddings, text_embeddings_2), (
            negative_text_embeddings,
            negative_text_embeddings_2,
        ) = self.encode_prompt(prompt, negative_prompt)

        # Concatenate embeddings
        encoder_hidden_states = torch.cat([negative_text_embeddings, text_embeddings], dim=0)
        pool_hidden_states = torch.cat([negative_text_embeddings_2, text_embeddings_2], dim=0)
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Initialize latents
        latents = (
            torch.randn(
                (1, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
                dtype=self.torch_dtype,
            )
            if latents is None
            else latents
        )

        # Denoising loop

        self.time_ids = get_time_ids((1024, 1024), (0, 0), (height, width)).to(self.device)

        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pool_hidden_states,
                        "time_ids": torch.cat([self.time_ids, self.time_ids], dim=0),
                    },
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        image = self.decode_latents(latents)

        # Save if path provided
        if output_path:
            self._save_image(image, output_path)

        return image

    def _save_image(self, image_tensor: torch.Tensor, output_path: str):
        """Save image tensor to file."""
        # Convert to PIL and save
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))
        pil_image = Image.fromarray(image_np)

        # Create directory if needed
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        pil_image.save(output_path)
        print(f"Image saved to: {output_path}")

    def simple_ddim_inversion(
        self,
        patch_wise: List[torch.Tensor],
        t: Union[int, torch.Tensor] = 249,
        acp_batch_size: int = 8,
        ##### DDIM-inv only use text condition #####
        text_embeddings_ddim_inv: torch.Tensor = None,
        text_embeddings_2_ddim_inv: torch.Tensor = None,
        add_time_ids_input: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        PixelRushV1 algorithm to compute the DDIM-inv z_{ddim_inv_step} from low-quality latent z_0
        """
        batch_patch_wise = [
            patch_wise[i : i + acp_batch_size] for i in range(0, len(patch_wise), acp_batch_size)
        ]
        ddim_inv_local_list = []
        for i, b_patch_wise in enumerate(tqdm(batch_patch_wise, desc="DDIM inversion")):
            timestep = torch.full((b_patch_wise.shape[0],), t, device=self.device, dtype=torch.long)
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet_turbo(
                    b_patch_wise,
                    timestep,
                    encoder_hidden_states=text_embeddings_ddim_inv[0 : b_patch_wise.shape[0]],
                    added_cond_kwargs={
                        "text_embeds": text_embeddings_2_ddim_inv[0 : b_patch_wise.shape[0]],
                        "time_ids": add_time_ids_input[0 : b_patch_wise.shape[0]],
                    },
                ).sample

            current_t = 0
            next_t = t
            alpha_t = self.alphas_cumprod[current_t]
            alpha_t_next = self.alphas_cumprod[next_t]
            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            b_patch_wise = (b_patch_wise - (1 - alpha_t).sqrt() * noise_pred) * (
                alpha_t_next.sqrt() / alpha_t.sqrt()
            ) + (1 - alpha_t_next).sqrt() * noise_pred
            ddim_inv_local_list.append(b_patch_wise)
        ddim_inv_local_list = torch.cat(ddim_inv_local_list, dim=0)
        return ddim_inv_local_list

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
        forward_blend_weight: float = 0.2,
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
            forward_blend_weight: Hyperparameter to blend the forward and ddim inversion path
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
        acp_batch_size = 8
        anchor_mean, anchor_std = (
            latent_high_resolution.mean(),
            latent_high_resolution.std(),
        )
        views, num_blocks_height, num_blocks_width = self.get_views(
            height=latent_high_resolution.shape[2],
            width=latent_high_resolution.shape[3],
            stride=64,
            window_size=128,
        )
        patch_wise = []
        for h_start, h_end, w_start, w_end in views:
            patch_wise.append(latent_high_resolution[:, :, h_start:h_end, w_start:w_end])
        patch_wise = torch.cat(patch_wise, dim=0)
        repeat_times = patch_wise.shape[0]

        # Prepare time_ids_input #
        add_time_ids_input = []
        self.time_ids = get_time_ids((1024, 1024), (0, 0), (1024, 1024)).to(self.device)
        for h_start, h_end, w_start, w_end in views:
            add_time_ids = self.time_ids.clone()
            add_time_ids[:, 2] = h_start * 8
            add_time_ids[:, 3] = w_start * 8
            add_time_ids_input.append(add_time_ids)
        add_time_ids_input = torch.cat(add_time_ids_input, dim=0)

        # Prepare text embeddings for sdxl-turbo #
        (text_embeddings, text_embeddings_2), (
            negative_text_embeddings,
            negative_text_embeddings_2,
        ) = self.encode_prompt(prompt)
        # DDIM-inv only use text condition
        text_embeddings_ddim_inv = einops.repeat(
            text_embeddings, "b seq d -> (repeat b) seq d", repeat=repeat_times
        )
        text_embeddings_2_ddim_inv = einops.repeat(
            text_embeddings_2, "b d -> (repeat b) d", repeat=repeat_times
        )

        # DDIM inversion #
        print(f"DDIM inversion to step: {ddim_inv_to_step}")
        self.ddim_scheduler.set_timesteps(4)
        ddim_inverted_patch_wise = [patch_wise.clone()]

        timestep = torch.full(
            (repeat_times,), ddim_inv_to_step, device=self.device, dtype=torch.long
        )
        forward_patch_wise = [patch_wise.clone()]
        noise = torch.randn_like(patch_wise)
        fw_patch_wise = self.ddim_scheduler.add_noise(patch_wise, noise, timestep)
        forward_patch_wise.append(fw_patch_wise)

        ddim_inverted_patch_wise.append(
            self.simple_ddim_inversion(
                patch_wise=patch_wise,
                t=ddim_inv_to_step,
                acp_batch_size=acp_batch_size,
                text_embeddings_ddim_inv=text_embeddings_ddim_inv,
                text_embeddings_2_ddim_inv=text_embeddings_2_ddim_inv,
                add_time_ids_input=add_time_ids_input,
            )
        )
        patch_wise = ddim_inverted_patch_wise[-1]

        # 3. Predict clean patches
        patch_wise = (
            forward_blend_weight * forward_patch_wise[-1]
            + (1 - forward_blend_weight) * ddim_inverted_patch_wise[-1]
        )
        batch_patch_wise = [
            patch_wise[i : i + acp_batch_size] for i in range(0, len(patch_wise), acp_batch_size)
        ]
        new_patch_wise = torch.zeros_like(patch_wise)
        batch_patch_wise = [
            patch_wise[i : i + acp_batch_size] for i in range(0, len(patch_wise), acp_batch_size)
        ]
        new_patch_wise = torch.zeros_like(patch_wise)
        for i, b_patch_wise in enumerate(tqdm(batch_patch_wise, desc="DDIM loop")):
            # Prepare timestep
            timestep = torch.full(
                (b_patch_wise.shape[0],),
                ddim_inv_to_step,
                device=self.device,
                dtype=torch.long,
            )

            # Expand latents for classifier-free guidance
            latent_model_input = self.ddim_scheduler.scale_model_input(
                b_patch_wise, ddim_inv_to_step
            )

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet_turbo(
                    latent_model_input,
                    ddim_inv_to_step,
                    encoder_hidden_states=text_embeddings_ddim_inv[0 : b_patch_wise.shape[0]],
                    added_cond_kwargs={
                        "text_embeds": text_embeddings_2_ddim_inv[0 : b_patch_wise.shape[0]],
                        "time_ids": add_time_ids_input[
                            i * acp_batch_size : i * acp_batch_size + b_patch_wise.shape[0]
                        ],
                    },
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
        """
        Get profiling information for the last pixelrush run.

        Returns:
            Dictionary containing profiling data
        """
        # This can be extended to store and return detailed profiling info
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
