# PixelRush Evaluation Workflow

This folder contains the evaluation workflow for PixelRush.

It supports:

1. Generating 1000 evaluation images from a prompt list.
2. Saving 1K, 2K, and 4K outputs.
3. Loading the provided reference stats file: `eval/reference_stats.npz`.
4. Computing FID and Inception Score (IS).

## Files

```text
eval/
├── reference_stats.npz          Provided reference FID statistics.
├── compute_reference_stats.py   Optional: build a compatible stats file from cached real features.
├── generate_pixelrush_eval.py   Generate image_1k, image_2k, image_4k.
├── metrics.py                   FID and Inception Score implementation.
├── evaluate.py                  Load reference stats and evaluate generated images.
├── run_eval_workflow.sh         End-to-end generation + evaluation launcher.
├── eval_prompt.txt              Example prompt list.
├── PROMPT_PROVENANCE.md         Provenance of prompts in eval_prompt.txt.
└── README.md                    This file.
```

## Reference Stats

The repository already includes:

```text
eval/reference_stats.npz
```

By default, `eval/evaluate.py` and `eval/run_eval_workflow.sh` load this file.


Run:

```bash
python eval/compute_reference_stats.py \
  --real_cache_dir /path/to/real_cache_dir \
  --output_npz eval/reference_stats.npz
```

The stats file stores:

```text
mu
sigma
num_images
source_features
```

The evaluator also accepts explicit `.npz` files with keys:

```text
mean
cov
```

## Prompt File Format and Provenance

Use a plain text file with one prompt per line:

```text
A cinematic portrait with dramatic lighting and crisp fine details.
A glossy soda can on a bright tabletop.
A modern kitchen appliance in a clean studio scene.
```

The generation script reads the first `--num_images` prompts. For paper evaluation, use `--num_images 1000`.

The bundled prompt list provenance is documented in [`PROMPT_PROVENANCE.md`](PROMPT_PROVENANCE.md).

## Generate 1K, 2K, and 4K Images

Single GPU:

```bash
python eval/generate_pixelrush_eval.py \
  --prompt_file eval/eval_prompt.txt \
  --output_dir eval_outputs/pixelrush_sdxl \
  --num_images 1000 \
  --stage all \
  --model sdxl-turbo
```

This creates:

```text
eval_outputs/pixelrush_sdxl/
├── image_1k/
├── image_2k/
├── image_4k/
└── eval_config.json
```

Use SD-Turbo instead of SDXL-Turbo:

```bash
python eval/generate_pixelrush_eval.py \
  --prompt_file eval/eval_prompt.txt \
  --output_dir eval_outputs/pixelrush_sd \
  --num_images 1000 \
  --stage all \
  --model sd-turbo
```

## Workflow Launcher

The workflow launcher is intentionally simple. Edit the variables at the top of `eval/run_eval_workflow.sh`, then run:

```bash
bash eval/run_eval_workflow.sh
```

Default settings:

```bash
PROMPT_FILE="eval/eval_prompt.txt"
OUTPUT_DIR="eval_outputs/pixelrush_sdxl"
REF_STATS="eval/reference_stats.npz"

NUM_IMAGES=1000
NUM_GPUS=8
MODEL="sdxl-turbo"
STAGE="all"

USE_TEMPORAL_SCALING=true
OVERWRITE=true
```

Choose one `STAGE`:

```text
all      Generate 1K, generate 2K/4K, then evaluate.
base     Generate only 1K images.
upscale  Generate only 2K/4K images from existing image_1k.
eval     Evaluate existing image_1k/image_2k/image_4k.
```

For temporal-scaling upscale only, keep:

```bash
STAGE="upscale"
USE_TEMPORAL_SCALING=true
OVERWRITE=true
```

The launcher validates image counts before evaluation.

## Evaluate Existing Generated Images

Evaluate one output folder with the provided reference stats:

```bash
python eval/evaluate.py \
  --image_dir eval_outputs/pixelrush_sdxl/image_4k \
  --output_json eval_outputs/pixelrush_sdxl/image_4k_metrics.json \
  --batch_size 32
```

Evaluate with a custom reference stats file:

```bash
python eval/evaluate.py \
  --image_dir eval_outputs/pixelrush_sdxl/image_4k \
  --ref_stats /path/to/reference_stats.npz \
  --output_json eval_outputs/pixelrush_sdxl/image_4k_metrics.json \
  --batch_size 32
```

Output JSON format:

```json
{
  "image_dir": "eval_outputs/pixelrush_sdxl/image_4k",
  "ref_stats_npz": "eval/reference_stats.npz",
  "num_images": 1000,
  "fid": 0.0,
  "inception_score_mean": 0.0,
  "inception_score_std": 0.0
}
```

## Reference Results

The released evaluation workflow gives the following results. Paper numbers are shown in parentheses.

| Resolution | IS | FID |
| --- | ---: | ---: |
| 2K | 14.32 (14.32) | 49.23 (50.13) |
| 4K | 13.64 (13.73) | 53.58 (54.67) |

Small differences from the paper can occur due to environment, package versions, GPU kernels, or image I/O differences.

## Notes

- Use smaller `--batch_size` if Inception evaluation runs out of GPU memory.
- The default generation uses SDXL base for 1K images, then PixelRush for 2K and 4K.
- For 4K generation, the script uses cascade upscaling: `1K -> 2K -> 4K`.
