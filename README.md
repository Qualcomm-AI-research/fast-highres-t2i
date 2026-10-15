# 🏆 [CVPR26 Highlight] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion

<div align="center">
  <a href="https://qualcomm-ai-research.github.io/PixelRush/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/html/2602.12769v1"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red&logo=arxiv"></a> &ensp;
</div>

**Official PyTorch implementation** of our paper:
["**PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion**"](https://arxiv.org/html/2602.12769v1)

Author List:
Hong-Phuc Lai,
Phong Nguyen,
Anh Tran

> **Abstract**:
Pre-trained diffusion models excel at generating high-quality images but remain inherently limited by their native training resolution. Recent training-free approaches have attempted to overcome this constraint by introducing interventions during the denoising process; however, these methods incur substantial computational overhead, often requiring more than five minutes to produce a single 4K image. In this paper, we present PixelRush, the first tuning-free framework for practical high-resolution text-to-image generation. Our method builds upon the established patch-based inference paradigm but eliminates the need for multiple inversion and regeneration cycles. Instead, PixelRush enables efficient patch-based denoising within a low-step regime. To address artifacts introduced by patch blending in few-step generation, we propose a seamless blending strategy. Furthermore, we mitigate over-smoothing effects through a noise injection mechanism. PixelRush delivers exceptional efficiency, generating 4K images in approximately 20 seconds representing a 10x to 35x speedup over state-of-the-art methods while maintaining superior visual fidelity. Extensive experiments validate both the performance gains and the quality of outputs achieved by our approach.

For more visual results and project details, please visit our [project page](https://qualcomm-ai-research.github.io/PixelRush/).

![image](./assets/new_teaser.png)

## 📑 Citation

If you find this repository helpful for your work, please consider citing our paper:

```bibtex
@inproceedings{lai2026pixelrush,
  title={PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion},
  author={Lai, Hong-Phuc and Nguyen, Phong and Tran, Anh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={35946--35955},
  year={2026}
}
```

## 💡 Introduction

PixelRush is a simple, training-free and fast framework for high-resolution generation. The current release focuses on a minimal inference workflow:

1. **Input image**: provide a base image path.
2. **Text prompt**: provide a prompt describing the desired high-resolution result.
3. **Backbone choice**: use either `sd-turbo` or `sdxl-turbo`.
4. **Upscale factor**: choose `2`, `4`, `8`, or another power-of-two factor.

The main design goal of this repository is to keep the code easy to read and easy to run.

## 🔧 Dependencies and Installation

The implementation is based on Python, PyTorch, Hugging Face Diffusers, and public model checkpoints.

Create an environment and install dependencies:

```bash
conda create --name pixelrush python=3.10
conda activate pixelrush
pip install -r requirements.txt
```

PixelRush uses public Hugging Face model repositories:

- `stabilityai/sd-turbo`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `stabilityai/sdxl-turbo`

If your environment requires Hugging Face authentication, log in before running inference:

```bash
huggingface-cli login
```

## 📁 Repository Structure

The main inference code is at `inference.py`.

```text
PixelRush
├── assets/test_image/          Images for quick testing.
├── eval/                       FID/IS metrics and 1K/2K/4K evaluation workflow.
├── src/
│   ├── pixelrushv1_sd_turbo.py     PixelRush with SD-Turbo.
│   └── pixelrushv1_sdxl_turbo.py   PixelRush with SDXL-Turbo.
├── inference.py                Minimal inference script.
├── requirements.txt            Required packages.
└── README.md                   This README file.
```

## 🏃 Applications

PixelRush can be used for prompt-guided image enhancement and high-resolution upsampling. For examples and more details, visit the [project page](https://qualcomm-ai-research.github.io/PixelRush/).

## 🖼️ Try PixelRush

Run SDXL-Turbo PixelRush on an example image:

```bash
python inference.py \
  --image assets/test_image/init_image.png \
  --prompt "A cinematic portrait with dramatic lighting and crisp fine details." \
  --model sdxl-turbo \
  --upscale 2 \
  --output output/pixelrush.png
```

Run SD-Turbo PixelRush:

```bash
python inference.py \
  --image assets/test_image/init_image.png \
  --prompt "A cinematic portrait with dramatic lighting and crisp fine details." \
  --model sd-turbo \
  --upscale 2 \
  --output output/pixelrush_sd.png
```

Arguments:

- `--image`: path to the input image.
- `--prompt`: text prompt.
- `--model`: choose `sd-turbo` or `sdxl-turbo`.
- `--upscale`: total upsampling factor, e.g. `2`, `4`, or `8`.
- `--output`: path to save the output image.

## 📊 Evaluation

We provide a complete evaluation workflow in [`eval/`](eval/README.md) for paper-style experiments. See [`eval/README.md`](eval/README.md) for details.

## Contacts

For questions or suggestions, feel free to contact the first-author at
(phuclai@qti.qualcomm.com) or (phuckstnk63dktd@gmail.com).

## License

This repository is released under the BSD-3-Clause-Clear license. See [LICENSE](LICENSE) for the full license text.
