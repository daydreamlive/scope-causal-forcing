# scope-causal-forcing

A [Daydream Scope](https://github.com/daydreamlive/scope) plugin that adds the [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) pipeline for real-time streaming video generation.

## What is Causal Forcing?

Causal Forcing (Tsinghua / Shengshu / UT Austin, Feb 2026) is the successor to Self-Forcing -- the training method behind Scope's built-in LongLive pipeline. It fixes a theoretical flaw in Self-Forcing's ODE initialization by using an autoregressive teacher instead of a bidirectional one, producing strictly better video quality at identical inference speed.

**Improvements over Self-Forcing (same GPU, same FPS):**

| Metric                | Improvement |
| --------------------- | ----------- |
| Dynamic Degree        | **+19.3%**  |
| VisionReward          | **+8.7%**   |
| Instruction Following | **+16.7%**  |

Both models run on Wan2.1-T2V-1.3B with 4-step denoising and KV-cached autoregressive generation.

## Requirements

- NVIDIA GPU with **20+ GB VRAM** (RTX 3090 / 4090 / 5090 or cloud equivalent)
- [Daydream Scope](https://github.com/daydreamlive/scope) installed

## Installation

```bash
uv pip install git+https://github.com/daydreamlive/scope-causal-forcing.git
```

Or for local development:

```bash
git clone https://github.com/daydreamlive/scope-causal-forcing.git
cd scope-causal-forcing
uv pip install -e .
```

The plugin registers automatically via entry points — restart Scope and `causal-forcing` will appear in the pipeline selector.

## Model Weights

On first load, Scope will download the required model weights:

| Model                     | Source                                                                  | Size    |
| ------------------------- | ----------------------------------------------------------------------- | ------- |
| Wan2.1-T2V-1.3B           | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | ~3 GB   |
| UMT5-XXL encoder          | [google/umt5-xxl](https://huggingface.co/google/umt5-xxl)               | ~10 GB  |
| Causal Forcing checkpoint | [zhuhz22/Causal-Forcing](https://huggingface.co/zhuhz22/Causal-Forcing) | ~3 GB   |
| Wan2.1 VAE                | (bundled with Wan2.1-T2V-1.3B)                                          | ~300 MB |

If you already use LongLive in Scope, the base Wan2.1 and UMT5 weights are shared — only the Causal Forcing checkpoint is an additional download.

## Configuration

| Parameter         | Default               | Description                                 |
| ----------------- | --------------------- | ------------------------------------------- |
| `height`          | 480                   | Output height in pixels                     |
| `width`           | 832                   | Output width in pixels                      |
| `denoising_steps` | [1000, 750, 500, 250] | 4-step warped denoising schedule            |
| `vae_type`        | wan                   | Full VAE (`wan`) or 75% pruned (`lightvae`) |
| `base_seed`       | 42                    | Random seed for reproducibility             |

## How It Works

Each frame is generated autoregressively with KV caching:

1. **Denoise** — 4-step spatial denoising loop (flow matching with warped timesteps)
2. **Cache** — Re-run at timestep=0 to write clean context into KV cache
3. **Decode** — Wan VAE decodes latent to pixels with temporal caching
4. **Advance** — Move to next frame

This is the same inference architecture as LongLive/Self-Forcing — only the weights differ.

## References

- **Paper**: [Causal Forcing: Autoregressive Distillation via Causal ODE](https://arxiv.org/abs/2602.02214)
- **Code**: [github.com/thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)
- **Weights**: [huggingface.co/zhuhz22/Causal-Forcing](https://huggingface.co/zhuhz22/Causal-Forcing)

## License

Apache-2.0 (same as the Causal Forcing model weights)
