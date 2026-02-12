"""Causal Forcing pipeline for real-time streaming video generation.

Adapted from https://github.com/thu-ml/Causal-Forcing
Uses Scope's existing Wan2.1 infrastructure with Causal Forcing checkpoint weights.

The DMD (Distribution Matching Distillation) checkpoint is designed for fast 4-step
inference WITHOUT Classifier-Free Guidance, matching the reference causal_inference.py.
"""

import logging
import os
import time
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.interface import Pipeline
from scope.core.pipelines.process import postprocess_chunk
from scope.core.pipelines.utils import validate_resolution
from scope.core.pipelines.wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from scope.core.pipelines.wan2_1.utils import initialize_crossattn_cache, initialize_kv_cache
from scope.core.pipelines.wan2_1.vae import create_vae

from .schema import CausalForcingConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig


def _import_causal_wan_model():
    """Import CausalWanModel from any available Wan2.1-based pipeline.

    Multiple Scope pipelines bundle their own copy of CausalWanModel. We try
    several in order of preference so the plugin is not tightly coupled to a
    single built-in pipeline.
    """
    sources = [
        "scope.core.pipelines.longlive.modules.causal_model",
        "scope.core.pipelines.streamdiffusionv2.modules.causal_model",
        "scope.core.pipelines.reward_forcing.modules.causal_model",
        "scope.core.pipelines.memflow.modules.causal_model",
        "scope.core.pipelines.krea_realtime_video.modules.causal_model",
    ]
    for module_path in sources:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            return mod.CausalWanModel
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not import CausalWanModel from any Scope pipeline. "
        "Ensure at least one Wan2.1-based pipeline is available."
    )

logger = logging.getLogger(__name__)

# VAE and patch embedding downsample factors
VAE_SPATIAL_DOWNSAMPLE = 8
PATCH_SPATIAL_DOWNSAMPLE = 2
TOTAL_SPATIAL_DOWNSAMPLE = VAE_SPATIAL_DOWNSAMPLE * PATCH_SPATIAL_DOWNSAMPLE  # 16

# Rolling attention window size (in frames).
# CausalWanModel requires local_attn_size != -1 for KV cache rolling eviction.
LOCAL_ATTN_SIZE = 21


class CausalForcingPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return CausalForcingConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        CausalWanModel = _import_causal_wan_model()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build config from kwargs (pipeline_manager passes config fields as kwargs)
        config = CausalForcingConfig(**kwargs)

        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=TOTAL_SPATIAL_DOWNSAMPLE,
        )

        model_dir = getattr(config, "model_dir", None)
        if model_dir is None:
            model_dir = str(get_model_file_path("Wan2.1-T2V-1.3B").parent)

        # Resolve Causal Forcing checkpoint path
        cf_ckpt_path = self._resolve_checkpoint_path(config)

        # Load generator: Wan2.1-1.3B base config + Causal Forcing weights
        # Causal Forcing uses sink_size=0; local_attn_size enables rolling KV cache eviction
        start = time.time()
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name="Wan2.1-T2V-1.3B",
            model_dir=model_dir,
            generator_path=cf_ckpt_path,
            generator_model_name="generator_ema",
            timestep_shift=5.0,
            sink_size=0,
            local_attn_size=LOCAL_ATTN_SIZE,
        )
        generator = generator.to(device=device, dtype=dtype)
        print(f"Loaded Causal Forcing generator in {time.time() - start:.3f}s")

        # Load text encoder (UMT5-XXL, shared with other Wan2.1 pipelines)
        text_encoder_path = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )
        tokenizer_path = str(
            get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
        )
        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name="Wan2.1-T2V-1.3B",
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        text_encoder = text_encoder.to(device=device, dtype=torch.bfloat16)
        print(f"Loaded text encoder in {time.time() - start:.3f}s")

        # Load VAE
        start = time.time()
        vae = create_vae(
            model_dir=model_dir,
            model_name="Wan2.1-T2V-1.3B",
            vae_type=config.vae_type,
        )
        vae = vae.to(device=device, dtype=dtype)
        print(f"Loaded VAE (type={config.vae_type}) in {time.time() - start:.3f}s")

        # Setup scheduler
        self.scheduler = generator.get_scheduler()

        # Warp denoising steps through the scheduler's shifted timestep table.
        # This maps raw indices [1000, 750, 500, 250] to the actual shifted timestep
        # values the model was trained with (e.g., [1000, 937.5, 833.3, 625.0]).
        # Reference: Causal-Forcing/model/base.py lines 20-24
        raw_steps = torch.tensor(config.denoising_steps, dtype=torch.long)
        scheduler_timesteps = torch.cat([
            self.scheduler.timesteps.cpu(),
            torch.tensor([0.0], dtype=torch.float32),
        ])
        self.denoising_step_list = scheduler_timesteps[1000 - raw_steps]
        print(f"[CF DEBUG] Raw denoising steps: {config.denoising_steps}")
        print(f"[CF DEBUG] Warped denoising steps: {self.denoising_step_list.tolist()}")
        print(f"[CF DEBUG] Scheduler timesteps range: [{self.scheduler.timesteps[0]:.2f}, {self.scheduler.timesteps[-1]:.2f}], len={len(self.scheduler.timesteps)}")
        print(f"[CF DEBUG] Scheduler sigmas range: [{self.scheduler.sigmas[0]:.6f}, {self.scheduler.sigmas[-1]:.6f}]")

        # Store components
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self.dtype = dtype

        # Resolution-dependent constants
        self.height = config.height
        self.width = config.width
        self.height_latent = config.height // VAE_SPATIAL_DOWNSAMPLE
        self.width_latent = config.width // VAE_SPATIAL_DOWNSAMPLE
        self.frame_seq_length = (
            (config.height // TOTAL_SPATIAL_DOWNSAMPLE)
            * (config.width // TOTAL_SPATIAL_DOWNSAMPLE)
        )

        # Framewise generation (1 latent frame per block)
        self.num_frame_per_block = 1

        # Streaming state - single KV cache (no CFG for DMD checkpoint)
        self.kv_cache = None
        self.crossattn_cache = None
        self.current_start_frame = 0
        self.conditional_dict = None
        self._current_prompt = None

    def _resolve_checkpoint_path(self, config) -> str:
        """Resolve the path to the Causal Forcing checkpoint file."""
        generator_path = getattr(config, "generator_path", None)
        if generator_path is not None:
            return generator_path

        cf_model_dir = str(get_model_file_path("Causal-Forcing"))
        return os.path.join(cf_model_dir, "framewise", "causal_forcing.pt")

    def _make_kv_cache(self, existing=None):
        return initialize_kv_cache(
            generator=self.generator,
            batch_size=1,
            dtype=self.dtype,
            device=self.device,
            local_attn_size=LOCAL_ATTN_SIZE,
            frame_seq_length=self.frame_seq_length,
            kv_cache_existing=existing,
        )

    def _make_crossattn_cache(self, existing=None):
        return initialize_crossattn_cache(
            generator=self.generator,
            batch_size=1,
            dtype=self.dtype,
            device=self.device,
            crossattn_cache_existing=existing,
        )

    def _initialize_caches(self):
        """Initialize KV and cross-attention caches."""
        self.kv_cache = self._make_kv_cache()
        self.crossattn_cache = self._make_crossattn_cache()
        self.current_start_frame = 0

    def _reset_caches(self):
        """Reset caches for a fresh generation sequence."""
        if self.kv_cache is not None:
            self.kv_cache = self._make_kv_cache(existing=self.kv_cache)
            self.crossattn_cache = self._make_crossattn_cache(
                existing=self.crossattn_cache
            )

        if hasattr(self.vae, "model") and hasattr(self.vae.model, "clear_cache"):
            self.vae.model.clear_cache()

        self.current_start_frame = 0

    def __call__(self, **kwargs) -> dict:
        init_cache = kwargs.get("init_cache", False)
        prompts = kwargs.get("prompts")

        # Initialize or reset caches
        if self.kv_cache is None:
            self._initialize_caches()
        elif init_cache:
            self._reset_caches()

        # Handle prompt changes: re-encode text when prompt changes
        if prompts and len(prompts) > 0:
            first_prompt = prompts[0]
            new_prompt = (
                first_prompt["text"]
                if isinstance(first_prompt, dict)
                else first_prompt
            )
            if new_prompt != self._current_prompt:
                self.conditional_dict = self.text_encoder(text_prompts=[new_prompt])
                self._current_prompt = new_prompt

        if self.conditional_dict is None:
            self.conditional_dict = self.text_encoder(text_prompts=[""])

        # WanVAE stream_decode requires >= 2 latent frames on the first batch
        num_frames = 2 if self.current_start_frame == 0 else self.num_frame_per_block

        # Generate block of frames
        denoised = self._denoise_block(num_frames)

        # Update KV cache with clean context (timestep=0)
        self._cache_clean_context(denoised, num_frames)

        # Advance frame pointer
        self.current_start_frame += num_frames

        # Decode latent to pixel space
        video = self.vae.decode_to_pixel(denoised, use_cache=True)
        print(f"[CF DEBUG] VAE output: shape={list(video.shape)}, min={video.min().item():.4f}, max={video.max().item():.4f}, mean={video.mean().item():.4f}")

        result = postprocess_chunk(video)
        print(f"[CF DEBUG] postprocess: shape={list(result.shape)}, min={result.min().item():.4f}, max={result.max().item():.4f}, mean={result.mean().item():.4f}")

        return {"video": result}

    def _denoise_block(self, num_frames: int) -> torch.Tensor:
        """Run the spatial denoising loop.

        Uses pred_x0 + stochastic re-noising matching the reference
        causal_inference.py. The DMD checkpoint produces good x0 predictions
        in 4 steps without CFG.

        Args:
            num_frames: Number of latent frames to generate in this block.

        Returns:
            Denoised latent prediction [B, F, C, H_lat, W_lat]
        """
        sample = torch.randn(
            [1, num_frames, 16, self.height_latent, self.width_latent],
            device=self.device,
            dtype=self.dtype,
        )

        current_start = self.current_start_frame * self.frame_seq_length
        num_steps = len(self.denoising_step_list)

        print(f"[CF DEBUG] _denoise_block: num_frames={num_frames}, current_start_frame={self.current_start_frame}, current_start={current_start}")
        print(f"[CF DEBUG] Initial noise: shape={list(sample.shape)}, min={sample.min().item():.4f}, max={sample.max().item():.4f}, mean={sample.mean().item():.4f}")

        for i, current_timestep in enumerate(self.denoising_step_list):
            timestep = (
                torch.ones(
                    [1, num_frames],
                    device=self.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            print(f"[CF DEBUG] Step {i}: timestep={current_timestep.item():.2f}, timestep_tensor dtype={timestep.dtype}, sample dtype={sample.dtype}")

            if i < num_steps - 1:
                flow_pred, pred_x0 = self.generator(
                    noisy_image_or_video=sample,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                )

                print(f"[CF DEBUG]   flow_pred: min={flow_pred.min().item():.4f}, max={flow_pred.max().item():.4f}, mean={flow_pred.mean().item():.4f}, has_nan={flow_pred.isnan().any().item()}")
                print(f"[CF DEBUG]   pred_x0:   min={pred_x0.min().item():.4f}, max={pred_x0.max().item():.4f}, mean={pred_x0.mean().item():.4f}, has_nan={pred_x0.isnan().any().item()}")

                # Stochastic re-noising to next timestep level
                next_timestep = self.denoising_step_list[i + 1]
                flat_x0 = pred_x0.flatten(0, 1)
                noise_for_renoise = torch.randn_like(flat_x0)
                t_for_noise = next_timestep * torch.ones(
                    [flat_x0.shape[0]],
                    device=flat_x0.device,
                    dtype=torch.long,
                )
                print(f"[CF DEBUG]   add_noise: next_t={next_timestep.item():.2f}, t_tensor dtype={t_for_noise.dtype}, t_values={t_for_noise.tolist()}")
                sample = self.scheduler.add_noise(
                    flat_x0,
                    noise_for_renoise,
                    t_for_noise,
                ).unflatten(0, pred_x0.shape[:2])
                print(f"[CF DEBUG]   re-noised: min={sample.min().item():.4f}, max={sample.max().item():.4f}, mean={sample.mean().item():.4f}")
            else:
                # Last step: output pred_x0 directly
                flow_pred, pred_x0 = self.generator(
                    noisy_image_or_video=sample,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                )
                print(f"[CF DEBUG]   flow_pred: min={flow_pred.min().item():.4f}, max={flow_pred.max().item():.4f}, mean={flow_pred.mean().item():.4f}, has_nan={flow_pred.isnan().any().item()}")
                print(f"[CF DEBUG]   pred_x0:   min={pred_x0.min().item():.4f}, max={pred_x0.max().item():.4f}, mean={pred_x0.mean().item():.4f}, has_nan={pred_x0.isnan().any().item()}")
                sample = pred_x0

        print(f"[CF DEBUG] Final denoised: min={sample.min().item():.4f}, max={sample.max().item():.4f}, mean={sample.mean().item():.4f}")
        return sample

    def _cache_clean_context(self, denoised: torch.Tensor, num_frames: int):
        """Re-run generator at timestep=0 to write clean context into KV cache."""
        context_timestep = torch.zeros(
            [1, num_frames],
            device=self.device,
            dtype=torch.int64,
        )
        current_start = self.current_start_frame * self.frame_seq_length

        self.generator(
            noisy_image_or_video=denoised,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start,
        )
