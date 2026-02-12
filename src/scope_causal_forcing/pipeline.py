"""Causal Forcing pipeline for real-time streaming video generation.

Adapted from https://github.com/thu-ml/Causal-Forcing
Uses Scope's existing Wan2.1 infrastructure with Causal Forcing checkpoint weights.
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

# Causal Forcing framewise: 4-step denoising schedule
DEFAULT_DENOISING_STEPS = [1000, 750, 500, 250]

# Wan2.1-1.3B architecture constants
NUM_TRANSFORMER_BLOCKS = 30
NUM_HEADS = 12
HEAD_DIM = 128
CROSS_ATTN_SEQ_LEN = 512

# VAE and patch embedding downsample factors
VAE_SPATIAL_DOWNSAMPLE = 8
PATCH_SPATIAL_DOWNSAMPLE = 2
TOTAL_SPATIAL_DOWNSAMPLE = VAE_SPATIAL_DOWNSAMPLE * PATCH_SPATIAL_DOWNSAMPLE  # 16


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
        # Causal Forcing uses sink_size=0 and no local attention windowing
        start = time.time()
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name="Wan2.1-T2V-1.3B",
            model_dir=model_dir,
            generator_path=cf_ckpt_path,
            generator_model_name="generator_ema",
            timestep_shift=5.0,
            sink_size=0,
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
        text_encoder = text_encoder.to(device=device)
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

        # Setup scheduler and warp denoising steps to flow matching sigmas
        self.scheduler = generator.get_scheduler()
        self.denoising_step_list = self._warp_denoising_steps(
            torch.tensor(config.denoising_steps, dtype=torch.long)
        )

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

        # KV cache pre-allocation size: enough for 21 latent frames
        self.max_cached_frames = 21
        self.kv_cache_size = self.max_cached_frames * self.frame_seq_length

        # Streaming state
        self.kv_cache = None
        self.crossattn_cache = None
        self.current_start_frame = 0
        self.conditional_dict = None
        self._current_prompt = None

    def _resolve_checkpoint_path(self, config) -> str:
        """Resolve the path to the Causal Forcing checkpoint file."""
        # Check if explicitly set on config
        generator_path = getattr(config, "generator_path", None)
        if generator_path is not None:
            return generator_path

        # Resolve from Scope's model directory
        cf_model_dir = str(get_model_file_path("Causal-Forcing"))
        return os.path.join(cf_model_dir, "framewise", "causal_forcing.pt")

    def _warp_denoising_steps(self, steps: torch.Tensor) -> torch.Tensor:
        """Warp integer denoising steps through the flow matching schedule.

        Maps step indices [1000, 750, 500, 250] to their corresponding sigma
        values in the flow matching schedule, which improves generation quality.
        """
        timesteps = torch.cat(
            (self.scheduler.timesteps.cpu(), torch.tensor([0.0], dtype=torch.float32))
        )
        return timesteps[1000 - steps]

    def _initialize_caches(self):
        """Initialize KV and cross-attention caches for autoregressive generation."""
        self.kv_cache = [
            {
                "k": torch.zeros(
                    [1, self.kv_cache_size, NUM_HEADS, HEAD_DIM],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.zeros(
                    [1, self.kv_cache_size, NUM_HEADS, HEAD_DIM],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "global_end_index": torch.tensor(
                    [0], dtype=torch.long, device=self.device
                ),
                "local_end_index": torch.tensor(
                    [0], dtype=torch.long, device=self.device
                ),
            }
            for _ in range(NUM_TRANSFORMER_BLOCKS)
        ]

        self.crossattn_cache = [
            {
                "k": torch.zeros(
                    [1, CROSS_ATTN_SEQ_LEN, NUM_HEADS, HEAD_DIM],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "v": torch.zeros(
                    [1, CROSS_ATTN_SEQ_LEN, NUM_HEADS, HEAD_DIM],
                    dtype=self.dtype,
                    device=self.device,
                ),
                "is_init": False,
            }
            for _ in range(NUM_TRANSFORMER_BLOCKS)
        ]

        self.current_start_frame = 0

    def _reset_caches(self):
        """Reset caches for a fresh generation sequence."""
        if self.kv_cache is not None:
            for cache in self.kv_cache:
                cache["global_end_index"].fill_(0)
                cache["local_end_index"].fill_(0)
            for cache in self.crossattn_cache:
                cache["is_init"] = False

        # Clear VAE decode cache for temporal consistency
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

        # Generate one block of frames
        denoised = self._denoise_block()

        # Update KV cache with clean context (timestep=0)
        self._cache_clean_context(denoised)

        # Advance frame pointer
        self.current_start_frame += self.num_frame_per_block

        # Decode latent to pixel space
        video = self.vae.decode_to_pixel(denoised, use_cache=True)

        return {"video": postprocess_chunk(video)}

    def _denoise_block(self) -> torch.Tensor:
        """Run the spatial denoising loop for one block of frames.

        Returns:
            Denoised latent prediction [B, F, C, H_lat, W_lat]
        """
        # Sample noise for current block
        noisy_input = torch.randn(
            [1, self.num_frame_per_block, 16, self.height_latent, self.width_latent],
            device=self.device,
            dtype=self.dtype,
        )

        current_start = self.current_start_frame * self.frame_seq_length
        denoised_pred = None

        for i, current_timestep in enumerate(self.denoising_step_list):
            timestep = (
                torch.ones(
                    [1, self.num_frame_per_block],
                    device=self.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start,
            )

            # Re-noise for next step (except at the final step)
            if i < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[i + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep
                    * torch.ones(
                        [self.num_frame_per_block],
                        device=self.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        return denoised_pred

    def _cache_clean_context(self, denoised: torch.Tensor):
        """Re-run generator at timestep=0 to write clean context into KV cache.

        This is essential for autoregressive quality: the KV cache should contain
        representations of clean (denoised) frames, not noisy intermediate states.
        """
        context_timestep = torch.zeros(
            [1, self.num_frame_per_block],
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
