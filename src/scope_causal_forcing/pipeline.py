"""Causal Forcing pipeline for real-time streaming video generation.

Adapted from https://github.com/thu-ml/Causal-Forcing
"""

import logging
import os
import time
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path
from scope.core.pipelines.blending import EmbeddingBlender, parse_transition_config
from scope.core.pipelines.enums import Quantization
from scope.core.pipelines.interface import Pipeline
from scope.core.pipelines.process import postprocess_chunk
from scope.core.pipelines.utils import validate_resolution
from scope.core.pipelines.wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from scope.core.pipelines.wan2_1.blocks.setup_caches import (
    set_all_modules_frame_seq_length,
    set_all_modules_max_attention_size,
)
from scope.core.pipelines.wan2_1.utils import initialize_crossattn_cache, initialize_kv_cache
from scope.core.pipelines.wan2_1.vae import create_vae

from .schema import CausalForcingConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig


def _import_causal_wan_model():
    """Import CausalWanModel from any available Wan2.1-based pipeline."""
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

VAE_SPATIAL_DOWNSAMPLE = 8
PATCH_SPATIAL_DOWNSAMPLE = 2
TOTAL_SPATIAL_DOWNSAMPLE = VAE_SPATIAL_DOWNSAMPLE * PATCH_SPATIAL_DOWNSAMPLE
LOCAL_ATTN_SIZE = 21
MAX_ROPE_SEQ_LEN = 1024


class CausalForcingPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return CausalForcingConfig

    def __init__(
        self,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        CausalWanModel = _import_causal_wan_model()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = CausalForcingConfig(**kwargs)

        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=TOTAL_SPATIAL_DOWNSAMPLE,
        )

        model_dir = getattr(config, "model_dir", None)
        if model_dir is None:
            model_dir = str(get_model_file_path("Wan2.1-T2V-1.3B").parent)

        cf_ckpt_path = self._resolve_checkpoint_path(config)

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
        # Reload with double-prefix stripping (model. + _fsdp_wrapped_module.)
        from scope.core.pipelines.utils import load_state_dict as _load_sd
        _sd = _load_sd(cf_ckpt_path)["generator_ema"]
        _sd = {k.removeprefix("model.").removeprefix("_fsdp_wrapped_module."): v for k, v in _sd.items()}
        _result = generator.model.load_state_dict(_sd, assign=True, strict=False)
        logger.info(
            "Reloaded weights: %d matched, %d missing, %d unexpected",
            len(_sd) - len(_result.unexpected_keys),
            len(_result.missing_keys),
            len(_result.unexpected_keys),
        )
        del _sd

        if quantization == Quantization.FP8_E4M3FN:
            generator = generator.to(dtype=dtype)
            start_q = time.time()
            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )
            quantize_(
                generator,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=device,
            )
            logger.info("Quantized generator to FP8 in %.3fs", time.time() - start_q)
        else:
            generator = generator.to(device=device, dtype=dtype)
        logger.info("Loaded Causal Forcing generator in %.3fs", time.time() - start)

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
        logger.info("Loaded text encoder in %.3fs", time.time() - start)

        start = time.time()
        vae = create_vae(
            model_dir=model_dir,
            model_name="Wan2.1-T2V-1.3B",
            vae_type=config.vae_type,
        )
        vae = vae.to(device=device, dtype=dtype)
        logger.info("Loaded VAE (type=%s) in %.3fs", config.vae_type, time.time() - start)

        self.scheduler = generator.get_scheduler()

        # Warp denoising steps through the scheduler's shifted timestep table
        raw_steps = torch.tensor(config.denoising_steps, dtype=torch.long)
        scheduler_timesteps = torch.cat([
            self.scheduler.timesteps.cpu(),
            torch.tensor([0.0], dtype=torch.float32),
        ])
        self.denoising_step_list = scheduler_timesteps[1000 - raw_steps]
        logger.info("Warped denoising steps: %s", self.denoising_step_list.tolist())

        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = device
        self.dtype = dtype

        self.height = config.height
        self.width = config.width
        self.height_latent = config.height // VAE_SPATIAL_DOWNSAMPLE
        self.width_latent = config.width // VAE_SPATIAL_DOWNSAMPLE
        self.frame_seq_length = (
            (config.height // TOTAL_SPATIAL_DOWNSAMPLE)
            * (config.width // TOTAL_SPATIAL_DOWNSAMPLE)
        )

        self.num_frame_per_block = 1

        for block in self.generator.model.blocks:
            block.self_attn.local_attn_size = LOCAL_ATTN_SIZE
            block.self_attn.num_frame_per_block = self.num_frame_per_block
        self.generator.model.local_attn_size = LOCAL_ATTN_SIZE
        set_all_modules_frame_seq_length(self.generator, self.frame_seq_length)
        set_all_modules_max_attention_size(
            self.generator, LOCAL_ATTN_SIZE * self.frame_seq_length
        )

        self.blender = EmbeddingBlender(device=device, dtype=dtype)
        self.base_seed = config.base_seed
        self._rng = torch.Generator(device=device).manual_seed(self.base_seed)

        self.kv_cache = None
        self.crossattn_cache = None
        self.current_start_frame = 0
        self._conditioning_embeds = None
        self._last_prompts_signature = None

        self._recache_buffer = None
        self._recache_buffer_count = 0

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

    def _init_recache_buffer(self):
        """Initialize (or reinitialize) the sliding window latent buffer."""
        self._recache_buffer = torch.zeros(
            [1, LOCAL_ATTN_SIZE, 16, self.height_latent, self.width_latent],
            dtype=self.dtype,
            device=self.device,
        )
        self._recache_buffer_count = 0

    def _update_recache_buffer(self, denoised: torch.Tensor):
        """Append denoised latent(s) to the sliding window buffer."""
        num_new = denoised.shape[1]
        self._recache_buffer = torch.cat(
            [
                self._recache_buffer[:, num_new:],
                denoised.detach(),
            ],
            dim=1,
        )
        self._recache_buffer_count = min(
            self._recache_buffer_count + num_new, LOCAL_ATTN_SIZE
        )

    def _reset_caches(self):
        """Reset caches for a fresh generation sequence."""
        if self.kv_cache is not None:
            self.kv_cache = self._make_kv_cache(existing=self.kv_cache)
            self.crossattn_cache = self._make_crossattn_cache(
                existing=self.crossattn_cache
            )

        if hasattr(self.vae, "model") and hasattr(self.vae.model, "clear_cache"):
            self.vae.model.clear_cache()

        self.blender.reset()
        self._conditioning_embeds = None
        self._last_prompts_signature = None
        self.current_start_frame = 0
        self._rng = torch.Generator(device=self.device).manual_seed(self.base_seed)
        self._init_recache_buffer()

    @staticmethod
    def _normalize_prompts(prompts) -> list[dict]:
        if not prompts:
            return []
        if isinstance(prompts, str):
            return [{"text": prompts, "weight": 1.0}]
        result = []
        for p in prompts:
            if isinstance(p, str):
                result.append({"text": p, "weight": 1.0})
            elif isinstance(p, dict):
                result.append({"text": p.get("text", ""), "weight": p.get("weight", 1.0)})
            else:
                result.append({"text": str(p), "weight": 1.0})
        return result

    def _update_conditioning(self, prompts, transition) -> bool:
        """Encode prompts and blend embeddings. Returns True if embeddings changed."""
        prompt_items = self._normalize_prompts(prompts)
        if not prompt_items:
            if self._conditioning_embeds is None:
                cond = self.text_encoder(text_prompts=[""])
                self._conditioning_embeds = cond["prompt_embeds"]
                return True
            return False

        current_signature = tuple((p["text"], p["weight"]) for p in prompt_items)
        conditioning_changed = current_signature != self._last_prompts_signature
        self._last_prompts_signature = current_signature

        embeds_updated = False

        if conditioning_changed:
            if self.blender.is_transitioning():
                self.blender.cancel_transition()

            texts = [item["text"] for item in prompt_items]
            weights = [item["weight"] for item in prompt_items]
            cond = self.text_encoder(text_prompts=texts)
            batched_embeds = cond["prompt_embeds"]
            embeds_list = [batched_embeds[i : i + 1] for i in range(batched_embeds.shape[0])]

            target_blend = self.blender.blend(
                embeddings=embeds_list,
                weights=weights,
                interpolation_method="linear",
                cache_result=False,
            )

            tc = parse_transition_config(transition)
            if tc.num_steps > 0 and self._conditioning_embeds is not None:
                self.blender.start_transition(
                    source_embedding=self._conditioning_embeds,
                    target_embedding=target_blend,
                    num_steps=tc.num_steps,
                    temporal_interpolation_method=tc.temporal_interpolation_method,
                )
                next_emb = self.blender.get_next_embedding()
                if next_emb is not None:
                    self._conditioning_embeds = next_emb.to(dtype=self.dtype)
                    embeds_updated = True
            else:
                self._conditioning_embeds = target_blend.to(dtype=self.dtype)
                embeds_updated = True

        elif self.blender.is_transitioning():
            next_emb = self.blender.get_next_embedding()
            if next_emb is not None:
                self._conditioning_embeds = next_emb.to(dtype=self.dtype)
                embeds_updated = True

        if self._conditioning_embeds is None:
            cond = self.text_encoder(text_prompts=[""])
            self._conditioning_embeds = cond["prompt_embeds"]
            embeds_updated = True

        return embeds_updated

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        init_cache = kwargs.get("init_cache", False)
        prompts = kwargs.get("prompts")
        transition = kwargs.get("transition")

        if self.kv_cache is None:
            self._initialize_caches()
            self._init_recache_buffer()
        elif init_cache:
            self._reset_caches()

        embeds_updated = self._update_conditioning(prompts, transition)
        conditional_dict = {"prompt_embeds": self._conditioning_embeds}

        # RoPE boundary → smooth reset via recache buffer
        if self.current_start_frame >= MAX_ROPE_SEQ_LEN - self.num_frame_per_block:
            logger.info(
                "RoPE boundary at frame %d, smooth reset via recache",
                self.current_start_frame,
            )
            self.kv_cache = self._make_kv_cache(existing=self.kv_cache)
            self.crossattn_cache = self._make_crossattn_cache(
                existing=self.crossattn_cache
            )
            self.current_start_frame = 0
            if self._recache_buffer_count > 0:
                self._recache_from_buffer(0, conditional_dict)
                self.current_start_frame = self._recache_buffer_count

        elif embeds_updated and self.crossattn_cache is not None:
            for entry in self.crossattn_cache:
                entry["is_init"] = False

        # VAE stream_decode requires >= 2 latent frames on first batch
        if self.current_start_frame == 0:
            frame0 = self._denoise_block(1, conditional_dict)
            self._cache_clean_context(frame0, 1, conditional_dict)
            self.current_start_frame += 1

            frame1 = self._denoise_block(1, conditional_dict)
            self._cache_clean_context(frame1, 1, conditional_dict)
            self.current_start_frame += 1

            denoised = torch.cat([frame0, frame1], dim=1)
        else:
            denoised = self._denoise_block(self.num_frame_per_block, conditional_dict)
            self._cache_clean_context(denoised, self.num_frame_per_block, conditional_dict)
            self.current_start_frame += self.num_frame_per_block

        self._update_recache_buffer(denoised)
        video = self.vae.decode_to_pixel(denoised, use_cache=True)

        return {"video": postprocess_chunk(video)}

    def _denoise_block(self, num_frames: int, conditional_dict: dict) -> torch.Tensor:
        """Run the spatial denoising loop (pred_x0 + stochastic re-noising)."""
        sample = torch.randn(
            [1, num_frames, 16, self.height_latent, self.width_latent],
            device=self.device,
            dtype=self.dtype,
            generator=self._rng,
        )

        current_start = self.current_start_frame * self.frame_seq_length
        num_steps = len(self.denoising_step_list)

        for i, current_timestep in enumerate(self.denoising_step_list):
            timestep = (
                torch.ones(
                    [1, num_frames],
                    device=self.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if i < num_steps - 1:
                _, pred_x0 = self.generator(
                    noisy_image_or_video=sample,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                )

                next_timestep = self.denoising_step_list[i + 1]
                flat_x0 = pred_x0.flatten(0, 1)
                renoise = torch.randn(
                    flat_x0.shape,
                    device=flat_x0.device,
                    dtype=flat_x0.dtype,
                    generator=self._rng,
                )
                sample = self.scheduler.add_noise(
                    flat_x0,
                    renoise,
                    next_timestep
                    * torch.ones(
                        [flat_x0.shape[0]],
                        device=flat_x0.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, pred_x0.shape[:2])
            else:
                _, pred_x0 = self.generator(
                    noisy_image_or_video=sample,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                )
                sample = pred_x0

        return sample

    def _recache_from_buffer(self, recache_start: int, conditional_dict: dict):
        """Re-encode buffered latents to rebuild KV cache at new positions."""
        n = self._recache_buffer_count
        if n == 0:
            return

        frames = self._recache_buffer[:, -n:].contiguous()

        for i in range(n):
            frame = frames[:, i : i + 1]
            ts = torch.zeros([1, 1], device=self.device, dtype=torch.int64)
            self.generator(
                noisy_image_or_video=frame,
                conditional_dict=conditional_dict,
                timestep=ts,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=(recache_start + i) * self.frame_seq_length,
            )

        logger.info(
            "Recached %d frames starting at position %d", n, recache_start
        )

    def _cache_clean_context(self, denoised: torch.Tensor, num_frames: int, conditional_dict: dict):
        context_timestep = torch.zeros(
            [1, num_frames],
            device=self.device,
            dtype=torch.int64,
        )
        current_start = self.current_start_frame * self.frame_seq_length

        self.generator(
            noisy_image_or_video=denoised,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start,
        )
