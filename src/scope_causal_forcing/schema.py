"""Configuration schema for Causal Forcing pipeline."""

from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.artifacts import Artifact, HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from scope.core.pipelines.common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from scope.core.pipelines.utils import VaeType


class CausalForcingConfig(BasePipelineConfig):
    """Configuration for Causal Forcing pipeline."""

    pipeline_id = "causal-forcing"
    pipeline_name = "Causal Forcing"
    pipeline_description = (
        "An autoregressive video diffusion model from Tsinghua, Shengshu and UT Austin. "
        "Trained using Causal Forcing on Wan2.1 1.3B, achieving 19% better dynamics and "
        "16% better instruction following over Self-Forcing, with identical inference speed. "
        "Supports framewise streaming generation and Rolling Forcing for long video."
    )
    docs_url = "https://github.com/thu-ml/Causal-Forcing"
    estimated_vram_gb = 20.0

    artifacts: ClassVar[list[Artifact]] = [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="zhuhz22/Causal-Forcing",
            files=["framewise/causal_forcing.pt"],
        ),
    ]

    supports_cache_management = True
    supports_quantization = True
    min_dimension = 16

    # Configuration fields
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
        json_schema_extra=ui_field_config(order=1, is_load_param=True, label="VAE"),
    )
    height: int = Field(
        default=480,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=832,
        ge=1,
        description="Output width in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    base_seed: int = Field(
        default=42,
        ge=0,
        description="Base random seed for reproducible generation",
        json_schema_extra=ui_field_config(order=3, is_load_param=True, label="Seed"),
    )
    manage_cache: bool = Field(
        default=True,
        description="Enable automatic cache management for performance optimization",
        json_schema_extra=ui_field_config(
            order=4, component="cache", is_load_param=True
        ),
    )
    guidance_scale: float = Field(
        default=6.0,
        ge=1.0,
        le=20.0,
        description="Classifier-Free Guidance scale. Higher values produce more prompt-adherent but less diverse output.",
        json_schema_extra=ui_field_config(order=5, label="CFG Scale"),
    )
    denoising_steps: list[int] = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
        json_schema_extra=ui_field_config(
            order=6, component="denoising_steps", is_load_param=True
        ),
    )
    quantization: None = Field(
        default=None,
        description="Quantization method for the diffusion model.",
        json_schema_extra=ui_field_config(
            order=6, component="quantization", is_load_param=True
        ),
    )

    modes = {
        "text": ModeDefaults(default=True),
    }
