"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel

from src.hooked_model.hooked_model_stableaudio import HookedStableAudioModel
from src.hooked_model.utils import get_timesteps
from src.sae.cache_activations_runner import CacheActivationsRunner
from src.sae.config import CacheActivationsRunnerConfig
from src.hooked_model.scheduler import DDIMScheduler


def run():
    config = CacheActivationsRunnerConfig(
        hook_names=[
            "transformer_blocks.11.attn2",
        ],
        dataset_type="csv",
        dataset_name="data/musiccaps_voice.csv", # "data/musiccaps_voice.csv" # "data/musiccaps_public.csv"
        dataset_duplicate_rows=4,
        column="caption",
        negative_prompt="Low quality, average quality.",
        model_name="stabilityai/stable-audio-open-1.0",
        num_inference_steps=100,
        audio_length_in_s=20.0,
        num_waveforms_per_prompt=1,
        guidance_scale=7.0,
        cache_every_n_timesteps=10,
        along_freqs=False
    )
    args = parse(config)
    accelerator = Accelerator()

    pipe = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype, use_safetensors=True).to(
        accelerator.device
    )
    model = StableAudioDiTModel.from_pretrained(
        args.model_name,
        subfolder="transformer",
        torch_dtype=args.dtype,
        use_safetensors=True,
    ).to(accelerator.device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.transformer = model
    # scheduler = pipe.scheduler
    # pipe.unet = model

    hooked_model = HookedStableAudioModel(
        model=model,
        scheduler=scheduler,
        encode_prompt=pipe.encode_prompt,
        get_timesteps=get_timesteps,
        pipeline=pipe,
        vae=pipe.vae,
        accelerator = accelerator
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
