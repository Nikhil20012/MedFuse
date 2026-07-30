"""
Sweep guidance scale values per DR grade to find the setting that
produces the most realistic synthetic images.

Too low (< 5) gives blurry, unconditional-looking outputs.
Too high (> 12) gives oversaturated, artifact-heavy images.
The sweet spot depends on the grade - severe DR with complex
pathology tends to need a different scale than healthy retinas.
"""

import argparse
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline, DDIMScheduler

from src.data.prompt_registry import get_prompt, get_label, NEGATIVE_PROMPT
from src.generation.generate import load_pipeline


SWEEP_VALUES = [5.0, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0]


def run_sweep(config_path: str, lora_checkpoint: str, num_samples: int = 4):
    """
    Generate a small grid of images at each guidance scale for
    visual comparison. Pick the best scale before bulk generation.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path("outputs/guidance_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(config["model"]["pretrained_model"], lora_checkpoint, device)

    for grade in range(5):
        prompt = get_prompt(grade)
        label = get_label(grade)
        print(f"\nGrade {grade} ({label})")

        for scale in SWEEP_VALUES:
            generator = torch.Generator(device=device).manual_seed(42)

            with torch.no_grad():
                result = pipe(
                    prompt=[prompt] * num_samples,
                    negative_prompt=[NEGATIVE_PROMPT] * num_samples,
                    guidance_scale=scale,
                    num_inference_steps=50,
                    generator=generator,
                )

            scale_dir = output_dir / f"grade_{grade}" / f"scale_{scale}"
            scale_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(result.images):
                img.save(scale_dir / f"sample_{i}.png", "PNG")

            print(f"  Scale {scale}: {num_samples} samples saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guidance scale sweep")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final")
    parser.add_argument("--num_samples", type=int, default=4)
    args = parser.parse_args()
    run_sweep(args.config, args.checkpoint, args.num_samples)
