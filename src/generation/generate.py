"""
Generate synthetic diabetic retinopathy fundus images using
the LoRA-adapted Stable Diffusion model.

Images are generated per DR grade, with the bulk of generation
targeting Grades 3 and 4 where real data is scarcest.
"""

import argparse
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import PeftModel
from tqdm import tqdm

from src.data.prompt_registry import get_prompt, get_label, NEGATIVE_PROMPT


def load_pipeline(
    pretrained_model: str,
    lora_checkpoint: str,
    device: str = "cuda",
) -> StableDiffusionPipeline:
    """Load SD pipeline and merge in the LoRA weights."""
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # Swap in DDIM for faster inference (50 steps vs 1000)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load LoRA adapter weights into the UNet
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_checkpoint)
    pipe.unet.eval()

    pipe = pipe.to(device)

    # Memory optimization for consumer GPUs
    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


def generate_grade(
    pipe: StableDiffusionPipeline,
    grade: int,
    num_images: int,
    output_dir: Path,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    batch_size: int = 4,
    seed: int = 42,
):
    """Generate synthetic fundus images for a single DR grade."""
    grade_dir = output_dir / f"grade_{grade}"
    grade_dir.mkdir(parents=True, exist_ok=True)

    prompt = get_prompt(grade)
    label = get_label(grade)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    generated = 0
    batch_idx = 0

    pbar = tqdm(total=num_images, desc=f"Grade {grade} ({label})")

    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)

        with torch.no_grad():
            result = pipe(
                prompt=[prompt] * current_batch,
                negative_prompt=[NEGATIVE_PROMPT] * current_batch,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        for i, img in enumerate(result.images):
            img_idx = generated + i
            save_path = grade_dir / f"synthetic_{grade}_{img_idx:04d}.png"
            img.save(save_path, "PNG")

        generated += current_batch
        batch_idx += 1
        pbar.update(current_batch)

    pbar.close()
    print(f"  Grade {grade} ({label}): {generated} images saved to {grade_dir}")


def generate_all(config_path: str, lora_checkpoint: str):
    """Generate synthetic images for all configured grades."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gen_cfg = config["generation"]
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading pipeline from {config['model']['pretrained_model']}")
    print(f"LoRA weights from {lora_checkpoint}")
    pipe = load_pipeline(
        config["model"]["pretrained_model"],
        lora_checkpoint,
        device=device,
    )

    for grade, num_images in gen_cfg["num_images_per_grade"].items():
        grade = int(grade)
        if num_images == 0:
            print(f"Skipping Grade {grade} (not configured for generation)")
            continue

        generate_grade(
            pipe=pipe,
            grade=grade,
            num_images=num_images,
            output_dir=output_dir,
            guidance_scale=gen_cfg["guidance_scale"],
            num_inference_steps=gen_cfg["num_inference_steps"],
        )

    print(f"\nAll synthetic images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFuse synthetic generation")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/final",
        help="Path to the LoRA checkpoint directory",
    )
    args = parser.parse_args()
    generate_all(args.config, args.checkpoint)
