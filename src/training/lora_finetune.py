"""
LoRA fine-tuning of Stable Diffusion 2.1 on diabetic retinopathy
fundus images, conditioned on DR severity grade.

Only the LoRA adapter weights are trained (~800K params).
The full UNet (860M params), VAE, and text encoder stay frozen.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import wandb
from src.data.eyepacs_loader import EyePACSDataset


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_lora(unet: UNet2DConditionModel, config: dict) -> UNet2DConditionModel:
    """Inject LoRA adapters into UNet cross-attention layers."""
    lora_config = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        target_modules=config["model"]["target_modules"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"LoRA trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return unet


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles both images and variable-length prompts."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "grades": [b["grade"] for b in batch],
    }


def train(config_path: str):
    config = load_config(config_path)
    train_cfg = config["training"]
    model_cfg = config["model"]

    accelerator = Accelerator(
        mixed_precision=train_cfg["mixed_precision"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    )

    # Load pretrained SD components
    model_id = model_cfg["pretrained_model"]
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze everything except the LoRA weights we're about to add
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    unet = setup_lora(unet, config)

    # Dataset
    dataset = EyePACSDataset(
        manifest_csv=f"{config['data']['data_dir']}/manifest.csv",
        image_size=config["data"]["image_size"],
        mode="diffusion",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
    )

    num_update_steps = math.ceil(
        len(dataloader) / train_cfg["gradient_accumulation_steps"]
    )
    # Use max_train_steps or full epochs, whichever is fewer
    max_steps = min(train_cfg["max_train_steps"], num_update_steps * 10)

    lr_scheduler = get_scheduler(
        train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"],
        num_training_steps=max_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Move frozen models to device
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    # W&B init
    if accelerator.is_main_process:
        wandb.init(
            project=config["wandb"]["project"],
            config=config,
            tags=config["wandb"]["tags"],
        )

    # Training loop
    global_step = 0
    progress = tqdm(total=max_steps, desc="Training", disable=not accelerator.is_main_process)

    unet.train()
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            with accelerator.accumulate(unet):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=torch.float16)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()

                # Forward diffusion - add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text prompts
                with torch.no_grad():
                    token_ids = tokenizer(
                        batch["prompts"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                    encoder_hidden = text_encoder(token_ids)[0].to(dtype=torch.float16)

                # Predict noise
                noise_pred = unet(
                    noisy_latents.to(dtype=torch.float16),
                    timesteps,
                    encoder_hidden,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.update(1)

            if accelerator.is_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step,
                })

            # Save checkpoint periodically
            if global_step % train_cfg["checkpointing_steps"] == 0:
                if accelerator.is_main_process:
                    save_dir = Path("checkpoints") / f"step_{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(unet)
                    unwrapped.save_pretrained(save_dir)
                    print(f"Saved checkpoint at step {global_step}")

    progress.close()

    # Save final model
    if accelerator.is_main_process:
        final_dir = Path("checkpoints") / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(final_dir)
        print(f"Training complete. Final model saved to {final_dir}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFuse LoRA fine-tuning")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to training config file",
    )
    args = parser.parse_args()
    train(args.config)
