"""
Generation quality evaluation using FID and LPIPS.

FID measures distributional similarity between real and synthetic sets.
LPIPS measures perceptual distance between individual image pairs.
Together they tell us whether the synthetic fundus images look
realistic at both the dataset level and the individual image level.
"""

import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision import transforms
from tqdm import tqdm

import wandb


def compute_fid(real_dir: str, synthetic_dir: str, device: str = "cuda") -> float:
    """
    FID between two directories of images.
    Lower is better. Good synthetic medical images typically
    land between 30-80 depending on the domain.
    """
    fid = calculate_fid_given_paths(
        [real_dir, synthetic_dir],
        batch_size=32,
        device=device,
        dims=2048,
    )
    return fid


def compute_lpips_stats(
    real_dir: str,
    synthetic_dir: str,
    num_pairs: int = 200,
    device: str = "cuda",
) -> dict:
    """
    Sample random (real, synthetic) pairs and compute LPIPS.
    Reports mean and std. Lower LPIPS means the synthetic images
    are perceptually closer to real ones.
    """
    loss_fn = lpips.LPIPS(net="alex").to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        # LPIPS expects [-1, 1] range
        transforms.Normalize([0.5], [0.5]),
    ])

    real_images = sorted(Path(real_dir).glob("*.png"))
    synth_images = sorted(Path(synthetic_dir).glob("*.png"))

    if not real_images or not synth_images:
        print(f"Warning: empty directory. Real: {len(real_images)}, Synth: {len(synth_images)}")
        return {"lpips_mean": float("nan"), "lpips_std": float("nan")}

    rng = np.random.default_rng(42)
    num_pairs = min(num_pairs, len(real_images), len(synth_images))

    real_indices = rng.choice(len(real_images), size=num_pairs, replace=False)
    synth_indices = rng.choice(len(synth_images), size=num_pairs, replace=False)

    scores = []
    for ri, si in tqdm(zip(real_indices, synth_indices), total=num_pairs, desc="LPIPS"):
        real_img = preprocess(Image.open(real_images[ri]).convert("RGB")).unsqueeze(0).to(device)
        synth_img = preprocess(Image.open(synth_images[si]).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            d = loss_fn(real_img, synth_img).item()
        scores.append(d)

    return {
        "lpips_mean": float(np.mean(scores)),
        "lpips_std": float(np.std(scores)),
    }


def evaluate_all_grades(
    real_base_dir: str,
    synthetic_base_dir: str,
    log_wandb: bool = True,
):
    """Run FID and LPIPS for each DR grade that has synthetic images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    real_base = Path(real_base_dir)
    synth_base = Path(synthetic_base_dir)

    for grade_dir in sorted(synth_base.iterdir()):
        if not grade_dir.is_dir() or not grade_dir.name.startswith("grade_"):
            continue

        grade = int(grade_dir.name.split("_")[1])
        real_grade_dir = real_base / f"grade_{grade}"

        if not real_grade_dir.exists():
            print(f"Skipping grade {grade}: no real images found at {real_grade_dir}")
            continue

        print(f"\nEvaluating Grade {grade}")

        fid = compute_fid(str(real_grade_dir), str(grade_dir), device)
        lpips_stats = compute_lpips_stats(str(real_grade_dir), str(grade_dir), device=device)

        results[grade] = {"fid": fid, **lpips_stats}
        print(f"  FID: {fid:.2f}")
        print(f"  LPIPS: {lpips_stats['lpips_mean']:.4f} +/- {lpips_stats['lpips_std']:.4f}")

        if log_wandb:
            wandb.log({
                f"eval/fid_grade_{grade}": fid,
                f"eval/lpips_mean_grade_{grade}": lpips_stats["lpips_mean"],
            })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFuse generation quality evaluation")
    parser.add_argument("--real_dir", type=str, default="data/processed")
    parser.add_argument("--synthetic_dir", type=str, default="data/synthetic")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if not args.no_wandb:
        wandb.init(project="medfuse", job_type="evaluation")

    evaluate_all_grades(args.real_dir, args.synthetic_dir, log_wandb=not args.no_wandb)
