"""
Ablation study across different synthetic-to-real augmentation ratios.

Trains the downstream classifier at 0%, 25%, 50%, 100%, and 200%
synthetic augmentation and compares per-grade recall and overall AUROC.

This is the key experiment: it quantifies exactly how much value
the generated data adds, and at what ratio it saturates or hurts.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

import wandb
from src.data.eyepacs_loader import SyntheticFundusDataset
from src.evaluation.downstream_classifier import train_classifier


def subsample_synthetic(
    full_synthetic_dir: str,
    output_dir: str,
    ratio: float,
    real_counts_per_grade: dict[int, int],
):
    """
    Create a subset of synthetic images at the given ratio
    relative to the real data count per grade.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    rng = np.random.default_rng(42)

    for grade, real_count in real_counts_per_grade.items():
        target_count = int(real_count * ratio)
        if target_count == 0:
            continue

        source_dir = Path(full_synthetic_dir) / f"grade_{grade}"
        if not source_dir.exists():
            continue

        available = sorted(source_dir.glob("*.png"))
        if not available:
            continue

        # Sample with replacement if we need more than available
        num_to_pick = min(target_count, len(available))
        selected = rng.choice(available, size=num_to_pick, replace=False)

        dest_dir = output_path / f"grade_{grade}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in selected:
            shutil.copy2(img_path, dest_dir / img_path.name)


def run_ablation(config_path: str, train_manifest: str, val_manifest: str):
    """Run the full ablation across all configured ratios."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ratios = config["evaluation"]["augmentation_ratios"]

    # Count real images per grade from the training manifest
    import pandas as pd
    train_df = pd.read_csv(train_manifest)
    real_counts = train_df["grade"].value_counts().to_dict()
    print(f"Real image counts per grade: {real_counts}")

    all_results = {}

    for ratio in ratios:
        run_name = f"ratio_{ratio:.0%}".replace("%", "pct")
        print(f"\n{'=' * 40}")
        print(f"Ablation: {ratio:.0%} synthetic augmentation")
        print(f"{'=' * 40}")

        synthetic_dir = None
        if ratio > 0:
            synthetic_dir = f"data/ablation_subsets/ratio_{ratio}"
            subsample_synthetic(
                "data/synthetic", synthetic_dir, ratio, real_counts,
            )

        wandb.init(
            project="medfuse",
            job_type="ablation",
            name=run_name,
            reinit=True,
        )

        metrics = train_classifier(
            train_manifest, val_manifest, config,
            synthetic_dir=synthetic_dir,
            run_name=run_name,
        )

        all_results[str(ratio)] = metrics
        wandb.finish()

    # Save consolidated results
    results_path = Path("outputs/ablation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAblation results saved to {results_path}")
    _print_summary(all_results)


def _print_summary(results: dict):
    """Print a readable comparison table."""
    print(f"\n{'Ratio':<12} {'Accuracy':<10} {'AUROC':<10} "
          f"{'Recall G3':<12} {'Recall G4':<12}")
    print("-" * 56)

    for ratio, metrics in results.items():
        g3_recall = metrics["per_grade"]["3"]["recall"] if "3" in metrics.get("per_grade", {}) else metrics.get("per_grade", {}).get(3, {}).get("recall", 0)
        g4_recall = metrics["per_grade"]["4"]["recall"] if "4" in metrics.get("per_grade", {}) else metrics.get("per_grade", {}).get(4, {}).get("recall", 0)

        print(
            f"{float(ratio):>8.0%}    "
            f"{metrics['accuracy']:<10.4f}"
            f"{metrics['weighted_auroc']:<10.4f}"
            f"{g3_recall:<12.4f}"
            f"{g4_recall:<12.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFuse ablation study")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--train_manifest", type=str, default="data/processed/train_manifest.csv")
    parser.add_argument("--val_manifest", type=str, default="data/processed/val_manifest.csv")
    args = parser.parse_args()
    run_ablation(args.config, args.train_manifest, args.val_manifest)
