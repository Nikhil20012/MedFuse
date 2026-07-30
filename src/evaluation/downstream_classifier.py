"""
Downstream 5-class DR grading classifier using EfficientNet-B0.

This is intentionally a simple classifier. The point isn't to build
the best DR screening model - it's to prove that synthetic data from
MedFuse measurably improves classification on the rare severe grades.
"""

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import wandb
from src.data.eyepacs_loader import EyePACSDataset, SyntheticFundusDataset
from src.data.prompt_registry import GRADE_LABELS


def build_classifier(num_classes: int = 5) -> nn.Module:
    """EfficientNet-B0 with a 5-class head for DR grading."""
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images = batch["pixel_values"].to(device)
        labels = torch.tensor(batch["grade"]).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict:
    """Compute per-grade metrics on the validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        images = batch["pixel_values"].to(device)
        labels = batch["grade"]

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    # Per-grade recall is the key metric
    report = classification_report(
        all_labels, all_preds,
        target_names=[GRADE_LABELS[g] for g in range(5)],
        output_dict=True,
        zero_division=0,
    )

    # Weighted AUROC across all grades
    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    except ValueError:
        auroc = 0.0

    return {
        "accuracy": report["accuracy"],
        "weighted_auroc": auroc,
        "per_grade": {
            grade: {
                "precision": report[GRADE_LABELS[grade]]["precision"],
                "recall": report[GRADE_LABELS[grade]]["recall"],
                "f1": report[GRADE_LABELS[grade]]["f1-score"],
            }
            for grade in range(5)
        },
    }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "grade": [b["grade"] for b in batch],
    }


def train_classifier(
    train_manifest: str,
    val_manifest: str,
    config: dict,
    synthetic_dir: str | None = None,
    run_name: str = "baseline",
) -> dict:
    """
    Train the downstream classifier, optionally augmenting with
    synthetic data. Returns final test metrics.
    """
    eval_cfg = config["evaluation"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Real training data
    train_dataset = EyePACSDataset(train_manifest, mode="classification")

    # Optionally add synthetic data
    if synthetic_dir and Path(synthetic_dir).exists():
        synth_dataset = SyntheticFundusDataset(synthetic_dir)
        train_dataset = ConcatDataset([train_dataset, synth_dataset])
        print(f"Training with {len(train_dataset)} images (real + synthetic)")
    else:
        print(f"Training with {len(train_dataset)} real images only")

    val_dataset = EyePACSDataset(val_manifest, mode="classification")

    train_loader = DataLoader(
        train_dataset, batch_size=eval_cfg["downstream_batch_size"],
        shuffle=True, num_workers=4, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_cfg["downstream_batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate_fn,
    )

    model = build_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eval_cfg["downstream_lr"])

    best_auroc = 0.0
    best_metrics = {}

    for epoch in range(eval_cfg["downstream_epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{eval_cfg['downstream_epochs']} - "
            f"Loss: {train_loss:.4f} - "
            f"Acc: {metrics['accuracy']:.4f} - "
            f"AUROC: {metrics['weighted_auroc']:.4f}"
        )

        wandb.log({
            f"{run_name}/train_loss": train_loss,
            f"{run_name}/accuracy": metrics["accuracy"],
            f"{run_name}/weighted_auroc": metrics["weighted_auroc"],
            f"{run_name}/recall_grade_3": metrics["per_grade"][3]["recall"],
            f"{run_name}/recall_grade_4": metrics["per_grade"][4]["recall"],
            f"{run_name}/epoch": epoch + 1,
        })

        if metrics["weighted_auroc"] > best_auroc:
            best_auroc = metrics["weighted_auroc"]
            best_metrics = metrics

    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedFuse downstream classifier")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--train_manifest", type=str, default="data/processed/train_manifest.csv")
    parser.add_argument("--val_manifest", type=str, default="data/processed/val_manifest.csv")
    parser.add_argument("--synthetic_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    wandb.init(project="medfuse", job_type="downstream", name=args.run_name)
    metrics = train_classifier(
        args.train_manifest, args.val_manifest, config,
        synthetic_dir=args.synthetic_dir, run_name=args.run_name,
    )
    print(f"\nBest metrics: {metrics}")
