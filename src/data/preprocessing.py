"""
Preprocessing pipeline for EyePACS fundus images.

Handles circle cropping (removes black borders around the fundus),
resizing to 512x512, and basic quality filtering for images that
are too dark or overexposed to be useful for training.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def circle_crop(img: Image.Image) -> Image.Image:
    """
    Crop the circular fundus region from the image.

    Fundus photographs have a circular field of view surrounded by
    black borders. Keeping those borders would teach the diffusion
    model to generate black frames instead of retinal features.
    """
    img_array = np.array(img)

    # Sum across color channels to get a brightness mask
    gray = np.mean(img_array, axis=2)

    # Threshold to find the illuminated fundus region
    mask = gray > 15
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop to the bounding box of the fundus circle
    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped


def is_usable(img: Image.Image, dark_threshold: float = 10.0,
              bright_threshold: float = 245.0) -> bool:
    """
    Quick quality check - reject images that are mostly black or blown out.

    EyePACS has ~5-10% of images with acquisition problems (too dark,
    too bright, severely out of focus). We skip these rather than
    contaminating the training set.
    """
    img_array = np.array(img).astype(np.float32)
    mean_brightness = img_array.mean()

    if mean_brightness < dark_threshold:
        return False
    if mean_brightness > bright_threshold:
        return False

    return True


def preprocess_eyepacs(
    raw_dir: str,
    output_dir: str,
    labels_csv: str,
    image_size: int = 512,
    max_samples_per_grade: int | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for EyePACS images.

    Reads raw images, applies circle cropping, resizes, filters
    unusable images, and saves the processed versions. Returns a
    DataFrame with file paths and grades for downstream use.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_csv)
    labels.columns = [c.strip().lower() for c in labels.columns]

    # Auto-detect dataset format
    # APTOS 2019: id_code, diagnosis
    # EyePACS: image, level
    if "id_code" in labels.columns and "diagnosis" in labels.columns:
        id_col, grade_col = "id_code", "diagnosis"
    elif "image" in labels.columns and "level" in labels.columns:
        id_col, grade_col = "image", "level"
    else:
        raise ValueError(
            f"Unrecognized CSV format. Expected (id_code, diagnosis) or "
            f"(image, level). Found: {list(labels.columns)}"
        )

    processed_records = []
    skipped = 0

    # Cap samples per grade if specified (for fine-tuning subset)
    grade_counts = {g: 0 for g in range(5)}

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Preprocessing"):
        image_name = row[id_col]
        grade = int(row[grade_col])

        if max_samples_per_grade and grade_counts[grade] >= max_samples_per_grade:
            continue

        # APTOS uses .png, EyePACS uses .jpeg
        img_path = raw_path / f"{image_name}.png"
        if not img_path.exists():
            img_path = raw_path / f"{image_name}.jpeg"
        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        if not is_usable(img):
            skipped += 1
            continue

        img = circle_crop(img)
        img = img.resize((image_size, image_size), Image.LANCZOS)

        # Save organized by grade for easier browsing
        grade_dir = output_path / f"grade_{grade}"
        grade_dir.mkdir(exist_ok=True)
        save_path = grade_dir / f"{image_name}.png"
        img.save(save_path, "PNG")

        processed_records.append({
            "image_path": str(save_path),
            "image_name": image_name,
            "grade": grade,
        })
        grade_counts[grade] += 1

    result_df = pd.DataFrame(processed_records)
    manifest_path = output_path / "manifest.csv"
    result_df.to_csv(manifest_path, index=False)

    print(f"Processed {len(result_df)} images, skipped {skipped}")
    for grade in range(5):
        count = grade_counts[grade]
        print(f"  Grade {grade}: {count} images")

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess EyePACS fundus images")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw EyePACS images")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save processed images")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to trainLabels.csv")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_samples_per_grade", type=int, default=None)
    args = parser.parse_args()

    preprocess_eyepacs(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        labels_csv=args.labels_csv,
        image_size=args.image_size,
        max_samples_per_grade=args.max_samples_per_grade,
    )