"""
Gradio demo for MedFuse.

Three tabs:
  1. Generate - create synthetic fundus images by DR grade
  2. Compare  - side-by-side real vs synthetic with quality scores
  3. Results  - ablation plot showing downstream improvement
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.prompt_registry import get_prompt, get_label, NEGATIVE_PROMPT
from src.generation.generate import load_pipeline


# Globals set at app startup
PIPELINE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_image(grade: int, guidance_scale: float, seed: int) -> Image.Image:
    """Generate a single synthetic fundus image."""
    prompt = get_prompt(grade)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.no_grad():
        result = PIPELINE(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            generator=generator,
        )

    return result.images[0]


def load_comparison_grid(grade: int, num_samples: int = 4) -> list[Image.Image]:
    """Load real and synthetic samples for side-by-side comparison."""
    images = []

    real_dir = Path("data/processed") / f"grade_{grade}"
    synth_dir = Path("data/synthetic") / f"grade_{grade}"

    real_images = sorted(real_dir.glob("*.png"))[:num_samples] if real_dir.exists() else []
    synth_images = sorted(synth_dir.glob("*.png"))[:num_samples] if synth_dir.exists() else []

    for img_path in real_images:
        images.append(Image.open(img_path))
    for img_path in synth_images:
        images.append(Image.open(img_path))

    return images


def plot_ablation_results() -> plt.Figure:
    """Plot the ablation study results if available."""
    results_path = Path("outputs/ablation_results.json")
    if not results_path.exists():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Run the ablation study first", ha="center", va="center")
        return fig

    with open(results_path) as f:
        results = json.load(f)

    ratios = []
    aurocs = []
    g3_recalls = []
    g4_recalls = []

    for ratio_str, metrics in sorted(results.items(), key=lambda x: float(x[0])):
        ratios.append(float(ratio_str) * 100)
        aurocs.append(metrics["weighted_auroc"])

        pg = metrics["per_grade"]
        g3 = pg.get("3", pg.get(3, {}))
        g4 = pg.get("4", pg.get(4, {}))
        g3_recalls.append(g3.get("recall", 0))
        g4_recalls.append(g4.get("recall", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ratios, aurocs, "o-", color="#534AB7", linewidth=2, markersize=8)
    ax1.set_xlabel("Synthetic augmentation (%)")
    ax1.set_ylabel("Weighted AUROC")
    ax1.set_title("Overall classification performance")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ratios, g3_recalls, "s-", color="#D85A30", linewidth=2, markersize=8, label="Grade 3 (Severe)")
    ax2.plot(ratios, g4_recalls, "^-", color="#1D9E75", linewidth=2, markersize=8, label="Grade 4 (Proliferative)")
    ax2.set_xlabel("Synthetic augmentation (%)")
    ax2.set_ylabel("Recall")
    ax2.set_title("Rare-grade recall improvement")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def build_app(model_path: str, pretrained_model: str) -> gr.Blocks:
    """Build the Gradio interface."""
    global PIPELINE

    PIPELINE = load_pipeline(pretrained_model, model_path, DEVICE)

    grade_choices = [(get_label(g), g) for g in range(5)]

    with gr.Blocks(title="MedFuse") as app:
        gr.Markdown("## MedFuse - Synthetic DR fundus generation with LoRA-adapted diffusion")

        with gr.Tab("Generate"):
            with gr.Row():
                grade_input = gr.Dropdown(
                    choices=grade_choices, value=3, label="DR grade",
                )
                guidance_input = gr.Slider(
                    minimum=3.0, maximum=15.0, value=7.5, step=0.5,
                    label="Guidance scale",
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)

            generate_btn = gr.Button("Generate")
            output_image = gr.Image(label="Synthetic fundus image", type="pil")

            generate_btn.click(
                fn=generate_image,
                inputs=[grade_input, guidance_input, seed_input],
                outputs=output_image,
            )

        with gr.Tab("Compare"):
            compare_grade = gr.Dropdown(
                choices=grade_choices, value=3, label="DR grade",
            )
            compare_btn = gr.Button("Load comparison")
            compare_gallery = gr.Gallery(label="Top row: real, Bottom row: synthetic", columns=4)

            compare_btn.click(
                fn=load_comparison_grid,
                inputs=[compare_grade],
                outputs=compare_gallery,
            )

        with gr.Tab("Results"):
            results_btn = gr.Button("Show ablation results")
            results_plot = gr.Plot(label="Augmentation ablation")

            results_btn.click(
                fn=plot_ablation_results,
                outputs=results_plot,
            )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedFuse Gradio demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final")
    parser.add_argument(
        "--pretrained_model", type=str,
        default="stabilityai/stable-diffusion-2-1-base",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.checkpoint, args.pretrained_model)
    app.launch(server_port=args.port, share=args.share)
