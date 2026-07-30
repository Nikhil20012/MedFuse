<div align="center">

# 👁️ MedFuse

**Synthetic diabetic retinopathy fundus generation with LoRA-adapted diffusion models**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/diffusers)
[![Stable Diffusion](https://img.shields.io/badge/Stable_Diffusion-2.1-A100FF?style=flat-square&logo=stability-ai&logoColor=white)](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Tracking-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-F97316?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

`IN PROGRESS`

</div>

---

Synthetic fundus image generation for diabetic retinopathy data augmentation.

Fine-tunes Stable Diffusion 2.1 with LoRA adapters on the APTOS 2019 dataset
to generate grade-conditioned retinal fundus images targeting underrepresented
severe DR classes. Validates generation quality through FID and LPIPS, then
proves downstream utility by measuring classification recall improvement on
rare grades using an EfficientNet-B0 ablation study across multiple
synthetic-to-real augmentation ratios.

## Problem

Diabetic retinopathy affects an estimated 103 million people worldwide and is
the leading cause of preventable blindness in working-age adults. Automated
screening models consistently underperform on the cases that matter most:
severe and proliferative DR.

The root cause is data imbalance. In the APTOS 2019 benchmark (3,662 fundus
images across 5 severity grades):

| Grade | Severity | Images | Share |
|---|---|---|---|
| 0 | No DR | 1,805 | 49.3% |
| 1 | Mild NPDR | 370 | 10.1% |
| 2 | Moderate NPDR | 999 | 27.3% |
| 3 | Severe NPDR | 193 | 5.3% |
| 4 | Proliferative DR | 295 | 8.1% |

Grade 3 (Severe) has only 193 images. Models trained on this distribution
struggle to learn the pathological features that characterize severe cases:
extensive hemorrhages, venous beading, cotton-wool spots, and
neovascularization. Traditional augmentation (rotations, flips, color jitter)
preserves existing sample morphology but cannot synthesize new instances of
these pathological patterns.

The same imbalance exists at larger scale in EyePACS (88,702 images, Grades
3-4 at just 4.4%), confirming this is a systemic problem in DR datasets rather
than a quirk of one benchmark.

## What makes this different

Most diffusion-based medical augmentation projects stop at generation quality
metrics. MedFuse runs a full downstream validation loop:

- Grade-conditioned generation using clinically grounded text prompts
  mapped to the ICDR severity scale
- Dual-track evaluation separating visual realism (FID, LPIPS) from
  clinical utility (downstream AUROC, per-grade recall)
- Ratio ablation study (0/25/50/100/200% synthetic) quantifying exactly
  where augmentation helps and where it saturates

## Architecture

```
APTOS 2019 (3,662 fundus images, 5 DR grades)
        |
  Preprocessing
  (circle crop, resize 512x512, quality filter)
        |
        v
  Stable Diffusion 2.1 + LoRA (rank 4)
  ~800K trainable params, full UNet frozen
        |
   Text conditioning via prompt registry
   (grade -> ICDR clinical description)
        |
        v
  DDIM generation (50 steps, guidance sweep 5.0-12.0)
        |
        +--------> Track A: Generation quality
        |             FID per grade
        |             LPIPS per grade
        |             Real vs synthetic visual grid
        |
        +--------> Track B: Downstream utility
                      EfficientNet-B0 (5-class DR grader)
                      |
                      +-> Baseline: real data only
                      +-> Augmented: real + synthetic
                      +-> Ratio ablation (5 levels)
                      +-> Per-grade recall (focus: Grades 3-4)
                      |
                   Weights & Biases
                   (all experiments tracked)
                      |
                      v
                   Gradio demo (HF Spaces)
                   Tab 1: Generate fundus by grade
                   Tab 2: Real vs synthetic comparison
                   Tab 3: Ablation results plot
```

## Tech stack

| Tool | Role |
|---|---|
| PyTorch | Deep learning framework |
| Hugging Face diffusers | Stable Diffusion pipeline and schedulers |
| Hugging Face PEFT | LoRA adapter injection and management |
| Hugging Face accelerate | Mixed-precision training, multi-GPU support |
| Weights & Biases | Experiment tracking, loss curves, generated samples |
| pytorch-fid | FID computation between real and synthetic sets |
| lpips | Perceptual similarity scoring |
| timm | EfficientNet-B0 for downstream classification |
| Gradio | Interactive demo interface |
| Hugging Face Spaces | Demo deployment |

## Project structure

```
MedFuse/
├── configs/
│   └── train_config.yaml           All hyperparameters in one place
├── src/
│   ├── data/
│   │   ├── eyepacs_loader.py       PyTorch datasets (diffusion + classification)
│   │   ├── preprocessing.py        Circle crop, resize, quality filter
│   │   └── prompt_registry.py      DR grade -> clinical text prompt mapping
│   ├── training/
│   │   └── lora_finetune.py        LoRA fine-tuning with Accelerate + W&B
│   ├── generation/
│   │   ├── generate.py             Batch synthetic generation per grade
│   │   └── guidance_sweep.py       Optimal guidance scale search
│   ├── evaluation/
│   │   ├── fid_lpips.py            Generation quality metrics
│   │   ├── downstream_classifier.py EfficientNet-B0 DR grader
│   │   └── ablation.py             Synthetic ratio sweep experiments
│   └── app/
│       └── gradio_app.py           Three-tab Gradio demo
├── notebooks/                       Colab-ready training notebooks
├── docs/                            Architecture and development notes
├── requirements.txt
├── LICENSE
└── README.md
```

## Data sources

| Source | What it provides |
|---|---|
| [APTOS 2019 (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection) | 3,662 fundus images across 5 DR grades (primary dataset) |
| [EyePACS (Kaggle)](https://www.kaggle.com/c/diabetic-retinopathy-detection) | 88,702 fundus images, same 5-grade scale (larger benchmark reference) |

## Setup

```bash
git clone https://github.com/Nikhil20012/MedFuse.git
cd MedFuse
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download APTOS 2019 from Kaggle (requires API credentials):

```bash
mkdir -p data/raw
kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
cd data/raw && unzip aptos2019-blindness-detection.zip
rm -rf test_images/ test.csv sample_submission.csv aptos2019-blindness-detection.zip
cd ../..
```

Preprocess:

```bash
python -m src.data.preprocessing \
    --raw_dir data/raw/train_images \
    --output_dir data/processed \
    --labels_csv data/raw/train.csv
```

## Usage

```bash
# Fine-tune LoRA adapters (requires GPU - use Colab/Kaggle notebook)
python -m src.training.lora_finetune --config configs/train_config.yaml

# Find best guidance scale per grade
python -m src.generation.guidance_sweep --checkpoint checkpoints/final

# Generate synthetic fundus images
python -m src.generation.generate --checkpoint checkpoints/final

# Evaluate generation quality
python -m src.evaluation.fid_lpips --real_dir data/processed --synthetic_dir data/synthetic

# Run downstream ablation study
python -m src.evaluation.ablation --config configs/train_config.yaml

# Launch demo
python -m src.app.gradio_app --checkpoint checkpoints/final --share
```

## Key design decisions

**Why LoRA over full fine-tuning?** APTOS per-grade subsets range from 193 to
1,805 images. Full fine-tuning of an 860M parameter UNet on this volume would
catastrophically forget pretrained knowledge. LoRA adds 0.1% trainable
parameters while preserving the model's generative priors.

**Why SD 2.1 over SDXL?** SDXL requires 6-8 GB VRAM for inference alone,
leaving little room for training on free-tier GPUs (T4 16GB). SD 2.1's
OpenCLIP text encoder also handles medical terminology better than SD 1.5's
CLIP ViT-L/14.

**Why DDIM over DDPM for inference?** DDIM produces comparable quality in 50
steps versus DDPM's 1000 steps, making batch generation feasible on consumer
hardware.

**Why EfficientNet-B0 for downstream?** The downstream classifier is
deliberately simple. A complex model might improve through architecture alone,
masking the effect of synthetic data augmentation. EfficientNet-B0 isolates
the variable we care about: does training data quality improve with synthetic
augmentation?

**Why FID + downstream AUROC instead of just FID?** FID measures
distributional similarity but does not tell you whether synthetic data is
useful for the actual task. A model could generate realistic-looking images
with incorrect pathological features. Downstream AUROC on held-out test data
directly measures clinical utility.

## Build progress

- [x] Project structure and configuration
- [x] Data preprocessing pipeline (circle crop, quality filter)
- [x] PyTorch datasets with grade-conditioned prompts
- [x] LoRA fine-tuning script with Accelerate and W&B
- [x] Synthetic generation with DDIM scheduler
- [x] Guidance scale sweep utility
- [x] FID and LPIPS evaluation
- [x] Downstream EfficientNet-B0 classifier
- [x] Augmentation ratio ablation study
- [x] Gradio demo (generate, compare, results tabs)
- [ ] LoRA training on Colab/Kaggle (needs GPU)
- [ ] Bulk synthetic generation
- [ ] Full ablation run with results
- [ ] Hugging Face Spaces deployment

## Author

**Nikhil Bharadwaj Yellapragada**
<br>
MS Data Analytics Engineering, Northeastern University

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nikhil-bharadwaj-yellapragada-48321a211/)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:yellapragada.n@northeastern.edu)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Nikhil20012)

## License

MIT. See [LICENSE](LICENSE) for details.