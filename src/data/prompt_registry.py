"""
Maps diabetic retinopathy severity grades to text prompts
for conditioning the diffusion model during training and inference.

Grade definitions follow the International Clinical Diabetic
Retinopathy (ICDR) severity scale used in the EyePACS dataset.
"""

DR_GRADE_PROMPTS = {
    0: (
        "a retinal fundus photograph of a healthy eye, "
        "no signs of diabetic retinopathy, "
        "clear retinal vasculature, normal macula"
    ),
    1: (
        "a retinal fundus photograph showing mild non-proliferative "
        "diabetic retinopathy with microaneurysms"
    ),
    2: (
        "a retinal fundus photograph showing moderate non-proliferative "
        "diabetic retinopathy with microaneurysms, dot and blot hemorrhages, "
        "and hard exudates"
    ),
    3: (
        "a retinal fundus photograph showing severe non-proliferative "
        "diabetic retinopathy with extensive hemorrhages, venous beading, "
        "cotton wool spots, and intraretinal microvascular abnormalities"
    ),
    4: (
        "a retinal fundus photograph showing proliferative diabetic "
        "retinopathy with neovascularization, vitreous hemorrhage, "
        "and fibrovascular proliferation"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, artifacts, text, watermark, "
    "out of focus, distorted, cartoon, illustration"
)

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def get_prompt(grade: int) -> str:
    """Return the conditioning prompt for a given DR grade."""
    if grade not in DR_GRADE_PROMPTS:
        raise ValueError(f"Invalid DR grade: {grade}. Must be 0-4.")
    return DR_GRADE_PROMPTS[grade]


def get_label(grade: int) -> str:
    """Return the human-readable label for a given DR grade."""
    return GRADE_LABELS[grade]
