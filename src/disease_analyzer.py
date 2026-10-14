"""Computer Vision Leaf Health & Disease Analysis Module for AgriSmart AI.

This module provides an empirical, locally-executable computer vision pipeline
to detect leaf lesions, chlorosis (yellowing), necrosis (browning/leaf spots),
and fungal surface symptoms using chromatic segmentation and Excess Green Index (ExG).

Attributed to: ICAR & FAO Plant Pathology & Agricultural Remote Sensing Guidelines.
"""

from typing import Dict, Any, Tuple
import io
import numpy as np
from PIL import Image

# Diagnostic disease profiles with symptoms and agricultural management advice
DISEASE_PROFILES: Dict[str, Dict[str, Any]] = {
    "healthy": {
        "condition": "Healthy Foliage",
        "severity": "None (0 - 5% affected)",
        "pathogen": "No active pathogen detected",
        "symptoms": "Uniform chlorophyll coloration with minimal to no chlorotic or necrotic spots.",
        "management": "Continue standard preventive cultural practices, balanced irrigation, and routine scouting.",
        "badge_color": "#2d6a4f",
    },
    "leaf_spot_blight": {
        "condition": "Fungal Leaf Spot / Early Blight (Alternaria / Cercospora spp.)",
        "severity": "Moderate to High",
        "pathogen": "Fungal pathogen favored by warm, humid foliage",
        "symptoms": "Dark brown to black necrotic spots, often surrounded by a chlorotic yellow halo.",
        "management": (
            "1. Remove and destroy infected lower leaves.\n"
            "2. Avoid overhead sprinkler irrigation to keep foliage dry.\n"
            "3. Apply organic copper fungicide or bio-agent (Trichoderma viride) if infection spreads.\n"
            "4. Ensure proper plant spacing for improved air circulation."
        ),
        "badge_color": "#b08968",
    },
    "severe_necrosis_late_blight": {
        "condition": "Severe Blight / Necrosis (Phytophthora infestans / Rhizoctonia)",
        "severity": "Severe",
        "pathogen": "Aggressive fungal or oomycete pathogen",
        "symptoms": "Large water-soaked irregular lesions rapidly turning dark brown/black with extensive leaf tissue decay.",
        "management": (
            "1. Immediately rogue out severely infected plants to prevent field-wide contagion.\n"
            "2. Improve soil drainage and avoid water stagnation.\n"
            "3. Apply preventive fungicide treatments as recommended by local agricultural extension.\n"
            "4. Practice strict 3-year crop rotation with non-host species."
        ),
        "badge_color": "#780000",
    },
    "chlorosis_deficiency": {
        "condition": "Chlorosis / Micronutrient Deficiency or Viral Stunting",
        "severity": "Mild to Moderate",
        "pathogen": "Nutritional imbalance (Nitrogen/Iron/Magnesium) or Mosaic Virus",
        "symptoms": "Loss of normal green pigmentation; pervasive pale yellowing between leaf veins with minimal necrosis.",
        "management": (
            "1. Conduct a laboratory soil test to confirm micro/macro nutrient availability.\n"
            "2. Apply foliar zinc/iron or balanced NPK if deficiency is diagnosed.\n"
            "3. Inspect for piercing-sucking vector insects (aphids, whiteflies) that transmit viral mosaics.\n"
            "4. Ensure soil pH is within optimal range (6.0 - 7.5) for nutrient uptake."
        ),
        "badge_color": "#d4a373",
    },
    "powdery_mildew": {
        "condition": "Powdery Mildew (Podosphaera / Erysiphe spp.)",
        "severity": "Moderate",
        "pathogen": "Obligate biotrophic fungus",
        "symptoms": "White to grayish powdery fungal patches on the upper leaf surface.",
        "management": (
            "1. Prune congested foliage to enhance sunlight penetration.\n"
            "2. Spray diluted potassium bicarbonate or wettable sulfur during early infection.\n"
            "3. Apply neem seed kernel extract (NSKE 5%) as an organic prophylactic deterrent."
        ),
        "badge_color": "#6c757d",
    },
}


def segment_and_analyze_leaf(image: Image.Image) -> Dict[str, Any]:
    """Perform chromatic segmentation and lesion analysis on a leaf image.

    Args:
        image: PIL Image object.

    Returns:
        Dict[str, Any]: Detailed pathological metrics, diagnosis, and segmented mask image.
    """
    # 1. Resize for consistent processing performance
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((300, 300))
    arr = np.array(image_resized, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    total_pixels = 300 * 300

    # 2. Agricultural Excess Green Index (ExG): ExG = 2*G - R - B
    # Used in precision agriculture computer vision to segment plant foliage
    exg = 2.0 * g - r - b

    # Intensity / Brightness
    intensity = (r + g + b) / 3.0

    # 3. Leaf Segmentation (Exclude extreme white background > 245 and deep black background < 15)
    # A leaf contains green foliage, yellow chlorotic zones, and brown/black necrotic lesions
    # Non-background pixels have chromatic variation and valid intensity
    is_not_white_bg = (intensity < 240.0) | ((r < 235.0) | (g < 235.0) | (b < 235.0))
    is_not_black_bg = intensity > 20.0
    leaf_mask = is_not_white_bg & is_not_black_bg

    leaf_pixel_count = int(np.sum(leaf_mask))

    if leaf_pixel_count < 2000:
        raise ValueError(
            "Could not detect clear agricultural leaf foliage in the uploaded photo. "
            "Please upload a close-up photo of a plant leaf with decent lighting."
        )

    # 4. Symptom Detection within the leaf mask:
    # A. Healthy Green Tissue: Dominant green channel and positive excess green
    healthy_mask = leaf_mask & (g > r) & (g > b) & (exg > 5.0)

    # B. Chlorosis (Yellowing tissue): High Red + High Green, low Blue
    chlorosis_mask = (
        leaf_mask
        & (~healthy_mask)
        & (r > 110.0)
        & (g > 100.0)
        & (b < 120.0)
        & (r > b)
        & (g > b)
    )

    # C. Necrosis (Browning / Dark lesion spots): Brown, dark red-brown, or dark grey/black lesions
    # Typical RGB for necrotic spots: Red > Green > Blue with low-to-medium intensity
    necrosis_mask = (
        leaf_mask
        & (~healthy_mask)
        & (~chlorosis_mask)
        & (
            ((r >= g) & (g >= b) & (intensity < 160.0))
            | ((intensity < 80.0) & (exg <= 0.0))
        )
    )

    # D. Powdery / Whitish Surface Mildew: High brightness, low color saturation
    diff_rg = np.abs(r - g)
    diff_gb = np.abs(g - b)
    powdery_mask = (
        leaf_mask
        & (~healthy_mask)
        & (~chlorosis_mask)
        & (~necrosis_mask)
        & (intensity > 170.0)
        & (diff_rg < 20.0)
        & (diff_gb < 20.0)
    )

    # Calculate proportions relative to total leaf surface
    healthy_pct = float(np.sum(healthy_mask) / leaf_pixel_count * 100.0)
    chlorosis_pct = float(np.sum(chlorosis_mask) / leaf_pixel_count * 100.0)
    necrosis_pct = float(np.sum(necrosis_mask) / leaf_pixel_count * 100.0)
    powdery_pct = float(np.sum(powdery_mask) / leaf_pixel_count * 100.0)

    # Residual damaged/indeterminate tissue
    damaged_pct = max(0.0, 100.0 - healthy_pct)

    # 5. Diagnostic Classification Logic
    if necrosis_pct > 18.0:
        diagnosis_key = "severe_necrosis_late_blight"
        confidence = min(96.0, 75.0 + (necrosis_pct * 0.9))
    elif necrosis_pct > 4.5:
        diagnosis_key = "leaf_spot_blight"
        confidence = min(94.0, 72.0 + (necrosis_pct * 1.5))
    elif powdery_pct > 12.0:
        diagnosis_key = "powdery_mildew"
        confidence = min(92.0, 70.0 + (powdery_pct * 1.2))
    elif chlorosis_pct > 14.0:
        diagnosis_key = "chlorosis_deficiency"
        confidence = min(91.0, 68.0 + (chlorosis_pct * 1.1))
    elif healthy_pct >= 82.0:
        diagnosis_key = "healthy"
        confidence = min(98.0, 80.0 + (healthy_pct * 0.18))
    else:
        # Default mild leaf spot / stress
        diagnosis_key = "leaf_spot_blight"
        confidence = 74.5

    profile = DISEASE_PROFILES[diagnosis_key]

    # 6. Generate Color-Coded Lesion Diagnostic Heatmap
    # Green = Healthy, Yellow = Chlorosis, Crimson/Red = Necrotic Spots, Gray = Powdery
    segmented_rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    segmented_rgb[leaf_mask] = [40, 50, 40]  # Neutral dark foliage background
    segmented_rgb[healthy_mask] = [45, 170, 75]  # Green: Healthy
    segmented_rgb[chlorosis_mask] = [240, 200, 30]  # Yellow: Chlorotic
    segmented_rgb[necrosis_mask] = [210, 35, 35]  # Red: Necrotic Lesions
    segmented_rgb[powdery_mask] = [220, 220, 240]  # Light Blue/Gray: Powdery mildew

    segmented_image = Image.fromarray(segmented_rgb)

    return {
        "diagnosis_key": diagnosis_key,
        "condition": profile["condition"],
        "severity": profile["severity"],
        "pathogen": profile["pathogen"],
        "confidence_pct": round(confidence, 1),
        "symptoms": profile["symptoms"],
        "management": profile["management"],
        "badge_color": profile["badge_color"],
        "metrics": {
            "healthy_tissue_pct": round(healthy_pct, 1),
            "damaged_tissue_pct": round(damaged_pct, 1),
            "necrotic_lesion_pct": round(necrosis_pct, 1),
            "chlorotic_yellowing_pct": round(chlorosis_pct, 1),
            "powdery_mold_pct": round(powdery_pct, 1),
        },
        "segmented_image": segmented_image,
        "disclaimer": (
            "Notice: AgriSmart AI Leaf Health Scanner is a diagnostic decision-support aid. "
            "Field conditions such as severe shadowing, leaf wetness, or non-leaf objects "
            "can influence chromatic readings. Consult certified local agricultural extension "
            "agents before spraying high-potency chemical pesticides."
        ),
    }
