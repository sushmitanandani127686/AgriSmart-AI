"""Prediction engine and crop profile repository for AgriSmart AI.

This module provides validated inference functions using the trained Random Forest
model, returns top prediction probabilities, and supplies curated, attributed
agronomic profiles for recommended crops.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

# Feature names in exact order expected by the model
FEATURE_NAMES: List[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

# Validation limits: (min_value, max_value, human_label, unit)
VALIDATION_CONSTRAINTS: Dict[str, Tuple[float, float, str, str]] = {
    "N": (0.0, 300.0, "Nitrogen (N)", "kg/ha"),
    "P": (0.0, 300.0, "Phosphorus (P)", "kg/ha"),
    "K": (0.0, 300.0, "Potassium (K)", "kg/ha"),
    "temperature": (-10.0, 60.0, "Temperature", "°C"),
    "humidity": (0.0, 100.0, "Relative Humidity", "%"),
    "ph": (0.0, 14.0, "Soil pH", "pH scale (0-14)"),
    "rainfall": (0.0, 4000.0, "Rainfall", "mm"),
}

# Curated agronomic database (Attributed to FAO & ICAR crop advisory guidelines)
CROP_PROFILES: Dict[str, Dict[str, Any]] = {
    "rice": {
        "display_name": "Rice (Oryza sativa)",
        "category": "Cereal / Grain",
        "soil_type": "Clayey loam, alluvial, water-retentive soils",
        "water_requirement": "High (1000 - 2500 mm). Needs standing water during early growth.",
        "season": "Kharif (Monsoon / Summer)",
        "growth_duration": "100 - 150 days",
        "key_nutrients": "High Nitrogen requirement; balanced Phosphorus & Potassium.",
        "management_tip": "Maintain puddling in fields to control weed growth and conserve moisture.",
    },
    "maize": {
        "display_name": "Maize / Corn (Zea mays)",
        "category": "Cereal / Grain",
        "soil_type": "Well-drained deep loamy and sandy-loam soils",
        "water_requirement": "Moderate (500 - 800 mm). Sensitive to waterlogging.",
        "season": "Kharif & Rabi (Can be grown in multiple seasons)",
        "growth_duration": "90 - 120 days",
        "key_nutrients": "Heavy Nitrogen feeder with moderate Potassium requirement.",
        "management_tip": "Ensure good drainage; avoid stagnant water around roots during flowering.",
    },
    "chickpea": {
        "display_name": "Chickpea / Bengal Gram (Cicer arietinum)",
        "category": "Pulse / Legume",
        "soil_type": "Deep well-aerated sandy loam to clay loam",
        "water_requirement": "Low to Moderate (350 - 500 mm). Drought tolerant.",
        "season": "Rabi (Winter season)",
        "growth_duration": "90 - 120 days",
        "key_nutrients": "Fixes atmospheric nitrogen; requires adequate Phosphorus for root nodulation.",
        "management_tip": "Avoid heavy irrigation; excessive moisture promotes root rot (wilt).",
    },
    "kidneybeans": {
        "display_name": "Kidney Beans / Rajma (Phaseolus vulgaris)",
        "category": "Pulse / Legume",
        "soil_type": "Deep, light, porous soils with rich organic matter",
        "water_requirement": "Moderate (500 - 650 mm)",
        "season": "Kharif / Spring in hills, Rabi in plains",
        "growth_duration": "90 - 130 days",
        "key_nutrients": "Poor nitrogen fixation compared to other legumes; responds well to starter Nitrogen and Phosphorus.",
        "management_tip": "Sensitive to saline or alkaline soils; maintains best vigor at pH 6.0 - 7.0.",
    },
    "pigeonpeas": {
        "display_name": "Pigeonpeas / Arhar (Cajanus cajan)",
        "category": "Pulse / Legume",
        "soil_type": "Well-drained deep loamy soil",
        "water_requirement": "Moderate (600 - 1000 mm). Highly drought tolerant once established.",
        "season": "Kharif (Early monsoon)",
        "growth_duration": "150 - 200 days",
        "key_nutrients": "Responds well to Phosphorus (DAP) application at sowing.",
        "management_tip": "Deep taproot system opens up hard soil horizons; ideal for intercropping.",
    },
    "mothbeans": {
        "display_name": "Moth Beans (Vigna aconitifolia)",
        "category": "Pulse / Legume",
        "soil_type": "Sandy, light, arid soils",
        "water_requirement": "Low (200 - 450 mm). Extremely drought-hardy.",
        "season": "Kharif",
        "growth_duration": "75 - 90 days",
        "key_nutrients": "Low nutrient requirement; fixes atmospheric nitrogen efficiently.",
        "management_tip": "Excellent soil cover crop to prevent wind erosion in arid regions.",
    },
    "mungbean": {
        "display_name": "Mungbean / Green Gram (Vigna radiata)",
        "category": "Pulse / Legume",
        "soil_type": "Loam to sandy loam with good internal drainage",
        "water_requirement": "Low to Moderate (350 - 550 mm)",
        "season": "Kharif, Spring, or Summer (short duration catch crop)",
        "growth_duration": "60 - 75 days",
        "key_nutrients": "Requires Rhizobium bio-fertilizer inoculation and Phosphorus.",
        "management_tip": "Short duration fits well in crop rotations between cereals.",
    },
    "blackgram": {
        "display_name": "Blackgram / Urad (Vigna mungo)",
        "category": "Pulse / Legume",
        "soil_type": "Heavy, water-retentive clay loam to loamy soils",
        "water_requirement": "Moderate (400 - 600 mm)",
        "season": "Kharif & Summer",
        "growth_duration": "70 - 90 days",
        "key_nutrients": "High Phosphorus requirement for root nodule development.",
        "management_tip": "Protect from heavy rain during pod maturation to prevent seed discoloration.",
    },
    "lentil": {
        "display_name": "Lentil / Masoor (Lens culinaris)",
        "category": "Pulse / Legume",
        "soil_type": "Alluvial, sandy-loam, and light clay soils",
        "water_requirement": "Low (300 - 450 mm)",
        "season": "Rabi (Cool winter season)",
        "growth_duration": "110 - 130 days",
        "key_nutrients": "Benefits from sulfur and phosphorus fertilization.",
        "management_tip": "Tolerates cold temperatures well; sensitive to water stagnation.",
    },
    "pomegranate": {
        "display_name": "Pomegranate (Punica granatum)",
        "category": "Horticulture / Fruit",
        "soil_type": "Deep loamy to light alluvial soils, tolerates moderate alkalinity",
        "water_requirement": "Moderate (500 - 700 mm). Semi-arid climate favorite.",
        "season": "Perennial shrub / tree (Bahar treatment regulates fruiting season)",
        "growth_duration": "Perennial (first harvest at 2 - 3 years)",
        "key_nutrients": "Balanced NPK with emphasis on Potassium and Micronutrients (Zinc, Boron).",
        "management_tip": "Regulated deficit irrigation improves fruit cracking resistance and sweetness.",
    },
    "banana": {
        "display_name": "Banana (Musa acuminata)",
        "category": "Horticulture / Fruit",
        "soil_type": "Deep, rich, fertile clay loam or alluvial soil with high humus",
        "water_requirement": "High (1500 - 2000 mm). Requires frequent irrigation.",
        "season": "Year-round planting in humid tropical regions",
        "growth_duration": "11 - 14 months",
        "key_nutrients": "Very high Potassium and Nitrogen demand; moderate Phosphorus.",
        "management_tip": "Provide windbreak protection and mulch generously to retain soil moisture.",
    },
    "mango": {
        "display_name": "Mango (Mangifera indica)",
        "category": "Horticulture / Fruit",
        "soil_type": "Deep alluvial, red or lateritic well-drained soils",
        "water_requirement": "Moderate (750 - 1500 mm). Needs dry spell for flowering.",
        "season": "Perennial tree; flowering in winter, harvest in summer",
        "growth_duration": "Perennial (grafted trees fruit in 3 - 4 years)",
        "key_nutrients": "Requires yearly organic manure and NPK top-dressing post-harvest.",
        "management_tip": "Withhold irrigation 2-3 months prior to flowering to encourage blossom buds.",
    },
    "grapes": {
        "display_name": "Grapes (Vitis vinifera)",
        "category": "Horticulture / Fruit",
        "soil_type": "Well-drained sandy loam or gravelly soils",
        "water_requirement": "Low to Moderate (500 - 750 mm)",
        "season": "Perennial vine; pruned in spring and autumn",
        "growth_duration": "Perennial (fruits annually after 2nd year)",
        "key_nutrients": "High Potassium demand during fruit filling; precise nitrogen control.",
        "management_tip": "Systematic trellising and canopy management ensure sun exposure and prevent mildew.",
    },
    "watermelon": {
        "display_name": "Watermelon (Citrullus lanatus)",
        "category": "Cucurbit / Fruit",
        "soil_type": "Sandy, sandy-loam, river-bed soils with good drainage",
        "water_requirement": "Moderate (400 - 600 mm). Sensitive to excess moisture during ripening.",
        "season": "Zaid / Summer season",
        "growth_duration": "80 - 100 days",
        "key_nutrients": "High Potassium requirement for fruit sugar accumulation; moderate Nitrogen.",
        "management_tip": "Reduce irrigation significantly 1-2 weeks before harvest to enhance sweetness.",
    },
    "muskmelon": {
        "display_name": "Muskmelon / Cantaloupe (Cucumis melo)",
        "category": "Cucurbit / Fruit",
        "soil_type": "Warm, well-drained sandy loam or silt loam soils",
        "water_requirement": "Moderate (400 - 600 mm)",
        "season": "Zaid / Summer season",
        "growth_duration": "75 - 90 days",
        "key_nutrients": "Potassium and Magnesium rich fertilization enhances fruit netting and aroma.",
        "management_tip": "Maintain dry mulch underneath maturing fruits to prevent ground rot.",
    },
    "apple": {
        "display_name": "Apple (Malus domestica)",
        "category": "Temperate Fruit",
        "soil_type": "Deep, loamy, well-drained soils rich in organic matter",
        "water_requirement": "Moderate to High (1000 - 1250 mm distributed across season)",
        "season": "Temperate perennial; requires winter chilling hours (< 7°C)",
        "growth_duration": "Perennial (harvested late summer / autumn)",
        "key_nutrients": "Balanced NPK plus Calcium and Boron to prevent fruit disorders.",
        "management_tip": "Prune during dormant winter period to maintain open sunlight penetration.",
    },
    "orange": {
        "display_name": "Orange / Citrus (Citrus sinensis)",
        "category": "Horticulture / Citrus",
        "soil_type": "Deep, light, well-drained loamy soils free of hardpans",
        "water_requirement": "Moderate (800 - 1200 mm)",
        "season": "Sub-tropical perennial; winter/spring harvest",
        "growth_duration": "Perennial",
        "key_nutrients": "High Nitrogen, Potassium, Zinc, and Iron micronutrient demand.",
        "management_tip": "Avoid flooding trunk base; use drip or ring irrigation to avert collar rot.",
    },
    "papaya": {
        "display_name": "Papaya (Carica papaya)",
        "category": "Horticulture / Fruit",
        "soil_type": "Rich, well-drained sandy loam or alluvial soil",
        "water_requirement": "Moderate (1000 - 1500 mm). Cannot tolerate waterlogging even for 24 hours.",
        "season": "Tropical herbaceous perennial",
        "growth_duration": "9 - 12 months from seed to first harvest",
        "key_nutrients": "Heavy feeder of Nitrogen, Phosphorus, and Potassium.",
        "management_tip": "Plant on raised mounds with drainage ditches to prevent root rot.",
    },
    "coconut": {
        "display_name": "Coconut (Cocos nucifera)",
        "category": "Plantation / Oilseed",
        "soil_type": "Coastal sandy, alluvial, or lateritic soils",
        "water_requirement": "High (1300 - 2300 mm). Thrives under high atmospheric humidity.",
        "season": "Tropical perennial palm; year-round continuous yield",
        "growth_duration": "Perennial (first harvest at 5 - 7 years, productive for 60+ years)",
        "key_nutrients": "Extremely high Potassium and Chloride (Sodium Chloride / Salt) response.",
        "management_tip": "Mulch palm basins with coconut coir pith or green leaves to retain root moisture.",
    },
    "cotton": {
        "display_name": "Cotton (Gossypium hirsutum)",
        "category": "Cash / Fiber Crop",
        "soil_type": "Deep black cotton soils (regur) with high moisture capacity",
        "water_requirement": "Moderate (600 - 1100 mm). Needs warm dry weather during boll opening.",
        "season": "Kharif",
        "growth_duration": "150 - 180 days",
        "key_nutrients": "Moderate Nitrogen, high Phosphorus and Potassium for boll strength.",
        "management_tip": "Avoid excessive nitrogen late in season which causes vegetative rank growth instead of bolls.",
    },
    "jute": {
        "display_name": "Jute / Golden Fiber (Corchorus olitorius)",
        "category": "Cash / Fiber Crop",
        "soil_type": "Alluvial, rich river-bed silty loam",
        "water_requirement": "High (1200 - 1800 mm). Requires warm, humid climate.",
        "season": "Kharif (Sown in pre-monsoon March-May)",
        "growth_duration": "120 - 150 days",
        "key_nutrients": "Responds strongly to Nitrogen for tall vegetative stalk height.",
        "management_tip": "Requires clean slow-moving water bodies nearby for retting (fiber extraction).",
    },
    "coffee": {
        "display_name": "Coffee (Coffea arabica / canephora)",
        "category": "Plantation / Beverage",
        "soil_type": "Deep, porous, rich volcanic or forest loam soils with abundant humus",
        "water_requirement": "High (1500 - 2500 mm) with a distinct dry period for blossom opening.",
        "season": "Perennial hill crop; harvested in winter (Nov - Feb)",
        "growth_duration": "Perennial (fruits from 3rd year onward)",
        "key_nutrients": "High demand for Potassium, Nitrogen, and Magnesium.",
        "management_tip": "Cultivated under two-tier shade trees; shade regulates microclimate and bean quality.",
    },
}


class PredictionEngine:
    """Encapsulates model loading, input validation, and inference."""

    def __init__(
        self,
        model_path: str = "models/crop_rf_model.joblib",
        metrics_path: str = "models/model_metrics.json",
    ):
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.model = None
        self.metrics_data: Dict[str, Any] = {}
        self.classes: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load serialized model and metrics file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please run `python train_model.py` first."
            )
        self.model = joblib.load(self.model_path)

        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                self.metrics_data = json.load(f)
            self.classes = self.metrics_data.get("target_classes", [])
        else:
            self.classes = getattr(self.model, "classes_", []).tolist()

    @staticmethod
    def validate_inputs(inputs: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate input values against physical and agricultural constraints.

        Args:
            inputs: Dictionary of feature names and numeric values.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of human-readable error messages).
        """
        errors = []
        for feature in FEATURE_NAMES:
            if feature not in inputs:
                errors.append(f"Missing required parameter: '{feature}'.")
                continue

            val = inputs[feature]
            if not isinstance(val, (int, float)) or np.isnan(val):
                errors.append(f"Value for '{feature}' must be a valid number.")
                continue

            min_val, max_val, label, unit = VALIDATION_CONSTRAINTS[feature]
            if val < min_val or val > max_val:
                errors.append(
                    f"{label} ({val} {unit}) is outside realistic bounds "
                    f"[{min_val} to {max_val} {unit}]."
                )

        return len(errors) == 0, errors

    def predict(
        self, inputs: Dict[str, float], top_k: int = 3
    ) -> Dict[str, Any]:
        """Perform validated crop prediction.

        Args:
            inputs: Dictionary of input features.
            top_k: Number of top candidate crops with probabilities to return.

        Returns:
            Dict[str, Any]: Prediction results, confidence, top alternatives, and profile.

        Raises:
            ValueError: If inputs fail validation constraints.
        """
        is_valid, errors = self.validate_inputs(inputs)
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        # Construct input vector in exact expected order
        input_data = pd.DataFrame([[inputs[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)

        # Predict primary class
        pred_label = str(self.model.predict(input_data)[0]).lower()

        # Predict class probabilities if model supports it
        top_candidates = []
        confidence_pct = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_data)[0]
            model_classes = self.model.classes_
            sorted_indices = np.argsort(probs)[::-1]

            top_candidates = [
                {
                    "crop": str(model_classes[idx]),
                    "confidence_pct": round(float(probs[idx]) * 100, 2),
                }
                for idx in sorted_indices[:top_k]
            ]
            confidence_pct = top_candidates[0]["confidence_pct"] if top_candidates else 0.0

        # Retrieve curated agronomic profile
        profile = CROP_PROFILES.get(
            pred_label,
            {
                "display_name": pred_label.capitalize(),
                "category": "Agricultural Crop",
                "soil_type": "Standard arable agricultural soil",
                "water_requirement": "Consult local extension services",
                "season": "Regional specific",
                "growth_duration": "Seasonal",
                "key_nutrients": "Balanced NPK",
                "management_tip": "Follow standard regional package of practices.",
            },
        )

        return {
            "predicted_crop": pred_label,
            "confidence_pct": confidence_pct,
            "top_candidates": top_candidates,
            "profile": profile,
            "inputs_evaluated": inputs,
            "attribution": "Curated agronomic reference: FAO Crop Water Management & ICAR Package of Practices",
            "scope_disclaimer": (
                "AgriSmart AI recommendations are based on multivariate pattern analysis "
                "of historical soil nutrient levels and climatic conditions. This system evaluates "
                "physiological suitability and does not establish guaranteed yields, real-time "
                "market profitability, or active pathogen resistance."
            ),
        }
