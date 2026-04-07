from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Any, Optional
import numpy as np

from utils.model_loader import model_store, VALID_MODEL_NAMES
from utils.logger import prediction_logger, error_logger

router = APIRouter(prefix="/predict", tags=["Predictions"])

# ── Module A: Concrete ────────────────────────────────────────────────────────
# Exact 8 features the RF was trained on
class ModuleAFeatures(BaseModel):
    Cement: float = Field(ge=0)
    Slag: float = Field(ge=0)
    FlyAsh: float = Field(ge=0)
    Water: float = Field(ge=0)
    Superplasticizer: float = Field(ge=0)
    CoarseAgg: float = Field(ge=0)
    FineAgg: float = Field(ge=0)
    Age: float = Field(ge=0)
    model_config = ConfigDict(populate_by_name=True)

# ── Module B: Steel ───────────────────────────────────────────────────────────
# Exact 13 features the XGBoost was trained on (original backend.txt schema)
class ModuleBFeatures(BaseModel):
    C: float = Field(alias="%C", ge=0.0, default=0.0)
    Mn: float = Field(alias="%Mn", ge=0.0, default=0.0)
    Si: float = Field(alias="%Si", ge=0.0, default=0.0)
    Cr: float = Field(alias="%Cr", ge=0.0, default=0.0)
    Ni: float = Field(alias="%Ni", ge=0.0, default=0.0)
    Mo: float = Field(alias="%Mo", ge=0.0, default=0.0)
    Cu: float = Field(alias="%Cu", ge=0.0, default=0.0)
    V: float = Field(alias="%V", ge=0.0, default=0.0)
    Al: float = Field(alias="%Al", ge=0.0, default=0.0)
    Ti: float = Field(alias="%Ti", ge=0.0, default=0.0)
    N: float = Field(alias="%N", ge=0.0, default=0.0)
    B: float = Field(alias="%B", ge=0.0, default=0.0)
    Reduction_Ratio: float = Field(ge=1.0, default=1.0)
    model_config = ConfigDict(populate_by_name=True)

# ── Module C: Mechanical Materials ───────────────────────────────────────────
# Dataset columns: Material, Su, Sy, E, G, mu, Ro, Use
# Inputs = all numeric EXCEPT Su, Sy (targets for reg) and Use (target for clf)
# → E, G, mu, Ro  (4 features — matches model's n_features_in_=4)
class ModuleCFeatures(BaseModel):
    E: float = Field(ge=0, description="Elastic Modulus (GPa)")
    G: float = Field(ge=0, description="Shear Modulus (GPa)")
    mu: float = Field(ge=0, description="Poisson's Ratio")
    Ro: float = Field(ge=0, description="Density (Mg/m³)")
    model_config = ConfigDict(populate_by_name=True)

class PredictRequest(BaseModel):
    features: dict

class PredictResponse(BaseModel):
    prediction: Any
    model_used: str
    confidence: Optional[float] = None
    metrics: Optional[dict] = None
    task: str
    target: str
    interpretation: Optional[str] = None
    decision: Optional[str] = None


@router.post("/{model_name}", response_model=PredictResponse)
def predict(model_name: str, request: PredictRequest):
    print("Received Input:", request.features)

    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Valid: {VALID_MODEL_NAMES}"
        )

    try:
        model = model_store.get_model(model_name)
    except RuntimeError as e:
        error_logger.error(f"Model retrieval failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    try:
        if model_name == "module_a_rf":
            parsed = ModuleAFeatures(**request.features)
            input_array = np.array([[
                parsed.Cement, parsed.Slag, parsed.FlyAsh, parsed.Water,
                parsed.Superplasticizer, parsed.CoarseAgg, parsed.FineAgg, parsed.Age
            ]])

        elif model_name == "module_b_xgb":
            parsed = ModuleBFeatures(**request.features)
            # Strict order the model was trained on (13 features)
            input_array = np.array([[
                parsed.C, parsed.Mn, parsed.Si, parsed.Cr, parsed.Ni,
                parsed.Mo, parsed.Cu, parsed.V, parsed.Al, parsed.Ti,
                parsed.N, parsed.B, parsed.Reduction_Ratio
            ]])

        elif model_name in ["module_c_reg", "module_c_clf"]:
            parsed = ModuleCFeatures(**request.features)
            # Order: E, G, mu, Ro  (4 features — matches n_features_in_=4)
            input_array = np.array([[
                parsed.E, parsed.G, parsed.mu, parsed.Ro
            ]])

    except ValidationError as e:
        error_logger.error(f"Input validation failed for {model_name}: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_logger.error(f"Input processing error for {model_name}: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    try:
        raw_prediction = model.predict(input_array)

        if model_name == "module_a_rf":
            prediction_value = float(round(raw_prediction[0], 4))
            confidence = None
            task = "regression"
            target = "Compressive Strength (MPa)"
            
            # Interpretation logic for Concrete
            if prediction_value < 20:
                interpretation = "Low Strength — Good for Residential Footings / Patios"
                decision = "Review Required"
            elif 20 <= prediction_value < 40:
                interpretation = "Standard Strength — Good for Residential / Commercial Buildings"
                decision = "Qualified"
            elif 40 <= prediction_value < 60:
                interpretation = "High Strength — Good for Bridges / Industrial Floors"
                decision = "Qualified"
            else:
                interpretation = "Ultra-High Strength — Good for High-Rise Structural Members / Specialized Infrastructure"
                decision = "Qualified"

        elif model_name == "module_b_xgb":
            prediction_value = float(round(raw_prediction[0], 4))
            confidence = None
            task = "regression"
            target = "UTS (MPa)"
            
            # Interpretation for Steel
            if prediction_value < 500:
                interpretation = "Mild Steel — General Structural Use"
            elif 500 <= prediction_value < 1000:
                interpretation = "High-Strength Carbon Steel — Structural / Automotive"
            else:
                interpretation = "Alloy Steel — High-Strength Industrial / Aerospace Grade"
            decision = "Qualified" if prediction_value > 300 else "Review Required"

        elif model_name == "module_c_reg":
            pred_array = raw_prediction[0]
            prediction_value = {
                "Su": float(round(pred_array[0], 4)),
                "Sy": float(round(pred_array[1], 4))
            }
            confidence = None
            task = "multi-output regression"
            target = "Su (MPa), Sy (MPa)"
            
            # Use regression results for interpretation
            su_val = prediction_value["Su"]
            if su_val < 400:
                interpretation = "Low Yield Material — Decorative / Non-Structural"
            else:
                interpretation = "Load-Bearing Material — Structural Components"
            decision = "Qualified" if su_val > 250 else "Review Required"

        elif model_name == "module_c_clf":
            prediction_value = str(raw_prediction[0])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_array)[0]
                confidence = float(round(max(proba), 4))
            else:
                confidence = None
            task = "classification"
            target = "Use (Not Used / Used)"
            label_map = {0: "Not Used", 1: "Used", False: "Not Used", True: "Used"}
            if raw_prediction[0] in label_map:
                prediction_value = label_map[raw_prediction[0]]
            
            # Decision System mapping
            interpretation = "Material Selection Analysis"
            decision = "Qualified for Project" if prediction_value == "Used" else "Not Recommended"

    except Exception as e:
        error_logger.error(f"Prediction failed for {model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    prediction_logger.info(
        f"model={model_name} | input={request.features} | prediction={prediction_value}"
    )

    return PredictResponse(
        prediction=prediction_value,
        model_used=model_name,
        confidence=confidence,
        metrics=model_store.get_metadata(model_name),
        task=task,
        target=target,
        interpretation=interpretation,
        decision=decision
    )
