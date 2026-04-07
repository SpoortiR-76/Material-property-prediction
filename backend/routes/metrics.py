from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from utils.model_loader import model_store, MODEL_METADATA, VALID_MODEL_NAMES
from utils.logger import system_logger, error_logger

router = APIRouter(prefix="/metrics", tags=["Metrics"])

class MetricsResponse(BaseModel):
    model_name: str
    description: str
    task: str
    target: str
    mae: Optional[Any] = None
    r2: Optional[Any] = None
    accuracy: Optional[float] = None
    features: list[str]
    is_loaded: bool

@router.get("/{model_name}", response_model=MetricsResponse)
def get_model_metrics(model_name: str):
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=404, 
            detail=f"Unknown model '{model_name}'. Valid: {VALID_MODEL_NAMES}"
        )
    
    meta = MODEL_METADATA[model_name]
    system_logger.info(f"Metrics requested for {model_name}")

    return MetricsResponse(
        model_name=model_name,
        description=meta["description"],
        task=meta["task"],
        target=meta["target"],
        mae=meta["mae"],
        r2=meta["r2"],
        accuracy=meta["accuracy"],
        features=meta["features"],
        is_loaded=model_store.is_loaded(model_name)
    )

@router.get("/", response_model=list[MetricsResponse])
def get_all_metrics():
    system_logger.info("Metrics requested for all models")
    responses = []
    for model_name in VALID_MODEL_NAMES:
        meta = MODEL_METADATA[model_name]
        responses.append(
            MetricsResponse(
                model_name=model_name,
                description=meta["description"],
                task=meta["task"],
                target=meta["target"],
                mae=meta["mae"],
                r2=meta["r2"],
                accuracy=meta["accuracy"],
                features=meta["features"],
                is_loaded=model_store.is_loaded(model_name)
            )
        )
    return responses
