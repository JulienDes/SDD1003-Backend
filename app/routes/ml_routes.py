from typing import Any, Callable, Dict, Optional
from fastapi import APIRouter, HTTPException
from app.services.machine_learning.train_models.train_linear_regression_model import (
    train_linear_regression_model,
)
from app.services.machine_learning.train_models.train_random_forest_regressor import (
    train_random_forest_model,
)
from app.services.machine_learning.train_models.train_knn_model import train_knn_model
from app.services.machine_learning.predict_models.predict_price_with_linear_regression import (
    predict_price_with_linear_regression,
)
from app.services.machine_learning.predict_models.predict_price_with_random_forest import (
    predict_price_with_random_forest,
)
from app.services.machine_learning.predict_models.predict_price_with_knn import (
    predict_price_with_knn,
)
from app.services.EDF.statics_analysis import compute_and_plot_edf

from app.schemas.models import ModelName, PredictRequest

router_ml = APIRouter(prefix="/ml", tags=["Machine-learning operations"])

# Maps each model type to its corresponding prediction function.
MODEL_FN_MAP: Dict[ModelName, Callable[..., float]] = {
    ModelName.linear_regression: predict_price_with_linear_regression,
    ModelName.random_forest: predict_price_with_random_forest,
    ModelName.knn: predict_price_with_knn,
}


@router_ml.post("/train-linear-regression", response_model=Dict[str, Any])
async def train_linear_regression_endpoint() -> Dict[str, Any]:
    """
    Exposes a route to trigger training of the linear regression model.
    The endpoint exists so model creation and refresh can be controlled externally,
    ensuring the system regenerates predictions based on current database data.
    """
    try:
        result = await train_linear_regression_model()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_ml.post("/train-random-forest-regressor", response_model=Dict[str, Any])
async def train_random_forest_regressor_endpoint() -> Dict[str, Any]:
    """
    Provides external access to the training routine for the random-forest model.
    This allows scheduled or manual retraining without touching backend code.
    """
    try:
        result = await train_random_forest_model()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_ml.post("/train-knn", response_model=Dict[str, Any])
async def train_knn_endpoint() -> Dict[str, Any]:
    """
    Exposes the training job for the KNN model so it can be refreshed on demand.
    Useful when dataset updates require model recalibration.
    """
    try:
        result = await train_knn_model()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_ml.post("/predict/{model_name}", response_model=Dict[str, Any])
async def predict_price_endpoint(
    model_name: ModelName, payload: PredictRequest
) -> Dict[str, Any]:
    """
    Receives a model name and input features, then delegates prediction to the
    correct model function. This endpoint standardizes inference across models
    so the frontend and API consumers call a single route regardless of model type.
    """
    try:
        predict_fn = MODEL_FN_MAP[model_name]

        predicted_price = predict_fn(
            type=payload.type,
            beds=payload.beds,
            bath=payload.bath,
            sqft=payload.propertysqft,
            state=payload.state,
        )

        return {
            "predicted_price": predicted_price,
            "model": model_name,
            "inputs": payload.model_dump(),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Modèle introuvable. Lance d'abord l'entraînement.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router_ml.post("/edf-analysis", response_model=Dict[str, Any])
async def edf_analysis_endpoint(
    collection_name: str = "new_york_updated",
    variable: str = "actual_price",
    save_dir: Optional[str] = None,
):
    """
    Triggers EDF-based statistical analysis and plot generation.
    The endpoint returns only metadata, keeping large plot files out of responses
    while enabling reproducible offline diagnostics.
    """
    try:
        await compute_and_plot_edf(
            collection_name=collection_name, variable=variable, save_dir=save_dir
        )

        return {
            "status": "success",
            "message": "EDF analysis completed successfully.",
            "collection": collection_name,
            "variable": variable,
            "saved_in": save_dir,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
