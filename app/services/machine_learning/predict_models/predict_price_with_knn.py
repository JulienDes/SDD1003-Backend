import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def predict_price_with_knn(
    type: str,
    beds: int,
    bath: int,
    sqft: float,
    state: str,
    model_path: Optional[str] = None,
):
    """
    Loads a previously trained KNN regression model and produces a price
    estimate for a single property.
    Ensures the same feature preparation and target transformation used
    during training is reproduced for consistent predictions.

    Parameters
    ----------
    type : str
        Housing type category (must match training categories).
    beds : int
        Number of bedrooms used as a numerical predictor.
    bath : int
        Number of bathrooms used as a numerical predictor.
    sqft : float
        Property square footage, treated as a key continuous variable.
    state : str
        Geographic state identifier for categorical matching.
    model_path : Optional[str]
        Custom path for loading a serialized model, if provided.

    Returns
    -------
    float
        Final predicted price on the original scale of the target variable.
    """

    # Determines the model storage path so the function stays portable.
    base_dir = Path(__file__).resolve().parent.parent  # app/machine_learning
    target_path = (
        Path(model_path)
        if model_path
        else base_dir / "models" / "knn_regressor" / "knn_real_estate.pkl"
    )

    # Loads the serialized model object containing estimator, columns, and settings.
    with open(target_path, "rb") as f:
        model_obj = pickle.load(f)

    # Extracts stored components to guarantee consistent inference logic.
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        model_columns = model_obj.get("columns")
        log_target = model_obj.get("log_target", False)
    else:
        raise ValueError("Format de modèle KNN invalide")

    # Constructs a one-row dataset mirroring the feature structure used at training time.
    data = {
        "STATE": state,
        "TYPE": type,
        "BEDS": beds,
        "BATH": bath,
        "PROPERTYSQFT": sqft,
    }

    df = pd.DataFrame([data])

    categorical_cols = ["STATE", "TYPE"]
    numeric_cols = ["BEDS", "BATH", "PROPERTYSQFT"]

    # Ensures numerical predictors are valid so the model does not ingest corrupted data.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isnull().any().any():
        raise ValueError("Invalid numeric inputs for prediction")

    # Normalizes categorical inputs so category encoding matches the training format.
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.split(",", n=1).str[0].str.strip()

    # Applies one-hot encoding to reproduce the categorical expansion seen during training.
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype=int,
    )

    # Aligns the encoded features with the model’s expected feature space.
    # Missing columns are added with zeros to preserve consistency.
    X = df.reindex(columns=model_columns, fill_value=0)

    # Produces a prediction in the transformed space (log or raw depending on training).
    prediction_transformed = model.predict(X)[0]

    # Applies inverse transformation when the model was trained on log(target).
    if log_target:
        prediction = np.expm1(prediction_transformed)
    else:
        prediction = prediction_transformed

    return float(prediction)
