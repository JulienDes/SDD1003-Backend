import pickle
from pathlib import Path
from typing import Optional

import pandas as pd


def predict_price_with_random_forest(
    type: str,
    beds: int,
    bath: int,
    sqft: float,
    state: str,
    model_path: Optional[str] = None,
):
    """
    Loads a previously trained RandomForestRegressor and produces a price
    prediction using the same preprocessing logic applied during training.

    Parameters
    ----------
    type : str
        Property type (categorical feature used to align with training).
    beds : int
        Number of bedrooms.
    bath : int
        Number of bathrooms.
    sqft : float
        Interior square footage of the property.
    state : str
        U.S. state or region category associated with the listing.
    model_path : Optional[str]
        Path to the saved model file. Uses default path when omitted.

    Returns
    -------
    float
        Predicted price for the given property attributes.
    """

    # Locates the ML root folder and resolves the default trained model path.
    base_dir = Path(__file__).resolve().parent.parent
    target_path = (
        Path(model_path)
        if model_path
        else base_dir / "models" / "random_forest" / "random_forest_real_estate.pkl"
    )

    # Loads the serialized model and metadata required for input alignment.
    with open(target_path, "rb") as f:
        model_obj = pickle.load(f)

    # Validates model structure and extracts estimator + encoded columns list.
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        model_columns = model_obj.get("columns")
    else:
        raise ValueError("Format de modèle Random Forest invalide")

    # Builds a single-row input matching the original feature schema.
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

    # Ensures numeric values are valid for inference; otherwise rejects input.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isnull().any().any():
        raise ValueError("Invalid numeric inputs for prediction")

    # Normalizes categorical fields to avoid misalignment due to formatting noise.
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.split(",", n=1).str[0].str.strip()

    # Reconstructs one-hot encoded categoricals to mirror training preprocessing.
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype=int,
    )

    # Reorders dataframe columns to match training schema and fills unknown dummies.
    X = df.reindex(columns=model_columns, fill_value=0)

    # Produces the final numerical price estimate from the trained model.
    prediction = model.predict(X)[0]

    return float(prediction)
