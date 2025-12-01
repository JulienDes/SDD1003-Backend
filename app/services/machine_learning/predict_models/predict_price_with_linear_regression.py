import pickle
from pathlib import Path
from typing import Optional

import pandas as pd


def predict_price_with_linear_regression(
    type: str,
    beds: int,
    bath: int,
    sqft: float,
    state: str,
    model_path: Optional[str] = None,
):
    """
    Loads a pre-trained linear regression model and produces a price prediction
    from the provided real-estate characteristics. Ensures features undergo
    the same preprocessing used during model training.

    Parameters
    ----------
    type : str
        Property type as used during training.
    beds : int
        Number of bedrooms.
    bath : int
        Number of bathrooms.
    sqft : float
        Interior living area.
    state : str
        State identifier used for categorical encoding.
    model_path : Optional[str]
        Custom model location. Defaults to the standard trained model path.

    Returns
    -------
    float
        Predicted price value.
    """

    # Resolves the default model directory if none is provided.
    base_dir = Path(__file__).resolve().parent.parent
    target_path = (
        Path(model_path)
        if model_path
        else base_dir
        / "models"
        / "linear_regression"
        / "linear_regression_real_estate.pkl"
    )

    # Loads the serialized model object containing estimator + feature columns.
    with open(target_path, "rb") as f:
        model_obj = pickle.load(f)

    # Verifies the model was saved using the expected dictionary structure.
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
        model_columns = model_obj.get("columns")
    else:
        return "Error"

    # Builds a single-row dataframe containing raw input features.
    data = {
        "STATE": state,
        "TYPE": type,
        "BEDS": beds,
        "BATH": bath,
        "PROPERTYSQFT": sqft,
    }

    df = pd.DataFrame([data])

    # Ensures numeric fields are consistent with model expectations.
    numeric_cols = ["BEDS", "BATH", "PROPERTYSQFT"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric_cols].isnull().any().any():
        raise ValueError("Invalid numeric inputs for prediction")

    # Normalizes categorical text to avoid mismatches with training categories.
    categorical_cols = ["STATE", "TYPE"]
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.split(",", n=1).str[0].str.strip()

    # One-hot encodes categorical fields using the same approach as training.
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
        dtype=int,
    )

    # Reorders and fills missing columns to match model training layout.
    X = df.reindex(columns=model_columns, fill_value=0)

    # Runs the prediction using the aligned feature matrix.
    prediction = model.predict(X)[0]

    return float(prediction)
