from pathlib import Path
from typing import Optional
import pickle

from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from app.utils.money_formatter import money_formatter
from app.db.database import db

# Explicit feature groups so numeric columns are never treated as categoricals
CATEGORICAL_COLS = ["STATE", "TYPE"]
NUMERIC_COLS = ["BEDS", "BATH", "PROPERTYSQFT"]
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS


async def train_random_forest_model(
    collection_name: str = "new_york_updated",
    features: list = FEATURES,
    target: str = "actual_price",
    model_dir: Optional[str] = None,
    model_filename: str = "random_forest_real_estate.pkl",
    n_estimators: int = 300,
    max_depth: Optional[int] = 20,
    random_state: int = 42,
):
    """
    Trains a RandomForestRegressor model for real-estate price estimation,
    saves the trained model and a diagnostic plot of Actual vs Predicted values.
    """

    # Fetch all real-estate documents from MongoDB for model training.
    collection = db[collection_name]
    documents = await collection.find().to_list(length=None)

    if len(documents) == 0:
        raise Exception("Aucune donnée trouvée dans MongoDB.")

    df = pd.DataFrame(documents)

    # Ensures required features and target exist in the dataset before training.
    for col in features + [target]:
        if col not in df.columns:
            raise Exception(f"Colonne absente dans MongoDB: {col}")

    # Keeps only fields needed for the model to avoid unintended information leakage.
    df = df[features + [target]].dropna()

    # Extracts input features and ground-truth labels used for supervised learning.
    X = df[features].copy()
    y = df[target]

    # Forces numeric fields into true numeric type to guarantee consistent model input representation.
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Filters out invalid rows to maintain model integrity after numeric conversion.
    X = X.dropna(subset=NUMERIC_COLS)
    y = y.loc[X.index]

    # Standardizes categorical fields by removing trailing segments (e.g., after commas),
    # ensuring categories remain consistent across rows.
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str).str.split(",", n=1).str[0].str.strip()

    # Transforms categorical fields into machine-readable binary indicators
    # using a controlled one-hot encoding on the predefined category list.
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLS,
        drop_first=False,
        dtype=int,
    )

    # Splits data into training and testing subsets to enable unbiased performance evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Defines a RandomForest model tuned for structured tabular regression.
    # Parameters balance model complexity and generalization.
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    # Fits the forest on the training data to learn non-linear relationships in housing features.
    model.fit(X_train, y_train)

    # Measures predictive strength on unseen test samples.
    score = model.score(X_test, y_test)

    # Resolves where to save the trained model, ensuring reproducible storage.
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(model_dir) if model_dir else base_dir / "models" / "random_forest"
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename

    # Persists both the model and its expected input schema for future predictions.
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "columns": X.columns.tolist(),
            },
            f,
        )

    # Produces predictions on held-out data to construct evaluation visuals.
    y_pred = model.predict(X_test)

    # Computes regression metrics to quantify model accuracy and calibration.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_str = money_formatter(mse)

    # Creates a diagnostic scatter plot showing how well predictions align with true values.
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_pred, alpha=0.6)

    # Draws the ideal prediction line, which serves as a reference indicator
    # for detecting bias or systematic under/over-estimation.
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "k--")

    ax.set_title("Random Forest - Actual vs. Predicted Values")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    # Adds textual performance indicators directly on the plot for quick visual assessment.
    ax.text(
        0.05,
        0.95,
        f"MSE: {mse_str}\nR²: {r2:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Formats axes in monetary scale to match domain context and interpretation.
    ax.xaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))

    # Exports the diagnostic plot to the same location as the serialized model.
    plot_path = target_dir / "actual_vs_predicted_random_forest.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "success",
        "model_path": str(model_path),
        "score_r2": score,
        "mse": mse,
        "plot_path": str(plot_path),
    }
