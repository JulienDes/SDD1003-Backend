from pathlib import Path
from typing import Optional
import pickle

from matplotlib.ticker import FuncFormatter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from app.utils.money_formatter import money_formatter
from app.db.database import db

CATEGORICAL_COLS = ["STATE", "TYPE"]
NUMERIC_COLS = ["BEDS", "BATH", "PROPERTYSQFT"]
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS


async def train_knn_model(
    collection_name: str = "new_york_updated",
    features: list = FEATURES,
    target: str = "actual_price",
    model_dir: Optional[str] = None,
    model_filename: str = "knn_real_estate.pkl",
    n_neighbors: int = 5,
    weights: str = "distance",
    metric: str = "minkowski",
    random_state: int = 42,
    remove_outliers: bool = True,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    log_target: bool = True,
):
    """
    Trains a KNN regression model for real-estate prices after optional
    outlier filtering and log-transform on the target.
    Saves the trained model and a diagnostic plot.
    """

    # Fetches the entire dataset from MongoDB to build the training frame.
    collection = db[collection_name]
    documents = await collection.find().to_list(length=None)

    if len(documents) == 0:
        raise Exception("Aucune donnée trouvée dans MongoDB.")

    df = pd.DataFrame(documents)

    # Ensures all required feature and target columns exist before training.
    for col in features + [target]:
        if col not in df.columns:
            raise Exception(f"Colonne absente dans MongoDB: {col}")

    # Restricts dataset to relevant columns and removes incomplete rows.
    df = df[features + [target]].dropna()

    # Optionally limits extreme target values to stabilize KNN distance behavior.
    if remove_outliers:
        q_low, q_high = df[target].quantile([lower_quantile, upper_quantile])
        df = df[(df[target] >= q_low) & (df[target] <= q_high)]

    # Splits into feature matrix and target vector.
    X = df[features].copy()
    y = df[target].astype(float)

    # Validates and coerces numeric fields to ensure consistency for KNN distances.
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drops rows with invalid numerical values and realigns target values.
    X = X.dropna(subset=NUMERIC_COLS)
    y = y.loc[X.index]

    # Normalizes categorical fields to a consistent label form before encoding.
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str).str.split(",", n=1).str[0].str.strip()

    # Converts categorical variables into sparse indicator columns for KNN input.
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLS,
        drop_first=False,
        dtype=int,
    )

    # Applies log transform to stabilize variance and reduce extreme-value influence.
    if log_target:
        y_train_target = np.log1p(y)  # log(1 + price)
    else:
        y_train_target = y

    # Creates train/test partitions to evaluate generalization performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_train_target, test_size=0.2, random_state=random_state
    )

    # Instantiates a distance-based regressor sensitive to feature scaling and density.
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Resolves and prepares the target directory for saving artifacts.
    base_dir = Path(__file__).resolve().parent.parent  # app/machine_learning
    target_dir = Path(model_dir) if model_dir else base_dir / "models" / "knn_regressor"
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename

    # Stores model and preprocessing metadata to enable consistent inference.
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "columns": X.columns.tolist(),
                "log_target": log_target,
                "remove_outliers": remove_outliers,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
            },
            f,
        )

    # Predicts with the trained model on the test split.
    y_pred_transformed = model.predict(X_test)

    # Converts predictions and ground truth back to real-price space for evaluation.
    if log_target:
        y_test_price = np.expm1(y_test)  # inverse de log1p
        y_pred_price = np.expm1(y_pred_transformed)
    else:
        y_test_price = y_test
        y_pred_price = y_pred_transformed

    # Computes evaluation metrics to quantify model accuracy.
    mse = mean_squared_error(y_test_price, y_pred_price)
    r2 = r2_score(y_test_price, y_pred_price)
    mse_str = money_formatter(mse)

    # Generates a scatter plot to visually assess predictive alignment.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_price, y_pred_price, alpha=0.6)

    max_val = max(max(y_test_price), max(y_pred_price))
    min_val = min(min(y_test_price), min(y_pred_price))
    ax.plot([min_val, max_val], [min_val, max_val], "k--")

    ax.set_title("KNN Regressor - Actual vs. Predicted Values")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    ax.text(
        0.05,
        0.95,
        f"MSE: {mse_str}\nR²: {r2:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Formats axis ticks to human-readable currency.
    ax.xaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))

    plot_path = target_dir / "actual_vs_predicted_knn.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "success",
        "model_path": str(model_path),
        "score_r2": r2,
        "mse": mse,
        "plot_path": str(plot_path),
    }
