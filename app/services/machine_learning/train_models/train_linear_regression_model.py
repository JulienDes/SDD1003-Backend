from pathlib import Path
from typing import Optional
import pickle

from matplotlib.ticker import FuncFormatter
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from app.db.database import db
from app.utils.money_formatter import money_formatter

# Explicit feature groups so numeric columns are never treated as categoricals
CATEGORICAL_COLS = ["STATE", "TYPE"]
NUMERIC_COLS = ["BEDS", "BATH", "PROPERTYSQFT"]
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS


async def train_linear_regression_model(
    collection_name: str = "new_york_updated",
    features: list = FEATURES,
    target: str = "actual_price",
    model_dir: Optional[str] = None,
    model_filename: str = "linear_regression_real_estate.pkl",
):
    collection = db[collection_name]
    documents = await collection.find().to_list(length=None)

    # Ensures training does not proceed without any available data.
    if len(documents) == 0:
        raise Exception("Aucune donnée trouvée dans MongoDB.")

    df = pd.DataFrame(documents)

    # Confirms that all required input and target columns exist in the dataset before training.
    for col in features + [target]:
        if col not in df.columns:
            raise Exception(f"Colonne absente dans MongoDB: {col}")

    # Keeps only the columns used for training to limit noise and unexpected data.
    df = df[features + [target]].dropna()

    # Separates the model inputs and outputs to prepare for preprocessing and training.
    X = df[features].copy()
    y = df[target]

    # Forces numeric columns into a proper numerical type to avoid incorrect encoding or model errors.
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Removes rows where numeric conversion failed to maintain model consistency.
    X = X.dropna(subset=NUMERIC_COLS)
    y = y.loc[X.index]

    # Standardizes categorical values by stripping extra information after commas.
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype(str).str.split(",", n=1).str[0].str.strip()

    # Converts categorical variables into binary indicators to allow regression to process them.
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLS,
        drop_first=False,
        dtype=int,
    )

    # Separates data into training and validation splits to evaluate predictive performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Trains a linear regression model to learn relationships between features and price.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Measures predictive performance using the built-in R² score.
    score = model.score(X_test, y_test)

    # Computes where trained models should be stored for later inference.
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = (
        Path(model_dir) if model_dir else base_dir / "models" / "linear_regression"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename

    # Stores both the trained model and the feature layout for consistent future predictions.
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "columns": X.columns.tolist(),
            },
            f,
        )

    # Generates predictions on the test subset for performance visualization.
    y_pred = model.predict(X_test)

    # Evaluates prediction accuracy through error and goodness-of-fit metrics.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Formats the mean squared error into a readable currency-styled format.
    mse_str = money_formatter(mse)

    # Prepares a plot comparing actual vs. predicted prices to visually assess model reliability.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)

    # Draws an ideal reference line representing perfect prediction alignment.
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "k--")

    ax.set_title("Actual vs. Predicted Values")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    # Displays evaluation metrics directly on the plot for quick interpretation.
    ax.text(
        0.05,
        0.95,
        f"MSE: {mse_str}\nR²: {r2:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Applies currency formatting so axes reflect real-estate pricing conventions.
    ax.xaxis.set_major_formatter(FuncFormatter(money_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))

    # Stores the visualization next to the trained model for debugging and reporting.
    plot_path = target_dir / "actual_vs_predicted.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"status": "success", "model_path": str(model_path), "score_r2": score}
