from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

from app.db.database import db


async def compute_and_plot_edf(
    collection_name: str = "new_york_updated",
    variable: str = "actual_price",
    save_dir: Optional[str] = None,
):
    """
    Connects to MongoDB Atlas, computes EDF for a quantitative variable,
    compares it to a theoretical normal CDF, and saves the plots.
    """
    collection = db[collection_name]

    # Identifies the directory of this module to build a stable reference point for output paths.
    base_dir = Path(__file__).resolve().parent

    # Determines where plot files should be written; defaults to a dedicated imgs folder for consistency.
    target_dir = Path(save_dir) if save_dir else base_dir / "imgs"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Retrieves all documents containing the target variable to ensure analyses are performed on valid numerical data.
    cursor = collection.find(
        {variable: {"$exists": True, "$ne": None}},
        {variable: 1, "_id": 0},
    )
    docs = await cursor.to_list(length=None)

    # Converts extracted values into a uniform numeric array, enabling vectorized statistical operations.
    values = np.array([doc[variable] for doc in docs], dtype=float)

    n = len(values)
    if n == 0:
        raise ValueError(f"No data found for variable '{variable}'.")

    # Orders values to construct the empirical distribution and align with theoretical CDF evaluations.
    sorted_values = np.sort(values)

    # Builds the empirical cumulative distribution by assigning each sorted point its cumulative rank proportion.
    edf = np.arange(1, n + 1) / n

    # Estimates normal distribution parameters from sample moments to provide a baseline theoretical comparison.
    mu = np.mean(values)
    sigma = np.std(values)

    # Computes the corresponding theoretical CDF values to assess goodness-of-fit visually
    cdf_theoretical = norm.cdf(sorted_values, loc=mu, scale=sigma)

    # Defines a reference price point to contextualize one specific quantile in both empirical and theoretical terms.
    x0 = 1_000_000

    # Locates the closest index for the reference value within the sorted dataset to extract its cumulative positions.
    idx = np.searchsorted(sorted_values, x0, side="right") - 1
    idx = max(0, min(idx, len(sorted_values) - 1))

    edf_x0 = edf[idx]
    cdf_x0 = cdf_theoretical[idx]

    # Establishes a fixed zoom range to standardize comparisons across multiple runs and variables.
    xmin, xmax = 50_000, 50_000_000

    # Creates an EDF-only plot to highlight the empirical cumulative structure of the distribution.
    plt.figure(figsize=(8, 5))
    plt.step(sorted_values, edf, where="post", label="Empirical CDF (EDF)")
    plt.title(f"Empirical Distribution Function for '{variable}'")
    plt.xlabel(variable)
    plt.ylabel("Probabilité cumulée")

    # Applies log-scaling to better represent wide numeric ranges and avoid visual compression of low-price values.
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(money_formatter))

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Saves the standalone EDF plot for further inspection or reporting.
    edf_path = target_dir / f"edf_{variable}.png"
    plt.savefig(edf_path)
    plt.close()

    # Produces a comparison plot to visually inspect how well the empirical distribution aligns with theoretical normality.
    plt.figure(figsize=(8, 5))
    plt.step(sorted_values, edf, where="post", label="Empirical CDF (EDF)")
    plt.plot(sorted_values, cdf_theoretical, label="Theoretical CDF (Normal)")
    plt.title(f"EDF vs Theoretical CDF (Normal) for '{variable}'")
    plt.xlabel(variable)
    plt.ylabel("Probabilité cumulée")

    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(money_formatter))

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Adds visual markers showing where the reference value lies on both empirical and theoretical curves.
    plt.axvline(x0, color="gray", linestyle="--", linewidth=1)
    plt.axhline(edf_x0, color="gray", linestyle="--", linewidth=1)
    plt.scatter([x0], [edf_x0], color="blue")
    plt.text(x0, edf_x0 + 0.03, f"EDF(1M$) = {edf_x0:.2f}", color="blue")
    plt.scatter([x0], [cdf_x0], color="orange")
    plt.text(x0, cdf_x0 - 0.05, f"CDF(1M$) = {cdf_x0:.2f}", color="orange")

    # Saves the comparative EDF vs CDF plot to illustrate normality assumptions and deviations.
    edf_cdf_path = target_dir / f"edf_vs_cdf_{variable}.png"
    plt.savefig(edf_cdf_path)
    plt.close()

    # Computes summary statistics to contextualize distribution shape, variability, and common percentiles.
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    p50 = float(np.percentile(values, 50))
    p80 = float(np.percentile(values, 80))
    p90 = float(np.percentile(values, 90))

    # Creates a human-readable report to accompany visual outputs and provide a persistent analysis summary.
    log_path = target_dir / f"edf_{variable}_summary.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"EDF analysis summary for variable '{variable}'\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of observations: {n}\n")
        f.write(f"Mean: {mu:.4f}\n")
        f.write(f"Standard deviation: {sigma:.4f}\n")
        f.write(f"Min: {minimum:.4f}\n")
        f.write(f"Max: {maximum:.4f}\n")
        f.write(f"Median (50th percentile): {p50:.4f}\n")
        f.write(f"80th percentile: {p80:.4f}\n")
        f.write(f"90th percentile: {p90:.4f}\n")
        f.write("\nZoom shown on plots: [50,000 ; 50,000,000]\n")
        f.write(f"EDF plot: {edf_path}\n")
        f.write(f"EDF vs CDF plot: {edf_cdf_path}\n")
        f.write(f"Log file path: {log_path}\n")


def money_formatter(x):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B $"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.1f}M $"
    elif x >= 1_000:
        return f"{x/1_000:.1f}K $"
    else:
        return f"{x:.0f} $"
