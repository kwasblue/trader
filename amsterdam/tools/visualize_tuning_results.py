#!/usr/bin/env python3
"""
Visualize Parameter Tuning Results

Analyzes tuning output and creates visualizations:
- Parameter importance analysis
- Convergence plots
- Parameter correlation heatmap
- Performance scatter plots

Usage:
    python visualize_tuning_results.py results/tuning_AAPL.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(json_path: Path) -> dict:
    """Load tuning results from JSON."""
    with open(json_path) as f:
        return json.load(f)


def plot_convergence(results: list[dict], output_dir: Path):
    """
    Plot optimization convergence over trials.

    Shows how the best objective score improves over time.
    """
    objectives = [r["objective_score"] for r in results]

    # Running best
    best_so_far = []
    current_best = -np.inf
    for obj in objectives:
        if obj > current_best:
            current_best = obj
        best_so_far.append(current_best)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # All trials
    ax1.plot(objectives, "o-", alpha=0.5, label="Trial objective")
    ax1.plot(best_so_far, "r-", linewidth=2, label="Best so far")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Objective Score")
    ax1.set_title("Optimization Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of objectives
    ax2.hist(objectives, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(best_so_far[-1], color="r", linestyle="--", linewidth=2, label=f"Best: {best_so_far[-1]:.3f}")
    ax2.set_xlabel("Objective Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Trial Scores")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=150)
    print(f"Saved convergence plot to {output_dir / 'convergence.png'}")


def plot_parameter_importance(results: list[dict], output_dir: Path):
    """
    Analyze which parameters have the most impact on performance.

    Uses variance-based sensitivity analysis.
    """
    # Extract parameters and scores
    df = pd.DataFrame([r["params"] for r in results])
    df["objective"] = [r["objective_score"] for r in results]

    # Calculate correlation with objective
    correlations = df.corr()["objective"].drop("objective").abs().sort_values(ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.plot(kind="barh", ax=ax)
    ax.set_xlabel("Absolute Correlation with Objective")
    ax.set_ylabel("Parameter")
    ax.set_title("Parameter Importance (Higher = More Impact on Performance)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_importance.png", dpi=150)
    print(f"Saved importance plot to {output_dir / 'parameter_importance.png'}")

    # Print top 5
    print("\nTop 5 Most Important Parameters:")
    for param, corr in correlations.head(5).items():
        print(f"  {param:30s}: {corr:.3f}")


def plot_parameter_distributions(results: list[dict], best_params: dict, output_dir: Path):
    """
    Plot distributions of parameter values tried, highlighting best.
    """
    df = pd.DataFrame([r["params"] for r in results])

    # Select continuous parameters for violin plots
    continuous_params = [
        "bayesian_prior_sharpe",
        "sharpe_smoothing_alpha",
        "scaling_alpha",
        "min_multiplier",
        "max_multiplier",
        "skip_trade_threshold",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(continuous_params):
        if param in df.columns:
            ax = axes[idx]

            # Violin plot
            ax.violinplot([df[param].values], vert=False, widths=0.7, showmeans=True, showextrema=True)

            # Mark best value
            best_val = best_params.get(param)
            if best_val is not None:
                ax.axvline(best_val, color="r", linestyle="--", linewidth=2, label=f"Best: {best_val:.3f}")

            ax.set_xlabel("Parameter Value")
            ax.set_title(param)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150)
    print(f"Saved distributions plot to {output_dir / 'parameter_distributions.png'}")


def plot_parameter_correlations(results: list[dict], output_dir: Path):
    """
    Show correlation heatmap between parameters.

    Helps identify if certain parameter combinations work well together.
    """
    df = pd.DataFrame([r["params"] for r in results])
    df["objective"] = [r["objective_score"] for r in results]

    # Correlation matrix
    corr = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Parameter Correlation Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_correlations.png", dpi=150)
    print(f"Saved correlation heatmap to {output_dir / 'parameter_correlations.png'}")


def plot_2d_slices(results: list[dict], best_params: dict, output_dir: Path):
    """
    Show 2D slices of objective surface for key parameter pairs.

    Helps visualize how pairs of parameters affect performance.
    """
    df = pd.DataFrame([r["params"] for r in results])
    df["objective"] = [r["objective_score"] for r in results]

    # Key parameter pairs to visualize
    param_pairs = [
        ("scaling_alpha", "max_multiplier"),
        ("min_multiplier", "skip_trade_threshold"),
        ("bayesian_prior_sharpe", "bayesian_prior_weight"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (param_x, param_y) in enumerate(param_pairs):
        ax = axes[idx]

        # Scatter plot
        scatter = ax.scatter(
            df[param_x],
            df[param_y],
            c=df["objective"],
            cmap="viridis",
            s=50,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

        # Mark best
        ax.scatter(
            best_params[param_x],
            best_params[param_y],
            c="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Best",
            zorder=10,
        )

        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f"{param_x} vs {param_y}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=ax, label="Objective")

    plt.tight_layout()
    plt.savefig(output_dir / "2d_slices.png", dpi=150)
    print(f"Saved 2D slices to {output_dir / '2d_slices.png'}")


def generate_report(data: dict, output_dir: Path):
    """Generate text summary report."""
    report = []
    report.append("=" * 70)
    report.append("TUNING RESULTS SUMMARY")
    report.append("=" * 70)
    report.append("")

    report.append(f"Symbol: {data['symbol']}")
    report.append(f"Number of trials: {data['n_trials']}")
    report.append(f"Walk-forward windows: {data['n_windows']}")
    report.append(f"Best objective score: {data['best_score']:.3f}")
    report.append("")

    report.append("BEST PARAMETERS:")
    report.append("-" * 70)
    for param, value in sorted(data["best_params"].items()):
        if isinstance(value, float):
            report.append(f"  {param:30s}: {value:.4f}")
        else:
            report.append(f"  {param:30s}: {value}")

    report.append("")
    report.append("PERFORMANCE STATISTICS:")
    report.append("-" * 70)

    all_scores = [r["objective_score"] for r in data["all_results"]]
    report.append(f"  Mean objective:    {np.mean(all_scores):.3f}")
    report.append(f"  Std objective:     {np.std(all_scores):.3f}")
    report.append(f"  Min objective:     {np.min(all_scores):.3f}")
    report.append(f"  Max objective:     {np.max(all_scores):.3f}")
    report.append(f"  Best objective:    {data['best_score']:.3f}")

    report.append("")
    report.append("=" * 70)

    # Save report
    report_text = "\n".join(report)
    print(report_text)

    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(report_text)

    print(f"\nSaved text report to {output_dir / 'summary_report.txt'}")


def main():
    """Generate all visualizations from tuning results."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_tuning_results.py <results.json>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    # Load results
    print(f"Loading results from {results_path}...")
    data = load_results(results_path)

    # Create output directory
    output_dir = results_path.parent / f"plots_{results_path.stem}"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving plots to {output_dir}/")

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_convergence(data["all_results"], output_dir)
    plot_parameter_importance(data["all_results"], output_dir)
    plot_parameter_distributions(data["all_results"], data["best_params"], output_dir)
    plot_parameter_correlations(data["all_results"], output_dir)
    plot_2d_slices(data["all_results"], data["best_params"], output_dir)

    # Generate summary report
    generate_report(data, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
