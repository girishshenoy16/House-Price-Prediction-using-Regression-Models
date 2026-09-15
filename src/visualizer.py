import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def plot_correlation_heatmap(X_engineered, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = X_engineered.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_model_comparison(cv_results, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    names = list(cv_results.keys())
    rmses = [cv_results[n]["cv_rmse"] for n in names]
    r2s = [cv_results[n]["cv_r2"] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2c3e50", "#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    ax1.barh(names, rmses, color=colors)
    ax1.set_xlabel("CV RMSE ($)")
    ax1.set_title("Model Comparison - RMSE")
    ax2.barh(names, r2s, color=colors)
    ax2.set_xlabel("R² Score")
    ax2.set_title("Model Comparison - R²")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_hold, y_pred, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_hold, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
    mn = min(y_hold.min(), y_pred.min())
    mx = max(y_hold.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title("Actual vs Predicted Sale Prices")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close()


def plot_feature_importance(model_export, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    coefs = model_export.get("coefficients")
    if coefs is None:
        return
    feature_names = model_export["feature_names"]
    importance = np.abs(coefs)
    top_n = min(15, len(importance))
    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in indices],
            [importance[i] for i in indices], color="#3498db")
    ax.set_xlabel("Absolute Coefficient Value")
    ax.set_title("Top Feature Importance (Absolute Coefficients)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
