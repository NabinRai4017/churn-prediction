import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import logging
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import seaborn as sns
import mlflow

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_model_comparison_report(
    model_comparison: Dict[str, Any]
) -> pd.DataFrame:
    """Creates a formatted model comparison report.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        DataFrame with model comparison metrics
    """
    models_data = model_comparison["models"]

    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in models_data.items():
        comparison_data.append({
            "Model": metrics["model_name"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
        })

    df = pd.DataFrame(comparison_data)

    # Sort by selection metric
    selection_metric = model_comparison["selection_metric"]
    metric_col_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
    }
    sort_col = metric_col_map.get(selection_metric, "F1 Score")
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Add rank column
    df.insert(0, "Rank", range(1, len(df) + 1))

    logger.info("\nModel Comparison Report:")
    logger.info(f"\n{df.to_string(index=False)}")

    return df


def create_confusion_matrix_report(
    model_comparison: Dict[str, Any]
) -> pd.DataFrame:
    """Creates confusion matrix report for all models.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        DataFrame with confusion matrix metrics
    """
    models_data = model_comparison["models"]

    confusion_data = []
    for model_name, metrics in models_data.items():
        tn = metrics["true_negatives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        tp = metrics["true_positives"]

        total = tn + fp + fn + tp

        confusion_data.append({
            "Model": metrics["model_name"],
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
            "TN Rate": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "FP Rate": fp / (tn + fp) if (tn + fp) > 0 else 0,
            "FN Rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "TP Rate": tp / (fn + tp) if (fn + tp) > 0 else 0,
        })

    df = pd.DataFrame(confusion_data)

    logger.info("\nConfusion Matrix Report:")
    logger.info(f"\n{df.to_string(index=False)}")

    return df


def create_feature_importance_report(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Creates a formatted feature importance report.

    Args:
        feature_importance: DataFrame with feature importance scores
        top_n: Number of top features to include

    Returns:
        DataFrame with top N important features
    """
    if feature_importance.empty:
        logger.warning("Feature importance data is empty")
        return pd.DataFrame()

    # Get top N features
    top_features = feature_importance.head(top_n).copy()

    # Add rank and percentage
    top_features.insert(0, "Rank", range(1, len(top_features) + 1))
    total_importance = feature_importance["importance"].sum()
    top_features["Importance %"] = (
        top_features["importance"] / total_importance * 100
    ).round(2)
    top_features["Cumulative %"] = top_features["Importance %"].cumsum().round(2)

    # Rename columns
    top_features = top_features.rename(columns={
        "feature": "Feature",
        "importance": "Importance Score",
    })

    logger.info(f"\nTop {top_n} Feature Importance Report:")
    logger.info(f"\n{top_features.to_string(index=False)}")

    return top_features


def create_executive_summary(
    model_comparison: Dict[str, Any],
    feature_importance: pd.DataFrame,
) -> str:
    """Creates an executive summary report.

    Args:
        model_comparison: Dictionary containing model metrics
        feature_importance: DataFrame with feature importance scores

    Returns:
        String containing the executive summary
    """
    best_model_name = model_comparison["best_model"]
    selection_metric = model_comparison["selection_metric"]
    best_metrics = model_comparison["models"][best_model_name]

    # Get top 5 features
    top_features = feature_importance.head(5)["feature"].tolist()

    summary = f"""
================================================================================
                        CHURN PREDICTION MODEL REPORT
                        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

EXECUTIVE SUMMARY
-----------------

1. BEST PERFORMING MODEL: {best_metrics["model_name"]}
   Selected based on: {selection_metric.replace("_", " ").title()}

2. KEY PERFORMANCE METRICS:
   - Accuracy:  {best_metrics["accuracy"]:.2%}
   - Precision: {best_metrics["precision"]:.2%}
   - Recall:    {best_metrics["recall"]:.2%}
   - F1 Score:  {best_metrics["f1_score"]:.2%}
   - ROC-AUC:   {best_metrics["roc_auc"]:.4f}

3. CONFUSION MATRIX SUMMARY:
   - True Negatives (Correctly predicted No Churn):  {best_metrics["true_negatives"]}
   - True Positives (Correctly predicted Churn):     {best_metrics["true_positives"]}
   - False Positives (Incorrectly predicted Churn):  {best_metrics["false_positives"]}
   - False Negatives (Missed Churns):                {best_metrics["false_negatives"]}

4. TOP 5 CHURN PREDICTORS:
   {chr(10).join([f"   {i+1}. {feat}" for i, feat in enumerate(top_features)])}

5. MODEL COMPARISON SUMMARY:
"""

    # Add model comparison
    for model_name, metrics in model_comparison["models"].items():
        is_best = " (BEST)" if model_name == best_model_name else ""
        summary += f"""
   {metrics["model_name"]}{is_best}:
   - F1 Score: {metrics["f1_score"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}
"""

    summary += """
6. BUSINESS RECOMMENDATIONS:
   - Focus retention efforts on customers with high churn risk scores
   - Monitor customers with month-to-month contracts closely
   - Consider incentives for customers using electronic check payments
   - Promote security and protection services to reduce churn
   - Target new customers (< 12 months tenure) with engagement programs

================================================================================
                              END OF REPORT
================================================================================
"""

    # Save and log to MLflow
    output_dir = "data/08_reporting"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = f"{output_dir}/executive_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    mlflow.log_artifact(summary_path, artifact_path="reports")

    logger.info(summary)
    return summary


def create_detailed_metrics_report(
    model_comparison: Dict[str, Any],
) -> pd.DataFrame:
    """Creates a detailed metrics report with all statistics.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        DataFrame with detailed metrics for all models
    """
    models_data = model_comparison["models"]

    detailed_data = []
    for model_name, metrics in models_data.items():
        tn = metrics["true_negatives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        tp = metrics["true_positives"]

        total = tn + fp + fn + tp
        positives = tp + fn
        negatives = tn + fp

        # Calculate additional metrics
        specificity = tn / negatives if negatives > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / negatives if negatives > 0 else 0  # False Positive Rate
        fnr = fn / positives if positives > 0 else 0  # False Negative Rate

        detailed_data.append({
            "Model": metrics["model_name"],
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision (PPV)": round(metrics["precision"], 4),
            "Recall (Sensitivity)": round(metrics["recall"], 4),
            "Specificity": round(specificity, 4),
            "F1 Score": round(metrics["f1_score"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "NPV": round(npv, 4),
            "FPR": round(fpr, 4),
            "FNR": round(fnr, 4),
            "Support (Churn)": positives,
            "Support (No Churn)": negatives,
            "Total Samples": total,
        })

    df = pd.DataFrame(detailed_data)

    logger.info("\nDetailed Metrics Report:")
    logger.info(f"\n{df.to_string(index=False)}")

    return df


def plot_model_comparison(
    model_comparison: Dict[str, Any],
) -> Dict[str, str]:
    """Creates a bar chart comparing model performance metrics.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        Dictionary with plot path info
    """
    import os

    models_data = model_comparison["models"]

    # Prepare data
    models = []
    metrics_data = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "ROC-AUC": [],
    }

    for _, metrics in models_data.items():
        models.append(metrics["model_name"])
        metrics_data["Accuracy"].append(metrics["accuracy"])
        metrics_data["Precision"].append(metrics["precision"])
        metrics_data["Recall"].append(metrics["recall"])
        metrics_data["F1 Score"].append(metrics["f1_score"])
        metrics_data["ROC-AUC"].append(metrics["roc_auc"])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.15
    multiplier = 0

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    for i, (metric, values) in enumerate(metrics_data.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i])
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        multiplier += 1

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right', ncols=5)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()

    # Save figure
    output_dir = "data/08_reporting/plots"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/model_comparison.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(filepath, artifact_path="plots")

    logger.info(f"Model comparison plot saved to {filepath}")

    return {"plot_path": filepath, "plot_name": "model_comparison"}


def plot_confusion_matrices(
    model_comparison: Dict[str, Any],
) -> Dict[str, str]:
    """Creates confusion matrix heatmaps for all models.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        Dictionary with plot path info
    """
    import os

    models_data = model_comparison["models"]
    n_models = len(models_data)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for idx, (_, metrics) in enumerate(models_data.items()):
        cm = np.array([
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]]
        ])

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            annot_kws={'size': 14}
        )
        axes[idx].set_title(f'{metrics["model_name"]}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_dir = "data/08_reporting/plots"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/confusion_matrices.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(filepath, artifact_path="plots")

    logger.info(f"Confusion matrices plot saved to {filepath}")

    return {"plot_path": filepath, "plot_name": "confusion_matrices"}


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
) -> Dict[str, str]:
    """Creates a horizontal bar chart of feature importance.

    Args:
        feature_importance: DataFrame with feature importance scores
        top_n: Number of top features to display

    Returns:
        Dictionary with plot path info
    """
    import os

    if feature_importance.empty:
        logger.warning("Feature importance data is empty")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No feature importance data available',
                ha='center', va='center', fontsize=12)
    else:
        # Get top N features
        top_features = feature_importance.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

        bars = ax.barh(
            top_features['feature'],
            top_features['importance'],
            color=colors
        )

        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest importance at top

        # Add value labels
        for bar, val in zip(bars, top_features['importance']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_dir = "data/08_reporting/plots"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/feature_importance.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(filepath, artifact_path="plots")

    logger.info(f"Feature importance plot saved to {filepath}")

    return {"plot_path": filepath, "plot_name": "feature_importance"}


def plot_metrics_radar(
    model_comparison: Dict[str, Any],
) -> Dict[str, str]:
    """Creates a radar chart comparing model metrics.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        Dictionary with plot path info
    """
    import os

    models_data = model_comparison["models"]

    # Metrics to plot
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    num_metrics = len(metrics_names)

    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx, (_, metrics) in enumerate(models_data.items()):
        values = [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"],
        ]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=metrics["model_name"],
                color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Model Metrics Comparison (Radar)', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    # Save figure
    output_dir = "data/08_reporting/plots"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/metrics_radar.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(filepath, artifact_path="plots")

    logger.info(f"Metrics radar plot saved to {filepath}")

    return {"plot_path": filepath, "plot_name": "metrics_radar"}


def plot_churn_prediction_summary(
    model_comparison: Dict[str, Any],
) -> Dict[str, str]:
    """Creates a summary visualization with key metrics for the best model.

    Args:
        model_comparison: Dictionary containing model metrics

    Returns:
        Dictionary with plot path info
    """
    import os

    best_model_name = model_comparison["best_model"]
    best_metrics = model_comparison["models"][best_model_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Confusion Matrix for best model
    ax1 = axes[0, 0]
    cm = np.array([
        [best_metrics["true_negatives"], best_metrics["false_positives"]],
        [best_metrics["false_negatives"], best_metrics["true_positives"]]
    ])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                annot_kws={'size': 16})
    ax1.set_title(f'Best Model: {best_metrics["model_name"]}\nConfusion Matrix',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    # 2. Key Metrics Bar Chart
    ax2 = axes[0, 1]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metrics_values = [
        best_metrics["accuracy"],
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["f1_score"],
        best_metrics["roc_auc"],
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax2.bar(metrics_names, metrics_values, color=colors)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score')
    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2%}', ha='center', fontsize=10)

    # 3. Prediction Distribution (Pie Chart)
    ax3 = axes[1, 0]
    correct = best_metrics["true_negatives"] + best_metrics["true_positives"]
    incorrect = best_metrics["false_positives"] + best_metrics["false_negatives"]

    ax3.pie([correct, incorrect], labels=['Correct', 'Incorrect'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
            explode=(0.05, 0), startangle=90)
    ax3.set_title('Prediction Accuracy Distribution', fontsize=12, fontweight='bold')

    # 4. Model Comparison
    ax4 = axes[1, 1]
    models = []
    f1_scores = []
    roc_aucs = []
    for _, metrics in model_comparison["models"].items():
        models.append(metrics["model_name"].replace(" ", "\n"))
        f1_scores.append(metrics["f1_score"])
        roc_aucs.append(metrics["roc_auc"])

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax4.bar(x - width/2, f1_scores, width, label='F1 Score', color='#3498db')
    bars2 = ax4.bar(x + width/2, roc_aucs, width, label='ROC-AUC', color='#e74c3c')
    ax4.set_ylabel('Score')
    ax4.set_title('All Models Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.set_ylim(0, 1.0)

    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', fontsize=9)

    plt.suptitle('Churn Prediction Model Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_dir = "data/08_reporting/plots"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/churn_summary_dashboard.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(filepath, artifact_path="plots")

    logger.info(f"Churn prediction summary plot saved to {filepath}")

    return {"plot_path": filepath, "plot_name": "churn_summary_dashboard"}
