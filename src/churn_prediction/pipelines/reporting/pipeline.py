from kedro.pipeline import Pipeline, node

from .node import (
    create_model_comparison_report,
    create_confusion_matrix_report,
    create_feature_importance_report,
    create_executive_summary,
    create_detailed_metrics_report,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metrics_radar,
    plot_churn_prediction_summary,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the reporting pipeline.

    This pipeline generates the following reports and visualizations:

    Reports:
    1. Model comparison report (metrics table)
    2. Confusion matrix report
    3. Feature importance report (top 20)
    4. Detailed metrics report
    5. Executive summary (text report)

    Visualizations:
    1. Model comparison bar chart
    2. Confusion matrices heatmaps
    3. Feature importance bar chart
    4. Metrics radar chart
    5. Churn prediction summary dashboard
    """
    return Pipeline(
        [
            # ============= REPORTS =============
            node(
                func=create_model_comparison_report,
                inputs="model_comparison",
                outputs="model_comparison_report",
                name="create_model_comparison_report_node",
            ),
            node(
                func=create_confusion_matrix_report,
                inputs="model_comparison",
                outputs="confusion_matrix_report",
                name="create_confusion_matrix_report_node",
            ),
            node(
                func=create_feature_importance_report,
                inputs="feature_importance",
                outputs="feature_importance_report",
                name="create_feature_importance_report_node",
            ),
            node(
                func=create_detailed_metrics_report,
                inputs="model_comparison",
                outputs="detailed_metrics_report",
                name="create_detailed_metrics_report_node",
            ),
            node(
                func=create_executive_summary,
                inputs=["model_comparison", "feature_importance"],
                outputs="executive_summary",
                name="create_executive_summary_node",
            ),
            # ============= VISUALIZATIONS =============
            node(
                func=plot_model_comparison,
                inputs="model_comparison",
                outputs="model_comparison_plot",
                name="plot_model_comparison_node",
            ),
            node(
                func=plot_confusion_matrices,
                inputs="model_comparison",
                outputs="confusion_matrices_plot",
                name="plot_confusion_matrices_node",
            ),
            node(
                func=plot_feature_importance,
                inputs="feature_importance",
                outputs="feature_importance_plot",
                name="plot_feature_importance_node",
            ),
            node(
                func=plot_metrics_radar,
                inputs="model_comparison",
                outputs="metrics_radar_plot",
                name="plot_metrics_radar_node",
            ),
            node(
                func=plot_churn_prediction_summary,
                inputs="model_comparison",
                outputs="churn_summary_plot",
                name="plot_churn_summary_node",
            ),
        ]
    )
