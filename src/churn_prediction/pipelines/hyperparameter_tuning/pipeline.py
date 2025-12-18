"""Hyperparameter tuning pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .node import (
    create_tuning_report,
    evaluate_tuned_model,
    identify_best_model,
    run_optuna_study,
    train_tuned_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the hyperparameter tuning pipeline.

    This pipeline tunes the best-performing model from baseline comparison
    using Optuna optimization with MLflow tracking.

    Returns:
        Pipeline: The hyperparameter tuning pipeline.
    """
    return pipeline(
        [
            node(
                func=identify_best_model,
                inputs="model_comparison",
                outputs="best_model_name",
                name="identify_best_model_node",
            ),
            node(
                func=run_optuna_study,
                inputs=[
                    "best_model_name",
                    "X_train",
                    "y_train",
                    "params:hyperparameter_tuning",
                ],
                outputs=["optuna_study", "best_hyperparameters"],
                name="run_optuna_study_node",
            ),
            node(
                func=train_tuned_model,
                inputs=[
                    "best_model_name",
                    "best_hyperparameters",
                    "X_train",
                    "y_train",
                    "params:hyperparameter_tuning",
                ],
                outputs="tuned_model",
                name="train_tuned_model_node",
            ),
            node(
                func=evaluate_tuned_model,
                inputs=[
                    "tuned_model",
                    "X_test",
                    "y_test",
                    "best_model_name",
                ],
                outputs="tuned_model_metrics",
                name="evaluate_tuned_model_node",
            ),
            node(
                func=create_tuning_report,
                inputs=[
                    "optuna_study",
                    "tuned_model_metrics",
                    "model_comparison",
                    "best_model_name",
                ],
                outputs="tuning_report",
                name="create_tuning_report_node",
            ),
        ],
        tags=["hyperparameter_tuning"],
    )
