from kedro.pipeline import Pipeline, node

from .node import (
    split_data,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    select_best_model,
    get_feature_importance,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the model training pipeline.

    This pipeline:
    1. Splits data into train/test sets
    2. Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting)
    3. Evaluates each model
    4. Selects the best model based on specified metric
    5. Extracts feature importance
    """
    return Pipeline(
        [
            # Data splitting
            node(
                func=split_data,
                inputs=["features_engineered", "params:model_training"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            # Train Logistic Regression
            node(
                func=train_logistic_regression,
                inputs=["X_train", "y_train", "params:model_training"],
                outputs="logistic_regression_model",
                name="train_logistic_regression_node",
            ),
            # Train Random Forest
            node(
                func=train_random_forest,
                inputs=["X_train", "y_train", "params:model_training"],
                outputs="random_forest_model",
                name="train_random_forest_node",
            ),
            # Train Gradient Boosting
            node(
                func=train_gradient_boosting,
                inputs=["X_train", "y_train", "params:model_training"],
                outputs="gradient_boosting_model",
                name="train_gradient_boosting_node",
            ),
            # Evaluate Logistic Regression
            node(
                func=lambda model, X, y: evaluate_model(model, X, y, "Logistic Regression"),
                inputs=["logistic_regression_model", "X_test", "y_test"],
                outputs="lr_metrics",
                name="evaluate_logistic_regression_node",
            ),
            # Evaluate Random Forest
            node(
                func=lambda model, X, y: evaluate_model(model, X, y, "Random Forest"),
                inputs=["random_forest_model", "X_test", "y_test"],
                outputs="rf_metrics",
                name="evaluate_random_forest_node",
            ),
            # Evaluate Gradient Boosting
            node(
                func=lambda model, X, y: evaluate_model(model, X, y, "Gradient Boosting"),
                inputs=["gradient_boosting_model", "X_test", "y_test"],
                outputs="gb_metrics",
                name="evaluate_gradient_boosting_node",
            ),
            # Select best model
            node(
                func=select_best_model,
                inputs=[
                    "lr_metrics",
                    "rf_metrics",
                    "gb_metrics",
                    "logistic_regression_model",
                    "random_forest_model",
                    "gradient_boosting_model",
                    "params:model_training",
                ],
                outputs=["best_model", "model_comparison"],
                name="select_best_model_node",
            ),
            # Get feature importance
            node(
                func=get_feature_importance,
                inputs=["best_model", "X_train"],
                outputs="feature_importance",
                name="get_feature_importance_node",
            ),
        ]
    )
