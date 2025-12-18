from kedro.pipeline import Pipeline, node

from .node import (
    split_data,
    apply_smote,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_voting_ensemble,
    train_xgboost,
    evaluate_model,
    select_best_model,
    get_feature_importance,
    XGBOOST_AVAILABLE,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the model training pipeline.

    This pipeline:
    1. Splits data into train/test sets
    2. Applies SMOTE to handle class imbalance (if enabled)
    3. Trains multiple models (LR, RF, GB, XGBoost if available, Voting Ensemble)
    4. Evaluates each model
    5. Selects the best model based on specified metric
    6. Extracts feature importance
    """
    # Base nodes that are always included
    base_nodes = [
        # Data splitting
        node(
            func=split_data,
            inputs=["features_engineered", "params:model_training"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        # Apply SMOTE for class imbalance
        node(
            func=apply_smote,
            inputs=["X_train", "y_train", "params:model_training"],
            outputs=["X_train_smote", "y_train_smote"],
            name="apply_smote_node",
        ),
        # Train Logistic Regression
        node(
            func=train_logistic_regression,
            inputs=["X_train_smote", "y_train_smote", "params:model_training"],
            outputs="logistic_regression_model",
            name="train_logistic_regression_node",
        ),
        # Train Random Forest
        node(
            func=train_random_forest,
            inputs=["X_train_smote", "y_train_smote", "params:model_training"],
            outputs="random_forest_model",
            name="train_random_forest_node",
        ),
        # Train Gradient Boosting
        node(
            func=train_gradient_boosting,
            inputs=["X_train_smote", "y_train_smote", "params:model_training"],
            outputs="gradient_boosting_model",
            name="train_gradient_boosting_node",
        ),
        # Train Voting Ensemble
        node(
            func=train_voting_ensemble,
            inputs=["X_train_smote", "y_train_smote", "params:model_training"],
            outputs="voting_ensemble_model",
            name="train_voting_ensemble_node",
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
        # Evaluate Voting Ensemble
        node(
            func=lambda model, X, y: evaluate_model(model, X, y, "Voting Ensemble"),
            inputs=["voting_ensemble_model", "X_test", "y_test"],
            outputs="ve_metrics",
            name="evaluate_voting_ensemble_node",
        ),
    ]

    # Add XGBoost nodes if available
    if XGBOOST_AVAILABLE:
        xgboost_nodes = [
            # Train XGBoost
            node(
                func=train_xgboost,
                inputs=["X_train_smote", "y_train_smote", "params:model_training"],
                outputs="xgboost_model",
                name="train_xgboost_node",
            ),
            # Evaluate XGBoost
            node(
                func=lambda model, X, y: evaluate_model(model, X, y, "XGBoost"),
                inputs=["xgboost_model", "X_test", "y_test"],
                outputs="xgb_metrics",
                name="evaluate_xgboost_node",
            ),
            # Select best model (with XGBoost)
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
                    "ve_metrics",
                    "voting_ensemble_model",
                    "xgb_metrics",
                    "xgboost_model",
                ],
                outputs=["best_model", "model_comparison"],
                name="select_best_model_node",
            ),
        ]
    else:
        xgboost_nodes = [
            # Select best model (without XGBoost)
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
                    "ve_metrics",
                    "voting_ensemble_model",
                ],
                outputs=["best_model", "model_comparison"],
                name="select_best_model_node",
            ),
        ]

    # Feature importance node
    final_nodes = [
        node(
            func=get_feature_importance,
            inputs=["best_model", "X_train_smote"],
            outputs="feature_importance",
            name="get_feature_importance_node",
        ),
    ]

    return Pipeline(base_nodes + xgboost_nodes + final_nodes)
