import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
import pandera as pa
from pandera.typing import DataFrame
import logging

from churn_prediction.schemas import (
    FeaturesEngineeredSchema,
    ModelInputSchema,
    TargetSchema,
)

logger = logging.getLogger(__name__)


@pa.check_types
def split_data(
    data: DataFrame[FeaturesEngineeredSchema], parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into training and test sets.

    Args:
        data: DataFrame with features and target (validated against FeaturesEngineeredSchema)
        parameters: Dictionary containing:
            - target_column: Name of target column
            - test_size: Fraction of data for testing
            - random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as DataFrames
    """
    target_column = parameters["target_column"]
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Target distribution in train: {y_train.value_counts(normalize=True).to_dict()}")

    # Log data split info to MLflow
    mlflow.log_params({
        "data_split_test_size": test_size,
        "data_split_random_state": random_state,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
    })

    # Convert Series to DataFrame for Parquet compatibility
    y_train_df = y_train.to_frame()
    y_test_df = y_test.to_frame()

    # Validate outputs
    ModelInputSchema.validate(X_train)
    ModelInputSchema.validate(X_test)
    TargetSchema.validate(y_train_df)
    TargetSchema.validate(y_test_df)

    logger.info("Data split validation passed")
    return X_train, X_test, y_train_df, y_test_df


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]
) -> LogisticRegression:
    """Trains a Logistic Regression model with MLflow logging.

    Args:
        X_train: Training features
        y_train: Training target (DataFrame with single column)
        parameters: Model parameters including class_weight

    Returns:
        Trained LogisticRegression model
    """
    model_params = parameters.get("logistic_regression", {})

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Get parameters
    C = model_params.get("C", 1.0)
    max_iter = model_params.get("max_iter", 1000)
    class_weight = model_params.get("class_weight", "balanced")
    random_state = parameters.get("random_state", 42)

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )

    model.fit(X_train, y)

    # Log parameters to MLflow
    mlflow.log_params({
        "lr_C": C,
        "lr_max_iter": max_iter,
        "lr_class_weight": str(class_weight),
    })

    logger.info("Logistic Regression model trained successfully")

    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]
) -> RandomForestClassifier:
    """Trains a Random Forest model with MLflow logging.

    Args:
        X_train: Training features
        y_train: Training target (DataFrame with single column)
        parameters: Model parameters

    Returns:
        Trained RandomForestClassifier model
    """
    model_params = parameters.get("random_forest", {})

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Get parameters
    n_estimators = model_params.get("n_estimators", 100)
    max_depth = model_params.get("max_depth", 10)
    min_samples_split = model_params.get("min_samples_split", 2)
    min_samples_leaf = model_params.get("min_samples_leaf", 1)
    class_weight = model_params.get("class_weight", "balanced")
    random_state = parameters.get("random_state", 42)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y)

    # Log parameters to MLflow
    mlflow.log_params({
        "rf_n_estimators": n_estimators,
        "rf_max_depth": max_depth,
        "rf_min_samples_split": min_samples_split,
        "rf_min_samples_leaf": min_samples_leaf,
        "rf_class_weight": str(class_weight),
    })

    logger.info("Random Forest model trained successfully")

    return model


def train_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]
) -> GradientBoostingClassifier:
    """Trains a Gradient Boosting model with MLflow logging.

    Args:
        X_train: Training features
        y_train: Training target (DataFrame with single column)
        parameters: Model parameters

    Returns:
        Trained GradientBoostingClassifier model
    """
    model_params = parameters.get("gradient_boosting", {})

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Get parameters
    n_estimators = model_params.get("n_estimators", 100)
    learning_rate = model_params.get("learning_rate", 0.1)
    max_depth = model_params.get("max_depth", 3)
    min_samples_split = model_params.get("min_samples_split", 2)
    min_samples_leaf = model_params.get("min_samples_leaf", 1)
    random_state = parameters.get("random_state", 42)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    model.fit(X_train, y)

    # Log parameters to MLflow
    mlflow.log_params({
        "gb_n_estimators": n_estimators,
        "gb_learning_rate": learning_rate,
        "gb_max_depth": max_depth,
        "gb_min_samples_split": min_samples_split,
        "gb_min_samples_leaf": min_samples_leaf,
    })

    logger.info("Gradient Boosting model trained successfully")

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluates a trained model on test data with MLflow logging.

    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test target (DataFrame with single column)
        model_name: Name of the model for logging

    Returns:
        Dictionary containing evaluation metrics
    """
    # Extract series from DataFrame
    y_true = y_test.iloc[:, 0]

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    # Create prefix for MLflow metrics
    prefix = model_name.lower().replace(" ", "_")

    # Log metrics to MLflow
    mlflow.log_metrics({
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_f1_score": metrics["f1_score"],
        f"{prefix}_roc_auc": metrics["roc_auc"],
    })

    logger.info(f"\n{model_name} Evaluation Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def select_best_model(
    lr_metrics: Dict[str, Any],
    rf_metrics: Dict[str, Any],
    gb_metrics: Dict[str, Any],
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier,
    gb_model: GradientBoostingClassifier,
    parameters: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Selects the best model and logs it to MLflow.

    Args:
        lr_metrics: Logistic Regression evaluation metrics
        rf_metrics: Random Forest evaluation metrics
        gb_metrics: Gradient Boosting evaluation metrics
        lr_model: Trained Logistic Regression model
        rf_model: Trained Random Forest model
        gb_model: Trained Gradient Boosting model
        parameters: Parameters containing selection_metric

    Returns:
        Tuple of (best_model, all_metrics_summary)
    """
    selection_metric = parameters.get("selection_metric", "f1_score")

    models = {
        "logistic_regression": (lr_model, lr_metrics),
        "random_forest": (rf_model, rf_metrics),
        "gradient_boosting": (gb_model, gb_metrics),
    }

    # Find best model
    best_model_name = max(
        models.keys(), key=lambda x: models[x][1][selection_metric]
    )
    best_model = models[best_model_name][0]
    best_metrics = models[best_model_name][1]

    # Create summary
    summary = {
        "selection_metric": selection_metric,
        "best_model": best_model_name,
        "models": {
            name: metrics for name, (_, metrics) in models.items()
        },
    }

    # Log best model info to MLflow
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_param("selection_metric", selection_metric)

    # Log best model metrics with 'best_' prefix
    mlflow.log_metrics({
        "best_accuracy": best_metrics["accuracy"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_f1_score": best_metrics["f1_score"],
        "best_roc_auc": best_metrics["roc_auc"],
    })

    # Log the best model to MLflow
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name="churn_prediction_model",
    )

    # Also log all individual models
    mlflow.sklearn.log_model(lr_model, artifact_path="logistic_regression_model")
    mlflow.sklearn.log_model(rf_model, artifact_path="random_forest_model")
    mlflow.sklearn.log_model(gb_model, artifact_path="gradient_boosting_model")

    logger.info(f"\nBest model: {best_model_name} (based on {selection_metric})")
    logger.info(f"  {selection_metric}: {best_metrics[selection_metric]:.4f}")
    logger.info("Models logged to MLflow")

    return best_model, summary


def get_feature_importance(
    model: Any, X_train: pd.DataFrame
) -> pd.DataFrame:
    """Extracts feature importance and logs to MLflow.

    Args:
        model: Trained model
        X_train: Training features (for column names)

    Returns:
        DataFrame with feature importance scores
    """
    feature_names = X_train.columns.tolist()

    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model doesn't have feature importance attribute")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Log top features to MLflow
    top_features = importance_df.head(10)
    for idx, row in top_features.iterrows():
        mlflow.log_metric(f"feature_importance_{row['feature'][:50]}", row["importance"])

    # Log feature importance as artifact
    importance_path = "data/07_model_output/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path, artifact_path="feature_importance")

    logger.info("\nTop 10 Important Features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df
