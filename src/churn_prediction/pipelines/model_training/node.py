import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import pandera as pa
from pandera.typing import DataFrame
import logging

# Try to import XGBoost, set flag for availability
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    # Catches ImportError and XGBoostError (library loading failures)
    XGBOOST_AVAILABLE = False
    XGBClassifier = None  # type: ignore

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


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies SMOTE to handle class imbalance in training data.

    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
    samples for the minority class to balance the dataset.

    Args:
        X_train: Training features
        y_train: Training target (DataFrame with single column)
        parameters: Parameters including random_state and smote settings

    Returns:
        Tuple of (X_train_resampled, y_train_resampled) as DataFrames
    """
    random_state = parameters.get("random_state", 42)
    smote_params = parameters.get("smote", {})

    # Check if SMOTE is enabled
    if not smote_params.get("enabled", True):
        logger.info("SMOTE is disabled, returning original data")
        return X_train, y_train

    sampling_strategy = smote_params.get("sampling_strategy", "auto")
    k_neighbors = smote_params.get("k_neighbors", 5)

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Log original class distribution
    original_dist = y.value_counts().to_dict()
    logger.info(f"Original class distribution: {original_dist}")

    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y)

    # Convert back to DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=y_train.columns)

    # Log resampled distribution
    resampled_dist = y_resampled_df.iloc[:, 0].value_counts().to_dict()
    logger.info(f"Resampled class distribution: {resampled_dist}")
    logger.info(f"Training samples increased from {len(X_train)} to {len(X_resampled_df)}")

    # Log SMOTE info to MLflow
    mlflow.log_params({
        "smote_enabled": True,
        "smote_sampling_strategy": str(sampling_strategy),
        "smote_k_neighbors": k_neighbors,
        "smote_original_samples": len(X_train),
        "smote_resampled_samples": len(X_resampled_df),
    })

    return X_resampled_df, y_resampled_df


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Selects top K features using statistical tests.

    Uses SelectKBest with ANOVA F-test to identify the most
    discriminative features for classification.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        parameters: Parameters including feature selection settings

    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_feature_names)
    """
    feature_selection_params = parameters.get("feature_selection", {})

    # Check if feature selection is enabled
    if not feature_selection_params.get("enabled", False):
        logger.info("Feature selection is disabled, using all features")
        return X_train, X_test, list(X_train.columns)

    k_features = feature_selection_params.get("k_features", 20)

    # Ensure k is not greater than number of features
    k_features = min(k_features, X_train.shape[1])

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Apply SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    # Convert back to DataFrame
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

    logger.info(f"Selected {k_features} features out of {X_train.shape[1]}")
    logger.info(f"Top selected features: {selected_features[:10]}")

    # Log feature selection info to MLflow
    mlflow.log_params({
        "feature_selection_enabled": True,
        "feature_selection_k": k_features,
        "feature_selection_original": X_train.shape[1],
    })

    return X_train_selected_df, X_test_selected_df, selected_features


def train_voting_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> VotingClassifier:
    """Trains a Voting Ensemble combining multiple classifiers.

    Combines Logistic Regression, Random Forest, Gradient Boosting,
    and XGBoost (if available) using soft voting for better predictions.

    Args:
        X_train: Training features
        y_train: Training target (DataFrame with single column)
        parameters: Model parameters for individual classifiers

    Returns:
        Trained VotingClassifier model
    """
    random_state = parameters.get("random_state", 42)
    ensemble_params = parameters.get("voting_ensemble", {})

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Get individual model parameters
    lr_params = parameters.get("logistic_regression", {})
    rf_params = parameters.get("random_forest", {})
    gb_params = parameters.get("gradient_boosting", {})
    xgb_params = parameters.get("xgboost", {})

    # Create base estimators
    lr = LogisticRegression(
        C=lr_params.get("C", 1.0),
        max_iter=lr_params.get("max_iter", 1000),
        class_weight=lr_params.get("class_weight", "balanced"),
        random_state=random_state,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", 10),
        min_samples_split=rf_params.get("min_samples_split", 2),
        min_samples_leaf=rf_params.get("min_samples_leaf", 1),
        class_weight=rf_params.get("class_weight", "balanced"),
        random_state=random_state,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=gb_params.get("n_estimators", 100),
        learning_rate=gb_params.get("learning_rate", 0.1),
        max_depth=gb_params.get("max_depth", 3),
        min_samples_split=gb_params.get("min_samples_split", 2),
        min_samples_leaf=gb_params.get("min_samples_leaf", 1),
        random_state=random_state,
    )

    # Build estimators list
    estimators = [
        ("logistic_regression", lr),
        ("random_forest", rf),
        ("gradient_boosting", gb),
    ]

    # Get voting type and weights
    voting = ensemble_params.get("voting", "soft")
    weights = ensemble_params.get("weights", [1, 1, 1])

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 100),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            max_depth=xgb_params.get("max_depth", 6),
            min_child_weight=xgb_params.get("min_child_weight", 1),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        estimators.append(("xgboost", xgb))
        # Extend weights for XGBoost
        if len(weights) == 3:
            weights = weights + [1]
        logger.info("XGBoost added to Voting Ensemble")
    else:
        logger.warning("XGBoost not available, Voting Ensemble will use 3 models")

    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )

    # Train the ensemble
    ensemble.fit(X_train, y)

    # Log parameters to MLflow
    mlflow.log_params({
        "ensemble_voting": voting,
        "ensemble_weights": str(weights),
        "ensemble_n_estimators": len(estimators),
        "ensemble_xgboost_included": XGBOOST_AVAILABLE,
    })

    logger.info(f"Voting Ensemble trained with {voting} voting and {len(estimators)} models")

    return ensemble


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


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]
) -> Any:
    """Trains an XGBoost model with MLflow logging.

    XGBoost handles class imbalance via scale_pos_weight parameter,
    which is calculated as the ratio of negative to positive samples.

    Args:
        X_train: Training features
        y_train: Training target (DataFrame with single column)
        parameters: Model parameters

    Returns:
        Trained XGBClassifier model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is not installed. Please install it with: pip install xgboost"
        )

    model_params = parameters.get("xgboost", {})

    # Extract series from DataFrame
    y = y_train.iloc[:, 0]

    # Calculate scale_pos_weight for class imbalance
    # ratio of negative to positive samples
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Get parameters
    n_estimators = model_params.get("n_estimators", 100)
    learning_rate = model_params.get("learning_rate", 0.1)
    max_depth = model_params.get("max_depth", 6)
    min_child_weight = model_params.get("min_child_weight", 1)
    subsample = model_params.get("subsample", 0.8)
    colsample_bytree = model_params.get("colsample_bytree", 0.8)
    random_state = parameters.get("random_state", 42)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    model.fit(X_train, y)

    # Log parameters to MLflow
    mlflow.log_params({
        "xgb_n_estimators": n_estimators,
        "xgb_learning_rate": learning_rate,
        "xgb_max_depth": max_depth,
        "xgb_min_child_weight": min_child_weight,
        "xgb_subsample": subsample,
        "xgb_colsample_bytree": colsample_bytree,
        "xgb_scale_pos_weight": scale_pos_weight,
    })

    logger.info(f"XGBoost model trained successfully (scale_pos_weight={scale_pos_weight:.2f})")

    return model


def find_optimal_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, optimize_for: str = "recall"
) -> Tuple[float, Dict[str, float]]:
    """Finds the optimal classification threshold for a given metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        optimize_for: Metric to optimize ('recall', 'f1_score', 'precision')

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate F1 for each threshold
    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (
        precision_arr[:-1] + recall_arr[:-1] + 1e-10
    )

    if optimize_for == "recall":
        # Find threshold that achieves high recall (e.g., >0.8) with best precision
        target_recall = 0.80
        valid_indices = recall_arr[:-1] >= target_recall
        if valid_indices.any():
            # Among thresholds with recall >= target, find one with best precision
            valid_f1 = f1_scores.copy()
            valid_f1[~valid_indices] = 0
            best_idx = np.argmax(valid_f1)
        else:
            # Fallback to highest recall
            best_idx = np.argmax(recall_arr[:-1])
    elif optimize_for == "f1_score":
        best_idx = np.argmax(f1_scores)
    elif optimize_for == "precision":
        best_idx = np.argmax(precision_arr[:-1])
    else:
        best_idx = np.argmax(f1_scores)

    optimal_threshold = thresholds[best_idx]

    metrics = {
        "threshold": optimal_threshold,
        "precision": precision_arr[best_idx],
        "recall": recall_arr[best_idx],
        "f1_score": f1_scores[best_idx],
    }

    return optimal_threshold, metrics


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
    ve_metrics: Dict[str, Any] = None,
    ve_model: VotingClassifier = None,
    xgb_metrics: Dict[str, Any] = None,
    xgb_model: Any = None,
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
        ve_metrics: Voting Ensemble evaluation metrics (optional)
        ve_model: Trained Voting Ensemble model (optional)
        xgb_metrics: XGBoost evaluation metrics (optional)
        xgb_model: Trained XGBoost model (optional)

    Returns:
        Tuple of (best_model, all_metrics_summary)
    """
    selection_metric = parameters.get("selection_metric", "f1_score")

    models = {
        "logistic_regression": (lr_model, lr_metrics),
        "random_forest": (rf_model, rf_metrics),
        "gradient_boosting": (gb_model, gb_metrics),
    }

    # Add voting ensemble if provided
    if ve_metrics is not None and ve_model is not None:
        models["voting_ensemble"] = (ve_model, ve_metrics)

    # Add XGBoost if provided
    if xgb_metrics is not None and xgb_model is not None:
        models["xgboost"] = (xgb_model, xgb_metrics)

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
    if ve_model is not None:
        mlflow.sklearn.log_model(ve_model, artifact_path="voting_ensemble_model")
    if xgb_model is not None:
        mlflow.sklearn.log_model(xgb_model, artifact_path="xgboost_model")

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
