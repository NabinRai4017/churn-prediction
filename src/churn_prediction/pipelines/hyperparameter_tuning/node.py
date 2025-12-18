"""Node functions for hyperparameter tuning pipeline."""

import logging
from typing import Any, Callable, Dict, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def identify_best_model(model_comparison: Dict[str, Any]) -> str:
    """Identifies the best model from baseline comparison.

    Args:
        model_comparison: Dictionary containing model comparison results
            from the model_training pipeline.

    Returns:
        Name of the best performing model (e.g., "random_forest").
    """
    best_model_name = model_comparison["best_model"]
    selection_metric = model_comparison["selection_metric"]
    best_score = model_comparison["models"][best_model_name][selection_metric]

    logger.info(f"Best baseline model: {best_model_name}")
    logger.info(f"Selection metric: {selection_metric} = {best_score:.4f}")

    return best_model_name


def _suggest_hyperparameters(
    trial: optuna.Trial,
    model_type: str,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Suggests hyperparameters for a given model type.

    Args:
        trial: Optuna trial object.
        model_type: Type of model (logistic_regression, random_forest, gradient_boosting).
        search_space: Dictionary defining the search space for hyperparameters.

    Returns:
        Dictionary of suggested hyperparameters.
    """
    params = {}

    for param_name, param_config in search_space.items():
        param_type = param_config["type"]

        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                param_config["low"],
                param_config["high"],
                log=param_config.get("log", False),
            )
        elif param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                param_config["low"],
                param_config["high"],
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config["choices"],
            )

    return params


def _create_model(model_type: str, params: Dict[str, Any], random_state: int) -> Any:
    """Creates a model instance with given hyperparameters.

    Args:
        model_type: Type of model to create.
        params: Hyperparameters for the model.
        random_state: Random state for reproducibility.

    Returns:
        Sklearn model instance.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            **params,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _get_scoring_metric(metric_name: str) -> str:
    """Maps metric name to sklearn scoring parameter.

    Args:
        metric_name: Name of the metric.

    Returns:
        Sklearn scoring string.
    """
    mapping = {
        "f1_score": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    return mapping.get(metric_name, "f1")


def run_optuna_study(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """Runs Optuna hyperparameter optimization study.

    Args:
        best_model_name: Name of the best baseline model to tune.
        X_train: Training features.
        y_train: Training target.
        parameters: Hyperparameter tuning configuration.

    Returns:
        Tuple of (Optuna study object, best hyperparameters dictionary).
    """
    n_trials = parameters.get("n_trials", 50)
    cv_folds = parameters.get("cv_folds", 5)
    random_state = parameters.get("random_state", 42)
    optimization_metric = parameters.get("optimization_metric", "f1_score")
    direction = parameters.get("direction", "maximize")
    pruning = parameters.get("pruning", True)
    timeout = parameters.get("timeout", None)

    # Get search space for the specific model
    search_space = parameters.get(best_model_name, {})

    if not search_space:
        raise ValueError(f"No search space defined for model: {best_model_name}")

    logger.info(f"Starting Optuna optimization for {best_model_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Cross-validation folds: {cv_folds}")
    logger.info(f"Optimization metric: {optimization_metric}")

    # Convert y_train to 1D array
    y_train_array = y_train.values.ravel() if hasattr(y_train, "values") else y_train.ravel()

    # Create objective function
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = _suggest_hyperparameters(trial, best_model_name, search_space)

        # Create model
        model = _create_model(best_model_name, params, random_state)

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = _get_scoring_metric(optimization_metric)

        try:
            scores = cross_val_score(
                model,
                X_train,
                y_train_array,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            mean_score = scores.mean()

            # Log to trial for pruning
            trial.report(mean_score, step=0)

            if pruning and trial.should_prune():
                raise optuna.TrialPruned()

            return mean_score

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"tune_{best_model_name}",
    )

    # Log study start to MLflow
    mlflow.log_param("tuning_model", best_model_name)
    mlflow.log_param("tuning_n_trials", n_trials)
    mlflow.log_param("tuning_cv_folds", cv_folds)
    mlflow.log_param("tuning_metric", optimization_metric)

    # Configure MLflow callback (with graceful fallback)
    callbacks = []
    try:
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri and not tracking_uri.startswith("mock"):
            mlflow_callback = MLflowCallback(
                tracking_uri=tracking_uri,
                metric_name=optimization_metric,
                create_experiment=False,
                mlflow_kwargs={"nested": True},
            )
            callbacks.append(mlflow_callback)
    except Exception as e:
        logger.warning(f"MLflow callback setup failed, running without trial logging: {e}")

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks if callbacks else None,
        show_progress_bar=True,
    )

    # Log best trial results
    best_trial = study.best_trial
    best_params = best_trial.params

    logger.info(f"Optimization complete!")
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best {optimization_metric}: {best_trial.value:.4f}")
    logger.info(f"Best params: {best_params}")

    # Log best results to MLflow
    mlflow.log_metric(f"best_cv_{optimization_metric}", best_trial.value)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    return study, best_params


def train_tuned_model(
    best_model_name: str,
    best_hyperparameters: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Any:
    """Trains the final model with optimized hyperparameters.

    Args:
        best_model_name: Name of the model to train.
        best_hyperparameters: Optimized hyperparameters from Optuna.
        X_train: Training features.
        y_train: Training target.
        parameters: Additional parameters (random_state).

    Returns:
        Trained sklearn model.
    """
    random_state = parameters.get("random_state", 42)

    logger.info(f"Training final {best_model_name} with tuned hyperparameters")
    logger.info(f"Hyperparameters: {best_hyperparameters}")

    # Create and train model
    model = _create_model(best_model_name, best_hyperparameters, random_state)

    y_train_array = y_train.values.ravel() if hasattr(y_train, "values") else y_train.ravel()
    model.fit(X_train, y_train_array)

    # Log model to MLflow
    mlflow.log_params({f"tuned_{k}": v for k, v in best_hyperparameters.items()})
    mlflow.sklearn.log_model(
        model,
        artifact_path="tuned_model",
        registered_model_name=f"churn_prediction_tuned_{best_model_name}",
    )

    logger.info("Tuned model training complete")

    return model


def evaluate_tuned_model(
    tuned_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    best_model_name: str,
) -> Dict[str, Any]:
    """Evaluates the tuned model on the test set.

    Args:
        tuned_model: Trained model with optimized hyperparameters.
        X_test: Test features.
        y_test: Test target.
        best_model_name: Name of the model.

    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("Evaluating tuned model on test set")

    y_test_array = y_test.values.ravel() if hasattr(y_test, "values") else y_test.ravel()

    # Make predictions
    y_pred = tuned_model.predict(X_test)
    y_pred_proba = tuned_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "model_name": f"Tuned {best_model_name}",
        "accuracy": float(accuracy_score(y_test_array, y_pred)),
        "precision": float(precision_score(y_test_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test_array, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test_array, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_array, y_pred_proba)),
    }

    # Log metrics to MLflow
    mlflow.log_metrics({
        "tuned_accuracy": metrics["accuracy"],
        "tuned_precision": metrics["precision"],
        "tuned_recall": metrics["recall"],
        "tuned_f1_score": metrics["f1_score"],
        "tuned_roc_auc": metrics["roc_auc"],
    })

    logger.info(f"Tuned model metrics:")
    for metric_name, value in metrics.items():
        if metric_name != "model_name":
            logger.info(f"  {metric_name}: {value:.4f}")

    return metrics


def create_tuning_report(
    optuna_study: optuna.Study,
    tuned_model_metrics: Dict[str, Any],
    model_comparison: Dict[str, Any],
    best_model_name: str,
) -> Dict[str, Any]:
    """Creates a comprehensive tuning report.

    Args:
        optuna_study: Completed Optuna study.
        tuned_model_metrics: Metrics from the tuned model.
        model_comparison: Baseline model comparison results.
        best_model_name: Name of the tuned model.

    Returns:
        Dictionary containing the tuning report.
    """
    logger.info("Creating hyperparameter tuning report")

    # Get baseline metrics
    baseline_metrics = model_comparison["models"][best_model_name]
    optimization_metric = model_comparison["selection_metric"]

    # Calculate improvement
    baseline_score = baseline_metrics[optimization_metric]
    tuned_score = tuned_model_metrics[optimization_metric]
    improvement = tuned_score - baseline_score
    improvement_pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0

    # Get study statistics
    n_trials = len(optuna_study.trials)
    n_completed = len([t for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED])

    # Create report
    report = {
        "model_name": best_model_name,
        "optimization_metric": optimization_metric,
        "baseline_score": baseline_score,
        "tuned_score": tuned_score,
        "improvement": improvement,
        "improvement_percentage": improvement_pct,
        "best_hyperparameters": optuna_study.best_params,
        "study_statistics": {
            "total_trials": n_trials,
            "completed_trials": n_completed,
            "pruned_trials": n_pruned,
            "best_trial_number": optuna_study.best_trial.number,
            "best_cv_score": optuna_study.best_value,
        },
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_model_metrics,
        "optimization_history": [
            {"trial": t.number, "value": t.value, "state": str(t.state)}
            for t in optuna_study.trials
            if t.value is not None
        ][:20],  # Keep top 20 for brevity
    }

    # Log summary to MLflow
    mlflow.log_metric("improvement", improvement)
    mlflow.log_metric("improvement_percentage", improvement_pct)

    logger.info(f"Tuning Report Summary:")
    logger.info(f"  Model: {best_model_name}")
    logger.info(f"  Baseline {optimization_metric}: {baseline_score:.4f}")
    logger.info(f"  Tuned {optimization_metric}: {tuned_score:.4f}")
    logger.info(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
    logger.info(f"  Total trials: {n_trials} (completed: {n_completed}, pruned: {n_pruned})")

    return report
