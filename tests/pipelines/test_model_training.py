"""Unit tests for the model training pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from churn_prediction.pipelines.model_training.node import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    select_best_model,
    get_feature_importance,
)


@pytest.fixture
def simple_train_data():
    """Simple training data for model tests."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
    })
    y = pd.DataFrame({
        "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    return X, y


@pytest.fixture
def simple_test_data():
    """Simple test data for evaluation tests."""
    np.random.seed(123)
    n_samples = 30

    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
    })
    y = pd.DataFrame({
        "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    return X, y


class TestTrainLogisticRegression:
    """Tests for train_logistic_regression function."""

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_returns_logistic_regression_model(self, mock_mlflow, simple_train_data):
        """Test that function returns a LogisticRegression model."""
        X_train, y_train = simple_train_data
        parameters = {
            "logistic_regression": {"C": 1.0, "max_iter": 100, "class_weight": "balanced"},
            "random_state": 42,
        }
        model = train_logistic_regression(X_train, y_train, parameters)
        assert isinstance(model, LogisticRegression)

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_model_can_predict(self, mock_mlflow, simple_train_data):
        """Test that the trained model can make predictions."""
        X_train, y_train = simple_train_data
        parameters = {
            "logistic_regression": {"C": 1.0, "max_iter": 100, "class_weight": "balanced"},
            "random_state": 42,
        }
        model = train_logistic_regression(X_train, y_train, parameters)
        predictions = model.predict(X_train)
        assert len(predictions) == len(X_train)
        assert set(predictions).issubset({0, 1})

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_logs_parameters_to_mlflow(self, mock_mlflow, simple_train_data):
        """Test that parameters are logged to MLflow."""
        X_train, y_train = simple_train_data
        parameters = {
            "logistic_regression": {"C": 1.0, "max_iter": 100, "class_weight": "balanced"},
            "random_state": 42,
        }
        train_logistic_regression(X_train, y_train, parameters)
        mock_mlflow.log_params.assert_called_once()


class TestTrainRandomForest:
    """Tests for train_random_forest function."""

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_returns_random_forest_model(self, mock_mlflow, simple_train_data):
        """Test that function returns a RandomForestClassifier model."""
        X_train, y_train = simple_train_data
        parameters = {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
            "random_state": 42,
        }
        model = train_random_forest(X_train, y_train, parameters)
        assert isinstance(model, RandomForestClassifier)

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_model_has_feature_importances(self, mock_mlflow, simple_train_data):
        """Test that the trained model has feature_importances_ attribute."""
        X_train, y_train = simple_train_data
        parameters = {
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
            "random_state": 42,
        }
        model = train_random_forest(X_train, y_train, parameters)
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X_train.shape[1]


class TestTrainGradientBoosting:
    """Tests for train_gradient_boosting function."""

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_returns_gradient_boosting_model(self, mock_mlflow, simple_train_data):
        """Test that function returns a GradientBoostingClassifier model."""
        X_train, y_train = simple_train_data
        parameters = {
            "gradient_boosting": {
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            },
            "random_state": 42,
        }
        model = train_gradient_boosting(X_train, y_train, parameters)
        assert isinstance(model, GradientBoostingClassifier)

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_model_can_predict_proba(self, mock_mlflow, simple_train_data):
        """Test that the trained model can predict probabilities."""
        X_train, y_train = simple_train_data
        parameters = {
            "gradient_boosting": {
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            },
            "random_state": 42,
        }
        model = train_gradient_boosting(X_train, y_train, parameters)
        proba = model.predict_proba(X_train)
        assert proba.shape == (len(X_train), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_returns_metrics_dict(self, mock_mlflow, simple_train_data, simple_test_data):
        """Test that function returns a dictionary with expected metrics."""
        X_train, y_train = simple_train_data
        X_test, y_test = simple_test_data

        # Train a simple model
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        metrics = evaluate_model(model, X_test, y_test, "Test Model")

        expected_keys = [
            "model_name", "accuracy", "precision", "recall",
            "f1_score", "roc_auc", "true_negatives", "false_positives",
            "false_negatives", "true_positives"
        ]
        for key in expected_keys:
            assert key in metrics

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_metrics_in_valid_range(self, mock_mlflow, simple_train_data, simple_test_data):
        """Test that metrics are in valid range [0, 1]."""
        X_train, y_train = simple_train_data
        X_test, y_test = simple_test_data

        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        metrics = evaluate_model(model, X_test, y_test, "Test Model")

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_confusion_matrix_values_non_negative(self, mock_mlflow, simple_train_data, simple_test_data):
        """Test that confusion matrix values are non-negative integers."""
        X_train, y_train = simple_train_data
        X_test, y_test = simple_test_data

        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        metrics = evaluate_model(model, X_test, y_test, "Test Model")

        assert metrics["true_negatives"] >= 0
        assert metrics["false_positives"] >= 0
        assert metrics["false_negatives"] >= 0
        assert metrics["true_positives"] >= 0


class TestSelectBestModel:
    """Tests for select_best_model function."""

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_selects_best_model(self, mock_mlflow, simple_train_data):
        """Test that function selects the model with highest metric."""
        X_train, y_train = simple_train_data

        # Create mock models and metrics
        lr_model = LogisticRegression(max_iter=100, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)

        lr_model.fit(X_train, y_train.iloc[:, 0])
        rf_model.fit(X_train, y_train.iloc[:, 0])
        gb_model.fit(X_train, y_train.iloc[:, 0])

        lr_metrics = {
            "model_name": "Logistic Regression", "f1_score": 0.60, "accuracy": 0.70,
            "precision": 0.65, "recall": 0.56, "roc_auc": 0.75
        }
        rf_metrics = {
            "model_name": "Random Forest", "f1_score": 0.65, "accuracy": 0.75,
            "precision": 0.70, "recall": 0.61, "roc_auc": 0.80
        }
        gb_metrics = {
            "model_name": "Gradient Boosting", "f1_score": 0.55, "accuracy": 0.72,
            "precision": 0.60, "recall": 0.51, "roc_auc": 0.78
        }

        parameters = {"selection_metric": "f1_score"}

        best_model, summary = select_best_model(
            lr_metrics, rf_metrics, gb_metrics,
            lr_model, rf_model, gb_model,
            parameters
        )

        # RF should be selected (highest f1_score)
        assert summary["best_model"] == "random_forest"

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_returns_summary_with_all_models(self, mock_mlflow, simple_train_data):
        """Test that summary contains metrics for all models."""
        X_train, y_train = simple_train_data

        lr_model = LogisticRegression(max_iter=100, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=10, random_state=42)

        lr_model.fit(X_train, y_train.iloc[:, 0])
        rf_model.fit(X_train, y_train.iloc[:, 0])
        gb_model.fit(X_train, y_train.iloc[:, 0])

        lr_metrics = {
            "model_name": "Logistic Regression", "f1_score": 0.60, "accuracy": 0.70,
            "precision": 0.65, "recall": 0.56, "roc_auc": 0.75
        }
        rf_metrics = {
            "model_name": "Random Forest", "f1_score": 0.65, "accuracy": 0.75,
            "precision": 0.70, "recall": 0.61, "roc_auc": 0.80
        }
        gb_metrics = {
            "model_name": "Gradient Boosting", "f1_score": 0.55, "accuracy": 0.72,
            "precision": 0.60, "recall": 0.51, "roc_auc": 0.78
        }

        parameters = {"selection_metric": "f1_score"}

        _, summary = select_best_model(
            lr_metrics, rf_metrics, gb_metrics,
            lr_model, rf_model, gb_model,
            parameters
        )

        assert "logistic_regression" in summary["models"]
        assert "random_forest" in summary["models"]
        assert "gradient_boosting" in summary["models"]


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_returns_dataframe(self, mock_mlflow, simple_train_data):
        """Test that function returns a DataFrame."""
        X_train, y_train = simple_train_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        result = get_feature_importance(model, X_train)
        assert isinstance(result, pd.DataFrame)

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_has_feature_and_importance_columns(self, mock_mlflow, simple_train_data):
        """Test that result has feature and importance columns."""
        X_train, y_train = simple_train_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        result = get_feature_importance(model, X_train)
        assert "feature" in result.columns
        assert "importance" in result.columns

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_sorted_by_importance_descending(self, mock_mlflow, simple_train_data):
        """Test that features are sorted by importance in descending order."""
        X_train, y_train = simple_train_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        result = get_feature_importance(model, X_train)
        importances = result["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    @patch("churn_prediction.pipelines.model_training.node.mlflow")
    def test_works_with_logistic_regression(self, mock_mlflow, simple_train_data):
        """Test that function works with LogisticRegression (uses coef_)."""
        X_train, y_train = simple_train_data

        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_train, y_train.iloc[:, 0])

        result = get_feature_importance(model, X_train)
        assert len(result) == X_train.shape[1]
