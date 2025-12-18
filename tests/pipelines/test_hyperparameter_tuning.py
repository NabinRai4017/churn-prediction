"""Unit tests for the hyperparameter tuning pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import optuna

from churn_prediction.pipelines.hyperparameter_tuning.node import (
    identify_best_model,
    run_optuna_study,
    train_tuned_model,
    evaluate_tuned_model,
    create_tuning_report,
    _suggest_hyperparameters,
    _create_model,
    _get_scoring_metric,
)


@pytest.fixture
def sample_model_comparison():
    """Sample model comparison from baseline training."""
    return {
        "selection_metric": "f1_score",
        "best_model": "random_forest",
        "models": {
            "logistic_regression": {
                "model_name": "Logistic Regression",
                "accuracy": 0.74,
                "precision": 0.51,
                "recall": 0.78,
                "f1_score": 0.62,
                "roc_auc": 0.84,
            },
            "random_forest": {
                "model_name": "Random Forest",
                "accuracy": 0.77,
                "precision": 0.55,
                "recall": 0.71,
                "f1_score": 0.65,
                "roc_auc": 0.85,
            },
            "gradient_boosting": {
                "model_name": "Gradient Boosting",
                "accuracy": 0.80,
                "precision": 0.64,
                "recall": 0.52,
                "f1_score": 0.57,
                "roc_auc": 0.84,
            },
        },
    }


@pytest.fixture
def sample_train_data():
    """Sample training data for tuning tests."""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "feature4": np.random.randn(n_samples),
        "feature5": np.random.randn(n_samples),
    })
    y = pd.DataFrame({
        "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    return X, y


@pytest.fixture
def sample_test_data():
    """Sample test data for evaluation."""
    np.random.seed(123)
    n_samples = 50

    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "feature4": np.random.randn(n_samples),
        "feature5": np.random.randn(n_samples),
    })
    y = pd.DataFrame({
        "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    })
    return X, y


@pytest.fixture
def sample_tuning_parameters():
    """Sample hyperparameter tuning parameters."""
    return {
        "n_trials": 5,  # Small number for testing
        "optimization_metric": "f1_score",
        "cv_folds": 3,
        "random_state": 42,
        "direction": "maximize",
        "pruning": False,
        "timeout": None,
        "random_forest": {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
            "max_depth": {"type": "int", "low": 2, "high": 10},
        },
        "logistic_regression": {
            "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        },
        "gradient_boosting": {
            "n_estimators": {"type": "int", "low": 10, "high": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        },
    }


class TestIdentifyBestModel:
    """Tests for identify_best_model function."""

    def test_identifies_correct_model(self, sample_model_comparison):
        """Test that the correct best model is identified."""
        result = identify_best_model(sample_model_comparison)
        assert result == "random_forest"

    def test_returns_string(self, sample_model_comparison):
        """Test that function returns a string."""
        result = identify_best_model(sample_model_comparison)
        assert isinstance(result, str)

    def test_handles_different_best_model(self):
        """Test identification when different model is best."""
        comparison = {
            "selection_metric": "accuracy",
            "best_model": "gradient_boosting",
            "models": {
                "logistic_regression": {"accuracy": 0.70},
                "random_forest": {"accuracy": 0.75},
                "gradient_boosting": {"accuracy": 0.80},
            },
        }
        result = identify_best_model(comparison)
        assert result == "gradient_boosting"


class TestSuggestHyperparameters:
    """Tests for _suggest_hyperparameters function."""

    def test_suggests_int_params(self):
        """Test suggestion of integer parameters."""
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 100

        search_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
        }

        result = _suggest_hyperparameters(trial, "random_forest", search_space)

        trial.suggest_int.assert_called_once_with("n_estimators", 50, 200)
        assert result["n_estimators"] == 100

    def test_suggests_float_params(self):
        """Test suggestion of float parameters."""
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_float.return_value = 1.5

        search_space = {
            "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        }

        result = _suggest_hyperparameters(trial, "logistic_regression", search_space)

        trial.suggest_float.assert_called_once_with("C", 0.1, 10.0, log=True)
        assert result["C"] == 1.5

    def test_suggests_categorical_params(self):
        """Test suggestion of categorical parameters."""
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_categorical.return_value = "gini"

        search_space = {
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
        }

        result = _suggest_hyperparameters(trial, "random_forest", search_space)

        trial.suggest_categorical.assert_called_once_with("criterion", ["gini", "entropy"])
        assert result["criterion"] == "gini"


class TestCreateModel:
    """Tests for _create_model function."""

    def test_creates_logistic_regression(self):
        """Test creation of LogisticRegression model."""
        from sklearn.linear_model import LogisticRegression

        params = {"C": 1.0, "max_iter": 100}
        model = _create_model("logistic_regression", params, random_state=42)

        assert isinstance(model, LogisticRegression)
        assert model.C == 1.0
        assert model.max_iter == 100

    def test_creates_random_forest(self):
        """Test creation of RandomForestClassifier model."""
        from sklearn.ensemble import RandomForestClassifier

        params = {"n_estimators": 50, "max_depth": 10}
        model = _create_model("random_forest", params, random_state=42)

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 10

    def test_creates_gradient_boosting(self):
        """Test creation of GradientBoostingClassifier model."""
        from sklearn.ensemble import GradientBoostingClassifier

        params = {"n_estimators": 50, "learning_rate": 0.1}
        model = _create_model("gradient_boosting", params, random_state=42)

        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1

    def test_raises_for_unknown_model(self):
        """Test that unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            _create_model("unknown_model", {}, random_state=42)


class TestGetScoringMetric:
    """Tests for _get_scoring_metric function."""

    def test_maps_f1_score(self):
        """Test mapping of f1_score metric."""
        assert _get_scoring_metric("f1_score") == "f1"

    def test_maps_accuracy(self):
        """Test mapping of accuracy metric."""
        assert _get_scoring_metric("accuracy") == "accuracy"

    def test_maps_roc_auc(self):
        """Test mapping of roc_auc metric."""
        assert _get_scoring_metric("roc_auc") == "roc_auc"

    def test_defaults_to_f1(self):
        """Test that unknown metrics default to f1."""
        assert _get_scoring_metric("unknown_metric") == "f1"


class TestRunOptunaStudy:
    """Tests for run_optuna_study function."""

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_returns_study_and_params(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that function returns study and best parameters."""
        # Mock get_tracking_uri to return a mock value that skips MLflowCallback
        mock_mlflow.get_tracking_uri.return_value = "mock://test"

        X_train, y_train = sample_train_data

        study, best_params = run_optuna_study(
            "random_forest",
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        assert isinstance(study, optuna.Study)
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_study_has_correct_direction(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that study has correct optimization direction."""
        mock_mlflow.get_tracking_uri.return_value = "mock://test"

        X_train, y_train = sample_train_data

        study, _ = run_optuna_study(
            "random_forest",
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_completes_all_trials(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that all trials are completed."""
        mock_mlflow.get_tracking_uri.return_value = "mock://test"

        X_train, y_train = sample_train_data

        study, _ = run_optuna_study(
            "random_forest",
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        # At least n_trials should be attempted
        assert len(study.trials) >= sample_tuning_parameters["n_trials"]

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_logs_to_mlflow(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that parameters are logged to MLflow."""
        mock_mlflow.get_tracking_uri.return_value = "mock://test"

        X_train, y_train = sample_train_data

        run_optuna_study(
            "random_forest",
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        mock_mlflow.log_param.assert_called()


class TestTrainTunedModel:
    """Tests for train_tuned_model function."""

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_returns_trained_model(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that function returns a trained model."""
        from sklearn.ensemble import RandomForestClassifier

        X_train, y_train = sample_train_data
        best_params = {"n_estimators": 50, "max_depth": 5}

        model = train_tuned_model(
            "random_forest",
            best_params,
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        assert isinstance(model, RandomForestClassifier)

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_model_can_predict(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that trained model can make predictions."""
        X_train, y_train = sample_train_data
        best_params = {"n_estimators": 50, "max_depth": 5}

        model = train_tuned_model(
            "random_forest",
            best_params,
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        predictions = model.predict(X_train)
        assert len(predictions) == len(X_train)
        assert set(predictions).issubset({0, 1})

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_logs_model_to_mlflow(self, mock_mlflow, sample_train_data, sample_tuning_parameters):
        """Test that model is logged to MLflow."""
        X_train, y_train = sample_train_data
        best_params = {"n_estimators": 50, "max_depth": 5}

        train_tuned_model(
            "random_forest",
            best_params,
            X_train,
            y_train,
            sample_tuning_parameters,
        )

        mock_mlflow.sklearn.log_model.assert_called_once()


class TestEvaluateTunedModel:
    """Tests for evaluate_tuned_model function."""

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_returns_metrics_dict(self, mock_mlflow, sample_train_data, sample_test_data):
        """Test that function returns metrics dictionary."""
        from sklearn.ensemble import RandomForestClassifier

        X_train, y_train = sample_train_data
        X_test, y_test = sample_test_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        metrics = evaluate_tuned_model(model, X_test, y_test, "random_forest")

        expected_keys = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for key in expected_keys:
            assert key in metrics

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_metrics_in_valid_range(self, mock_mlflow, sample_train_data, sample_test_data):
        """Test that metrics are in valid range [0, 1]."""
        from sklearn.ensemble import RandomForestClassifier

        X_train, y_train = sample_train_data
        X_test, y_test = sample_test_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        metrics = evaluate_tuned_model(model, X_test, y_test, "random_forest")

        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert 0 <= metrics[key] <= 1

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_logs_metrics_to_mlflow(self, mock_mlflow, sample_train_data, sample_test_data):
        """Test that metrics are logged to MLflow."""
        from sklearn.ensemble import RandomForestClassifier

        X_train, y_train = sample_train_data
        X_test, y_test = sample_test_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        evaluate_tuned_model(model, X_test, y_test, "random_forest")

        mock_mlflow.log_metrics.assert_called_once()


class TestCreateTuningReport:
    """Tests for create_tuning_report function."""

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_returns_report_dict(self, mock_mlflow, sample_model_comparison):
        """Test that function returns report dictionary."""
        # Create a mock study
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)

        tuned_metrics = {
            "model_name": "Tuned random_forest",
            "accuracy": 0.80,
            "precision": 0.60,
            "recall": 0.75,
            "f1_score": 0.67,
            "roc_auc": 0.87,
        }

        report = create_tuning_report(
            study,
            tuned_metrics,
            sample_model_comparison,
            "random_forest",
        )

        assert isinstance(report, dict)
        assert "model_name" in report
        assert "baseline_score" in report
        assert "tuned_score" in report
        assert "improvement" in report

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_calculates_improvement(self, mock_mlflow, sample_model_comparison):
        """Test that improvement is calculated correctly."""
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)

        tuned_metrics = {
            "model_name": "Tuned random_forest",
            "accuracy": 0.80,
            "precision": 0.60,
            "recall": 0.75,
            "f1_score": 0.70,  # Baseline was 0.65
            "roc_auc": 0.87,
        }

        report = create_tuning_report(
            study,
            tuned_metrics,
            sample_model_comparison,
            "random_forest",
        )

        expected_improvement = 0.70 - 0.65  # tuned - baseline
        assert abs(report["improvement"] - expected_improvement) < 0.001

    @patch("churn_prediction.pipelines.hyperparameter_tuning.node.mlflow")
    def test_includes_study_statistics(self, mock_mlflow, sample_model_comparison):
        """Test that study statistics are included."""
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)

        tuned_metrics = {
            "model_name": "Tuned random_forest",
            "f1_score": 0.70,
        }

        report = create_tuning_report(
            study,
            tuned_metrics,
            sample_model_comparison,
            "random_forest",
        )

        assert "study_statistics" in report
        assert "total_trials" in report["study_statistics"]
        assert "completed_trials" in report["study_statistics"]
        assert "best_trial_number" in report["study_statistics"]
