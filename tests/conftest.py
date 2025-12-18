"""Pytest fixtures for churn prediction pipeline tests."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_raw_data():
    """Create sample raw customer data for testing."""
    return pd.DataFrame({
        "customerID": ["1001", "1002", "1003", "1004", "1005"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "tenure": [12, 0, 36, 24, 48],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "No phone service", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No", "No internet service", "No", "Yes"],
        "OnlineBackup": ["No", "Yes", "No internet service", "Yes", "No"],
        "DeviceProtection": ["Yes", "No", "No internet service", "Yes", "Yes"],
        "TechSupport": ["No", "No", "No internet service", "Yes", "No"],
        "StreamingTV": ["Yes", "Yes", "No internet service", "No", "Yes"],
        "StreamingMovies": ["No", "Yes", "No internet service", "Yes", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "No", "Yes", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
        ],
        "MonthlyCharges": [29.85, 70.70, 20.05, 45.25, 89.50],
        "TotalCharges": ["350.50", "", "721.80", "1085.00", "4296.00"],
        "Churn": ["No", "Yes", "No", "Yes", "No"],
    })


@pytest.fixture
def sample_data_no_id():
    """Sample data after dropping customerID."""
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 24, 36],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["Yes", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
        ],
        "MonthlyCharges": [29.85, 70.70, 20.05],
        "TotalCharges": ["350.50", "1697.40", "721.80"],
        "Churn": ["No", "Yes", "No"],
    })


@pytest.fixture
def sample_preprocessed_data():
    """Sample preprocessed data for feature engineering tests."""
    return pd.DataFrame({
        "gender": [1, 0, 1],
        "SeniorCitizen": [0, 1, 0],
        "Partner": [1, 0, 1],
        "Dependents": [0, 0, 1],
        "tenure": [-0.5, 0.0, 0.5],  # Scaled
        "PhoneService": [1, 1, 0],
        "PaperlessBilling": [1, 0, 0],
        "MonthlyCharges": [-0.8, 0.5, -1.0],  # Scaled
        "TotalCharges": [-0.6, 0.3, -0.2],  # Scaled
        "Churn": [0, 1, 0],
        # One-hot encoded columns
        "Contract_One year": [0, 1, 0],
        "Contract_Two year": [0, 0, 1],
        "InternetService_Fiber optic": [0, 1, 0],
        "InternetService_No": [0, 0, 1],
        "OnlineSecurity_No internet service": [0, 0, 1],
        "OnlineSecurity_Yes": [1, 0, 0],
        "OnlineBackup_No internet service": [0, 0, 1],
        "OnlineBackup_Yes": [0, 1, 0],
        "DeviceProtection_No internet service": [0, 0, 1],
        "DeviceProtection_Yes": [1, 0, 0],
        "TechSupport_No internet service": [0, 0, 1],
        "TechSupport_Yes": [0, 0, 0],
        "StreamingTV_No internet service": [0, 0, 1],
        "StreamingTV_Yes": [1, 1, 0],
        "StreamingMovies_No internet service": [0, 0, 1],
        "StreamingMovies_Yes": [0, 1, 0],
        "PaymentMethod_Credit card (automatic)": [0, 0, 0],
        "PaymentMethod_Electronic check": [1, 0, 0],
        "PaymentMethod_Mailed check": [0, 1, 0],
        "MultipleLines_No phone service": [0, 0, 0],
        "MultipleLines_Yes": [0, 0, 0],
    })


@pytest.fixture
def sample_features_engineered():
    """Sample data after feature engineering."""
    preprocessed = pd.DataFrame({
        "gender": [1, 0, 1],
        "SeniorCitizen": [0, 1, 0],
        "Partner": [1, 0, 1],
        "Dependents": [0, 0, 1],
        "tenure": [-0.5, 0.0, 0.5],
        "PhoneService": [1, 1, 0],
        "PaperlessBilling": [1, 0, 0],
        "MonthlyCharges": [-0.8, 0.5, -1.0],
        "TotalCharges": [-0.6, 0.3, -0.2],
        "Churn": [0, 1, 0],
        "Contract_One year": [0, 1, 0],
        "Contract_Two year": [0, 0, 1],
    })
    # Add engineered features
    preprocessed["total_services"] = [3.0, 4.0, 1.0]
    preprocessed["has_internet"] = [1, 1, 0]
    preprocessed["has_streaming"] = [1, 1, 0]
    preprocessed["has_security_services"] = [1, 0, 0]
    preprocessed["security_services_count"] = [2.0, 0.0, 0.0]
    preprocessed["has_multiple_lines"] = [0, 0, 0]
    preprocessed["tenure_group"] = [1, 2, 3]
    preprocessed["is_new_customer"] = [0, 0, 0]
    preprocessed["is_loyal_customer"] = [0, 0, 1]
    preprocessed["is_month_to_month"] = [1, 0, 0]
    preprocessed["has_long_contract"] = [0, 1, 1]
    preprocessed["uses_electronic_check"] = [1, 0, 0]
    preprocessed["has_auto_payment"] = [0, 0, 1]
    preprocessed["charge_per_service"] = [-0.2, 0.1, -1.0]
    preprocessed["is_high_charges"] = [0, 1, 0]
    preprocessed["charge_tenure_ratio"] = [1.6, 0.5, -2.0]
    preprocessed["avg_monthly_spend"] = [1.2, 0.3, -0.4]
    preprocessed["tenure_normalized"] = [-0.007, 0.0, 0.007]
    preprocessed["value_consistency"] = [1.0, 1.0, 1.0]
    preprocessed["high_risk_combo"] = [0, 0, 0]
    preprocessed["churn_risk_score"] = [4.0, 2.0, 1.0]
    preprocessed["low_engagement"] = [0, 0, 0]
    preprocessed["senior_high_charges"] = [0, 1, 0]
    preprocessed["has_family"] = [1, 0, 1]
    return preprocessed


@pytest.fixture
def sample_model_parameters():
    """Sample model training parameters."""
    return {
        "target_column": "Churn",
        "test_size": 0.2,
        "random_state": 42,
        "selection_metric": "f1_score",
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 100,
            "class_weight": "balanced",
        },
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        },
        "gradient_boosting": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
    }


@pytest.fixture
def sample_train_test_data(sample_features_engineered):
    """Sample train/test split data."""
    # Create larger dataset for proper splitting
    df = pd.concat([sample_features_engineered] * 50, ignore_index=True)
    # Add some variation
    df["tenure"] = np.random.randn(len(df))
    df["MonthlyCharges"] = np.random.randn(len(df))
    df["Churn"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

    X = df.drop(columns=["Churn"])
    y = df[["Churn"]]

    # Simple split
    train_size = int(0.8 * len(df))
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_metrics():
    """Sample model evaluation metrics."""
    return {
        "model_name": "Test Model",
        "accuracy": 0.80,
        "precision": 0.75,
        "recall": 0.70,
        "f1_score": 0.72,
        "roc_auc": 0.85,
        "true_negatives": 100,
        "false_positives": 20,
        "false_negatives": 15,
        "true_positives": 65,
    }


@pytest.fixture
def sample_model_comparison():
    """Sample model comparison dictionary."""
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
                "true_negatives": 800,
                "false_positives": 230,
                "false_negatives": 80,
                "true_positives": 290,
            },
            "random_forest": {
                "model_name": "Random Forest",
                "accuracy": 0.77,
                "precision": 0.55,
                "recall": 0.71,
                "f1_score": 0.62,
                "roc_auc": 0.84,
                "true_negatives": 850,
                "false_positives": 180,
                "false_negatives": 110,
                "true_positives": 260,
            },
            "gradient_boosting": {
                "model_name": "Gradient Boosting",
                "accuracy": 0.80,
                "precision": 0.64,
                "recall": 0.52,
                "f1_score": 0.57,
                "roc_auc": 0.84,
                "true_negatives": 900,
                "false_positives": 130,
                "false_negatives": 180,
                "true_positives": 190,
            },
        },
    }


@pytest.fixture
def sample_feature_importance():
    """Sample feature importance DataFrame."""
    return pd.DataFrame({
        "feature": [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract_Two year",
            "is_month_to_month",
            "total_services",
            "churn_risk_score",
            "has_security_services",
            "is_new_customer",
            "uses_electronic_check",
        ],
        "importance": [0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04],
    })
