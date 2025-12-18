"""Unit tests for the data processing pipeline."""

import pytest
import pandas as pd
import numpy as np

from churn_prediction.pipelines.data_processing.node import (
    drop_customer_id,
    convert_total_charges,
    handle_missing_values,
    encode_binary_columns,
    encode_multiclass_columns,
    scale_numerical_features,
)


class TestDropCustomerId:
    """Tests for drop_customer_id function."""

    def test_removes_customer_id_column(self, sample_raw_data):
        """Test that customerID column is removed."""
        result = drop_customer_id(sample_raw_data)
        assert "customerID" not in result.columns

    def test_preserves_other_columns(self, sample_raw_data):
        """Test that other columns are preserved."""
        result = drop_customer_id(sample_raw_data)
        expected_cols = [c for c in sample_raw_data.columns if c != "customerID"]
        assert list(result.columns) == expected_cols

    def test_preserves_row_count(self, sample_raw_data):
        """Test that row count is preserved."""
        result = drop_customer_id(sample_raw_data)
        assert len(result) == len(sample_raw_data)


class TestConvertTotalCharges:
    """Tests for convert_total_charges function."""

    def test_converts_to_numeric(self, sample_data_no_id):
        """Test that TotalCharges is converted to numeric."""
        result = convert_total_charges(sample_data_no_id)
        assert result["TotalCharges"].dtype == np.float64

    def test_handles_empty_strings(self):
        """Test that empty strings become NaN."""
        df = pd.DataFrame({
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [0],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["Yes"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [29.85],
            "TotalCharges": [""],
            "Churn": ["No"],
        })
        result = convert_total_charges(df)
        assert pd.isna(result["TotalCharges"].iloc[0])

    def test_preserves_valid_values(self, sample_data_no_id):
        """Test that valid numeric strings are converted correctly."""
        result = convert_total_charges(sample_data_no_id)
        # First row has "350.50"
        assert result["TotalCharges"].iloc[0] == 350.50


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    def test_fills_missing_for_new_customers(self):
        """Test that new customers (tenure=0) get TotalCharges=0."""
        df = pd.DataFrame({
            "tenure": [0, 12],
            "MonthlyCharges": [29.85, 50.0],
            "TotalCharges": [np.nan, 600.0],
            "Churn": ["No", "Yes"],
            "gender": ["Male", "Female"],
        })
        result = handle_missing_values(df)
        assert result.loc[result["tenure"] == 0, "TotalCharges"].iloc[0] == 0

    def test_fills_with_median_for_others(self):
        """Test that other missing values are filled with median."""
        df = pd.DataFrame({
            "tenure": [12, 24, 36],
            "MonthlyCharges": [29.85, 50.0, 70.0],
            "TotalCharges": [np.nan, 1200.0, 2520.0],
            "Churn": ["No", "Yes", "No"],
            "gender": ["Male", "Female", "Male"],
        })
        result = handle_missing_values(df)
        # Median of [1200, 2520] = 1860
        assert result["TotalCharges"].iloc[0] == 1860.0

    def test_no_missing_values_after(self):
        """Test that no missing values remain after processing."""
        df = pd.DataFrame({
            "tenure": [0, 12],
            "MonthlyCharges": [29.85, 50.0],
            "TotalCharges": [np.nan, 600.0],
            "Churn": ["No", "Yes"],
            "gender": ["Male", "Female"],
        })
        result = handle_missing_values(df)
        assert result["TotalCharges"].isna().sum() == 0


class TestEncodeBinaryColumns:
    """Tests for encode_binary_columns function."""

    def test_encodes_gender(self):
        """Test gender encoding (Female=0, Male=1)."""
        df = pd.DataFrame({
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["Yes", "No"],
            "PhoneService": ["Yes", "No"],
            "PaperlessBilling": ["Yes", "No"],
            "Churn": ["Yes", "No"],
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [600.0, 1440.0],
        })
        result = encode_binary_columns(df)
        assert result["gender"].iloc[0] == 1  # Male
        assert result["gender"].iloc[1] == 0  # Female

    def test_encodes_churn(self):
        """Test Churn encoding (No=0, Yes=1)."""
        df = pd.DataFrame({
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["Yes", "No"],
            "PhoneService": ["Yes", "No"],
            "PaperlessBilling": ["Yes", "No"],
            "Churn": ["Yes", "No"],
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [600.0, 1440.0],
        })
        result = encode_binary_columns(df)
        assert result["Churn"].iloc[0] == 1  # Yes
        assert result["Churn"].iloc[1] == 0  # No

    def test_all_binary_columns_are_numeric(self):
        """Test that all binary columns become numeric."""
        df = pd.DataFrame({
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "PhoneService": ["Yes"],
            "PaperlessBilling": ["No"],
            "Churn": ["Yes"],
            "tenure": [12],
            "MonthlyCharges": [50.0],
            "TotalCharges": [600.0],
        })
        result = encode_binary_columns(df)
        binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
        for col in binary_cols:
            assert result[col].dtype in [np.int64, np.int32, int]


class TestEncodeMulticlassColumns:
    """Tests for encode_multiclass_columns function."""

    def test_creates_dummy_columns(self):
        """Test that one-hot encoding creates expected columns."""
        df = pd.DataFrame({
            "gender": [1, 0],
            "Contract": ["Month-to-month", "One year"],
            "InternetService": ["DSL", "Fiber optic"],
            "tenure": [12, 24],
        })
        result = encode_multiclass_columns(df)
        # Contract should have One year and Two year columns (drop_first=True)
        assert "Contract_One year" in result.columns or "Contract_Two year" in result.columns
        # Original Contract column should be removed
        assert "Contract" not in result.columns

    def test_drop_first_reduces_columns(self):
        """Test that drop_first=True reduces column count."""
        df = pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year"],
            "tenure": [12, 24, 36],
        })
        result = encode_multiclass_columns(df)
        # Should have 2 dummy columns for Contract (3 categories - 1)
        contract_cols = [c for c in result.columns if c.startswith("Contract_")]
        assert len(contract_cols) == 2


class TestScaleNumericalFeatures:
    """Tests for scale_numerical_features function."""

    def test_scales_tenure(self):
        """Test that tenure is scaled."""
        df = pd.DataFrame({
            "gender": [1, 0, 1, 0, 1],
            "SeniorCitizen": [0, 1, 0, 1, 0],
            "Partner": [1, 0, 1, 0, 1],
            "Dependents": [0, 1, 0, 1, 0],
            "PhoneService": [1, 1, 0, 1, 0],
            "PaperlessBilling": [1, 0, 1, 0, 1],
            "Churn": [0, 1, 0, 1, 0],
            "tenure": [12, 24, 36, 48, 60],
            "MonthlyCharges": [30.0, 60.0, 90.0, 45.0, 75.0],
            "TotalCharges": [360.0, 1440.0, 3240.0, 2160.0, 4500.0],
        })
        result = scale_numerical_features(df)
        # Scaled values should have mean ~0 and std ~1
        assert abs(result["tenure"].mean()) < 0.1
        # Use ddof=0 for population std which sklearn uses
        assert abs(result["tenure"].std(ddof=0) - 1.0) < 0.1

    def test_preserves_non_numerical_columns(self):
        """Test that non-numerical columns are preserved."""
        df = pd.DataFrame({
            "gender": [1, 0],
            "SeniorCitizen": [0, 1],
            "Partner": [1, 0],
            "Dependents": [0, 1],
            "PhoneService": [1, 1],
            "PaperlessBilling": [1, 0],
            "Churn": [0, 1],
            "tenure": [12, 24],
            "MonthlyCharges": [30.0, 60.0],
            "TotalCharges": [360.0, 1440.0],
        })
        result = scale_numerical_features(df)
        # Binary columns should be unchanged
        assert list(result["gender"]) == [1, 0]
        assert list(result["Churn"]) == [0, 1]
