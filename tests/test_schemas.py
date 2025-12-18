"""Unit tests for Pandera schemas."""

import pytest
import pandas as pd
import numpy as np
import pandera as pa

from churn_prediction.schemas import (
    RawCustomerSchema,
    CustomerNoIdSchema,
    CustomerNumericSchema,
    CustomerCleanedSchema,
    CustomerBinaryEncodedSchema,
    PreprocessedCustomerSchema,
    FeaturesEngineeredSchema,
    ModelInputSchema,
    TargetSchema,
)


class TestRawCustomerSchema:
    """Tests for RawCustomerSchema validation."""

    def test_valid_data_passes(self, sample_raw_data):
        """Test that valid raw data passes validation."""
        # Should not raise
        RawCustomerSchema.validate(sample_raw_data)

    def test_missing_column_fails(self, sample_raw_data):
        """Test that missing required column fails validation."""
        df = sample_raw_data.drop(columns=["customerID"])
        with pytest.raises(pa.errors.SchemaError):
            RawCustomerSchema.validate(df)

    def test_invalid_gender_fails(self, sample_raw_data):
        """Test that invalid gender value fails validation."""
        df = sample_raw_data.copy()
        df.loc[0, "gender"] = "Other"
        with pytest.raises(pa.errors.SchemaError):
            RawCustomerSchema.validate(df)

    def test_invalid_senior_citizen_fails(self, sample_raw_data):
        """Test that invalid SeniorCitizen value fails validation."""
        df = sample_raw_data.copy()
        df.loc[0, "SeniorCitizen"] = 2
        with pytest.raises(pa.errors.SchemaError):
            RawCustomerSchema.validate(df)

    def test_negative_tenure_fails(self, sample_raw_data):
        """Test that negative tenure fails validation."""
        df = sample_raw_data.copy()
        df.loc[0, "tenure"] = -1
        with pytest.raises(pa.errors.SchemaError):
            RawCustomerSchema.validate(df)

    def test_invalid_churn_fails(self, sample_raw_data):
        """Test that invalid Churn value fails validation."""
        df = sample_raw_data.copy()
        df.loc[0, "Churn"] = "Maybe"
        with pytest.raises(pa.errors.SchemaError):
            RawCustomerSchema.validate(df)


class TestCustomerNoIdSchema:
    """Tests for CustomerNoIdSchema validation."""

    def test_valid_data_passes(self, sample_data_no_id):
        """Test that valid data without customerID passes validation."""
        CustomerNoIdSchema.validate(sample_data_no_id)

    def test_with_customer_id_still_passes(self, sample_raw_data):
        """Test that extra columns are allowed (strict=False)."""
        # Schema has strict=False, so extra columns are allowed
        df = sample_raw_data.drop(columns=["customerID"])
        CustomerNoIdSchema.validate(df)


class TestCustomerCleanedSchema:
    """Tests for CustomerCleanedSchema validation."""

    def test_valid_data_passes(self):
        """Test that valid cleaned data passes validation."""
        df = pd.DataFrame({
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [600.0, 1440.0],
            "Churn": ["No", "Yes"],
            "gender": ["Male", "Female"],
        })
        CustomerCleanedSchema.validate(df)

    def test_null_total_charges_fails(self):
        """Test that null TotalCharges fails validation."""
        df = pd.DataFrame({
            "tenure": [12],
            "MonthlyCharges": [50.0],
            "TotalCharges": [np.nan],
            "Churn": ["No"],
            "gender": ["Male"],
        })
        with pytest.raises(pa.errors.SchemaError):
            CustomerCleanedSchema.validate(df)


class TestCustomerBinaryEncodedSchema:
    """Tests for CustomerBinaryEncodedSchema validation."""

    def test_valid_data_passes(self):
        """Test that valid binary encoded data passes validation."""
        df = pd.DataFrame({
            "gender": [1, 0],
            "SeniorCitizen": [0, 1],
            "Partner": [1, 0],
            "Dependents": [0, 1],
            "PhoneService": [1, 0],
            "PaperlessBilling": [1, 0],
            "Churn": [0, 1],
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [600.0, 1440.0],
        })
        CustomerBinaryEncodedSchema.validate(df)

    def test_non_binary_gender_fails(self):
        """Test that non-binary gender value fails validation."""
        df = pd.DataFrame({
            "gender": [2, 0],  # Invalid: 2 is not in [0, 1]
            "SeniorCitizen": [0, 1],
            "Partner": [1, 0],
            "Dependents": [0, 1],
            "PhoneService": [1, 0],
            "PaperlessBilling": [1, 0],
            "Churn": [0, 1],
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 60.0],
            "TotalCharges": [600.0, 1440.0],
        })
        with pytest.raises(pa.errors.SchemaError):
            CustomerBinaryEncodedSchema.validate(df)


class TestPreprocessedCustomerSchema:
    """Tests for PreprocessedCustomerSchema validation."""

    def test_valid_data_passes(self, sample_preprocessed_data):
        """Test that valid preprocessed data passes validation."""
        PreprocessedCustomerSchema.validate(sample_preprocessed_data)

    def test_allows_extra_one_hot_columns(self, sample_preprocessed_data):
        """Test that extra columns from one-hot encoding are allowed."""
        df = sample_preprocessed_data.copy()
        df["NewColumn_Yes"] = [1, 0, 1]
        # Should not raise because strict=False
        PreprocessedCustomerSchema.validate(df)


class TestFeaturesEngineeredSchema:
    """Tests for FeaturesEngineeredSchema validation."""

    def test_valid_data_passes(self, sample_features_engineered):
        """Test that valid feature engineered data passes validation."""
        FeaturesEngineeredSchema.validate(sample_features_engineered)

    def test_invalid_tenure_group_fails(self, sample_features_engineered):
        """Test that invalid tenure_group value fails validation."""
        df = sample_features_engineered.copy()
        df.loc[0, "tenure_group"] = 5  # Invalid: should be 0-3
        with pytest.raises(pa.errors.SchemaError):
            FeaturesEngineeredSchema.validate(df)

    def test_invalid_churn_risk_score_fails(self, sample_features_engineered):
        """Test that churn_risk_score > 6 fails validation."""
        df = sample_features_engineered.copy()
        df.loc[0, "churn_risk_score"] = 7.0  # Invalid: should be 0-6
        with pytest.raises(pa.errors.SchemaError):
            FeaturesEngineeredSchema.validate(df)

    def test_invalid_binary_feature_fails(self, sample_features_engineered):
        """Test that non-binary value in binary feature fails validation."""
        df = sample_features_engineered.copy()
        df.loc[0, "has_internet"] = 2  # Invalid: should be 0 or 1
        with pytest.raises(pa.errors.SchemaError):
            FeaturesEngineeredSchema.validate(df)


class TestModelInputSchema:
    """Tests for ModelInputSchema validation."""

    def test_valid_data_passes(self):
        """Test that valid model input data passes validation."""
        df = pd.DataFrame({
            "tenure": [0.5, -0.3, 0.1],
            "MonthlyCharges": [0.2, -0.5, 0.8],
            "TotalCharges": [0.3, -0.2, 0.5],
            "feature1": [1, 0, 1],
        })
        ModelInputSchema.validate(df)

    def test_null_values_fail(self):
        """Test that null values fail validation."""
        df = pd.DataFrame({
            "tenure": [0.5, np.nan, 0.1],
            "MonthlyCharges": [0.2, -0.5, 0.8],
            "TotalCharges": [0.3, -0.2, 0.5],
        })
        with pytest.raises(pa.errors.SchemaError):
            ModelInputSchema.validate(df)


class TestTargetSchema:
    """Tests for TargetSchema validation."""

    def test_valid_data_passes(self):
        """Test that valid target data passes validation."""
        df = pd.DataFrame({"Churn": [0, 1, 0, 1, 0]})
        TargetSchema.validate(df)

    def test_invalid_churn_value_fails(self):
        """Test that invalid Churn value fails validation."""
        df = pd.DataFrame({"Churn": [0, 2, 0]})  # 2 is invalid
        with pytest.raises(pa.errors.SchemaError):
            TargetSchema.validate(df)

    def test_null_churn_fails(self):
        """Test that null Churn value fails validation."""
        df = pd.DataFrame({"Churn": [0, None, 1]})
        with pytest.raises(pa.errors.SchemaError):
            TargetSchema.validate(df)

    def test_extra_columns_fail(self):
        """Test that extra columns fail validation (strict=True)."""
        df = pd.DataFrame({
            "Churn": [0, 1, 0],
            "ExtraColumn": [1, 2, 3],
        })
        with pytest.raises(pa.errors.SchemaError):
            TargetSchema.validate(df)
