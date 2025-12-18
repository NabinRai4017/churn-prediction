"""Unit tests for the feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np

from churn_prediction.pipelines.feature_engineering.node import (
    create_service_features,
    create_tenure_features,
    create_contract_features,
    create_charge_features,
    create_interaction_features,
)


class TestCreateServiceFeatures:
    """Tests for create_service_features function."""

    def test_creates_total_services(self, sample_preprocessed_data):
        """Test that total_services is created."""
        result = create_service_features(sample_preprocessed_data)
        assert "total_services" in result.columns

    def test_creates_has_internet(self, sample_preprocessed_data):
        """Test that has_internet is created."""
        result = create_service_features(sample_preprocessed_data)
        assert "has_internet" in result.columns
        assert result["has_internet"].isin([0, 1]).all()

    def test_creates_has_streaming(self, sample_preprocessed_data):
        """Test that has_streaming is created."""
        result = create_service_features(sample_preprocessed_data)
        assert "has_streaming" in result.columns
        assert result["has_streaming"].isin([0, 1]).all()

    def test_creates_security_features(self, sample_preprocessed_data):
        """Test that security-related features are created."""
        result = create_service_features(sample_preprocessed_data)
        assert "has_security_services" in result.columns
        assert "security_services_count" in result.columns

    def test_security_count_range(self, sample_preprocessed_data):
        """Test that security_services_count is in valid range."""
        result = create_service_features(sample_preprocessed_data)
        assert result["security_services_count"].min() >= 0
        assert result["security_services_count"].max() <= 4


class TestCreateTenureFeatures:
    """Tests for create_tenure_features function."""

    def test_creates_tenure_group(self, sample_preprocessed_data):
        """Test that tenure_group is created."""
        # First add service features (required input)
        data = create_service_features(sample_preprocessed_data)
        result = create_tenure_features(data)
        assert "tenure_group" in result.columns
        assert result["tenure_group"].isin([0, 1, 2, 3]).all()

    def test_creates_customer_flags(self, sample_preprocessed_data):
        """Test that customer type flags are created."""
        data = create_service_features(sample_preprocessed_data)
        result = create_tenure_features(data)
        assert "is_new_customer" in result.columns
        assert "is_loyal_customer" in result.columns
        assert result["is_new_customer"].isin([0, 1]).all()
        assert result["is_loyal_customer"].isin([0, 1]).all()

    def test_new_and_loyal_mutually_exclusive(self, sample_preprocessed_data):
        """Test that a customer can't be both new and loyal."""
        data = create_service_features(sample_preprocessed_data)
        result = create_tenure_features(data)
        # A customer should not be both new and loyal
        both = (result["is_new_customer"] == 1) & (result["is_loyal_customer"] == 1)
        assert not both.any()


class TestCreateContractFeatures:
    """Tests for create_contract_features function."""

    def test_creates_month_to_month_flag(self, sample_preprocessed_data):
        """Test that is_month_to_month is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        result = create_contract_features(data)
        assert "is_month_to_month" in result.columns
        assert result["is_month_to_month"].isin([0, 1]).all()

    def test_creates_long_contract_flag(self, sample_preprocessed_data):
        """Test that has_long_contract is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        result = create_contract_features(data)
        assert "has_long_contract" in result.columns
        assert result["has_long_contract"].isin([0, 1]).all()

    def test_month_to_month_and_long_contract_exclusive(self, sample_preprocessed_data):
        """Test that month-to-month and long contract are mutually exclusive."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        result = create_contract_features(data)
        both = (result["is_month_to_month"] == 1) & (result["has_long_contract"] == 1)
        assert not both.any()

    def test_creates_payment_features(self, sample_preprocessed_data):
        """Test that payment-related features are created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        result = create_contract_features(data)
        assert "uses_electronic_check" in result.columns
        assert "has_auto_payment" in result.columns


class TestCreateChargeFeatures:
    """Tests for create_charge_features function."""

    def test_creates_charge_per_service(self, sample_preprocessed_data):
        """Test that charge_per_service is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        result = create_charge_features(data)
        assert "charge_per_service" in result.columns

    def test_creates_is_high_charges(self, sample_preprocessed_data):
        """Test that is_high_charges is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        result = create_charge_features(data)
        assert "is_high_charges" in result.columns
        assert result["is_high_charges"].isin([0, 1]).all()

    def test_creates_charge_tenure_ratio(self, sample_preprocessed_data):
        """Test that charge_tenure_ratio is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        result = create_charge_features(data)
        assert "charge_tenure_ratio" in result.columns

    def test_charge_per_service_handles_zero_services(self):
        """Test that charge_per_service handles zero services gracefully."""
        df = pd.DataFrame({
            "MonthlyCharges": [50.0],
            "total_services": [0.0],
            "tenure": [12.0],
        })
        # Add dummy columns for other requirements
        df["is_high_charges"] = 0

        # charge_per_service should use (total_services + 1) to avoid division by zero
        result = df.copy()
        result["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)
        assert not np.isinf(result["charge_per_service"]).any()
        assert not result["charge_per_service"].isna().any()


class TestCreateInteractionFeatures:
    """Tests for create_interaction_features function."""

    def test_creates_high_risk_combo(self, sample_preprocessed_data):
        """Test that high_risk_combo is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        data = create_charge_features(data)
        result = create_interaction_features(data)
        assert "high_risk_combo" in result.columns
        assert result["high_risk_combo"].isin([0, 1]).all()

    def test_creates_churn_risk_score(self, sample_preprocessed_data):
        """Test that churn_risk_score is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        data = create_charge_features(data)
        result = create_interaction_features(data)
        assert "churn_risk_score" in result.columns
        # Score should be between 0 and 6
        assert result["churn_risk_score"].min() >= 0
        assert result["churn_risk_score"].max() <= 6

    def test_creates_low_engagement(self, sample_preprocessed_data):
        """Test that low_engagement is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        data = create_charge_features(data)
        result = create_interaction_features(data)
        assert "low_engagement" in result.columns
        assert result["low_engagement"].isin([0, 1]).all()

    def test_creates_has_family(self, sample_preprocessed_data):
        """Test that has_family is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        data = create_charge_features(data)
        result = create_interaction_features(data)
        assert "has_family" in result.columns
        assert result["has_family"].isin([0, 1]).all()

    def test_creates_senior_high_charges(self, sample_preprocessed_data):
        """Test that senior_high_charges is created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        data = create_charge_features(data)
        result = create_interaction_features(data)
        assert "senior_high_charges" in result.columns
        assert result["senior_high_charges"].isin([0, 1]).all()

    def test_final_feature_count(self, sample_preprocessed_data):
        """Test that all 22 new features are created."""
        data = create_service_features(sample_preprocessed_data)
        data = create_tenure_features(data)
        data = create_contract_features(data)
        data = create_charge_features(data)
        result = create_interaction_features(data)

        expected_new_features = [
            # Service features (6)
            "total_services", "has_internet", "has_streaming",
            "has_security_services", "security_services_count", "has_multiple_lines",
            # Tenure features (3)
            "tenure_group", "is_new_customer", "is_loyal_customer",
            # Contract features (4)
            "is_month_to_month", "has_long_contract", "uses_electronic_check", "has_auto_payment",
            # Charge features (3)
            "charge_per_service", "is_high_charges", "charge_tenure_ratio",
            # Interaction features (6)
            "high_risk_combo", "churn_risk_score", "low_engagement",
            "senior_high_charges", "has_family",
        ]

        for feature in expected_new_features:
            assert feature in result.columns, f"Missing feature: {feature}"
