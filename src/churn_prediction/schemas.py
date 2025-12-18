"""Pandera schema definitions for data validation across pipelines.

This module defines DataFrame schemas for validating data at each stage
of the churn prediction pipeline.
"""

import pandera as pa
from pandera.typing import Series
import pandas as pd


# =============================================================================
# RAW DATA SCHEMA
# =============================================================================

class RawCustomerSchema(pa.DataFrameModel):
    """Schema for raw customer data from telco-customer-churn.csv."""

    customerID: Series[str] = pa.Field(nullable=False, unique=True)
    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(
        isin=["Yes", "No", "No phone service"]
    )
    InternetService: Series[str] = pa.Field(
        isin=["DSL", "Fiber optic", "No"]
    )
    OnlineSecurity: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    OnlineBackup: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    DeviceProtection: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    TechSupport: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    StreamingTV: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    StreamingMovies: Series[str] = pa.Field(
        isin=["Yes", "No", "No internet service"]
    )
    Contract: Series[str] = pa.Field(
        isin=["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[str]  # String in raw data, may contain empty strings
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        name = "RawCustomerSchema"
        strict = False  # Allow extra columns
        coerce = True


# =============================================================================
# DATA PROCESSING SCHEMAS
# =============================================================================

class CustomerNoIdSchema(pa.DataFrameModel):
    """Schema after dropping customerID."""

    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[str]
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        name = "CustomerNoIdSchema"
        strict = False
        coerce = True


class CustomerNumericSchema(pa.DataFrameModel):
    """Schema after converting TotalCharges to numeric."""

    tenure: Series[int] = pa.Field(ge=0)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(nullable=True)  # May have NaN
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        name = "CustomerNumericSchema"
        strict = False
        coerce = True


class CustomerCleanedSchema(pa.DataFrameModel):
    """Schema after handling missing values."""

    tenure: Series[int] = pa.Field(ge=0)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(ge=0, nullable=False)
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        name = "CustomerCleanedSchema"
        strict = False
        coerce = True


class CustomerBinaryEncodedSchema(pa.DataFrameModel):
    """Schema after binary encoding."""

    gender: Series[int] = pa.Field(isin=[0, 1])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[int] = pa.Field(isin=[0, 1])
    Dependents: Series[int] = pa.Field(isin=[0, 1])
    PhoneService: Series[int] = pa.Field(isin=[0, 1])
    PaperlessBilling: Series[int] = pa.Field(isin=[0, 1])
    Churn: Series[int] = pa.Field(isin=[0, 1])
    tenure: Series[int] = pa.Field(ge=0)
    MonthlyCharges: Series[float] = pa.Field(ge=0)
    TotalCharges: Series[float] = pa.Field(ge=0)

    class Config:
        name = "CustomerBinaryEncodedSchema"
        strict = False
        coerce = True


class PreprocessedCustomerSchema(pa.DataFrameModel):
    """Schema for fully preprocessed data (after scaling)."""

    # Binary encoded columns
    gender: Series[int] = pa.Field(isin=[0, 1])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[int] = pa.Field(isin=[0, 1])
    Dependents: Series[int] = pa.Field(isin=[0, 1])
    PhoneService: Series[int] = pa.Field(isin=[0, 1])
    PaperlessBilling: Series[int] = pa.Field(isin=[0, 1])
    Churn: Series[int] = pa.Field(isin=[0, 1])

    # Scaled numerical columns (can be negative after standardization)
    tenure: Series[float]
    MonthlyCharges: Series[float]
    TotalCharges: Series[float]

    class Config:
        name = "PreprocessedCustomerSchema"
        strict = False  # Allow one-hot encoded columns
        coerce = True


# =============================================================================
# FEATURE ENGINEERING SCHEMAS
# =============================================================================

class FeaturesEngineeredSchema(pa.DataFrameModel):
    """Schema for data after feature engineering."""

    # Original preprocessed columns
    Churn: Series[int] = pa.Field(isin=[0, 1])
    tenure: Series[float]
    MonthlyCharges: Series[float]
    TotalCharges: Series[float]

    # Service features
    total_services: Series[float] = pa.Field(ge=0)
    has_internet: Series[int] = pa.Field(isin=[0, 1])
    has_streaming: Series[int] = pa.Field(isin=[0, 1])
    has_security_services: Series[int] = pa.Field(isin=[0, 1])
    security_services_count: Series[float] = pa.Field(ge=0, le=4)
    has_multiple_lines: Series[int] = pa.Field(isin=[0, 1])

    # Tenure features
    tenure_group: Series[int] = pa.Field(isin=[0, 1, 2, 3])
    is_new_customer: Series[int] = pa.Field(isin=[0, 1])
    is_loyal_customer: Series[int] = pa.Field(isin=[0, 1])

    # Contract features
    is_month_to_month: Series[int] = pa.Field(isin=[0, 1])
    has_long_contract: Series[int] = pa.Field(isin=[0, 1])
    uses_electronic_check: Series[int] = pa.Field(isin=[0, 1])
    has_auto_payment: Series[int] = pa.Field(isin=[0, 1])

    # Charge features
    charge_per_service: Series[float]
    is_high_charges: Series[int] = pa.Field(isin=[0, 1])
    charge_tenure_ratio: Series[float]

    # Interaction features
    high_risk_combo: Series[int] = pa.Field(isin=[0, 1])
    churn_risk_score: Series[float] = pa.Field(ge=0, le=6)
    low_engagement: Series[int] = pa.Field(isin=[0, 1])
    senior_high_charges: Series[int] = pa.Field(isin=[0, 1])
    has_family: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        name = "FeaturesEngineeredSchema"
        strict = False  # Allow one-hot encoded columns
        coerce = True


# =============================================================================
# MODEL TRAINING SCHEMAS
# =============================================================================

class ModelInputSchema(pa.DataFrameModel):
    """Schema for model input features (X_train, X_test)."""

    # Core columns should be numeric
    tenure: Series[float] = pa.Field(nullable=False)
    MonthlyCharges: Series[float] = pa.Field(nullable=False)
    TotalCharges: Series[float] = pa.Field(nullable=False)

    class Config:
        name = "ModelInputSchema"
        strict = False  # Allow all feature columns
        coerce = True


class TargetSchema(pa.DataFrameModel):
    """Schema for target variable (y_train, y_test)."""

    Churn: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        name = "TargetSchema"
        strict = True
        coerce = True


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw customer data."""
    return RawCustomerSchema.validate(df)


def validate_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate preprocessed customer data."""
    return PreprocessedCustomerSchema.validate(df)


def validate_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate feature engineered data."""
    return FeaturesEngineeredSchema.validate(df)


def validate_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate model input features."""
    return ModelInputSchema.validate(df)


def validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """Validate target variable."""
    return TargetSchema.validate(df)
