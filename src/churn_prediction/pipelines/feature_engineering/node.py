import pandas as pd
import numpy as np
import pandera as pa
from pandera.typing import DataFrame
import logging

from churn_prediction.schemas import (
    PreprocessedCustomerSchema,
    FeaturesEngineeredSchema,
)

logger = logging.getLogger(__name__)


@pa.check_types
def create_service_features(
    data: DataFrame[PreprocessedCustomerSchema],
) -> pd.DataFrame:
    """Creates service-related features.

    Features created:
    - total_services: Count of all services subscribed
    - has_internet: Whether customer has internet service
    - has_streaming: Whether customer has any streaming service
    - has_security_services: Whether customer has any security/protection service
    - security_services_count: Count of security services
    - has_multiple_lines: Whether customer has multiple phone lines
    """
    df = data.copy()

    # Identify Yes service columns
    yes_service_cols = [
        col for col in df.columns
        if col.endswith("_Yes")
        and any(
            x in col
            for x in [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
        )
    ]

    # Total services count
    df["total_services"] = df[yes_service_cols].sum(axis=1)
    df["total_services"] = df["total_services"] + df["PhoneService"]

    # Has internet service
    no_internet_cols = [col for col in df.columns if "No internet service" in col]
    df["has_internet"] = (~df[no_internet_cols].any(axis=1)).astype(int)
    df["total_services"] = df["total_services"] + df["has_internet"]

    # Has streaming services
    streaming_cols = [
        col
        for col in df.columns
        if ("StreamingTV_Yes" in col or "StreamingMovies_Yes" in col)
    ]
    df["has_streaming"] = df[streaming_cols].any(axis=1).astype(int)

    # Has security/protection services
    security_cols = [
        col
        for col in df.columns
        if any(
            x in col
            for x in [
                "OnlineSecurity_Yes",
                "OnlineBackup_Yes",
                "DeviceProtection_Yes",
                "TechSupport_Yes",
            ]
        )
    ]
    df["has_security_services"] = df[security_cols].any(axis=1).astype(int)
    df["security_services_count"] = df[security_cols].sum(axis=1)

    # Has multiple lines
    multiple_lines_cols = [col for col in df.columns if "MultipleLines_Yes" in col]
    if multiple_lines_cols:
        df["has_multiple_lines"] = df[multiple_lines_cols].any(axis=1).astype(int)
    else:
        df["has_multiple_lines"] = 0

    return df


def create_tenure_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates tenure-based features.

    Features created:
    - tenure_group: Categorical grouping (0=new, 1=short, 2=mid, 3=long)
    - is_new_customer: Bottom 25% tenure
    - is_loyal_customer: Top 25% tenure
    """
    df = data.copy()

    # Calculate tenure quartiles
    tenure_q25 = df["tenure"].quantile(0.25)
    tenure_q50 = df["tenure"].quantile(0.50)
    tenure_q75 = df["tenure"].quantile(0.75)

    # Tenure group
    def get_tenure_group(tenure):
        if tenure <= tenure_q25:
            return 0  # New customer
        elif tenure <= tenure_q50:
            return 1  # Short-term
        elif tenure <= tenure_q75:
            return 2  # Mid-term
        else:
            return 3  # Long-term

    df["tenure_group"] = df["tenure"].apply(get_tenure_group)

    # Binary tenure flags
    df["is_new_customer"] = (df["tenure"] <= tenure_q25).astype(int)
    df["is_loyal_customer"] = (df["tenure"] > tenure_q75).astype(int)

    return df


def create_contract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates contract and payment related features.

    Features created:
    - is_month_to_month: High churn risk indicator
    - has_long_contract: Has 1 or 2 year contract
    - uses_electronic_check: Often associated with higher churn
    - has_auto_payment: Automatic payment method
    """
    df = data.copy()

    # Month-to-month contract (high risk)
    df["is_month_to_month"] = (
        (df["Contract_One year"] == 0) & (df["Contract_Two year"] == 0)
    ).astype(int)

    # Has long-term contract
    df["has_long_contract"] = (
        (df["Contract_One year"] == 1) | (df["Contract_Two year"] == 1)
    ).astype(int)

    # Electronic check payment
    electronic_check_col = [col for col in df.columns if "Electronic check" in col]
    if electronic_check_col:
        df["uses_electronic_check"] = df[electronic_check_col[0]].astype(int)
    else:
        df["uses_electronic_check"] = 0

    # Automatic payment
    auto_payment_cols = [col for col in df.columns if "automatic" in col.lower()]
    if auto_payment_cols:
        df["has_auto_payment"] = df[auto_payment_cols].any(axis=1).astype(int)
    else:
        df["has_auto_payment"] = 0

    return df


def create_charge_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates charge-related features.

    Features created:
    - charge_per_service: Monthly charges divided by total services
    - is_high_charges: Above 75th percentile monthly charges
    - charge_tenure_ratio: Monthly charges relative to tenure
    - avg_monthly_spend: Average monthly spend over tenure
    - tenure_normalized: Tenure normalized to 0-1 range (max ~72 months)
    - value_consistency: Ratio of expected vs actual total charges
    """
    df = data.copy()

    # Charge per service
    df["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)

    # High monthly charges flag
    monthly_q75 = df["MonthlyCharges"].quantile(0.75)
    df["is_high_charges"] = (df["MonthlyCharges"] > monthly_q75).astype(int)

    # Charge-tenure ratio
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1e-6)

    # Average monthly spend over tenure (TotalCharges / tenure)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1e-6)

    # Normalized tenure (0-1 range, max tenure is ~72 months)
    df["tenure_normalized"] = df["tenure"] / 72.0

    # Value consistency: how consistent is the customer's spending
    # Expected total = MonthlyCharges * tenure, actual = TotalCharges
    expected_total = df["MonthlyCharges"] * (df["tenure"] + 1e-6)
    df["value_consistency"] = df["TotalCharges"] / (expected_total + 1e-6)

    return df


@pa.check_types
def create_interaction_features(data: pd.DataFrame) -> DataFrame[FeaturesEngineeredSchema]:
    """Creates interaction and risk combination features.

    Features created:
    - high_risk_combo: New + month-to-month + no security
    - churn_risk_score: Composite risk score (0-6)
    - low_engagement: Few services + no long contract
    - senior_high_charges: Senior citizen with high charges
    - has_family: Has partner or dependents
    """
    df = data.copy()

    # High risk combination
    df["high_risk_combo"] = (
        (df["is_new_customer"] == 1)
        & (df["is_month_to_month"] == 1)
        & (df["has_security_services"] == 0)
    ).astype(int)

    # Churn risk score
    df["churn_risk_score"] = (
        df["is_new_customer"]
        + df["is_month_to_month"]
        + df["uses_electronic_check"]
        + df["is_high_charges"]
        + (1 - df["has_security_services"])
        + df["PaperlessBilling"]
    )

    # Low engagement
    df["low_engagement"] = (
        (df["total_services"] <= 2) & (df["has_long_contract"] == 0)
    ).astype(int)

    # Senior with high charges
    df["senior_high_charges"] = (
        (df["SeniorCitizen"] == 1) & (df["is_high_charges"] == 1)
    ).astype(int)

    # Has family
    df["has_family"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    logger.info("New features created: 25")
    return df
