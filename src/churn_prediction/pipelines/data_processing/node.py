import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandera as pa
from pandera.typing import DataFrame
import logging

from churn_prediction.schemas import (
    RawCustomerSchema,
    CustomerNoIdSchema,
    CustomerNumericSchema,
    CustomerCleanedSchema,
    CustomerBinaryEncodedSchema,
    PreprocessedCustomerSchema,
)

logger = logging.getLogger(__name__)


@pa.check_types
def drop_customer_id(
    data: DataFrame[RawCustomerSchema],
) -> DataFrame[CustomerNoIdSchema]:
    """Drops the customerID column as it's not predictive."""
    return data.drop(columns=["customerID"])


@pa.check_types
def convert_total_charges(
    data: DataFrame[CustomerNoIdSchema],
) -> DataFrame[CustomerNumericSchema]:
    """Converts TotalCharges from object to float64, handling empty strings."""
    df = data.copy()
    # Replace empty strings with NaN, then convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info(f"Converted TotalCharges to numeric. NaN count: {df['TotalCharges'].isna().sum()}")
    return df


@pa.check_types
def handle_missing_values(
    data: DataFrame[CustomerNumericSchema],
) -> DataFrame[CustomerCleanedSchema]:
    """Handles missing values in TotalCharges.

    For new customers (tenure=0), TotalCharges is set to 0.
    For others, missing values are filled with median.
    """
    df = data.copy()

    # For customers with tenure=0, set TotalCharges to 0
    mask_new_customers = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    df.loc[mask_new_customers, "TotalCharges"] = 0

    # For remaining missing values, fill with median
    median_value = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

    logger.info(f"Missing values handled. Remaining NaN: {df['TotalCharges'].isna().sum()}")
    return df


@pa.check_types
def encode_binary_columns(
    data: DataFrame[CustomerCleanedSchema],
) -> DataFrame[CustomerBinaryEncodedSchema]:
    """Encodes binary categorical columns to 0/1.

    Binary columns: gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn
    """
    df = data.copy()

    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "Churn": {"No": 0, "Yes": 1},
    }

    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    logger.info("Binary columns encoded successfully")
    return df


def encode_multiclass_columns(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes multi-class categorical columns.

    Multi-class columns: MultipleLines, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies, Contract, PaymentMethod

    Note: No strict schema validation here due to dynamic column creation.
    """
    df = data.copy()

    multiclass_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    # Filter to only existing columns
    cols_to_encode = [col for col in multiclass_cols if col in df.columns]

    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    logger.info(f"One-hot encoding complete. New shape: {df.shape}")
    return df


@pa.check_types
def scale_numerical_features(data: pd.DataFrame) -> DataFrame[PreprocessedCustomerSchema]:
    """Scales numerical features using StandardScaler.

    Numerical columns: tenure, MonthlyCharges, TotalCharges
    """
    df = data.copy()

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Filter to only existing columns
    cols_to_scale = [col for col in numerical_cols if col in df.columns]

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    logger.info("Numerical features scaled successfully")
    return df


@pa.check_types
def preprocess_data(
    data: DataFrame[RawCustomerSchema],
) -> DataFrame[PreprocessedCustomerSchema]:
    """Complete preprocessing pipeline in a single function.

    Steps:
    1. Drop customerID
    2. Convert TotalCharges to numeric
    3. Handle missing values
    4. Encode binary columns
    5. One-hot encode multi-class columns
    6. Scale numerical features
    """
    df = data.copy()

    # Step 1: Drop customerID
    df = drop_customer_id(df)

    # Step 2: Convert TotalCharges to numeric
    df = convert_total_charges(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Encode binary columns
    df = encode_binary_columns(df)

    # Step 5: One-hot encode multi-class columns
    df = encode_multiclass_columns(df)

    # Step 6: Scale numerical features
    df = scale_numerical_features(df)

    return df
