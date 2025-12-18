from kedro.pipeline import Pipeline, node

from .node import (
    drop_customer_id,
    convert_total_charges,
    handle_missing_values,
    encode_binary_columns,
    encode_multiclass_columns,
    scale_numerical_features,
    preprocess_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the data processing pipeline.

    This pipeline performs the following preprocessing steps:
    1. Drop customerID column
    2. Convert TotalCharges to numeric
    3. Handle missing values
    4. Encode binary categorical columns
    5. One-hot encode multi-class categorical columns
    6. Scale numerical features
    """
    return Pipeline(
        [
            node(
                func=drop_customer_id,
                inputs="customers",
                outputs="customers_no_id",
                name="drop_customer_id_node",
            ),
            node(
                func=convert_total_charges,
                inputs="customers_no_id",
                outputs="customers_numeric",
                name="convert_total_charges_node",
            ),
            node(
                func=handle_missing_values,
                inputs="customers_numeric",
                outputs="customers_cleaned",
                name="handle_missing_values_node",
            ),
            node(
                func=encode_binary_columns,
                inputs="customers_cleaned",
                outputs="customers_binary_encoded",
                name="encode_binary_columns_node",
            ),
            node(
                func=encode_multiclass_columns,
                inputs="customers_binary_encoded",
                outputs="customers_encoded",
                name="encode_multiclass_columns_node",
            ),
            node(
                func=scale_numerical_features,
                inputs="customers_encoded",
                outputs="preprocessed_customers",
                name="scale_numerical_features_node",
            ),
        ]
    )
