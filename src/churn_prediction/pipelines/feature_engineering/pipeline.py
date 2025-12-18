from kedro.pipeline import Pipeline, node

from .node import (
    create_service_features,
    create_tenure_features,
    create_contract_features,
    create_charge_features,
    create_interaction_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Creates the feature engineering pipeline.

    This pipeline creates the following feature groups:
    1. Service features (total_services, has_internet, has_streaming, etc.)
    2. Tenure features (tenure_group, is_new_customer, is_loyal_customer)
    3. Contract features (is_month_to_month, has_long_contract, etc.)
    4. Charge features (charge_per_service, is_high_charges, etc.)
    5. Interaction features (high_risk_combo, churn_risk_score, etc.)
    """
    return Pipeline(
        [
            node(
                func=create_service_features,
                inputs="preprocessed_customers",
                outputs="customers_with_service_features",
                name="create_service_features_node",
            ),
            node(
                func=create_tenure_features,
                inputs="customers_with_service_features",
                outputs="customers_with_tenure_features",
                name="create_tenure_features_node",
            ),
            node(
                func=create_contract_features,
                inputs="customers_with_tenure_features",
                outputs="customers_with_contract_features",
                name="create_contract_features_node",
            ),
            node(
                func=create_charge_features,
                inputs="customers_with_contract_features",
                outputs="customers_with_charge_features",
                name="create_charge_features_node",
            ),
            node(
                func=create_interaction_features,
                inputs="customers_with_charge_features",
                outputs="features_engineered",
                name="create_interaction_features_node",
            ),
        ]
    )
