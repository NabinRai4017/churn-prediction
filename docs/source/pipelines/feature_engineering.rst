Feature Engineering Pipeline
============================

The feature engineering pipeline creates derived features from the preprocessed
data to improve model performance and capture important business patterns.

Pipeline Overview
-----------------

.. code-block:: text

   preprocessed_customers
       → create_service_features
       → create_tenure_features
       → create_contract_features
       → create_charge_features
       → create_interaction_features
       → features_engineered

Running the Pipeline
--------------------

.. code-block:: bash

   kedro run --pipeline feature_engineering

Features Created
----------------

The pipeline creates **22 new features** organized into 5 categories.

Service Features (6 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_service_features``

+------------------------+----------------------------------------------------+
| Feature                | Description                                        |
+========================+====================================================+
| total_services         | Count of all services subscribed (0-8)             |
+------------------------+----------------------------------------------------+
| has_internet           | Whether customer has internet service (0/1)        |
+------------------------+----------------------------------------------------+
| has_streaming          | Has StreamingTV or StreamingMovies (0/1)           |
+------------------------+----------------------------------------------------+
| has_security_services  | Has any security/protection service (0/1)          |
+------------------------+----------------------------------------------------+
| security_services_count| Count of security services (0-4)                   |
+------------------------+----------------------------------------------------+
| has_multiple_lines     | Has multiple phone lines (0/1)                     |
+------------------------+----------------------------------------------------+

**Business Insight:** Customers with more services tend to be more engaged and
less likely to churn. Security services especially indicate commitment.

Tenure Features (3 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_tenure_features``

+------------------------+----------------------------------------------------+
| Feature                | Description                                        |
+========================+====================================================+
| tenure_group           | Categorical: 0=new, 1=short, 2=mid, 3=long         |
+------------------------+----------------------------------------------------+
| is_new_customer        | Bottom 25% tenure (0/1)                            |
+------------------------+----------------------------------------------------+
| is_loyal_customer      | Top 25% tenure (0/1)                               |
+------------------------+----------------------------------------------------+

**Business Insight:** New customers (low tenure) have significantly higher
churn rates. Tenure is one of the strongest predictors of churn.

Contract Features (4 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_contract_features``

+------------------------+----------------------------------------------------+
| Feature                | Description                                        |
+========================+====================================================+
| is_month_to_month      | Has month-to-month contract (0/1)                  |
+------------------------+----------------------------------------------------+
| has_long_contract      | Has 1-year or 2-year contract (0/1)                |
+------------------------+----------------------------------------------------+
| uses_electronic_check  | Pays via electronic check (0/1)                    |
+------------------------+----------------------------------------------------+
| has_auto_payment       | Uses automatic payment method (0/1)                |
+------------------------+----------------------------------------------------+

**Business Insight:** Month-to-month contracts have 3-4x higher churn than
annual contracts. Electronic check payment is associated with higher churn.

Charge Features (3 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_charge_features``

+------------------------+----------------------------------------------------+
| Feature                | Description                                        |
+========================+====================================================+
| charge_per_service     | MonthlyCharges / (total_services + 1)              |
+------------------------+----------------------------------------------------+
| is_high_charges        | Above 75th percentile monthly charges (0/1)        |
+------------------------+----------------------------------------------------+
| charge_tenure_ratio    | MonthlyCharges / tenure (value perception proxy)   |
+------------------------+----------------------------------------------------+

**Business Insight:** Customers who pay more relative to their service usage
may perceive lower value and be more likely to churn.

Interaction Features (6 features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_interaction_features``

+------------------------+----------------------------------------------------+
| Feature                | Description                                        |
+========================+====================================================+
| high_risk_combo        | New + month-to-month + no security (0/1)           |
+------------------------+----------------------------------------------------+
| churn_risk_score       | Composite risk score (0-6)                         |
+------------------------+----------------------------------------------------+
| low_engagement         | Few services + no long contract (0/1)              |
+------------------------+----------------------------------------------------+
| senior_high_charges    | Senior citizen with high charges (0/1)             |
+------------------------+----------------------------------------------------+
| has_family             | Has partner or dependents (0/1)                    |
+------------------------+----------------------------------------------------+

**Churn Risk Score Components:**

.. code-block:: text

   churn_risk_score = (
       is_new_customer           +
       is_month_to_month         +
       uses_electronic_check     +
       is_high_charges           +
       (1 - has_security_services) +
       PaperlessBilling
   )

**Business Insight:** Combining multiple risk factors captures customers
with the highest churn probability. The ``high_risk_combo`` flag identifies
the most at-risk segment.

Pipeline Nodes
--------------

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Node
     - Input
     - Output
   * - create_service_features
     - preprocessed_customers
     - customers_with_service_features
   * - create_tenure_features
     - customers_with_service_features
     - customers_with_tenure_features
   * - create_contract_features
     - customers_with_tenure_features
     - customers_with_contract_features
   * - create_charge_features
     - customers_with_contract_features
     - customers_with_charge_features
   * - create_interaction_features
     - customers_with_charge_features
     - features_engineered

Output Data
-----------

**Dataset:** ``features_engineered``

**Location:** ``data/04_feature/features_engineered.parquet``

The output contains all original preprocessed features plus the 22 new
engineered features, ready for model training.

Feature Importance
------------------

Based on model training results, the most predictive engineered features typically are:

1. ``tenure_group`` / ``is_new_customer`` - Tenure-related features
2. ``is_month_to_month`` - Contract type
3. ``churn_risk_score`` - Composite risk indicator
4. ``total_services`` - Service engagement level
5. ``has_security_services`` - Security service indicator
