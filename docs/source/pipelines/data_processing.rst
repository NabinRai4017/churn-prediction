Data Processing Pipeline
========================

The data processing pipeline transforms raw customer data into a clean, preprocessed
format ready for feature engineering and model training.

Pipeline Overview
-----------------

.. code-block:: text

   customers (raw)
       → drop_customer_id
       → convert_total_charges
       → handle_missing_values
       → encode_binary_columns
       → encode_multiclass_columns
       → scale_numerical_features
       → preprocessed_customers

Running the Pipeline
--------------------

.. code-block:: bash

   kedro run --pipeline data_processing

Input Data
----------

**Dataset:** ``customers`` (Telco Customer Churn dataset)

**Location:** ``data/01_raw/telco-customer-churn.csv``

**Columns:**

* ``customerID`` - Unique customer identifier
* ``gender`` - Male/Female
* ``SeniorCitizen`` - 0/1
* ``Partner`` - Yes/No
* ``Dependents`` - Yes/No
* ``tenure`` - Months with company
* ``PhoneService`` - Yes/No
* ``MultipleLines`` - Yes/No/No phone service
* ``InternetService`` - DSL/Fiber optic/No
* ``OnlineSecurity`` - Yes/No/No internet service
* ``OnlineBackup`` - Yes/No/No internet service
* ``DeviceProtection`` - Yes/No/No internet service
* ``TechSupport`` - Yes/No/No internet service
* ``StreamingTV`` - Yes/No/No internet service
* ``StreamingMovies`` - Yes/No/No internet service
* ``Contract`` - Month-to-month/One year/Two year
* ``PaperlessBilling`` - Yes/No
* ``PaymentMethod`` - Electronic check/Mailed check/Bank transfer/Credit card
* ``MonthlyCharges`` - Monthly charge amount
* ``TotalCharges`` - Total charges to date
* ``Churn`` - Yes/No (target variable)

Pipeline Nodes
--------------

1. Drop Customer ID
~~~~~~~~~~~~~~~~~~~

**Function:** ``drop_customer_id``

Removes the ``customerID`` column as it provides no predictive value.

.. code-block:: python

   def drop_customer_id(data: pd.DataFrame) -> pd.DataFrame:
       return data.drop(columns=["customerID"])

2. Convert Total Charges
~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``convert_total_charges``

Converts ``TotalCharges`` from string to numeric type. The raw data contains
empty strings for new customers (tenure=0), which are converted to NaN.

.. code-block:: python

   def convert_total_charges(data: pd.DataFrame) -> pd.DataFrame:
       df = data.copy()
       df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
       return df

3. Handle Missing Values
~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``handle_missing_values``

Handles missing values in ``TotalCharges``:

* New customers (tenure=0): Set ``TotalCharges`` to 0
* Others: Fill with median value

.. code-block:: python

   def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
       df = data.copy()
       mask_new_customers = (df["tenure"] == 0) & (df["TotalCharges"].isna())
       df.loc[mask_new_customers, "TotalCharges"] = 0
       median_value = df["TotalCharges"].median()
       df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
       return df

4. Encode Binary Columns
~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``encode_binary_columns``

Converts binary categorical columns to 0/1 encoding:

+-------------------+---------------------+
| Column            | Mapping             |
+===================+=====================+
| gender            | Female=0, Male=1    |
+-------------------+---------------------+
| Partner           | No=0, Yes=1         |
+-------------------+---------------------+
| Dependents        | No=0, Yes=1         |
+-------------------+---------------------+
| PhoneService      | No=0, Yes=1         |
+-------------------+---------------------+
| PaperlessBilling  | No=0, Yes=1         |
+-------------------+---------------------+
| Churn             | No=0, Yes=1         |
+-------------------+---------------------+

5. Encode Multiclass Columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``encode_multiclass_columns``

Applies one-hot encoding to multi-class categorical columns using
``pd.get_dummies`` with ``drop_first=True`` to avoid multicollinearity.

**Columns encoded:**

* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaymentMethod

6. Scale Numerical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``scale_numerical_features``

Applies StandardScaler (z-score normalization) to numerical columns:

* ``tenure``
* ``MonthlyCharges``
* ``TotalCharges``

Output Data
-----------

**Dataset:** ``preprocessed_customers``

**Location:** ``data/03_primary/preprocessed_customers.parquet``

The output contains all original features transformed plus one-hot encoded
columns, ready for feature engineering.

Intermediate Datasets
---------------------

+---------------------------+-----------------------------------------------+
| Dataset                   | Description                                   |
+===========================+===============================================+
| customers_no_id           | After dropping customerID                     |
+---------------------------+-----------------------------------------------+
| customers_numeric         | After TotalCharges conversion                 |
+---------------------------+-----------------------------------------------+
| customers_cleaned         | After handling missing values                 |
+---------------------------+-----------------------------------------------+
| customers_binary_encoded  | After binary encoding                         |
+---------------------------+-----------------------------------------------+
| customers_encoded         | After one-hot encoding                        |
+---------------------------+-----------------------------------------------+
| preprocessed_customers    | Final output after scaling                    |
+---------------------------+-----------------------------------------------+

Data Validation
---------------

The data processing pipeline uses **Pandera** for schema-based validation at each step:

**Validation Schemas:**

+---------------------------+-----------------------------------------------+
| Schema                    | Validates                                     |
+===========================+===============================================+
| RawCustomerSchema         | Input data (21 columns, correct types)        |
+---------------------------+-----------------------------------------------+
| CustomerNoIdSchema        | After dropping customerID                     |
+---------------------------+-----------------------------------------------+
| CustomerNumericSchema     | TotalCharges is numeric                       |
+---------------------------+-----------------------------------------------+
| CustomerCleanedSchema     | No missing values in TotalCharges             |
+---------------------------+-----------------------------------------------+
| CustomerBinaryEncodedSchema| Binary columns are 0/1 integers              |
+---------------------------+-----------------------------------------------+
| PreprocessedCustomerSchema| Final preprocessed data                       |
+---------------------------+-----------------------------------------------+

**Validation Rules:**

* **Type checking**: Ensures correct data types (int, float, string)
* **Value constraints**: Validates ranges (e.g., tenure >= 0, SeniorCitizen in [0,1])
* **Categorical validation**: Checks allowed values (e.g., gender in ["Male", "Female"])
* **Null checking**: Enforces non-null constraints after cleaning

**Example usage with decorator:**

.. code-block:: python

   @pa.check_types
   def drop_customer_id(
       data: DataFrame[RawCustomerSchema],
   ) -> DataFrame[CustomerNoIdSchema]:
       return data.drop(columns=["customerID"])

Validation errors are raised immediately when data doesn't conform to the schema,
making it easy to identify and fix data quality issues.
