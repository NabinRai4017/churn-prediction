.. churn_prediction documentation master file

Churn Prediction Project Documentation
======================================

A machine learning project built with `Kedro <https://kedro.org>`_ to predict customer churn
using telco customer data. The project implements a complete ML pipeline from data preprocessing
to model training, hyperparameter optimization, and reporting, with MLflow integration for
experiment tracking, Optuna for hyperparameter tuning, and Pandera for data validation.

Overview
--------

This project predicts whether a customer will churn (leave the service) based on their
demographics, account information, and service usage patterns. It uses the Telco Customer
Churn dataset and implements three classification models:

* **Logistic Regression** - Baseline linear model
* **Random Forest** - Ensemble tree-based model
* **Gradient Boosting** - Sequential boosting model

Key Features
------------

* **Modular Kedro Pipelines** - Organized into 5 distinct pipelines
* **Automated Feature Engineering** - Creates 22 derived features
* **Model Comparison** - Trains and evaluates multiple models
* **Optuna Hyperparameter Tuning** - Automated optimization with 50 trials and cross-validation
* **Pandera Data Validation** - Schema-based DataFrame validation throughout the pipeline
* **MLflow Integration** - Tracks experiments, parameters, and metrics
* **Comprehensive Reporting** - Generates reports and visualizations

Project Structure
-----------------

.. code-block:: text

   churn_prediction/
   ├── conf/                          # Configuration files
   │   └── base/
   │       ├── catalog.yml            # Data catalog definitions
   │       ├── parameters.yml         # Pipeline parameters
   │       └── mlflow.yml             # MLflow configuration
   ├── data/                          # Data directory (layered)
   │   ├── 01_raw/                    # Raw input data
   │   ├── 02_intermediate/           # Intermediate processed data
   │   ├── 03_primary/                # Primary preprocessed data
   │   ├── 04_feature/                # Feature engineered data
   │   ├── 05_model_input/            # Train/test splits
   │   ├── 06_models/                 # Trained model files
   │   ├── 07_model_output/           # Model metrics and outputs
   │   └── 08_reporting/              # Reports and visualizations
   └── src/churn_prediction/
       ├── schemas.py                 # Pandera data validation schemas
       └── pipelines/
           ├── data_processing/       # Data preprocessing pipeline
           ├── feature_engineering/   # Feature creation pipeline
           ├── model_training/        # Model training pipeline
           ├── hyperparameter_tuning/ # Optuna hyperparameter optimization
           └── reporting/             # Reporting pipeline

Quick Start
-----------

**Install dependencies:**

.. code-block:: bash

   pip install -r requirements.txt

**Run the complete pipeline:**

.. code-block:: bash

   kedro run

**Run individual pipelines:**

.. code-block:: bash

   kedro run --pipeline data_processing
   kedro run --pipeline feature_engineering
   kedro run --pipeline model_training
   kedro run --pipeline hyperparameter_tuning
   kedro run --pipeline reporting

**View MLflow experiments:**

.. code-block:: bash

   mlflow ui --port 5000

Then open http://127.0.0.1:5000 in your browser.

Pipeline Documentation
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Pipelines

   pipelines/data_processing
   pipelines/feature_engineering
   pipelines/model_training
   pipelines/hyperparameter_tuning
   pipelines/reporting

Data Validation
---------------

This project uses **Pandera** for schema-based DataFrame validation throughout the pipeline.
Schemas are defined in ``src/churn_prediction/schemas.py`` and validate data at each
transformation step.

**Validation Schemas:**

* ``RawCustomerSchema`` - Validates raw input data (21 columns)
* ``CustomerNoIdSchema`` - After dropping customerID
* ``CustomerCleanedSchema`` - After handling missing values
* ``CustomerBinaryEncodedSchema`` - After binary encoding
* ``PreprocessedCustomerSchema`` - Fully preprocessed data
* ``FeaturesEngineeredSchema`` - After feature engineering (22 new features)
* ``ModelInputSchema`` - Training/test features (no nulls)
* ``TargetSchema`` - Binary target variable (0/1)

**Validation Features:**

* Type checking for each column
* Value constraints (ranges, allowed values)
* Null checking where required
* Automatic validation via ``@pa.check_types`` decorator

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
