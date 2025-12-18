Model Training Pipeline
=======================

The model training pipeline trains multiple classification models, evaluates
their performance, and selects the best model based on configurable metrics.
All experiments are tracked with MLflow.

Pipeline Overview
-----------------

.. code-block:: text

   features_engineered
       → split_data → X_train, X_test, y_train, y_test
       → train_logistic_regression → logistic_regression_model
       → train_random_forest → random_forest_model
       → train_gradient_boosting → gradient_boosting_model
       → evaluate_model (x3) → lr_metrics, rf_metrics, gb_metrics
       → select_best_model → best_model, model_comparison
       → get_feature_importance → feature_importance

Running the Pipeline
--------------------

.. code-block:: bash

   kedro run --pipeline model_training

Models Trained
--------------

1. Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~

A linear model that serves as the baseline. Fast to train and highly interpretable.

**Default Parameters:**

.. code-block:: yaml

   logistic_regression:
     C: 1.0
     max_iter: 1000
     class_weight: "balanced"

2. Random Forest
~~~~~~~~~~~~~~~~

An ensemble of decision trees using bagging. Good balance of accuracy and interpretability.

**Default Parameters:**

.. code-block:: yaml

   random_forest:
     n_estimators: 100
     max_depth: 10
     min_samples_split: 2
     min_samples_leaf: 1
     class_weight: "balanced"

3. Gradient Boosting
~~~~~~~~~~~~~~~~~~~~

Sequential boosting of weak learners. Often achieves highest accuracy.

**Default Parameters:**

.. code-block:: yaml

   gradient_boosting:
     n_estimators: 100
     learning_rate: 0.1
     max_depth: 3
     min_samples_split: 2
     min_samples_leaf: 1

Pipeline Nodes
--------------

1. Split Data
~~~~~~~~~~~~~

**Function:** ``split_data``

Splits data into training and test sets using stratified sampling to maintain
class balance.

**Parameters:**

* ``test_size``: Fraction for testing (default: 0.2)
* ``random_state``: Random seed (default: 42)
* ``target_column``: Name of target column (default: "Churn")

**MLflow Logging:**

* ``data_split_test_size``
* ``train_samples``
* ``test_samples``
* ``n_features``

2. Train Models
~~~~~~~~~~~~~~~

**Functions:** ``train_logistic_regression``, ``train_random_forest``, ``train_gradient_boosting``

Each training function:

1. Extracts model-specific parameters from configuration
2. Trains the model on training data
3. Logs hyperparameters to MLflow

**MLflow Logging (per model):**

* Model-specific hyperparameters with prefix (``lr_``, ``rf_``, ``gb_``)

3. Evaluate Model
~~~~~~~~~~~~~~~~~

**Function:** ``evaluate_model``

Evaluates each trained model on the test set.

**Metrics Calculated:**

+-----------------+-----------------------------------------------+
| Metric          | Description                                   |
+=================+===============================================+
| accuracy        | Overall prediction accuracy                   |
+-----------------+-----------------------------------------------+
| precision       | Positive predictive value                     |
+-----------------+-----------------------------------------------+
| recall          | True positive rate / sensitivity              |
+-----------------+-----------------------------------------------+
| f1_score        | Harmonic mean of precision and recall         |
+-----------------+-----------------------------------------------+
| roc_auc         | Area under ROC curve                          |
+-----------------+-----------------------------------------------+
| true_negatives  | Correctly predicted non-churners              |
+-----------------+-----------------------------------------------+
| false_positives | Non-churners predicted as churners            |
+-----------------+-----------------------------------------------+
| false_negatives | Churners predicted as non-churners            |
+-----------------+-----------------------------------------------+
| true_positives  | Correctly predicted churners                  |
+-----------------+-----------------------------------------------+

**MLflow Logging:**

* All metrics with model prefix (e.g., ``logistic_regression_f1_score``)

4. Select Best Model
~~~~~~~~~~~~~~~~~~~~

**Function:** ``select_best_model``

Compares all models and selects the best one based on the configured selection metric.

**Default Selection Metric:** ``f1_score``

**Why F1 Score?**

For churn prediction, both precision and recall matter:

* **High Precision**: Avoid wasting retention resources on non-churners
* **High Recall**: Identify as many churners as possible

F1 balances both concerns. Can be changed to ``roc_auc`` or ``recall`` if needed.

**MLflow Logging:**

* ``best_model`` parameter
* ``selection_metric`` parameter
* Best model metrics with ``best_`` prefix
* All models logged as artifacts
* Best model registered to MLflow Model Registry

5. Get Feature Importance
~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``get_feature_importance``

Extracts feature importance from the best model:

* **Tree models**: Uses ``feature_importances_`` attribute
* **Linear models**: Uses absolute coefficient values

**MLflow Logging:**

* Top 10 feature importances as metrics
* Feature importance CSV as artifact

Configuration
-------------

**File:** ``conf/base/parameters.yml``

.. code-block:: yaml

   model_training:
     target_column: "Churn"
     test_size: 0.2
     random_state: 42
     selection_metric: "f1_score"

     logistic_regression:
       C: 1.0
       max_iter: 1000
       class_weight: "balanced"

     random_forest:
       n_estimators: 100
       max_depth: 10
       min_samples_split: 2
       min_samples_leaf: 1
       class_weight: "balanced"

     gradient_boosting:
       n_estimators: 100
       learning_rate: 0.1
       max_depth: 3
       min_samples_split: 2
       min_samples_leaf: 1

Output Datasets
---------------

**Models:**

+---------------------------+-------------------------------------------+
| Dataset                   | Location                                  |
+===========================+===========================================+
| logistic_regression_model | data/06_models/logistic_regression.pkl    |
+---------------------------+-------------------------------------------+
| random_forest_model       | data/06_models/random_forest.pkl          |
+---------------------------+-------------------------------------------+
| gradient_boosting_model   | data/06_models/gradient_boosting.pkl      |
+---------------------------+-------------------------------------------+
| best_model                | data/06_models/best_model.pkl             |
+---------------------------+-------------------------------------------+

**Metrics:**

+---------------------------+-------------------------------------------+
| Dataset                   | Location                                  |
+===========================+===========================================+
| lr_metrics                | data/07_model_output/lr_metrics.json      |
+---------------------------+-------------------------------------------+
| rf_metrics                | data/07_model_output/rf_metrics.json      |
+---------------------------+-------------------------------------------+
| gb_metrics                | data/07_model_output/gb_metrics.json      |
+---------------------------+-------------------------------------------+
| model_comparison          | data/07_model_output/model_comparison.json|
+---------------------------+-------------------------------------------+
| feature_importance        | data/07_model_output/feature_importance   |
+---------------------------+-------------------------------------------+

MLflow Integration
------------------

All training runs are logged to MLflow:

**View experiments:**

.. code-block:: bash

   mlflow ui --port 5000

**Registered Model:**

The best model is registered to the MLflow Model Registry as ``churn_prediction_model``.

**Artifacts:**

* ``best_model/`` - Best model files
* ``logistic_regression_model/`` - LR model files
* ``random_forest_model/`` - RF model files
* ``gradient_boosting_model/`` - GB model files
* ``feature_importance/`` - Feature importance CSV

Class Imbalance Handling
------------------------

The dataset has imbalanced classes (~27% churn). This is handled by:

1. **Stratified splitting**: Maintains class ratio in train/test
2. **class_weight="balanced"**: Adjusts class weights inversely proportional to frequency
3. **F1 score selection**: Balances precision and recall

Data Validation
---------------

The model training pipeline uses **Pandera** for schema-based validation:

**Input Validation:**

* ``FeaturesEngineeredSchema``: Validates input data has all 22 engineered features

**Output Validation:**

* ``ModelInputSchema``: Ensures training/test features have no null values
* ``TargetSchema``: Validates target is binary (0 or 1)

**Example validation decorator:**

.. code-block:: python

   @pa.check_types
   def split_data(
       data: DataFrame[FeaturesEngineeredSchema], parameters: Dict[str, Any]
   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
       # Validates input automatically
       ...
       # Manual output validation
       ModelInputSchema.validate(X_train)
       TargetSchema.validate(y_train_df)

Validation ensures data quality at critical pipeline boundaries, catching errors early.
