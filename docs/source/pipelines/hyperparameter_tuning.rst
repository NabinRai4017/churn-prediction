Hyperparameter Tuning Pipeline
==============================

The hyperparameter tuning pipeline uses **Optuna** to automatically optimize hyperparameters
for the best-performing model from the baseline comparison. It includes full MLflow integration
for tracking individual trials.

Overview
--------

This pipeline:

1. Identifies the best model from baseline comparison
2. Runs Optuna optimization with 50 trials
3. Uses TPE (Tree-structured Parzen Estimator) sampler
4. Performs 5-fold stratified cross-validation
5. Trains final model with optimized hyperparameters
6. Creates detailed comparison report

Pipeline Nodes
--------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Node
     - Description
   * - ``identify_best_model``
     - Reads ``model_comparison.json`` to identify the best baseline model
   * - ``run_optuna_study``
     - Runs Optuna optimization for 50 trials with cross-validation
   * - ``train_tuned_model``
     - Trains final model using optimized hyperparameters
   * - ``evaluate_tuned_model``
     - Evaluates tuned model on test set
   * - ``create_tuning_report``
     - Creates comparison report (baseline vs tuned performance)

Optuna Configuration
--------------------

The hyperparameter search is configured in ``conf/base/parameters.yml``:

.. code-block:: yaml

   hyperparameter_tuning:
     n_trials: 50              # Number of Optuna trials
     optimization_metric: "recall"  # Metric to optimize
     cv_folds: 5               # Cross-validation folds
     random_state: 42          # Random seed for reproducibility
     direction: "maximize"     # Optimization direction
     pruning: true             # Enable trial pruning

Search Spaces
-------------

Each model type has its own hyperparameter search space:

Logistic Regression
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Parameter
     - Type
     - Range
   * - ``C``
     - float (log)
     - 0.001 - 100.0
   * - ``max_iter``
     - int
     - 100 - 2000

Random Forest
^^^^^^^^^^^^^

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Parameter
     - Type
     - Range
   * - ``n_estimators``
     - int
     - 50 - 300
   * - ``max_depth``
     - int
     - 3 - 20
   * - ``min_samples_split``
     - int
     - 2 - 20
   * - ``min_samples_leaf``
     - int
     - 1 - 10

Gradient Boosting
^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Parameter
     - Type
     - Range
   * - ``n_estimators``
     - int
     - 50 - 300
   * - ``learning_rate``
     - float (log)
     - 0.01 - 0.3
   * - ``max_depth``
     - int
     - 2 - 10
   * - ``min_samples_split``
     - int
     - 2 - 20
   * - ``min_samples_leaf``
     - int
     - 1 - 10

MLflow Integration
------------------

The pipeline integrates with MLflow for comprehensive tracking:

**Study-Level Logging:**

* ``tuning_model`` - Name of model being tuned
* ``tuning_n_trials`` - Number of optimization trials
* ``tuning_cv_folds`` - Cross-validation folds
* ``tuning_metric`` - Optimization metric

**Trial-Level Logging (via MLflowCallback):**

* Each trial logged as nested MLflow run
* Hyperparameters for each trial
* Cross-validation score
* Trial status (completed/pruned)

**Best Model Logging:**

* ``best_cv_{metric}`` - Best cross-validation score
* ``best_*`` parameters - Optimized hyperparameters
* Registered model: ``churn_prediction_tuned_{model_name}``

Output Datasets
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Dataset
     - Description
   * - ``optuna_study``
     - Serialized Optuna study object (``data/06_models/optuna_study.pkl``)
   * - ``best_hyperparameters``
     - Optimized hyperparameters (``data/07_model_output/best_hyperparameters.json``)
   * - ``tuned_model``
     - Trained model with best params (``data/06_models/tuned_model.pkl``)
   * - ``tuned_model_metrics``
     - Test set metrics (``data/07_model_output/tuned_model_metrics.json``)
   * - ``tuning_report``
     - Full comparison report (``data/07_model_output/tuning_report.json``)

Tuning Report
-------------

The tuning report includes:

* Model name and optimization metric
* Baseline score vs tuned score
* Absolute and percentage improvement
* Best hyperparameters found
* Study statistics (trials, completed, pruned)
* Optimization history (top 20 trials)

Usage
-----

Run the hyperparameter tuning pipeline after model training:

.. code-block:: bash

   # First run model training to generate baseline
   kedro run --pipeline model_training

   # Then run hyperparameter tuning
   kedro run --pipeline hyperparameter_tuning

Or run the complete pipeline:

.. code-block:: bash

   kedro run

Example Output
--------------

Sample tuning report summary:

.. code-block:: json

   {
     "model_name": "random_forest",
     "optimization_metric": "recall",
     "baseline_score": 0.62,
     "tuned_score": 0.67,
     "improvement": 0.05,
     "improvement_percentage": 8.06,
     "best_hyperparameters": {
       "n_estimators": 187,
       "max_depth": 12,
       "min_samples_split": 5,
       "min_samples_leaf": 2
     },
     "study_statistics": {
       "total_trials": 50,
       "completed_trials": 48,
       "pruned_trials": 2
     }
   }
