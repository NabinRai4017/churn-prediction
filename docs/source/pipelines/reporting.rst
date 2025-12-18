Reporting Pipeline
==================

The reporting pipeline generates comprehensive reports and visualizations
summarizing model performance and insights from the churn prediction analysis.

Pipeline Overview
-----------------

.. code-block:: text

   model_comparison, feature_importance
       → Reports:
           → create_model_comparison_report
           → create_confusion_matrix_report
           → create_feature_importance_report
           → create_detailed_metrics_report
           → create_executive_summary
       → Visualizations:
           → plot_model_comparison
           → plot_confusion_matrices
           → plot_feature_importance
           → plot_metrics_radar
           → plot_churn_prediction_summary

Running the Pipeline
--------------------

.. code-block:: bash

   kedro run --pipeline reporting

Reports Generated
-----------------

1. Model Comparison Report
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_model_comparison_report``

**Output:** ``data/08_reporting/model_comparison_report.csv``

A ranked comparison of all trained models showing:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Models are sorted by the selection metric (default: Recall).

2. Confusion Matrix Report
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_confusion_matrix_report``

**Output:** ``data/08_reporting/confusion_matrix_report.csv``

Detailed confusion matrix statistics for each model:

+-----------------+-------------------------------------------+
| Metric          | Description                               |
+=================+===========================================+
| True Negatives  | Correctly predicted non-churners          |
+-----------------+-------------------------------------------+
| False Positives | Non-churners wrongly predicted as churn   |
+-----------------+-------------------------------------------+
| False Negatives | Churners missed (predicted as no churn)   |
+-----------------+-------------------------------------------+
| True Positives  | Correctly predicted churners              |
+-----------------+-------------------------------------------+
| TN Rate         | True Negative Rate (Specificity)          |
+-----------------+-------------------------------------------+
| FP Rate         | False Positive Rate                       |
+-----------------+-------------------------------------------+
| FN Rate         | False Negative Rate                       |
+-----------------+-------------------------------------------+
| TP Rate         | True Positive Rate (Recall)               |
+-----------------+-------------------------------------------+

3. Feature Importance Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_feature_importance_report``

**Output:** ``data/08_reporting/feature_importance_report.csv``

Top 20 most important features with:

* Rank
* Feature name
* Importance score
* Percentage of total importance
* Cumulative percentage

4. Detailed Metrics Report
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_detailed_metrics_report``

**Output:** ``data/08_reporting/detailed_metrics_report.csv``

Extended metrics for each model including:

* Accuracy, Precision, Recall, F1 Score, ROC-AUC
* Specificity (True Negative Rate)
* NPV (Negative Predictive Value)
* FPR (False Positive Rate)
* FNR (False Negative Rate)
* Support counts for each class

5. Executive Summary
~~~~~~~~~~~~~~~~~~~~

**Function:** ``create_executive_summary``

**Output:** ``data/08_reporting/executive_summary.txt``

A text-based executive report containing:

* Best model identification and rationale
* Key performance metrics summary
* Confusion matrix interpretation
* Top 5 churn predictors
* Model comparison overview
* Business recommendations

This report is also logged to MLflow as an artifact.

Visualizations Generated
------------------------

All plots are saved to ``data/08_reporting/plots/`` and logged to MLflow.

1. Model Comparison Bar Chart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``plot_model_comparison``

**Output:** ``model_comparison.png``

A grouped bar chart comparing all five metrics (Accuracy, Precision, Recall,
F1 Score, ROC-AUC) across all trained models. Includes value labels for
precise comparison.

2. Confusion Matrices Heatmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``plot_confusion_matrices``

**Output:** ``confusion_matrices.png``

Side-by-side heatmap visualizations of confusion matrices for all models.
Uses a blue color scale with annotated cell counts.

3. Feature Importance Bar Chart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``plot_feature_importance``

**Output:** ``feature_importance.png``

A horizontal bar chart showing the top 15 most important features.
Uses a viridis color gradient with importance scores labeled.

4. Metrics Radar Chart
~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``plot_metrics_radar``

**Output:** ``metrics_radar.png``

A radar (spider) chart overlaying all models on the same plot.
Useful for quickly identifying which model excels in specific metrics.

5. Churn Prediction Summary Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function:** ``plot_churn_prediction_summary``

**Output:** ``churn_summary_dashboard.png``

A 4-panel dashboard for the best model containing:

* **Panel 1**: Confusion matrix heatmap
* **Panel 2**: Key metrics bar chart
* **Panel 3**: Prediction accuracy pie chart (correct vs incorrect)
* **Panel 4**: Model comparison (F1 Score and ROC-AUC)

Output Files
------------

**Reports (CSV/TXT):**

+---------------------------+--------------------------------------------------+
| File                      | Location                                         |
+===========================+==================================================+
| model_comparison_report   | data/08_reporting/model_comparison_report.csv    |
+---------------------------+--------------------------------------------------+
| confusion_matrix_report   | data/08_reporting/confusion_matrix_report.csv    |
+---------------------------+--------------------------------------------------+
| feature_importance_report | data/08_reporting/feature_importance_report.csv  |
+---------------------------+--------------------------------------------------+
| detailed_metrics_report   | data/08_reporting/detailed_metrics_report.csv    |
+---------------------------+--------------------------------------------------+
| executive_summary         | data/08_reporting/executive_summary.txt          |
+---------------------------+--------------------------------------------------+

**Visualizations (PNG):**

+---------------------------+------------------------------------------------------+
| File                      | Location                                             |
+===========================+======================================================+
| model_comparison          | data/08_reporting/plots/model_comparison.png         |
+---------------------------+------------------------------------------------------+
| confusion_matrices        | data/08_reporting/plots/confusion_matrices.png       |
+---------------------------+------------------------------------------------------+
| feature_importance        | data/08_reporting/plots/feature_importance.png       |
+---------------------------+------------------------------------------------------+
| metrics_radar             | data/08_reporting/plots/metrics_radar.png            |
+---------------------------+------------------------------------------------------+
| churn_summary_dashboard   | data/08_reporting/plots/churn_summary_dashboard.png  |
+---------------------------+------------------------------------------------------+

MLflow Artifacts
----------------

All reports and visualizations are logged to MLflow:

* ``artifacts/reports/`` - Executive summary
* ``artifacts/plots/`` - All visualization PNG files

View artifacts in the MLflow UI:

.. code-block:: bash

   mlflow ui --port 5000

Sample Output
-------------

**Executive Summary Preview:**

.. code-block:: text

   ================================================================================
                           CHURN PREDICTION MODEL REPORT
   ================================================================================

   EXECUTIVE SUMMARY
   -----------------

   1. BEST PERFORMING MODEL: Logistic Regression
      Selected based on: Recall

   2. KEY PERFORMANCE METRICS:
      - Accuracy:  80.21%
      - Precision: 67.35%
      - Recall:    52.48%
      - F1 Score:  59.01%
      - ROC-AUC:   0.8432

   3. TOP 5 CHURN PREDICTORS:
      1. tenure
      2. Contract_Two year
      3. MonthlyCharges
      4. total_services
      5. is_month_to_month

   6. BUSINESS RECOMMENDATIONS:
      - Focus retention efforts on customers with high churn risk scores
      - Monitor customers with month-to-month contracts closely
      - Consider incentives for customers using electronic check payments
      - Promote security and protection services to reduce churn
      - Target new customers (< 12 months tenure) with engagement programs
