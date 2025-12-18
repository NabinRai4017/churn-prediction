# Churn Prediction

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow)](https://mlflow.org)
[![Optuna](https://img.shields.io/badge/Optuna-hyperparameter_tuning-blue)](https://optuna.org)
[![Pandera](https://img.shields.io/badge/Pandera-data_validation-green)](https://pandera.readthedocs.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A machine learning project to predict customer churn using the Telco Customer Churn dataset. Built with Kedro for pipeline orchestration and MLflow for experiment tracking.

## Overview

This project predicts whether a telecom customer will churn (leave the service) based on their demographics, account information, and service usage patterns. It implements a complete ML pipeline from data preprocessing to model deployment, with comprehensive reporting and visualizations.

### Key Features

- **Modular Kedro Pipelines** - 5 distinct pipelines for data processing, feature engineering, model training, hyperparameter tuning, and reporting
- **Multiple ML Models** - Logistic Regression, Random Forest, and Gradient Boosting classifiers
- **Automated Feature Engineering** - Creates 22 derived features from raw data
- **Optuna Hyperparameter Tuning** - Automated hyperparameter optimization with 50 trials and cross-validation
- **Pandera Data Validation** - Schema-based DataFrame validation throughout the pipeline
- **MLflow Integration** - Full experiment tracking, parameter logging, and model registry
- **Comprehensive Reporting** - Generates reports, visualizations, and executive summaries

## Project Structure

```
churn-prediction/
├── conf/                              # Configuration files
│   ├── base/
│   │   ├── catalog.yml                # Data catalog definitions
│   │   ├── parameters.yml             # Pipeline parameters
│   │   └── mlflow.yml                 # MLflow configuration
│   └── local/                         # Local overrides (gitignored)
├── data/                              # Data directory (layered)
│   ├── 01_raw/                        # Raw input data
│   ├── 02_intermediate/               # Intermediate processed data
│   ├── 03_primary/                    # Primary preprocessed data
│   ├── 04_feature/                    # Feature engineered data
│   ├── 05_model_input/                # Train/test splits
│   ├── 06_models/                     # Trained model files
│   ├── 07_model_output/               # Model metrics and outputs
│   └── 08_reporting/                  # Reports and visualizations
├── docs/                              # Sphinx documentation
├── notebooks/                         # Jupyter notebooks for exploration
├── src/churn_prediction/
│   ├── schemas.py                     # Pandera data validation schemas
│   └── pipelines/
│       ├── data_processing/           # Data preprocessing pipeline
│       ├── feature_engineering/       # Feature creation pipeline
│       ├── model_training/            # Model training pipeline
│       ├── hyperparameter_tuning/     # Optuna hyperparameter optimization
│       └── reporting/                 # Reporting pipeline
└── tests/                             # Unit tests (108 tests)
```

## Installation

### Prerequisites

- Python 3.11+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Place `telco-customer-churn.csv` in `data/01_raw/`
   - Dataset available from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Usage

### Run Complete Pipeline

```bash
kedro run
```

### Run Individual Pipelines

```bash
# Data preprocessing
kedro run --pipeline data_processing

# Feature engineering
kedro run --pipeline feature_engineering

# Model training
kedro run --pipeline model_training

# Hyperparameter tuning (requires model_training outputs)
kedro run --pipeline hyperparameter_tuning

# Generate reports and visualizations
kedro run --pipeline reporting
```

### View MLflow Experiments

```bash
mlflow ui --port 5000
```

Then open http://127.0.0.1:5000 in your browser.

### Visualize Pipeline

```bash
kedro viz
```

## Pipelines

### 1. Data Processing Pipeline

Transforms raw customer data into clean, preprocessed format.

| Node | Description |
|------|-------------|
| `drop_customer_id` | Removes non-predictive customer ID |
| `convert_total_charges` | Converts TotalCharges to numeric |
| `handle_missing_values` | Handles missing values (median/zero fill) |
| `encode_binary_columns` | Binary encoding for Yes/No columns |
| `encode_multiclass_columns` | One-hot encoding for categorical columns |
| `scale_numerical_features` | StandardScaler normalization |

### 2. Feature Engineering Pipeline

Creates 22 derived features organized into 5 categories:

| Category | Features |
|----------|----------|
| **Service Features** | total_services, has_internet, has_streaming, has_security_services, security_services_count, has_multiple_lines |
| **Tenure Features** | tenure_group, is_new_customer, is_loyal_customer |
| **Contract Features** | is_month_to_month, has_long_contract, uses_electronic_check, has_auto_payment |
| **Charge Features** | charge_per_service, is_high_charges, charge_tenure_ratio |
| **Interaction Features** | high_risk_combo, churn_risk_score, low_engagement, senior_high_charges, has_family |

### 3. Model Training Pipeline

Trains and evaluates multiple classification models:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear model with balanced class weights |
| **Random Forest** | Ensemble of 100 decision trees |
| **Gradient Boosting** | Sequential boosting with 100 estimators |

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)

**Model Selection:** Best model selected based on F1 Score (configurable)

### 4. Hyperparameter Tuning Pipeline

Automated hyperparameter optimization using Optuna with MLflow integration.

| Node | Description |
|------|-------------|
| `identify_best_model` | Identifies best model from baseline comparison |
| `run_optuna_study` | Runs 50 Optuna trials with TPE sampler |
| `train_tuned_model` | Trains final model with optimized hyperparameters |
| `evaluate_tuned_model` | Evaluates tuned model on test set |
| `create_tuning_report` | Creates comparison report (baseline vs tuned) |

**Features:**
- TPE (Tree-structured Parzen Estimator) sampler for efficient search
- 5-fold stratified cross-validation
- Automatic pruning of unpromising trials
- MLflow callback for trial-level tracking
- Comprehensive tuning report with improvement metrics

**Search Spaces:**

| Model | Hyperparameters |
|-------|-----------------|
| Logistic Regression | C (0.001-100, log scale), max_iter (100-2000) |
| Random Forest | n_estimators (50-300), max_depth (3-20), min_samples_split (2-20), min_samples_leaf (1-10) |
| Gradient Boosting | n_estimators (50-300), learning_rate (0.01-0.3, log scale), max_depth (2-10) |

### 5. Reporting Pipeline

Generates comprehensive reports and visualizations:

**Reports:**
- Model comparison report (CSV)
- Confusion matrix report (CSV)
- Feature importance report (CSV)
- Detailed metrics report (CSV)
- Executive summary (TXT)

**Visualizations:**
- Model comparison bar chart
- Confusion matrices heatmaps
- Feature importance bar chart
- Metrics radar chart
- Churn prediction summary dashboard

## Data Validation with Pandera

This project uses Pandera for schema-based DataFrame validation throughout the pipeline, ensuring data quality and catching errors early.

### Schema Definitions

| Schema | Purpose |
|--------|---------|
| `RawCustomerSchema` | Validates raw input data (21 columns) |
| `CustomerNoIdSchema` | After dropping customerID |
| `CustomerNumericSchema` | After numeric conversion |
| `CustomerCleanedSchema` | After handling missing values |
| `CustomerBinaryEncodedSchema` | After binary encoding |
| `PreprocessedCustomerSchema` | Fully preprocessed data |
| `FeaturesEngineeredSchema` | After feature engineering (validates 22 new features) |
| `ModelInputSchema` | Training/test features (no nulls allowed) |
| `TargetSchema` | Target variable (binary 0/1) |

### Validation Features

- **Type checking** - Ensures correct data types for each column
- **Value constraints** - Validates ranges (e.g., tenure >= 0, Churn in [0,1])
- **Null checking** - Enforces non-null constraints where required
- **Categorical validation** - Validates allowed values (e.g., gender in ["Male", "Female"])
- **Decorator-based** - Uses `@pa.check_types` for automatic validation

### Example Schema

```python
class FeaturesEngineeredSchema(pa.DataFrameModel):
    Churn: Series[int] = pa.Field(isin=[0, 1])
    tenure_group: Series[int] = pa.Field(isin=[0, 1, 2, 3])
    churn_risk_score: Series[float] = pa.Field(ge=0, le=6)
    has_internet: Series[int] = pa.Field(isin=[0, 1])
    # ... 22 engineered features validated
```

## Configuration

### Model Parameters

Edit `conf/base/parameters.yml`:

```yaml
model_training:
  target_column: "Churn"
  test_size: 0.2
  random_state: 42
  selection_metric: "f1_score"  # Options: accuracy, precision, recall, f1_score, roc_auc

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
```

### Hyperparameter Tuning Configuration

Edit `conf/base/parameters.yml`:

```yaml
hyperparameter_tuning:
  n_trials: 50
  optimization_metric: "f1_score"
  cv_folds: 5
  direction: "maximize"
  pruning: true

  random_forest:
    n_estimators:
      type: "int"
      low: 50
      high: 300
    max_depth:
      type: "int"
      low: 3
      high: 20
```

### MLflow Configuration

Edit `conf/base/mlflow.yml`:

```yaml
server:
  mlflow_tracking_uri: mlruns  # Or remote server URI

tracking:
  experiment:
    name: churn_prediction
```

## Results

Sample model performance (results may vary):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 74.4% | 51.1% | 78.1% | 61.8% | 0.843 |
| Random Forest | 76.8% | 54.9% | 70.6% | 61.8% | 0.836 |
| Gradient Boosting | 79.6% | 64.3% | 51.6% | 57.3% | 0.836 |

**Top Churn Predictors:**
1. Tenure (customer lifetime)
2. Contract type (month-to-month vs annual)
3. Monthly charges
4. Total services subscribed
5. Payment method (electronic check)

## Documentation

Build and view Sphinx documentation:

```bash
cd docs
make html
open build/html/index.html
```

## Testing

Run unit tests:

```bash
pytest
```

With coverage:

```bash
pytest --cov=src/churn_prediction
```

## Notebooks

Exploratory notebooks are available in the `notebooks/` directory:

- `01_data_analysis_for_preprocessing.ipynb` - Data exploration and preprocessing analysis
- `02_feature_engineering.ipynb` - Feature engineering exploration

Run notebooks with Kedro context:

```bash
kedro jupyter notebook
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Kedro](https://kedro.org/) - Pipeline framework
- [MLflow](https://mlflow.org/) - Experiment tracking
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [Pandera](https://pandera.readthedocs.io/) - Data validation
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - Data source
