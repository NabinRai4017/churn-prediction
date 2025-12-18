"""Integration tests for the Kedro project."""

import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from churn_prediction.pipeline_registry import register_pipelines


class TestKedroProject:
    """Tests for Kedro project structure and basic functionality."""

    def test_project_bootstraps(self):
        """Test that the Kedro project can be bootstrapped."""
        bootstrap_project(Path.cwd())
        # If no exception is raised, the test passes

    def test_catalog_loads(self):
        """Test that the data catalog can be loaded."""
        bootstrap_project(Path.cwd())
        with KedroSession.create(project_path=Path.cwd()) as session:
            context = session.load_context()
            catalog = context.catalog
            # Check that key datasets are defined (use _datasets for older Kedro versions)
            dataset_list = list(catalog._datasets.keys()) if hasattr(catalog, '_datasets') else catalog.list()
            assert "customers" in dataset_list
            assert "preprocessed_customers" in dataset_list
            assert "features_engineered" in dataset_list

    def test_pipelines_exist(self):
        """Test that all expected pipelines are registered."""
        # Use register_pipelines directly from the pipeline registry module
        pipelines = register_pipelines()
        expected_pipelines = [
            "data_processing",
            "feature_engineering",
            "model_training",
            "reporting",
        ]
        for pipeline_name in expected_pipelines:
            assert pipeline_name in pipelines, f"Pipeline '{pipeline_name}' not found"
