"""
MLflow experiment tracking utilities.
Provides a clean interface for logging experiments.
"""

import os
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

from src.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for ML pipeline.
    
    Example usage:
        tracker = MLflowTracker(experiment_name="credit_score")
        with tracker.start_run(run_name="rf_baseline"):
            tracker.log_params({"n_estimators": 500})
            tracker.log_metrics({"accuracy": 0.85})
            tracker.log_model(model, "model")
    """
    
    def __init__(
        self,
        experiment_name: str = "credit_score_prediction",
        tracking_uri: str = "mlruns",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
            artifact_location: Location for storing artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self._run = None
        
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Set up MLflow configuration."""
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location
            )
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """
        Context manager for MLflow run.
        
        Args:
            run_name: Name of the run
            tags: Tags to attach to the run
            nested: Whether this is a nested run
        """
        self._run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )
        logger.info(f"Started MLflow run: {run_name or self._run.info.run_id}")
        
        try:
            yield self._run
        finally:
            mlflow.end_run()
            logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        for key, value in params.items():
            # MLflow doesn't accept None values
            if value is not None:
                mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log sklearn model to MLflow.
        
        Args:
            model: Trained sklearn model
            artifact_path: Path within artifact store
            registered_model_name: Name for model registry
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name,
            **kwargs
        )
        logger.info(f"Logged model to {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file or directory as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts."""
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug(f"Logged artifacts from: {local_dir}")
    
    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)
        logger.debug(f"Logged figure: {artifact_file}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """Log a dictionary as JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)
        logger.debug(f"Logged dict to: {artifact_file}")
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for current run."""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    @staticmethod
    def load_model(model_uri: str):
        """Load a model from MLflow."""
        return mlflow.sklearn.load_model(model_uri)
    
    def get_best_run(
        self,
        metric: str = "f1_macro",
        ascending: bool = False
    ) -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            ascending: If True, lower is better
        
        Returns:
            Best run or None
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) > 0:
            return runs.iloc[0]
        return None


def init_tracker(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Initialize MLflow tracker with config defaults.
    
    Args:
        experiment_name: Override experiment name
        tracking_uri: Override tracking URI
    
    Returns:
        Configured MLflowTracker instance
    """
    from src.config import get_config
    
    config = get_config()
    
    return MLflowTracker(
        experiment_name=experiment_name or config.mlflow.experiment_name,
        tracking_uri=tracking_uri or config.mlflow.tracking_uri
    )
