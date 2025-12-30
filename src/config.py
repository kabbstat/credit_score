"""
Configuration loader for the MLOps pipeline.
Centralizes all configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
    
    Returns:
        Dictionary containing configuration.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


@dataclass
class DataConfig:
    """Data configuration."""
    openml_dataset_id: int = 46441
    raw_data_path: str = "data/raw/data.csv"
    processed_data_path: str = "data/processed/processed.csv"
    features_path: str = "data/processed/features.txt"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "RandomForestClassifier"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1
    })
    cv_folds: int = 5
    model_path: str = "models/model.pkl"


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "mlruns"
    experiment_name: str = "credit_score_prediction"
    registry_uri: str = "models/mlflow_registry"
    log_model: bool = True
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        raw_config = load_config(config_path)
        
        return cls(
            data=DataConfig(**raw_config.get("data", {})),
            model=ModelConfig(
                name=raw_config.get("model", {}).get("name", "RandomForestClassifier"),
                params=raw_config.get("model", {}).get("params", {}),
                cv_folds=raw_config.get("model", {}).get("cv_folds", 5),
                model_path=raw_config.get("model", {}).get("model_path", "models/model.pkl")
            ),
            mlflow=MLflowConfig(**raw_config.get("mlflow", {}))
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_yaml()
    return _config
