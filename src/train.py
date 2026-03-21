"""
Enhanced model training with MLflow tracking.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prepro import create_feature_list
from src.config import get_config, load_config
from src.logger import setup_logging, get_logger
from src.tracking import MLflowTracker


# Initialize logger
logger = setup_logging(
    name="model_training",
    level="INFO",
    log_file="logs/training.log",
    console=True
)


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load processed data."""
    if data_path is None:
        config = get_config()
        data_path = config.data.processed_data_path
    
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
    return data


def stratified_sample(df: pd.DataFrame, stratify_col: str, frac: float = 1.0) -> pd.DataFrame:
    """Stratified sampling of dataframe."""
    return df.groupby(stratify_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac)
    ).reset_index(drop=True)


def get_model(model_name: str, params: dict):
    """
    Get model instance by name.
    Supporte: RandomForest, GradientBoosting, XGBoost, LogisticRegression (L1/Lasso).
    """
    models = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "LogisticRegressionLasso": lambda **kw: LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, **kw
        ),
    }

    if _XGBOOST_AVAILABLE:
        models["XGBClassifier"] = XGBClassifier

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](**params)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            ))
            # Gini coefficient (= 2×AUC − 1) — métrique standard scoring bancaire
            metrics["gini_coefficient"] = round(2 * metrics["roc_auc_ovr"] - 1, 4)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC / Gini: {e}")

    return metrics


def train_model(
    use_mlflow: bool = True,
    run_name: str = None
) -> tuple:
    """
    Train model with full MLOps pipeline.
    
    Args:
        use_mlflow: Whether to use MLflow tracking
        run_name: Name for the MLflow run
    
    Returns:
        Tuple of (model, metrics)
    """
    config = get_config()
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Load data
    db = load_data()
    df = stratified_sample(db, 'target', frac=1)
    
    # Get features
    feature_info = create_feature_list(df)
    features = feature_info['features']
    cat_features = feature_info['categorical']
    num_features = feature_info['numerical']
    
    logger.info(f"Total features: {len(features)}")
    logger.info(f"Categorical: {len(cat_features)}, Numerical: {len(num_features)}")
    
    # Prepare data
    available_features = [f for f in features if f in df.columns]
    data = df[available_features].copy()
    
    if 'target' not in data.columns:
        raise ValueError("Target column not found")
    
    y = data['target']
    X = data.drop(columns=['target'])
    
    # Update feature lists
    cat_features_final = [f for f in cat_features if f in X.columns]
    num_features_final = [f for f in num_features if f in X.columns]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Build preprocessor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features_final),
        ('num', StandardScaler(), num_features_final)
    ])
    
    # Get model
    classifier = get_model(config.model.name, config.model.params)
    
    # Build pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    logger.info(f"Training {config.model.name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Cross-validation
    logger.info(f"Running {config.model.cv_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=config.model.cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())
    
    # Log results
    logger.info("-" * 40)
    logger.info("EVALUATION METRICS")
    logger.info("-" * 40)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # MLflow tracking
    if use_mlflow:
        logger.info("Logging to MLflow...")
        tracker = MLflowTracker(
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri
        )
        
        run_name = run_name or f"{config.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with tracker.start_run(run_name=run_name):
            # Log parameters
            tracker.log_params({
                "model_name": config.model.name,
                "test_size": config.data.test_size,
                "cv_folds": config.model.cv_folds,
                "n_features": len(available_features) - 1,
                "n_cat_features": len(cat_features_final),
                "n_num_features": len(num_features_final),
                **config.model.params
            })
            
            # Log metrics
            tracker.log_metrics(metrics)
            
            # Log model
            tracker.log_model(model, "model")
            
            # Log tags
            tracker.set_tags({
                "stage": "training",
                "dataset": "openml_46441",
                "framework": "sklearn"
            })
        
        logger.info("MLflow logging complete")
    
    # Save model locally
    model_dir = Path(config.model.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, config.model.model_path)
    logger.info(f"Model saved to {config.model.model_path}")
    
    # Save metrics
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = metrics_dir / "train_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return model, metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train credit score model")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    
    args = parser.parse_args()
    
    model, metrics = train_model(
        use_mlflow=not args.no_mlflow,
        run_name=args.run_name
    )
    
    return model, metrics


if __name__ == "__main__":
    main()
