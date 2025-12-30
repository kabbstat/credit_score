import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


def load_model(model_path: str = "models/model.pkl"):
    """Load trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def load_test_data(data_path: str = "data/processed/processed.csv"):
    """Load and prepare test data."""
    data = pd.read_csv(data_path)
    return data


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro')),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro')),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted')),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted')),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted')),
    }
    
    # Add ROC AUC if probabilities are available
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            ))
        except Exception as e:
            print(f"Could not compute ROC AUC: {e}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path: str = "plots/confusion_matrix.png"):
    """Generate and save confusion matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=sorted(y_true.unique()),
        yticklabels=sorted(y_true.unique())
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_feature_importance(model, feature_names, output_path: str = "plots/feature_importance.png"):
    """Plot feature importance from the model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Get feature importance from the classifier
        classifier = model.named_steps['classifier']
        importances = classifier.feature_importances_
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        
        # Try to get transformed feature names
        try:
            transformed_names = preprocessor.get_feature_names_out()
        except:
            transformed_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': transformed_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 20 Feature Importances', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Feature importance plot saved to {output_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


def save_metrics(metrics: dict, output_path: str = "metrics/metrics.json"):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred))


def main():
    from data_prepro import create_feature_list
    from sklearn.model_selection import train_test_split
    
    print("Loading model and data...")
    model = load_model()
    data = load_test_data()
    
    # Get features
    feature_info = create_feature_list(data)
    features = feature_info['features']
    available_features = [f for f in features if f in data.columns]
    
    # Prepare data
    df = data[available_features].copy()
    y = df['target']
    X = df.drop(columns=['target'])
    
    # Use the same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Evaluating on {len(X_test)} samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None
    
    # Compute and save metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    save_metrics(metrics)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print_classification_report(y_test, y_pred)
    
    # Generate plots
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, X_test.columns.tolist())
    
    return metrics


if __name__ == "__main__":
    main()
