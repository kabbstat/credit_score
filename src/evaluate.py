import os
import sys
import json
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
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

sys.path.insert(0, str(Path(__file__).parent.parent))


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


# ─── SHAP Explainability ───────────────────────────────────────

def _prepare_shap_data(model, X_test: pd.DataFrame, max_samples: int = 100):
    """Transforme X_test et retourne (X_df, classifier, feature_names)."""
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Petit échantillon stratifié pour la vitesse
    n = min(max_samples, len(X_test))
    X_sample = X_test.sample(n=n, random_state=42)
    X_transformed = preprocessor.transform(X_sample)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    return X_df, classifier


def _get_shap_values(classifier, X_df: pd.DataFrame):
    """
    Calcule les SHAP values avec les options de performance:
    - approximate=True  → path-dependent approximation (10-50× plus rapide)
    - check_additivity=False → skip la vérification coûteuse
    """
    import shap

    model_type = type(classifier).__name__

    if model_type in ('RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier'):
        # background = résumé de 50 points (kmeans) pour accélérer
        background = shap.sample(X_df, min(50, len(X_df)))
        explainer = shap.TreeExplainer(classifier, data=background)
        shap_values = explainer.shap_values(X_df, check_additivity=False, approximate=True)
    else:
        background = shap.sample(X_df, min(50, len(X_df)))
        explainer = shap.LinearExplainer(classifier, background)
        shap_values = explainer.shap_values(X_df)

    # Pour multiclass: liste ou array 3D (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        return shap_abs_mean, shap_values[0]
    if shap_values.ndim == 3:
        # Newer SHAP: shape (n_samples, n_features, n_classes)
        shap_abs_mean = np.mean(np.abs(shap_values), axis=2)
        return shap_abs_mean, shap_values[:, :, 0]
    return shap_values, shap_values


def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    output_path: str = "plots/shap_summary.png",
    max_samples: int = 100,
):
    """
    SHAP Bar Plot — importance globale des features.
    Utilise approximate=True pour RandomForest (10-50× plus rapide).
    max_samples=100 suffit pour une estimation stable de l'importance.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  SHAP: computing on {min(max_samples, len(X_test))} samples (approximate mode)...")

    try:
        X_df, classifier = _prepare_shap_data(model, X_test, max_samples)
        shap_abs, _ = _get_shap_values(classifier, X_df)

        # Bar plot: mean |SHAP| par feature
        mean_abs = np.mean(np.abs(shap_abs), axis=0)
        feature_names = X_df.columns.tolist()
        top_idx = np.argsort(mean_abs)[-15:]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            [feature_names[i] for i in top_idx],
            mean_abs[top_idx],
            color='steelblue', edgecolor='white'
        )
        ax.set_xlabel("mean(|SHAP value|)", fontsize=12)
        ax.set_title("SHAP Feature Importance — Explainabilité Scoring Crédit", fontsize=13)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary saved to {output_path}")
        return True

    except Exception as e:
        print(f"SHAP summary failed: {e}")
        return None


def plot_shap_beeswarm(
    model,
    X_test: pd.DataFrame,
    output_path: str = "plots/shap_beeswarm.png",
    max_samples: int = 100,
):
    """
    SHAP Beeswarm — impact directionnel des features.
    Réduit à 100 samples pour éviter le blocage sur RandomForest 500 arbres.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  SHAP beeswarm: computing on {min(max_samples, len(X_test))} samples...")

    try:
        X_df, classifier = _prepare_shap_data(model, X_test, max_samples)
        _, shap_for_beeswarm = _get_shap_values(classifier, X_df)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_for_beeswarm, X_df, show=False, max_display=15)
        plt.title("SHAP Beeswarm — Impact directionnel sur le score crédit", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP beeswarm saved to {output_path}")
    except Exception as e:
        print(f"SHAP beeswarm failed: {e}")


# ─── Rapport HTML ──────────────────────────────────────────────

def _img_to_base64(img_path: str) -> str:
    """Encode une image en base64 pour l'embedding HTML."""
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_report(
    metrics: dict,
    output_path: str = "metrics/evaluation_report.html",
):
    """
    Rapport HTML interactif: métriques + plots embarqués.
    Format de reporting professionnel pour la présentation métier / conformité.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Embed images
    def img_tag(path, title=""):
        b64 = _img_to_base64(path)
        if not b64:
            return f"<p><em>{title} — image non disponible</em></p>"
        return (
            f'<figure>'
            f'<img src="data:image/png;base64,{b64}" alt="{title}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);">'
            f'<figcaption style="text-align:center;color:#666;margin-top:8px;">{title}</figcaption>'
            f'</figure>'
        )

    # Metrics table rows
    metric_rows = ""
    for k, v in metrics.items():
        badge_color = "#1a9850" if isinstance(v, float) and v >= 0.7 else "#fdae61" if isinstance(v, float) and v >= 0.5 else "#4575b4"
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        metric_rows += (
            f'<tr>'
            f'<td style="padding:10px;font-weight:500;">{k}</td>'
            f'<td style="padding:10px;text-align:center;">'
            f'<span style="background:{badge_color};color:white;padding:4px 12px;border-radius:12px;font-weight:600;">{val_str}</span>'
            f'</td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Évaluation — Credit Score Model</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; color: #333; }}
        .header {{ background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2rem; margin-bottom: 8px; }}
        .header p {{ opacity: 0.85; font-size: 1rem; }}
        .container {{ max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
        .card h2 {{ color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 12px; margin-bottom: 20px; font-size: 1.3rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #e8eaf6; color: #1a237e; padding: 12px; text-align: left; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8f9ff; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; }}
        .badge-green {{ background: #e8f5e9; color: #1a9850; }}
        .footer {{ text-align: center; color: #888; padding: 30px; font-size: 0.85rem; }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Rapport d'Évaluation — Credit Score Model</h1>
        <p>Pipeline MLOps | Mohamed Kabbaj — Data Scientist Scoring & Détection de Fraude</p>
        <p style="margin-top:8px;opacity:0.7;">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
    </div>

    <div class="container">

        <!-- Métriques -->
        <div class="card">
            <h2>📊 Métriques d'Évaluation</h2>
            <table>
                <thead><tr><th>Métrique</th><th>Valeur</th></tr></thead>
                <tbody>{metric_rows}</tbody>
            </table>
        </div>

        <!-- Plots principaux -->
        <div class="card">
            <h2>📈 Matrice de Confusion & Importance des Features</h2>
            <div class="grid">
                {img_tag("plots/confusion_matrix.png", "Matrice de Confusion")}
                {img_tag("plots/feature_importance.png", "Importance des Features (RandomForest)")}
            </div>
        </div>

        <!-- SHAP -->
        <div class="card">
            <h2>🔍 Explainabilité SHAP — Conformité Réglementaire</h2>
            <p style="color:#666;margin-bottom:20px;">
                Les valeurs SHAP quantifient la contribution de chaque feature à la décision de scoring.
                Conformité RGPD: justification des décisions automatisées.
            </p>
            <div class="grid">
                {img_tag("plots/shap_summary.png", "SHAP Feature Importance (bar)")}
                {img_tag("plots/shap_beeswarm.png", "SHAP Beeswarm — Impact directionnel")}
            </div>
        </div>

        <!-- Scorecard -->
        <div class="card">
            <h2>🏦 Métriques Scorecard Bancaire (WoE/IV)</h2>
            <div class="grid">
                {img_tag("plots/iv_ranking.png", "Information Value — Sélection de Variables")}
                {img_tag("plots/ks_curve.png", "Courbe KS — Discrimination Good/Poor")}
            </div>
        </div>

        <!-- Monitoring PSI -->
        <div class="card">
            <h2>🔔 Monitoring de Dérive (PSI)</h2>
            <div class="grid">
                {img_tag("plots/psi_dashboard.png", "Dashboard PSI — Stabilité des Features")}
                {img_tag("plots/resampling_comparison.png", "Comparaison des Stratégies de Rééchantillonnage")}
            </div>
        </div>

    </div>

    <div class="footer">
        <p>Credit Score MLOps Pipeline — Mohamed Kabbaj | Télécom Paris MS IA & MLOps</p>
        <p>Stack: Python · Scikit-learn · XGBoost · SHAP · MLflow · DVC · FastAPI · Docker</p>
    </div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


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

    # SHAP explainability
    print("\nGenerating SHAP explanations...")
    plot_shap_summary(model, X_test)
    plot_shap_beeswarm(model, X_test)

    # Scorecard metrics (Gini + KS)
    try:
        from src.scorecard import compute_scorecard_metrics
        classes = list(model.classes_)
        classes_lower = [str(c).lower() for c in classes]
        if 'good' in classes_lower:
            good_idx = classes_lower.index('good')
            y_proba_good = y_proba[:, good_idx]
            scorecard_metrics = compute_scorecard_metrics(y_test, y_proba_good, good_class=classes[good_idx])
            metrics.update(scorecard_metrics)
            save_metrics(metrics)
        else:
            print(f"Scorecard metrics skipped: 'good' class not found in {classes}")
    except Exception as e:
        print(f"Scorecard metrics failed: {e}")

    # HTML report
    generate_html_report(metrics)

    return metrics


if __name__ == "__main__":
    main()
