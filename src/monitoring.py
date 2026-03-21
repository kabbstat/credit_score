"""
Monitoring de dérive modèle: PSI, Data Drift.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ─── PSI ───────────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Population Stability Index (PSI).

    Mesure le déplacement de la distribution entre développement et production.
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Interprétation réglementaire (Bâle II):
        PSI < 0.10  → Stable — pas de dérive significative
        0.10-0.25   → Dérive modérée — surveillance recommandée
        PSI > 0.25  → Dérive significative — recalibration nécessaire

    Args:
        expected: Distribution de référence (train/développement)
        actual:   Distribution actuelle (production/test)
        bins:     Nombre de bins pour la discrétisation
        epsilon:  Correction pour éviter log(0)

    Returns:
        Valeur PSI (float)
    """
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    def get_distribution(data, bp):
        counts = np.histogram(data, bins=bp)[0]
        total = max(len(data), 1)
        return (counts + epsilon) / (total + epsilon * len(counts))

    dist_expected = get_distribution(expected, breakpoints)
    dist_actual = get_distribution(actual, breakpoints)

    psi = float(np.sum((dist_actual - dist_expected) * np.log(dist_actual / dist_expected)))
    return round(psi, 4)


def interpret_psi(psi_value: float) -> str:
    """Interprétation réglementaire du PSI."""
    if psi_value < 0.10:
        return "Stable"
    elif psi_value < 0.25:
        return "Dérive modérée — surveiller"
    else:
        return "Dérive significative — recalibration"


# ─── Feature Drift ─────────────────────────────────────────────

def compute_feature_drift(
    train_data: pd.DataFrame,
    new_data: pd.DataFrame,
    features: list,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calcule le PSI pour chaque feature entre train et données nouvelles.

    Returns:
        DataFrame avec PSI, interprétation et statut d'alerte par feature.
    """
    results = []

    for feature in features:
        if feature not in train_data.columns or feature not in new_data.columns:
            continue

        train_col = train_data[feature].dropna()
        new_col = new_data[feature].dropna()

        if train_col.dtype not in ['float64', 'int64', 'uint8'] or len(train_col) == 0:
            continue

        psi_val = compute_psi(train_col.values, new_col.values, bins=bins)

        results.append({
            'feature': feature,
            'psi': psi_val,
            'statut': interpret_psi(psi_val),
            'alerte': psi_val >= 0.25,
            'avertissement': 0.10 <= psi_val < 0.25,
            'n_train': len(train_col),
            'n_new': len(new_col),
            'mean_train': round(float(train_col.mean()), 4),
            'mean_new': round(float(new_col.mean()), 4),
            'drift_mean_pct': round(
                float(abs(new_col.mean() - train_col.mean()) / (abs(train_col.mean()) + 1e-6) * 100), 2
            ),
        })

    return (
        pd.DataFrame(results)
        .sort_values('psi', ascending=False)
        .reset_index(drop=True)
    )


def compute_score_psi(
    train_scores: np.ndarray,
    new_scores: np.ndarray,
    bins: int = 10,
) -> dict:
    """
    PSI sur les scores du modèle — indicateur clé de dérive modèle.
    Déclenche la recalibration en production (pipeline CIH Bank).
    """
    psi_val = compute_psi(train_scores, new_scores, bins=bins)
    return {
        'score_psi': psi_val,
        'statut': interpret_psi(psi_val),
        'alerte': psi_val >= 0.25,
        'mean_score_train': round(float(np.mean(train_scores)), 4),
        'mean_score_new': round(float(np.mean(new_scores)), 4),
        'std_score_train': round(float(np.std(train_scores)), 4),
        'std_score_new': round(float(np.std(new_scores)), 4),
    }


# ─── Dashboard PSI ─────────────────────────────────────────────

def plot_psi_dashboard(
    drift_df: pd.DataFrame,
    output_path: str = "plots/psi_dashboard.png",
):
    """Dashboard de monitoring PSI — outil de reporting réglementaire."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = drift_df.head(20).copy()
    colors = df_plot['psi'].apply(
        lambda x: '#d73027' if x >= 0.25 else '#fdae61' if x >= 0.10 else '#1a9850'
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(df_plot['feature'], df_plot['psi'], color=colors, edgecolor='white')

    ax.axvline(x=0.10, color='orange', linestyle='--', linewidth=1.5, label='Seuil modéré (0.10)')
    ax.axvline(x=0.25, color='red', linestyle='--', linewidth=1.5, label='Seuil critique (0.25)')

    for bar, row in zip(bars, df_plot.itertuples()):
        icon = "⚠" if row.psi >= 0.25 else "~" if row.psi >= 0.10 else "✓"
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{icon} {row.psi:.3f}', va='center', fontsize=9
        )

    ax.set_xlabel('PSI (Population Stability Index)', fontsize=12)
    ax.set_title('Dashboard Monitoring PSI — Détection de Dérive', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PSI dashboard saved to {output_path}")


def plot_distribution_drift(
    train_data: pd.DataFrame,
    new_data: pd.DataFrame,
    feature: str,
    output_path: str = "plots/drift_distribution.png",
):
    """Superposition des distributions train vs production pour une feature."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        train_data[feature].dropna(), bins=40, alpha=0.6,
        color='steelblue', label='Train (référence)', density=True
    )
    ax.hist(
        new_data[feature].dropna(), bins=40, alpha=0.6,
        color='tomato', label='Nouvelles données', density=True
    )
    ax.set_title(f'Drift de distribution — {feature}', fontsize=13)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Densité', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Distribution drift saved to {output_path}")


# ─── Rapport Complet ───────────────────────────────────────────

def generate_monitoring_report(
    train_data: pd.DataFrame,
    new_data: pd.DataFrame,
    features: list,
    output_dir: str = "metrics",
) -> dict:
    """
    Rapport de monitoring complet: PSI par feature + statut global.
    Conformité: traçabilité réglementaire des alertes de dérive.

    Returns:
        dict avec statut global, alertes, et PSI par feature
    """
    os.makedirs(output_dir, exist_ok=True)

    num_features = [
        f for f in features
        if f in train_data.columns
        and train_data[f].dtype in ['float64', 'int64', 'uint8']
    ]

    print(f"Computing PSI on {len(num_features)} numerical features...")
    drift_df = compute_feature_drift(train_data, new_data, num_features)

    n_alert = int((drift_df['psi'] >= 0.25).sum())
    n_warning = int(((drift_df['psi'] >= 0.10) & (drift_df['psi'] < 0.25)).sum())
    n_stable = int((drift_df['psi'] < 0.10).sum())

    global_status = (
        'ALERTE' if n_alert > 0 else
        'AVERTISSEMENT' if n_warning > 0 else
        'STABLE'
    )

    report = {
        'timestamp': datetime.now().isoformat(),
        'n_features_monitored': len(drift_df),
        'n_alert': n_alert,
        'n_warning': n_warning,
        'n_stable': n_stable,
        'global_status': global_status,
        'feature_drift': drift_df.to_dict('records'),
    }

    # Save JSON report
    report_path = os.path.join(output_dir, 'monitoring_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Monitoring report saved to {report_path}")

    # PSI dashboard
    plot_psi_dashboard(drift_df, "plots/psi_dashboard.png")

    # Distribution drift for most drifted feature
    if not drift_df.empty:
        top_drift_feature = drift_df.iloc[0]['feature']
        plot_distribution_drift(
            train_data, new_data, top_drift_feature,
            f"plots/drift_{top_drift_feature}.png"
        )

    print(f"\n=== STATUT MONITORING ===")
    print(f"Statut global        : {global_status}")
    print(f"Features en alerte   : {n_alert}")
    print(f"Features en warning  : {n_warning}")
    print(f"Features stables     : {n_stable}")
    if n_alert > 0:
        alerts = drift_df[drift_df['alerte']]['feature'].tolist()
        print(f"Features alertées    : {alerts}")

    return report


# ─── Point d'entrée ────────────────────────────────────────────

def main():
    """Monitoring train vs test split (simulation post-déploiement)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import get_config
    from src.data_prepro import create_feature_list
    from sklearn.model_selection import train_test_split

    config = get_config()
    df = pd.read_csv(config.data.processed_data_path)

    feature_info = create_feature_list(df)
    available = [f for f in feature_info['features'] if f in df.columns]

    train_df, test_df = train_test_split(
        df[available], test_size=0.2, random_state=42, stratify=df['target']
    )

    print(f"Train: {len(train_df)} samples | Test (prod): {len(test_df)} samples")
    report = generate_monitoring_report(train_df, test_df, available)

    return report


if __name__ == "__main__":
    main()
