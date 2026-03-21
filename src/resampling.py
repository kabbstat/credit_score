"""
Gestion des données déséquilibrées: SMOTE, Undersampling, comparaison.
Expertise: CIH Bank — optimisation rappel/faux positifs en scoring crédit
et détection de fraude sur données fortement déséquilibrées.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek


# ─── Stratégies de rééchantillonnage ───────────────────────────

def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple:
    """
    SMOTE (Synthetic Minority Over-sampling Technique).
    Génère des exemples synthétiques pour les classes minoritaires.

    Utilisé en production CIH Bank pour la détection de fraude.
    """
    smote = SMOTE(random_state=random_state, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"SMOTE — Avant: {dict(Counter(y))} → Après: {dict(Counter(y_res))}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_borderline_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple:
    """
    Borderline-SMOTE: génère des exemples synthétiques uniquement
    près de la frontière de décision.
    Meilleure performance que SMOTE classique sur données financières.
    """
    smote = BorderlineSMOTE(random_state=random_state, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Borderline-SMOTE — Avant: {dict(Counter(y))} → Après: {dict(Counter(y_res))}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_undersampling(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple:
    """
    Undersampling aléatoire: réduit la classe majoritaire.
    Utile quand le dataset est très grand et SMOTE trop coûteux.
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    print(f"Undersampling — Avant: {dict(Counter(y))} → Après: {dict(Counter(y_res))}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_smote_tomek(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple:
    """
    SMOTETomek: SMOTE (oversampling) + Tomek Links (nettoyage frontière).
    Stratégie recommandée pour minimiser les faux positifs en scoring crédit.
    """
    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X, y)
    print(f"SMOTETomek — Avant: {dict(Counter(y))} → Après: {dict(Counter(y_res))}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ─── Comparaison des stratégies ────────────────────────────────

def compare_resampling_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 3,
    output_dir: str = "metrics",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare les stratégies de rééchantillonnage via cross-validation.

    Métriques clés:
        - F1-macro: performance globale
        - Recall: minimisation des faux négatifs (coût métier élevé)

    Returns:
        DataFrame comparatif trié par F1-macro décroissant
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    base_model = LogisticRegression(
        max_iter=1000, random_state=random_state, class_weight='balanced'
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Only use numerical features for resampling comparison
    num_cols = X.select_dtypes(include=['float64', 'int64', 'uint8']).columns.tolist()
    X_num = X[num_cols].fillna(0)

    strategies = {"Baseline (class_weight='balanced')": (X_num, y)}

    for name, func in [
        ("SMOTE", apply_smote),
        ("Undersampling", apply_undersampling),
        ("SMOTETomek", apply_smote_tomek),
    ]:
        try:
            X_res, y_res = func(X_num, y, random_state=random_state)
            strategies[name] = (X_res, y_res)
        except Exception as e:
            print(f"{name} failed: {e}")

    results = []
    scaler = StandardScaler()

    for strategy_name, (X_s, y_s) in strategies.items():
        print(f"\nEvaluation: {strategy_name}...")
        X_scaled = scaler.fit_transform(X_s)
        try:
            f1_scores = cross_val_score(
                base_model, X_scaled, y_s, cv=cv, scoring='f1_macro', n_jobs=-1
            )
            recall_scores = cross_val_score(
                base_model, X_scaled, y_s, cv=cv, scoring='recall_macro', n_jobs=-1
            )
            results.append({
                'strategie': strategy_name,
                'f1_macro_mean': round(float(f1_scores.mean()), 4),
                'f1_macro_std': round(float(f1_scores.std()), 4),
                'recall_macro_mean': round(float(recall_scores.mean()), 4),
                'n_samples': len(X_s),
                'class_distribution': str(dict(Counter(y_s))),
            })
        except Exception as e:
            print(f"  Evaluation failed: {e}")

    results_df = pd.DataFrame(results).sort_values('f1_macro_mean', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_json(
        os.path.join(output_dir, 'resampling_comparison.json'),
        orient='records', indent=2
    )
    print(f"\nResampling comparison saved to {output_dir}/resampling_comparison.json")

    return results_df


# ─── Visualisations ────────────────────────────────────────────

def plot_class_distribution(
    y_before: pd.Series,
    y_after: pd.Series,
    strategy_name: str = "SMOTE",
    output_path: str = "plots/class_distribution.png",
):
    """Distribution des classes avant/après rééchantillonnage."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = ['#2196F3', '#4CAF50', '#F44336']

    for ax, (y_data, title) in zip(axes, [
        (y_before, "Avant rééchantillonnage"),
        (y_after, f"Après {strategy_name}"),
    ]):
        counts = Counter(y_data)
        classes = sorted(counts.keys())
        values = [counts[c] for c in classes]
        colors = palette[:len(classes)]

        bars = ax.bar(classes, values, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Classe de crédit', fontsize=11)
        ax.set_ylabel("Nombre d'échantillons", fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    plt.suptitle('Distribution des Classes — Rééchantillonnage', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class distribution saved to {output_path}")


def plot_resampling_comparison(
    results_df: pd.DataFrame,
    output_path: str = "plots/resampling_comparison.png",
):
    """Comparaison visuelle des stratégies de rééchantillonnage."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(results_df))
    colors = ['#1a9850' if i == 0 else '#4575b4' for i in x]

    bars = ax.bar(
        results_df['strategie'],
        results_df['f1_macro_mean'],
        color=colors, edgecolor='white', linewidth=1.5
    )

    ax.errorbar(
        x=list(x),
        y=results_df['f1_macro_mean'],
        yerr=results_df['f1_macro_std'],
        fmt='none', color='black', capsize=5, linewidth=2
    )

    for bar, val in zip(bars, results_df['f1_macro_mean']):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10
        )

    ax.set_xlabel('Stratégie de rééchantillonnage', fontsize=12)
    ax.set_ylabel('F1-Macro (CV 3-fold)', fontsize=12)
    ax.set_title('Comparaison des Stratégies de Rééchantillonnage', fontsize=14)
    ax.set_ylim(0, results_df['f1_macro_mean'].max() + 0.05)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Resampling comparison saved to {output_path}")


# ─── Point d'entrée ────────────────────────────────────────────

def main():
    """Analyse complète de rééchantillonnage."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import get_config
    from src.data_prepro import create_feature_list

    config = get_config()
    df = pd.read_csv(config.data.processed_data_path)

    feature_info = create_feature_list(df)
    available = [f for f in feature_info['features'] if f in df.columns]

    X = df[available].drop(columns=['target'])
    y = df['target']

    print(f"Distribution originale: {dict(Counter(y))}")

    results = compare_resampling_strategies(X, y)
    plot_resampling_comparison(results)

    print("\n=== COMPARAISON DES STRATÉGIES ===")
    print(results.to_string(index=False))

    return results


if __name__ == "__main__":
    main()
