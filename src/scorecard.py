"""
Scorecard: WoE, IV, Gini, KS — métriques de scoring crédit.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


# ─── WoE / IV ──────────────────────────────────────────────────

def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target_binary: pd.Series,
    bins: int = 10,
) -> tuple:
    """
    Calcule le Weight of Evidence (WoE) et l'Information Value (IV).

    WoE(bin) = ln(P(Events in bin) / P(Non-Events in bin))
    IV = Σ (P(Events) - P(Non-Events)) × WoE

    IV interprétation:
        < 0.02   → Pouvoir prédictif nul
        0.02-0.1 → Faible
        0.1-0.3  → Moyen
        0.3-0.5  → Fort
        > 0.5    → Suspect (overfitting potentiel)

    Returns:
        (woe_df, iv_value)
    """
    series = df[feature].copy()

    # Discrétisation des variables continues
    if series.dtype in ['float64', 'int64', 'uint8']:
        try:
            series = pd.qcut(series, q=bins, duplicates='drop')
        except Exception:
            series = pd.cut(series, bins=bins)

    cross = pd.crosstab(series, target_binary)

    if cross.shape[1] < 2:
        return pd.DataFrame(), 0.0

    cross.columns = ['non_event', 'event']
    cross['non_event'] = cross['non_event'].replace(0, 0.5)
    cross['event'] = cross['event'].replace(0, 0.5)

    total_events = cross['event'].sum()
    total_non_events = cross['non_event'].sum()

    cross['dist_event'] = cross['event'] / total_events
    cross['dist_non_event'] = cross['non_event'] / total_non_events
    cross['woe'] = np.log(cross['dist_event'] / cross['dist_non_event'])
    cross['iv_bin'] = (cross['dist_event'] - cross['dist_non_event']) * cross['woe']

    iv = float(cross['iv_bin'].sum())
    return cross.reset_index(), iv


def compute_all_iv(
    df: pd.DataFrame,
    features: list,
    target_binary: pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    """Calcule l'IV pour toutes les features — sélection de variables bancaire."""
    results = []
    for feat in features:
        try:
            _, iv = compute_woe_iv(df, feat, target_binary, bins)
            power = (
                "Nul" if iv < 0.02 else
                "Faible" if iv < 0.1 else
                "Moyen" if iv < 0.3 else
                "Fort" if iv < 0.5 else
                "Suspect"
            )
            results.append({'feature': feat, 'iv': round(iv, 4), 'pouvoir_predictif': power})
        except Exception as e:
            results.append({'feature': feat, 'iv': None, 'pouvoir_predictif': f"Erreur: {e}"})

    return (
        pd.DataFrame(results)
        .sort_values('iv', ascending=False)
        .reset_index(drop=True)
    )


# ─── Gini ──────────────────────────────────────────────────────

def compute_gini(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calcule le coefficient de Gini = 2×AUC − 1.
    Mesure standard de discrimination en scoring bancaire.

    Interprétation:
        Gini > 0.30 → modèle acceptable
        Gini > 0.50 → bon modèle
        Gini > 0.70 → excellent modèle
    """
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true_binary, y_score)
    return float(round(2 * auc - 1, 4))


# ─── KS Statistic ──────────────────────────────────────────────

def compute_ks_statistic(y_true_binary: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Calcule la statistique KS (Kolmogorov-Smirnov).
    Mesure l'écart maximal entre CDF des bons et mauvais clients.

    Interprétation:
        KS > 0.20 → modèle acceptable
        KS > 0.40 → bon modèle
    """
    sorted_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true_binary[sorted_idx]
    y_score_sorted = y_score[sorted_idx]

    n_events = int(y_true_binary.sum())
    n_non_events = len(y_true_binary) - n_events

    cumulative_events = np.cumsum(y_true_sorted) / max(n_events, 1)
    cumulative_non_events = np.cumsum(1 - y_true_sorted) / max(n_non_events, 1)

    ks_values = np.abs(cumulative_events - cumulative_non_events)
    ks_max = float(ks_values.max())
    ks_idx = int(ks_values.argmax())

    ks_stat, p_value = stats.ks_2samp(
        y_score[y_true_binary == 1],
        y_score[y_true_binary == 0]
    )

    return {
        'ks_statistic': round(ks_max, 4),
        'ks_threshold': round(float(y_score_sorted[ks_idx]), 4),
        'ks_p_value': round(float(p_value), 6),
        'cumulative_events': cumulative_events,
        'cumulative_non_events': cumulative_non_events,
        'sorted_scores': y_score_sorted,
    }


# ─── Plots ─────────────────────────────────────────────────────

def plot_ks_curve(ks_result: dict, output_path: str = "plots/ks_curve.png"):
    """Courbe KS — outil de validation standard en scoring bancaire."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n = len(ks_result['sorted_scores'])
    x = np.linspace(0, 1, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, ks_result['cumulative_events'], 'b-', label='Bons clients (Events)', linewidth=2)
    ax.plot(x, ks_result['cumulative_non_events'], 'r-', label='Mauvais clients (Non-Events)', linewidth=2)

    ks_val = ks_result['ks_statistic']
    ax.set_title(f'Courbe KS — Statistique KS = {ks_val:.4f}', fontsize=14)
    ax.set_xlabel('Population triée par score décroissant', fontsize=11)
    ax.set_ylabel('Taux cumulé', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"KS curve saved to {output_path}")


def plot_iv_ranking(iv_df: pd.DataFrame, output_path: str = "plots/iv_ranking.png"):
    """Ranking des features par Information Value."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = iv_df.dropna(subset=['iv']).head(20).copy()
    colors = df_plot['iv'].apply(
        lambda x: '#d73027' if x > 0.5 else
                  '#1a9850' if x > 0.3 else
                  '#fdae61' if x > 0.1 else
                  '#4575b4' if x > 0.02 else '#bababa'
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_plot['feature'], df_plot['iv'], color=colors)

    for threshold, label in [(0.02, 'Nul'), (0.1, 'Faible'), (0.3, 'Moyen'), (0.5, 'Fort')]:
        ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
        ax.text(threshold, -0.8, label, ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Information Value (IV)', fontsize=12)
    ax.set_title('Ranking des Features par Information Value (WoE/IV)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"IV ranking saved to {output_path}")


# ─── Analyse Complète ──────────────────────────────────────────

def run_scorecard_analysis(
    df: pd.DataFrame,
    features: list,
    target_col: str = 'target',
    good_class: str = 'Good',
    output_dir: str = "metrics",
) -> dict:
    """
    Analyse scorecard complète: WoE/IV ranking + métriques de discrimination.

    Args:
        df: DataFrame avec features et target multiclasse
        features: Liste de features à analyser
        target_col: Colonne cible (Good/Standard/Poor)
        good_class: Classe 'bon client' pour la binarisation
        output_dir: Dossier de sortie métriques

    Returns:
        dict avec IV summary, Gini (one-vs-rest) et KS statistique
    """
    # Binarisation: Good=1, reste=0 (logique bancaire: approuver ou refuser)
    target_binary = (df[target_col] == good_class).astype(int)

    num_features = [
        f for f in features
        if f in df.columns
        and df[f].dtype in ['float64', 'int64', 'uint8']
        and f != target_col
    ]

    print(f"Analyse WoE/IV sur {len(num_features)} features numériques...")
    iv_df = compute_all_iv(df, num_features, target_binary)

    os.makedirs(output_dir, exist_ok=True)
    iv_path = os.path.join(output_dir, 'iv_summary.json')
    iv_df.to_json(iv_path, orient='records', indent=2)
    print(f"IV summary saved to {iv_path}")

    plot_iv_ranking(iv_df, "plots/iv_ranking.png")

    return {
        'iv_summary': iv_df.to_dict('records'),
        'top_features_by_iv': iv_df[iv_df['iv'].notna()].head(5)['feature'].tolist(),
    }


def compute_scorecard_metrics(
    y_true: pd.Series,
    y_proba_good: np.ndarray,
    good_class: str = 'Good',
    output_dir: str = "metrics",
) -> dict:
    """
    Calcule Gini et KS pour la classe 'Good' (one-vs-rest).
    Compatible avec sortie de train.py (multiclasse).
    """
    y_binary = (y_true == good_class).astype(int).values

    gini = compute_gini(y_binary, y_proba_good)
    ks_result = compute_ks_statistic(y_binary, y_proba_good)

    metrics = {
        'gini_coefficient': gini,
        'ks_statistic': ks_result['ks_statistic'],
        'ks_p_value': ks_result['ks_p_value'],
        'ks_threshold': ks_result['ks_threshold'],
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'scorecard_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # KS curve
    plot_ks_curve(ks_result, "plots/ks_curve.png")

    print(f"\n=== MÉTRIQUES SCORECARD ===")
    print(f"Gini coefficient : {gini:.4f}")
    print(f"KS statistic     : {ks_result['ks_statistic']:.4f}")
    print(f"KS p-value       : {ks_result['ks_p_value']:.6f}")

    return metrics


def main():
    """Point d'entrée: analyse scorecard complète."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import get_config
    from src.data_prepro import create_feature_list

    config = get_config()
    print("Loading processed data...")
    df = pd.read_csv(config.data.processed_data_path)

    feature_info = create_feature_list(df)
    features = feature_info['features']

    print("Running scorecard analysis...")
    result = run_scorecard_analysis(df, features)

    print("\n=== TOP FEATURES PAR IV ===")
    iv_df = pd.DataFrame(result['iv_summary'])
    print(iv_df.head(10).to_string(index=False))

    return result


if __name__ == "__main__":
    main()
