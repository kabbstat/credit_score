import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data = pd.read_csv("./data/raw")
    return data

def explore_data (data): 
    print("exploring data")
    print("shape of the dataset:", data.shape)
    print("columns of the dataset :", data.columns)
    print("\n pourcentage de missing values")
    missing_percentage = data.isna().mean().sort_values(ascending=False)*100
    print(missing_percentage[missing_percentage>0])
    return {
        'categorical_features' : data.select_dtypes(include=['object']).columns.tolist(),
        'numerical_features' : data.select_dtypes(include=['int64','float64','uint8']).columns.tolist()
    }

def missing_val(data):
    for col in data.columns:
        if data[col].isnull().any():  # CORRIGÉ: .isnull() au lieu de .isnull
            if data[col].dtype in ['int64', 'float64', 'uint8']:  # CORRIGÉ: Vérifier si c'est numérique avant d'utiliser median()
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
    return data
def preprocess_data(data):
    data_pro = data.copy()
    # convert credit_history_age to numeric values months
    extracted = data_pro['credit_history_age'].str.extract(r'(?P<Years>\d+)\s*Years\s*and\s*(?P<Months>\d+)\s*Months').astype(float)
    data_pro['total_month'] = extracted['Years'] *12 + extracted['Months']
    # deal with some problem feature in the dataset 
    data_pro['payment_behaviour'] = data_pro['payment_behaviour'].replace('!@9#%8', 'unknown')
    data_pro['credit_mix'] = data_pro['credit_mix'].replace('_',pd.NA)
    # creating some new feature - new credit scores 
    data_pro['debt_to_income'] = data_pro['outstanding_debt'] / data_pro['annual_income']
    data_pro['emi_to_income_ratio'] = data_pro['total_emi_per_month'] / data_pro['monthly_inhand_salary']
    data_pro['loan_to_income_ratio'] = data_pro['outstanding_debt'] / data_pro['monthly_inhand_salary']
    data_pro['delayed_payment_freq'] = data_pro['num_of_delayed_payment'] / data_pro['num_of_loan']
    # new scores
    data_pro['credit_efficiency'] = data_pro['credit_utilization_ratio'] *(1- data_pro['delay_from_due_date'])
    data_pro['disposable_income_factor'] = (data_pro['monthly_inhand_salary'] - data_pro['total_emi_per_month'])/data_pro['monthly_inhand_salary']
    data_pro['payment_discipline_score'] = 1 - (data_pro['num_of_delayed_payment'] / (data_pro['num_credit_inquiries'] + 1))
    #data_pro = data_pro.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
    return data_pro

def create_feature_list(data):
    critical_features = ['total_month','outstanding_debt','num_of_delayed_payment','payment_behaviour','credit_utilization_ratio']
    important_features = ['annual_income', 'total_emi_per_month', 'monthly_inhand_salary','delay_from_due_date', 'credit_mix']
    engineered_features= [ 'debt_to_income', 'emi_to_income_ratio', 'loan_to_income_ratio', 'delayed_payment_freq', 'credit_efficiency', 'disposable_income_factor', 'payment_discipline_score']
    all_features = critical_features + important_features + engineered_features
    available_features = [f for f in all_features if f in data.columns]
    numerical_features = data[available_features].select_dtypes(include=['int64','float64','uint8']).columns.tolist()
    categorical_features = data[available_features].select_dtypes(include=['object']).columns.tolist()
    return {
        'features' : available_features , 
        'numerical' : numerical_features , 
        'categorical' : categorical_features
    }

def visualize_features_vs_target(data, target_column, feature_lists=None, figsize=(15, 10)):
    """
    Visualise les relations entre les différentes features et la variable target
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Le dataset contenant les features et la target
    target_column : str
        Le nom de la colonne target
    feature_lists : dict, optional
        Dictionnaire contenant les listes de features numériques et catégorielles
    figsize : tuple
        Taille de la figure
    """
    
    # Si feature_lists n'est pas fourni, le créer automatiquement
    if feature_lists is None:
        feature_lists = create_feature_list(data)
    
    numerical_features = feature_lists['numerical']
    categorical_features = feature_lists['categorical']
    
    # Vérifier que la target existe
    if target_column not in data.columns:
        print(f"Erreur: La colonne target '{target_column}' n'existe pas dans le dataset")
        return
    
    print(f"=== ANALYSE DES FEATURES VS TARGET ({target_column}) ===\n")
    
    # 1. Visualisation des features numériques
    if numerical_features:
        print(f"📊 FEATURES NUMÉRIQUES ({len(numerical_features)} features)")
        
        # Calculer le nombre de lignes et colonnes pour la grille
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], n_rows * 4))
        
        # CORRIGÉ: Gestion des cas où n_rows = 1
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                # Vérifier si la feature existe dans le dataset
                if feature in data.columns:
                    # Boxplot pour chaque catégorie de target
                    sns.boxplot(data=data, x=target_column, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} vs {target_column}')
                    axes[i].tick_params(axis='x', rotation=45)
                else:
                    axes[i].text(0.5, 0.5, f'Feature {feature}\nnon disponible', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{feature} (Non disponible)')
        
        # Masquer les axes vides
        for i in range(len(numerical_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Statistiques descriptives par target
        print("\n📈 STATISTIQUES DESCRIPTIVES PAR TARGET:")
        for feature in numerical_features:
            if feature in data.columns:
                print(f"\n{feature}:")
                stats = data.groupby(target_column)[feature].agg(['mean', 'median', 'std']).round(3)
                print(stats)
    
    # 2. Visualisation des features catégorielles
    if categorical_features:
        print(f"\n📊 FEATURES CATÉGORIELLES ({len(categorical_features)} features)")
        
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], n_rows * 5))
        
        # CORRIGÉ: Gestion des cas où n_rows = 1
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes):
                if feature in data.columns:
                    # Graphique en barres empilées
                    ct = pd.crosstab(data[feature], data[target_column], normalize='index') * 100
                    ct.plot(kind='bar', ax=axes[i], stacked=True)
                    axes[i].set_title(f'{feature} vs {target_column} (%)')
                    axes[i].set_ylabel('Pourcentage')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(title=target_column)
                else:
                    axes[i].text(0.5, 0.5, f'Feature {feature}\nnon disponible', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{feature} (Non disponible)')
        
        # Masquer les axes vides
        for i in range(len(categorical_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Test du Chi-2 pour les features catégorielles
        print("\n🔍 TESTS D'INDÉPENDANCE (Chi-2):")
        for feature in categorical_features:
            if feature in data.columns:
                try:
                    contingency_table = pd.crosstab(data[feature], data[target_column])
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    print(f"{feature}: Chi2={chi2:.3f}, p-value={p_value:.3f}")
                    if p_value < 0.05:
                        print(f"  ✅ Relation significative avec {target_column}")
                    else:
                        print(f"  ❌ Pas de relation significative avec {target_column}")
                except Exception as e:
                    print(f"{feature}: Erreur dans le test Chi-2 - {str(e)}")
    
    # 3. Matrice de corrélation pour les features numériques
    if numerical_features and len(numerical_features) > 1:
        print(f"\n📊 MATRICE DE CORRÉLATION")
        
        # Ajouter la target si elle est numérique
        features_for_corr = numerical_features.copy()
        if data[target_column].dtype in ['int64', 'float64', 'uint8']:
            features_for_corr.append(target_column)
        
        available_features = [f for f in features_for_corr if f in data.columns]
        
        if len(available_features) > 1:
            corr_matrix = data[available_features].corr()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Matrice de corrélation des features numériques')
            plt.tight_layout()
            plt.show()
            
            # Afficher les corrélations les plus fortes avec la target
            if target_column in corr_matrix.columns:
                target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
                print(f"\n🎯 CORRÉLATIONS AVEC {target_column}:")
                for feature, corr in target_corr.items():
                    if feature != target_column:
                        print(f"{feature}: {corr:.3f}")
    
    # 4. Résumé des insights
    print(f"\n📝 RÉSUMÉ:")
    print(f"- Features numériques analysées: {len(numerical_features)}")
    print(f"- Features catégorielles analysées: {len(categorical_features)}")
    print(f"- Variable target: {target_column}")
    print(f"- Classes dans la target: {data[target_column].unique()}")
    print(f"- Distribution de la target:")
    target_dist = data[target_column].value_counts(normalize=True) * 100
    for class_name, percentage in target_dist.items():
        print(f"  {class_name}: {percentage:.1f}%")

# Exemple d'utilisation
def main():
    # Charger et préprocesser les données
    data = load_data()
    data_processed = preprocess_data(data)
    data_processed = missing_val(data_processed)
    
    # Créer la liste des features
    feature_lists = create_feature_list(data_processed)
    
    # Visualiser les relations avec la target
    # Remplacer 'target' par le nom de votre variable target
    visualize_features_vs_target(data_processed, 'target', feature_lists)

if __name__ == "__main__":
    main()