import pandas as pd
from data_prepro import create_feature_list
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data():
    data = pd.read_csv("./data/processed/processed.csv")
    return data

def stratified_sample(df, stratify_col, frac=1):
    return df.groupby(stratify_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac)).reset_index(drop=True)

def main():
    db = load_data()
    df = stratified_sample(db, 'target', frac=1)
    
    # Obtenir les features (une seule fois)
    feature_info = create_feature_list(df)
    features = feature_info['features']
    cat_features = feature_info['categorical']
    num_features = feature_info['numerical']
    
    print(f"Features disponibles: {features}")
    print(f"Features catégorielles: {cat_features}")
    print(f"Features numériques: {num_features}")
    
    # Préparer les données
    # S'assurer que toutes les features sont disponibles dans df
    available_features = [f for f in features if f in df.columns]
    data = df[available_features].copy()
    
    # Séparer X et y
    if 'target' in data.columns:
        y = data['target']
        X = data.drop(columns=['target'])
    else:
        raise ValueError("La colonne 'target' n'est pas trouvée dans les données")
    
    # Mettre à jour les listes de features après avoir retiré 'target'
    cat_features_final = [f for f in cat_features if f in X.columns]
    num_features_final = [f for f in num_features if f in X.columns]
    
    print(f"Features catégorielles finales: {cat_features_final}")
    print(f"Features numériques finales: {num_features_final}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features_final),
            ('num', StandardScaler(), num_features_final)])
    
    # Créer le pipeline
    model = Pipeline([('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42))])

    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Faire des prédictions
    y_pred = model.predict(X_test)
    
    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()