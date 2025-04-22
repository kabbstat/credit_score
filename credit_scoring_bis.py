import openml
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score,RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def load_data():
    #load the dataset from openml
    dataset = openml.datasets.get_dataset(46441)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    df = pd.DataFrame(X)
    df['target'] = y
    return df
def explore_data(df):
    """explorer les informations essentielles du datset"""
    print("shape of the dataset:", df.shape)
    print("columns of the dataset :", df.columns)
    print('\n percentage of missing values:')
    missing_percentage = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_percentage[missing_percentage>0])
    return {
        'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
        'numerical_features': df.select_dtypes(include=['int64', 'float64', 'uint8']).columns.tolist()
    }
def preprocess_data(df):
    df_processed = df.copy()
    # Convert 'credit_history_age' to numeric values 'months'
    extracted = df_processed['credit_history_age'].str.extract(r'(?P<Years>\d+)\s*Years\s*and\s*(?P<Months>\d+)\s*Months').astype(float)
    df_processed['total_month'] = extracted['Years'] * 12 + extracted['Months']
    # Treatment of some problematique in the dataset
    df_processed['payment_behaviour'] = df_processed['payment_behaviour'].replace('!@9#%8', 'unknown')
    df_processed['credit_mix'] = df_processed['credit_mix'].replace('_', pd.NA)
    # feature engineering - creating new credit scores
    df_processed['debt_to_income'] = df_processed['outstanding_debt'] / df_processed['annual_income']
    df_processed['emi_to_income_ratio'] = df_processed['total_emi_per_month'] / df_processed['monthly_inhand_salary']
    df_processed['loan_to_income_ratio'] = df_processed['outstanding_debt'] / df_processed['monthly_inhand_salary']
    df_processed['delayed_payment_freq'] = df_processed['num_of_delayed_payment'] / df_processed['num_of_loan']
    # new scores
    df_processed['credit_efficiency'] = df_processed['credit_utilization_ratio'] *(1- df_processed['delay_from_due_date'])
    df_processed['disposable_income_factor'] = (df_processed['monthly_inhand_salary'] - df_processed['total_emi_per_month'])/df_processed['monthly_inhand_salary']
    df_processed['payment_discipline_score'] = 1 - (df_processed['num_of_delayed_payment'] / (df_processed['num_credit_inquiries'] + 1))
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
    return df_processed
def create_feature_list(df):
    critical_features = ['total_month','outstanding_debt','num_of_delayed_payment','payment_behaviour','credit_utilization_ratio']
    important_features = ['annual_income', 'total_emi_per_month', 'monthly_inhand_salary','delay_from_due_date', 'credit_mix']
    engineered_features = [ 'debt_to_income', 'emi_to_income_ratio', 'loan_to_income_ratio', 'delayed_payment_freq', 'credit_efficiency', 'disposable_income_factor', 'payment_discipline_score']
    numerical_features = df[critical_features + important_features + engineered_features].select_dtypes(include=['int64','float64','uint8']).columns.tolist()
    categorical_features = df[critical_features + important_features + engineered_features].select_dtypes(include =['object']).columns.tolist()
    return { 'features' : critical_features+ important_features +engineered_features,
            'numerical': numerical_features,
            'categorical': categorical_features
    }
# some plots 
'''
def plot_features_cat_target(df, feature, title, hue=None, palette='Set2'):
    plt.figure(figsize=(12, 7))  
    ax = sns.countplot(   x=feature,   hue=hue,   data=df,  palette=palette,  edgecolor='black',   linewidth=1.2)
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():,.0f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),  ha='center',  va='bottom',  fontsize=10,   color='black', xytext=(0, 5), textcoords='offset points' )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(feature, fontsize=14, labelpad=10)
    plt.ylabel('Count', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    if hue: plt.legend( title='Target Variable',title_fontsize=12,fontsize=11,frameon=True,framealpha=0.8, edgecolor='black' )
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
def plot_features_num_target(df, feature, target, title, palette='Set2'):
    plt.figure(figsize=(12, 7))  
    ax = sns.boxplot(x=target, y=feature, data =df , palette=palette, edgecolor='black', linewidth=1.2)
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():,.0f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),  ha='center',  va='bottom',  fontsize=10,   color='black', xytext=(0, 5), textcoords='offset points' )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Target Variable', fontsize=14, labelpad=10)
    plt.ylabel(feature, fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
plot_features_num_target('df_processed','total_month','target','Total months of credit history vs Target')
plot_features_num_target('df_processed','outstanding_debt','target','Distribution of Outstanding_debt vs Target')
plot_features_cat_target('df_processed','payment_behaviour','Distribution of payment behaviour vs Target')
plot_features_num_target('df_processed','num_of_delayed_payment','target','Distribution of number of delayed payment vs target')
plot_features_num_target('df_processed','credit_utilization_ratio','target','Distribution of credit utilization ratio vs target')
'''
def build_preprocessor(numerical_features, categorical_features):
    numerical_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
                                       ('scaler',StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('encoder',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('numerical', numerical_transformer, numerical_features),
                                      ('categorical', categorical_transformer, categorical_features)])
    return preprocessor
def build_model(preprocessor, use_smote = True):
    if use_smote:
        pipeline = ImbPipeline([('preprocessor',preprocessor),
                                ('smote',SMOTE(random_state=42)),
                                ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])
    else:
        pipeline = Pipeline([('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])
    return pipeline
def tune_hyperparameters(pipeline, X,y):
    param_grid ={
        'classifier__n_estimators':[100,200,300],
        'classifier__max_depth':[None, 10,20,30],
        'classifier__min_samples_split': [2,5,10],
        'classifier__min_samples_leaf': [1,2,4],
        'classifier__class_weight': ['balanced', None]
    }
    cv = StratifiedKFold(n_splits=5, shuffle= True, random_state=42)
    search = RandomizedSearchCV(pipeline, param_distributions= param_grid, n_iter=20, scoring ='roc_auc', cv =cv , n_jobs=-1, random_state= 42)
    search.fit(X,y)
    print("Best hyperparameters:", search.best_params_)
    return search.best_estimator_
def evaluate_model(model, X, y):
    """evaluate the model using cross-validation and print the classification report"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
    scores = cross_validate(model, X,y, cv= cv, scoring = scoring, return_train_score=True)
    print("Cross-validation scores:")
    for metric in scoring:
        print(f"{metric}: {scores['test_'+ metric].mean():.4f} (+/- {scores['test_'+ metric].std():.4f})")
    return scores
def plot_feature_importance(model,feature_names):
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names= model.named_steps['preprocessor'].get_feature_names_out()
        importances_df = pd.DataFrame({'feature':feature_names, 'importance': importances}).sort_values(by= 'importance', ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importances_df.head(20), palette='viridis')
        plt.title('Top 20 Feature Importances', fontsize=16, pad=20)
        plt.xlabel('Importance', fontsize=14, labelpad=10)
        plt.ylabel('Feature', fontsize=14, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()  
        plt.show()
        return importances_df
    else:
        print("Model does not have feature importances.")
        return None 
def create_credit_score(model, X, feature_info):
    try:
        proba = model.predict_proba(X)[:, 1]
        credit_score = 300 + (1-proba)*550
        return credit_score
    except:
        print("Model does not support probability predictions.")
        return None
def main():
    print("Loading data...")
    df = load_data()
    print("Exploring data...")  
    data_info = explore_data(df)
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    feature_info = create_feature_list(df_processed)
    X= df_processed[feature_info['features']]
    y= df_processed['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
    print("Building preprocessor...")
    preprocessor = build_preprocessor(feature_info['numerical'], feature_info['categorical'])
    pipeline = build_model(preprocessor, use_smote=True)
    print("Tuning hyperparameters...")
    best_model = tune_hyperparameters(pipeline, X_train , y_train)
    print('Evaluating model...')
    scores = evaluate_model(best_model, X_test, y_test)
    print("Feature importance:")
    importances = plot_feature_importance(best_model, feature_info['features'])
    print("Creating credit score...")
    credit_score = create_credit_score(best_model, X_test, feature_info)
    if credit_score is not None :
        print(f'the mean of the credit score is : {credit_score.mean():.2f}')
        print(f'the median of the credit score is : {credit_score.median():.2f}')
        print(f'the whole interval of the credit score is : [{credit_score.min():.2f}, {credit_score.max():.2f}]')
        plt.figure(figsize=(12, 7))
        sns.histplot(credit_score, bins=30, kde=True, color='blue', edgecolor='black')
        plt.title('Distribution of Credit Score', fontsize=16, pad=20)
        plt.xlabel('Credit Score', fontsize=14, labelpad=10)
        plt.ylabel('Frequency', fontsize=14, labelpad=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    return {
        'model': best_model,
        'feature_importance': importances,
        'scores': scores,
        'credit_score': credit_score
    }
if __name__ == "__main__":
    main()