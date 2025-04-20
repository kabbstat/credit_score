import openml
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
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
    