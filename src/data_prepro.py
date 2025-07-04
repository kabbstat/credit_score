import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data():
    data = pd.read_csv("./data/raw/data.csv")
    return data
'''
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
'''
'''
def missing_val(data):
    for col in data.columns:
        if data[col].isnull().any():  
            if data[col].dtype in ['int64', 'float64', 'uint8']: 
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
    return data
'''

def preprocess_data(data):
    data_pro = data.copy()
    # convert credit_history_age to numeric values months
    extracted = data_pro['credit_history_age'].str.extract(r'(?P<Years>\d+)\s*Years\s*and\s*(?P<Months>\d+)\s*Months').astype(float)
    data_pro['total_month'] = extracted['Years'] *12 + extracted['Months']
    # deal with some problem feature in the dataset 
    data_pro['payment_behaviour'] = data_pro['payment_behaviour'].replace('!@9#%8', 'unknown')
    data_pro['credit_mix'] = data_pro['credit_mix'].replace('_',pd.NA)
    return data_pro
def deal_missing_values(data):
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype in ['int64','float64','uint8']:
                imputer_quant = SimpleImputer(strategy='median',missing_values=np.nan) 
                data[col] = imputer_quant.fit_transform(data[[col]]).ravel()
            else: 
                imputer_quali = SimpleImputer(strategy='most_frequent',missing_values=pd.NA)
                data[col] = imputer_quali.fit_transform(data[[col]]).ravel()
    return data


def feature_eng(data_pro):
        # creating some new feature - new credit scores 
    data_pro['debt_to_income'] = data_pro['outstanding_debt'] / data_pro['annual_income']
    data_pro['emi_to_income_ratio'] = data_pro['total_emi_per_month'] / data_pro['monthly_inhand_salary']
    data_pro['loan_to_income_ratio'] = data_pro['outstanding_debt'] / data_pro['monthly_inhand_salary']
    data_pro['delayed_payment_freq'] = data_pro['num_of_delayed_payment'] / data_pro['num_of_loan']
    # new scores
    data_pro['credit_efficiency'] = data_pro['credit_utilization_ratio'] *(1- data_pro['delay_from_due_date'])
    data_pro['disposable_income_factor'] = (data_pro['monthly_inhand_salary'] - data_pro['total_emi_per_month'])/data_pro['monthly_inhand_salary']
    data_pro['payment_discipline_score'] = 1 - (data_pro['num_of_delayed_payment'] / (data_pro['num_credit_inquiries'] + 1))
    data_pro.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
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


# Exemple d'utilisation
def main():
    raw = load_data()

    pre = preprocess_data(raw)
    clean = deal_missing_values(pre)

    eng = feature_eng(clean)

    feats = create_feature_list(eng)
    print("Features sélectionnées :", feats)

    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    clean.to_csv(os.path.join(out_dir, 'processed.csv'), index=False)
    print(f"Données prétraitées enregistrées dans {out_dir}/processed.csv")

if __name__ == "__main__":
    main()