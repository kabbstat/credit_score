import openml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier , HistGradientBoostingClassifier
# loading the data from openML
dataset = openml.datasets.get_dataset(46441)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
df = pd.DataFrame(X)
df['target']=y
# checking the data types   
df.dtypes
df['target'].value_counts()
df.describe()
df.head()
df.info()   
# checking the distribution of the target variable  
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom', fontsize=10)
plt.title('Distribution of the target variable (credit risk)')
plt.xlabel('Target variable')
plt.ylabel('Count')
plt.show()
# listing the categorical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features)
# listing the numerical features    
numerical_features = df.select_dtypes(include=['int64', 'float64','uint8']).columns.tolist()
print("Numerical features:", numerical_features)
# PLOTING THE DISTRIBUTION OF THE categorical FEATURES with target variable
'''
def plot_categorical_distribution(feature, title, hue=None):
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=feature, hue=hue, data=df)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()))
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Count')
    if hue:
        plt.legend(title='Target variable')
    plt.show()
   
plot_categorical_distribution('occupation', 'Distribution of type_of_loan with target variable')
for feature in categorical_features:
    plot_categorical_distribution(feature, f'Distribution of {feature} with target variable', hue='target')
plt.figure(figsize=(12, 8))
'''
# checking for missing values in the numerical features
'''
for feature in numerical_features:
    missing_count = df[feature].isna().mean()
    print(f"{feature}: {missing_count} missing values ({missing_count * 100:.2f}%)")
# checking the missing values in the categorical features   
for feature in categorical_features:
    missing_count = df[feature].isna().mean()
    print(f"{feature}: {missing_count} missing values ({missing_count * 100:.2f}%)")
'''

## critical features
critical_features = ['credit_history_age','outstanding_debt','num_of_delayed_payment','payment_behaviour','credit_utilization_ratio']
# checking the missing values in the critical features
for feature in critical_features:
    missing_count = df[feature].isna().mean()
    print(f"{feature}: {missing_count} missing values ({missing_count * 100:.2f}%)")
# transforming credit_history_age to months
extracted = df['credit_history_age'].str.extract(r'(?P<Years>\d+)\s*Years\s*and\s*(?P<Months>\d+)\s*Months').astype(float)
df['total_month'] = extracted['Years'] * 12 + extracted['Months']
# dealing with missing values in the critical features
# credit_history_age
df['total_month'].fillna(df['total_month'].median(), inplace=True)
critical_features_preprocessing = Pipeline([('imputer', KNNImputer(n_neighbors=3, weights='distance')),
                                 'scaler', StandardScaler()])
# Num_of_delayed_payment
df['num_of_delayed_payment'].fillna(df['num_of_delayed_payment'].median(),inplace=True)
# payment_behaviour
df['payment_behaviuour']= df['paymnet_behaviour'].replace('!@9#%8','unknown')
# ploting the distribution of the critical features with target variable 
def plot_critical_features_cat_target(feature, title, hue=None, palette='Set2'):
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
def plot_critical_features_num_target(feature, target, title, palette='Set2'):
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        x=target, y=feature, data=df, hue=target,  palette=palette,showmeans=True,
          meanprops={"marker":"o", "markerfacecolor":"white", "markersize":"10", "markeredgecolor":"black"},linewidth=1.5
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(target, fontsize=14, labelpad=10)
    plt.ylabel(feature, fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if feature == 'num_of_delayed_payment':
        plt.ylim(0, df[feature].quantile(0.95))  # Zoom sur 95% des données
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()  # Supprime la légende redondante
    
    plt.tight_layout()
    plt.show()
plot_critical_features_num_target('total_month', 'target', 'Distribution of credit_history_age with target variable')
plot_critical_features_num_target('num_of_delayed_payment', 'target', 'Distribution of num_of_delayed_payment with target variable')
plot_critical_features_cat_target('payment_behaviour', 'Distribution of payment_behaviour with target variable', hue='target')
plot_critical_features_num_target('credit_utilization_ratio', 'target', 'Distribution of credit_utilization_ratio with target variable')
plot_critical_features_num_target('outstanding_debt', 'target', 'Distribution of outstanding_debt with target variable')

# checking the distribution of the critical features with target variable
for feature in critical_features:
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=feature, hue='target', data=df)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()))
    plt.title(f'Distribution of {feature} with target variable')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Target variable')
    plt.show()
# checking the distribution of the credit_history_age with target variable  
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='credit_history_age', hue='target', data=df)
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()))
plt.title('Distribution of credit_history_age with target variable')
#drop unnecessary columns for ML classification
df.drop(columns=['id', 'customer_id'], inplace=True)     

# using KNN imputer for numerical features
numerical_preprocessing = Pipeline([('imputer', KNNImputer(n_neighbors=3, weights='distance')),
                                 'scaler', StandardScaler()])
categorical_preprocessing = Pipeline([('encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                                      'imputer', KNNImputer(n_neighbors=3, weights='distance')])
# creating the preprocessor
preprocessor = ColumnTransformer([('numerical',numerical_preprocessing, numerical_features),
                                  ('categorical', categorical_preprocessing, categorical_features)])
# creating the pipeline
classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
classifier = HistGradientBoostingClassifier(random_state=42, n_jobs=-1)
model = Pipeline([('preprocessor', preprocessor),
                           ('classifier', classifier)])
# feature engineering 
# dropping the target variable from the features
X = df.drop(columns=['target'])


