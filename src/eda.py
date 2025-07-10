import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def load_data():
    data = pd.read_csv("./data/processed/processed.csv")
    return data

def explore_data(data):
    print("Exploring data")
    print("shape of the dataset:", data.shape)
    print("columns of the dataset:", data.columns)
    print(data.head(10))
    # describe tge dataset
    print("\nDescriptive statistics of the dataset:")
    print(data.describe())
    print(data.info())

def plot_indicators(data):
    print("Plotting indicators")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='target', data=data)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.show()
    # Plotting categorical features vs target as option
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()


