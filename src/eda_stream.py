import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io


st.set_page_config(page_title="EDA Credit Card Default", layout="wide")
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
# start streamlit app

@st.cache_data
def load_data():
    data = pd.read_csv("./data/processed/processed.csv")
    return data
data = load_data()
if data is None:
    st.stop("data not found, please check the data path")
st.title("Exploratory Data Analysis of Credit scoring Prediction Dataset")
st.sidebar.header("EDA options")
show_data_info = st.sidebar.checkbox("Show Data Info", value=True)
show_target_variable = st.sidebar.checkbox("Show target variable", value = True)
show_numercial_distribution = st.sidebar.checkbox("Show numerical distribution", value=True)
show_categorical_distribution = st.sidebar.checkbox("Show categorical distribution with option with target variable")

if show_data_info:
    with st.expander("Data Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("shape of the dataset:", data.shape)
        with col2:
            st.write("columns of the dataset:", list(data.columns))
        st.subheader("view of the 10 first rows of the dataset")
        st.dataframe(data.head(20), use_container_width=True)
        st.subheader("Descriptive statistics of the dataset")
        st.dataframe(data.describe(), use_container_width=True)
        st.subheader("Dataset Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
if show_target_variable:
    st.subheader("Distribution of Target Variable")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='target', data=data)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target')
    plt.ylabel('Count')
    st.pyplot(plt)
if show_numercial_distribution:
    st.subheader("Distribution des Variables Numériques")
    numerical_cols = data.select_dtypes(include=['number']).columns
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3  # Adjust rows for 3 plots per row
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), constrained_layout=True)
    axes = axes.flatten() if n_cols > 1 else [axes]
    for i, col in enumerate(numerical_cols):
        sns.histplot(data[col], bins=30, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f"Distribution de {col}", fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Nombre", fontsize=10)
        axes[i].tick_params(axis='both', labelsize=8)

    # Remove empty subplots
    for i in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[i])

    st.pyplot(fig)
if show_categorical_distribution:
    st.subheader("Analyse of categorcial features")
    exclude_features = ['customer_id', 'id','name','ssn','credit_history_age','type_of_loan','month']  # Features to exclude from categorical analysis
    categorical_features = [col for col in data.select_dtypes(include=['object']).columns if col not in exclude_features]
# Define visualization options
    viz_options = (
      ["Distribution de " + col for col in categorical_features] +
         [f"{col} vs Target" for col in categorical_features if col != 'target']
)
    selected_viz = st.selectbox("Sélectionner une visualisation", ["Aucune"] + viz_options)

    def count_plot(column, title, hue=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=column, data=data, hue=hue, palette='Set3')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
        plt.title(title, fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Nombre", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    if selected_viz != "Aucune":
        if " vs Target" in selected_viz:
            column = selected_viz.replace(" vs Target","")
            count_plot(column, f"Distribution of {column} vs Target", hue="target")
        else:
            column = selected_viz.replace("Distribution de ", "")
            count_plot(column, f"Distribution of {column}")
st.sidebar.header("Model Evaluation Options")
Modelisation = st.sidebar.checkbox("Modelisation", value=True)
if Modelisation:
    st.subheader("Model evaluation")
    




