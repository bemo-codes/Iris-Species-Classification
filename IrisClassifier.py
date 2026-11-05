#############################################################################
#
#   Importing required libraries
#
#############################################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#############################################################################
#
#   Function name:      load_data
#   Input:              data_path (string)
#   Description:        Loads the dataset from CSV file and caches it for 
#                       optimal performance in Streamlit.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

#############################################################################
#
#   Function name:      plot_visualization
#   Input:              df (DataFrame)
#   Description:        Generates a Matplotlib figure for Petal Length vs Width.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_visualization(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sp in df["species"].unique():
        temp = df[df["species"] == sp]
        ax.scatter(temp["petal length (cm)"], temp["petal width (cm)"], label=sp)

    ax.set_title("Petal Length vs Petal Width")
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.legend()
    ax.grid(True)
    return fig

#############################################################################
#
#   Function name:      train_and_evaluate
#   Input:              df (DataFrame), test_size (float), max_depth (int)
#   Description:        Trains Decision Tree Classifier based on interactive 
#                       sidebar inputs and computes evaluation metrics.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def train_and_evaluate(df, test_size, max_depth):
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
    X = df[feature_cols]
    Y = df["species"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)

    return model, acc, cm, report

#############################################################################
#
#   Function name:      main
#   Input:              None
#   Description:        Main entry point managing Streamlit layout, tabs, 
#                       and real-time prediction interface.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def main():
    st.set_page_config(page_title="Iris ML Classifier", page_icon="🌺", layout="wide")

    st.title("🌺 Iris Species Classification Dashboard")
    st.write("Interactive Machine Learning dashboard built with Streamlit and Scikit-Learn.")

    # Sidebar hyperparameters
    st.sidebar.header("⚙️ Model Configuration")
    test_size = st.sidebar.slider("Test Split Size", 0.1, 0.9, 0.5, 0.05)
    max_depth = st.sidebar.slider("Max Tree Depth", 1, 10, 5)

    # Load dataset
    try:
        df = load_data("iris.csv")
    except Exception:
        st.error("Error: `iris.csv` file not found in the root directory!")
        return

    # Train model dynamically
    model, acc, cm, report = train_and_evaluate(df, test_size, max_depth)

    # UI Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset & EDA", 
        "📈 Visualization", 
        "🎯 Model Metrics", 
        "🔮 Real-time Prediction"
    ])

    with tab1:
        st.subheader("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Preview:**")
            st.dataframe(df.head())
            st.write(f"**Total Records:** `{df.shape[0]}` rows | `{df.shape[1]}` columns")
        with col2:
            st.write("**Species Distribution:**")
            st.dataframe(df["species"].value_counts())

        st.write("**Statistical Summary:**")
        st.dataframe(df.describe())

    with tab2:
        st.subheader("Feature Relationships")
        fig = plot_visualization(df)
        st.pyplot(fig)

    with tab3:
        st.subheader("Model Evaluation")
        st.metric(label="Classifier Accuracy", value=f"{acc * 100:.2f}%")

        col_cm, col_rep = st.columns(2)
        with col_cm:
            st.write("**Confusion Matrix Heatmap:**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=df["species"].unique(),
                        yticklabels=df["species"].unique(), ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        with col_rep:
            st.write("**Classification Report:**")
            st.text(report)

    with tab4:
        st.subheader("Predict Custom Measurements")
        st.write("Adjust flower dimensions to test live classification:")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sepal_len = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
            sepal_wid = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
        with col_s2:
            petal_len = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
            petal_wid = st.number_input("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)

        if st.button("Classify Species", type="primary"):
            sample = [[sepal_len, sepal_wid, petal_len, petal_wid]]
            prediction = model.predict(sample)[0]
            st.success(f"Predicted Class: **{prediction.upper()}**")

#############################################################################
#
#   Starter of main
#
#############################################################################

if __name__ == "__main__":
    main()