import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_leaflet import leaflet_marker_cluster

# Dummy data for demonstration purposes
@st.cache_data
def load_data():
    df = pd.DataFrame({
        'lat': np.random.uniform(40, 60, 100),
        'lon': np.random.uniform(-120, -80, 100),
        'class': np.random.choice(['A', 'B', 'C'], 100),
        'confidence': np.random.uniform(0.5, 1.0, 100)
    })
    return df

def main():
    st.set_page_config(page_title="ML Workflow Manager", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Overview", "View Classification", "Maps", "Model Evaluation"])

    if page == "Project Overview":
        project_overview()
    elif page == "View Classification":
        view_classification()
    elif page == "Maps":
        maps()
    elif page == "Model Evaluation":
        model_evaluation()

def project_overview():
    st.title("Project Overview")
    st.write("""
    Welcome to the ML Workflow Manager project. This application provides an overview of our machine learning pipeline,
    including data ingestion, processing, model training, evaluation, deployment, and monitoring.
    
    Use the sidebar to navigate through different sections of the application.
    """)

def view_classification():
    st.title("View Classification")
    df = load_data()
    st.write("Sample of classification results:")
    st.dataframe(df.head(10))

    st.subheader("Classification Distribution")
    fig = px.pie(df, names='class', title='Distribution of Classes')
    st.plotly_chart(fig)

def maps():
    st.title("Maps")
    df = load_data()

    st.subheader("Classification Map")
    leaflet_marker_cluster(df, 'lat', 'lon', popup=['class', 'confidence'])

def model_evaluation():
    st.title("Model Evaluation")
    
    # Dummy metrics for demonstration
    accuracy = 0.85
    precision = 0.82
    recall = 0.88
    f1_score = 0.85

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1_score:.2f}")

    st.subheader("Confusion Matrix")
    confusion_matrix = np.array([[80, 10, 5], [5, 85, 10], [10, 5, 90]])
    fig = px.imshow(confusion_matrix, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['A', 'B', 'C'],
                    y=['A', 'B', 'C'])
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
