import streamlit as st
import pickle
import plotly.graph_objects as go
from PIL import Image

# Load training history data (Python model)
with open("dropout_GAP_history.pkl", "rb") as f:
    simple_history = pickle.load(f)

# Load confusion matrix figure (Matplotlib figure)
with open("conf_matrix_fig.pkl", "rb") as f:
    conf_matrix_fig = pickle.load(f)

# Function to plot training history interactively with Plotly
def plot_training_history_interactive(history):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(1, len(history["train_loss"]) + 1)), 
                             y=history["train_loss"],
                             mode='lines+markers',
                             name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, len(history["val_loss"]) + 1)), 
                             y=history["val_loss"],
                             mode='lines+markers',
                             name='Val Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, len(history["train_acc"]) + 1)), 
                             y=history["train_acc"],
                             mode='lines+markers',
                             name='Train Accuracy',
                             yaxis='y2'))
    fig.add_trace(go.Scatter(x=list(range(1, len(history["val_acc"]) + 1)), 
                             y=history["val_acc"],
                             mode='lines+markers',
                             name='Val Accuracy',
                             yaxis='y2'))

    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy (%)", overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# Sidebar Navigation with 3 Pages
page = st.sidebar.selectbox("Choose a page", ["Home Page", "Python Model Results", "R Model Results"])

if page == "Home Page":
    st.image("satellite_image.jpg", use_container_width=True)
    st.title("Satellite Image Classification Project")
    st.markdown("""
    Welcome!  
    With the increase of technology in space, satellite imagery has become a massive tool 
    for providing critical visual data of our planet and many others. Satellite imagery 
    is used for environment monitoring, urban/agricultural planning and land use. However, 
    the volume and complexity of the satellite images make it difficult to analyze and 
    categorize them for effective use. 

    This problem stems from the common “automatic image classification” issue that is prevalent in many areas of the world. More specifically, 
    the task at hand is to classify an image taken by a satellite into 4 distinct categories, 
    including “water”, “cloudy”, “green_area” and “desert”. Each image belongs to one of 
    these categories and our goal is to accurately classify the raw image. While this report 
    focuses on general land-type classification, this approach can be extended to more specific
    tasks, given more detailed images. For example, if the images were captured in more 
    targeted regions, models similar to the ones in this report, could be used for analyzing
    natural disasters, monitoring specific wildlife habitats or other highly detailed issues. 

    This is a supervised image classification problem, making a Convolution Neural Network a natural fit.
    The network will take the RGB satellite images as an input, and will assign each image a
    label as its output. By developing and evaluating a CNN classifier, this project aims
    to explore how deep learning can be applied and modified to best classify satellite images,
    with a goal to improve the accuracy of geographic data applications.
    """)

elif page == "Python Model Results":
    st.title("Python Model Results – Dropout + GAP CNN")

    st.subheader("Training History")
    interactive_fig = plot_training_history_interactive(simple_history)
    st.plotly_chart(interactive_fig, use_container_width=True)

    st.subheader("Confusion Matrix")
    st.pyplot(conf_matrix_fig)

elif page == "R Model Results":
    st.title("R Model Results")

    st.subheader("Accuracy Over Time")
    st.image("R_model_accuracy.png", use_container_width=True)

    st.subheader("Loss Over Time")
    st.image("R_model_loss.png", use_container_width=True)

    st.subheader("Class-Level Statistics")
    st.image("R_model_class_stats.png", use_container_width=True)
