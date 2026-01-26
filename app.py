import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import base64
from gtts import gTTS
import tempfile

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Curriculum Gap Analysis", layout="wide")
st.title("Curriculum Gap Identification Dashboard")

# =========================
# BACKGROUND IMAGE FUNCTION
# =========================
def set_bg(image_file):
    if image_file is not None:
        bg_base64 = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)

# =========================
# SIDEBAR UPLOADS
# =========================
with st.sidebar:
    st.header("Customization")
    bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])
    video_file = st.file_uploader("Upload Demo Video", type=["mp4", "mov"])

    if bg_file:
        set_bg(bg_file)

# =========================
# MAIN CONTENT
# =========================
if video_file:
    st.sidebar.header("Demo Video")
    st.sidebar.video(video_file)

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # =========================
    # DATA VISUALIZATION
    # =========================
    st.markdown("---")
    st.header("Data Visualizations")

    selected_col = st.selectbox("Select a column to visualize", df.columns)

    if selected_col:
        col_data = df[selected_col]

        # Pie Chart
        fig_pie, ax_pie = plt.subplots()
        pie_data = col_data.value_counts().head(10)
        ax_pie.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%')
        st.pyplot(fig_pie)

        # Bar Chart
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(x=pie_data
