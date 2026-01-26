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
        sns.barplot(x=pie_data.index, y=pie_data.values, ax=ax_bar)
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)

    # =========================
    # MACHINE LEARNING
    # =========================
    st.markdown("---")
    st.header("Machine Learning Analysis")

    target = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target])

    if target and features:
        X = pd.get_dummies(df[features], drop_first=True)
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )

        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
        model = LogisticRegression(max_iter=200) if model_choice == "Logistic Regression" else RandomForestClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.text(classification_report(y_test, y_pred))

        # =========================
        # PREDICTION + VOICE
        # =========================
        st.markdown("---")
        st.subheader("Make a Prediction")

        input_data = {}
        for col in features:
            if df[col].dtype in ["int64", "float64"]:
                input_data[col] = st.number_input(col, value=float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(col, df[col].unique())

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df, drop_first=True)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Class: {prediction}")
            st.balloons()

            # Voice (Streamlit Cloud SAFE)
            tts = gTTS(text=f"The predicted class is {prediction}", lang="en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name)
