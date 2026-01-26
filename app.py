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

# ✅ Cloud-safe voice
from gtts import gTTS
import tempfile

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Curriculum Gap Analysis", layout="wide")
st.title("📊 Curriculum Gap Identification Dashboard")

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
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🎨 Customization")
    bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])
    video_file = st.file_uploader("Upload Demo Video", type=["mp4", "mov"])

    if bg_file:
        set_bg(bg_file)

if video_file:
    st.sidebar.video(video_file)

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # =========================
    # DATA VISUALIZATION
    # =========================
    st.markdown("---")
    st.header("📈 Data Visualizations")

    selected_col = st.selectbox("Select a column to visualize", df.columns)

    if selected_col:
        col_data = df[selected_col]

        # Pie Chart
        st.subheader("🥧 Pie Chart")
        fig1, ax1 = plt.subplots()
        pie_data = col_data.value_counts().head(10)
        ax1.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        # Bar Chart
        st.subheader("📊 Bar Chart")
        fig2, ax2 = plt.subplots()
        sns.barplot(x=pie_data.index, y=pie_data.values, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # =========================
    # MACHINE LEARNING
    # =========================
    st.markdown("---")
    st.header("🤖 Machine Learning Analysis")

    target = st.selectbox("Select Target Column", df.columns)
    features = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target]
    )

    if target and features:
        X = df[features]
        y = df[target]

        X_encoded = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

            model_choice = st.selectbox(
                "Choose Model", ["Logistic Regression", "Random Forest"]
            )

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # =========================
            # RESULTS
            # =========================
            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=model.classes_,
                        yticklabels=model.classes_,
                        ax=ax3)
            st.pyplot(fig3)

            # =========================
            # PREDICTION + VOICE
            # =========================
            st.markdown("---")
            st.subheader("🔮 Make a Prediction")

            input_data = {}
            cols = st.columns(2)

            for i, col in enumerate(features):
                with cols[i % 2]:
                    if df[col].dtype in ["int64", "float64"]:
                        input_data[col] = st.number_input(
                            f"Enter {col}", value=float(df[col].mean())
                        )
                    else:
                        input_data[col] = st.selectbox(
                            f"Select {col}", df[col].unique()
                        )

            if st.button("Predict", type="primary"):
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df, drop_first=True)
                input_encoded = input_encoded.reindex(
                    columns=X_encoded.columns, fill_value=0
                )

                prediction = model.predict(input_encoded)[0]

                st.balloons()
                st.success(f"✅ Predicted Class: {prediction}")

                # ✅ AI VOICE (NO ERROR)
                tts = gTTS(text=f"The predicted class is {prediction}", lang="en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tts.save(f.name)
                    st.audio(f.name)

                # Animation
                st.markdown("""
                <div style="text-align:center;
                            font-size:24px;
                            font-weight:bold;
                            color:green;">
                    🎉 Prediction Successful!
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
