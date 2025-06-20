import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ Localhost App Setup
st.set_page_config(page_title="Local Solar Predictor", layout="wide")

# ===============================
# 🔐 Password Authentication
# ===============================
PASSWORD = "SolarCast-2025"
st.sidebar.title("🔒 Login Required")
user_password = st.sidebar.text_input("Enter Access Password", type="password")
if user_password != PASSWORD:
    st.sidebar.warning("Invalid password.")
    st.stop()

# ===============================
# 🎨 App UI
# ===============================
st.title("☀️ LSTM NEXT 12-HOUR PREDICTOR")
st.markdown("Upload your CSV file for local testing of our solar forecasting model.")
st.markdown("<h4 style='text-align: center; color: gray;'>FYP by <b>Stuart Ssenabulya</b> and <b>Juliet Tusabe</b></h4>", unsafe_allow_html=True)

# ===============================
# 🔧 Load Model and Scaler
# ===============================
MODEL_PATH = "optimized_model3.h5"
SCALER_PATH = "scaler_model3.pkl"
WINDOW_SIZE = 24

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model (.h5) file not found!")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error("❌ Scaler file not found!")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# ===============================
# 📁 File Upload
# ===============================
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📁 Upload your test CSV (date & solar output)", type=["csv"])
with col2:
    true_file = st.file_uploader("📂 Upload actual 12-hour values (optional)", type=["csv"])

start_date = st.date_input("📅 Dataset Start Date", pd.to_datetime("2021-05-01"))
prediction_date = st.date_input("📅 Prediction Date", pd.to_datetime("2024-07-31"))

# ===============================
# 🧹 Preprocessing
# ===============================
def preprocess(df):
    try:
        if not all(col in df.columns for col in ['date', 'solar output']):
            st.error("CSV must contain 'date' and 'solar output' columns.")
            return None, None, None

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date', 'solar output'], inplace=True)
        df = df[df['date'] >= pd.to_datetime(start_date)]

        df.rename(columns={'date': 'ds', 'solar output': 'y'}, inplace=True)
        df = df.set_index('ds')
        df = df.between_time('06:00', '18:00')
        df.reset_index(inplace=True)

        values = df['y'].values.reshape(-1, 1)
        scaled = scaler.transform(values)

        if len(scaled) < WINDOW_SIZE:
            st.error("⛔ Not enough rows (need at least 24 after filtering).")
            return None, None, None

        last_24 = scaled[-WINDOW_SIZE:]
        X_input = last_24.reshape((1, WINDOW_SIZE, 1))
        return X_input, df, values[-12:] if len(values) >= 12 else values
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None

# ===============================
# 🔮 Forecasting Logic
# ===============================
def predict_next_12_hours(X_input):
    preds_scaled = []
    current_input = X_input.copy()
    for _ in range(12):
        next_pred = model.predict(current_input, verbose=0)
        preds_scaled.append(next_pred[0][0])
        current_input = np.append(current_input[:, 1:, :], [[[next_pred[0][0]]]], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_unscaled = scaler.inverse_transform(preds_scaled)
    return preds_unscaled.flatten()

# ===============================
# 📊 Evaluation
# ===============================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nrmse = rmse / (max(y_true) - min(y_true))
    return rmse, mae, r2, nrmse

def evaluate_segments(y_true, y_pred):
    return (
        mean_absolute_error(y_true[:4], y_pred[:4]),
        mean_absolute_error(y_true[4:8], y_pred[4:8]),
        mean_absolute_error(y_true[8:12], y_pred[8:12])
    )

# ===============================
# 🚀 Main Execution
# ===============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_input, cleaned_df, y_last12 = preprocess(df)

    if X_input is not None:
        y_pred = predict_next_12_hours(X_input)

        forecast_times = pd.date_range(
            start=pd.to_datetime(prediction_date.strftime("%Y-%m-%d") + " 06:00"),
            periods=12, freq='h'
        )
        forecast_df = pd.DataFrame({
            'Hourly Timestamp': forecast_times,
            'Predicted Solar Output (MW)': y_pred
        })

        st.success("✅ Forecast complete!")
        st.subheader("📊 12-Hour Forecast Output")
        st.dataframe(forecast_df)

        # Plotting
        st.subheader("📈 Forecast Plot")
        plt.figure(figsize=(10, 4))
        plt.plot(forecast_times, y_pred, marker='o', color='orange', label='Predicted')

        if true_file is not None:
            try:
                actual_df = pd.read_csv(true_file)
                if 'solar output' in actual_df.columns:
                    y_true = actual_df['solar output'].dropna().values[:12]
                    if len(y_true) >= 12:
                        plt.plot(forecast_times, y_true, marker='x', color='blue', label='Actual')
                        rmse, mae, r2, nrmse = evaluate(y_true, y_pred)
                        mae1, mae2, mae3 = evaluate_segments(y_true, y_pred)

                        st.subheader("📌 Evaluation Results")
                        st.write(f"**RMSE:** {rmse:.4f}")
                        st.write(f"**MAE:** {mae:.4f}")
                        st.write(f"**R² Score:** {r2:.4f}")
                        st.write(f"**nRMSE:** {nrmse:.4f}")
                        st.markdown("**Segmented MAE:**")
                        st.write(f"1–4 hrs: {mae1:.4f}, 5–8 hrs: {mae2:.4f}, 9–12 hrs: {mae3:.4f}")
            except:
                st.warning("⚠️ Issue loading actual values.")

        plt.title("Solar Output Forecast")
        plt.xlabel("Hour")
        plt.ylabel("MW")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

# ===============================
# 🎨 Custom CSS Styling
# ===============================
custom_css = """
<style>
    body {
        background: linear-gradient(to right, #f0f2f6, #ffffff);
    }
    .stApp h1 {
        font-family: 'Arial Black', sans-serif;
        color: #ff9900;
        text-align: center;
    }
    .stApp h4, .stApp h2, .stApp h3 {
        color: #333333;
        text-align: center;
    }
    .stDataFrame {
        border: 2px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
    }
    section[data-testid="stSidebar"] {
        background-color: #f6f6f9;
    }
    button {
        background-color: #ff9900 !important;
        color: white !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.sidebar.image("logo.png", use_container_width=True)
