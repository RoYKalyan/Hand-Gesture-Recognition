import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from streamlit_js_eval import streamlit_js_eval as st_js_eval

# Set Streamlit page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Gesture Prediction with Real-Time RSSI Data",
    page_icon="📡",
    layout="wide",
)

# Gesture definitions
GESTURES = ["swipe", "push-pull", "circular", "unidentified"]

# Load the trained model
model_path = "models/random_forest_model.pkl"
if os.path.exists(model_path):
    selected_model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}. Ensure the file exists.")
    st.stop()

# Preprocess live RSSI data
def preprocess_live_rssi(data):
    """Preprocess live RSSI data to match training preprocessing."""
    try:
        df = pd.DataFrame(data, columns=['rssi'])
        df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='10ms')
        df_resampled = df.resample('10ms').mean()
        df_resampled['rssi'] = df_resampled['rssi'].rolling(window=3, min_periods=1).mean()
        df_resampled['rssi'] = df_resampled['rssi'].interpolate()
        sequence = df_resampled['rssi'].values
        if len(sequence) < 101:
            sequence = np.pad(sequence, (0, 101 - len(sequence)), mode='constant', constant_values=-100)
        # Dynamically calculate mean and standard deviation
        mean_rssi = np.mean(sequence)
        std_rssi = np.std(sequence)
        sequence = (sequence - mean_rssi) / std_rssi
        return sequence[:101]
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return np.full(101, -100)

# Predict gesture
def predict_gesture(sequence, model):
    """Predict gesture based on the preprocessed sequence."""
    try:
        processed_sequence = np.array(sequence).reshape(1, -1)
        prediction = model.predict(processed_sequence)[0]
        if prediction < len(GESTURES) - 1:
            return GESTURES[prediction]
        else:
            return "unidentified"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"

# Main Streamlit application
def main():
    st.title("Gesture Prediction with Real-Time RSSI Data")
    st.markdown(
        "This app uses real-time Wi-Fi signal strength (RSSI) data captured directly from your browser to predict gestures in real-time."
    )

    # Execute JavaScript to fetch network signal strength
    js_code = """
    async function getSignalStrength() {
        if ('connection' in navigator) {
            return navigator.connection.downlink || -100;
        }
        return -100; // Default if no API available
    }
    getSignalStrength();
    """
    signal_strength = st_js_eval(js_code, key="network_signal")

    if signal_strength is not None:
        st.info(f"Captured Signal Strength: {signal_strength} dBm")

    # Capture and process data
    st.markdown("Press the 'Start Capture' button to begin capturing live RSSI data.")
    chart_placeholder = st.empty()
    prediction_placeholder = st.empty()

    # Initialize data containers
    time_series = []
    rssi_series = []

    # Capture control
    capturing = False
    if st.button("Start Capture"):
        capturing = True

    while capturing and len(rssi_series) < 101:
        new_rssi = signal_strength if signal_strength is not None else -100
        current_time = pd.Timestamp.now()

        # Append new data
        time_series.append(current_time)
        rssi_series.append(new_rssi)

        # Create DataFrame for plotting
        data = pd.DataFrame({'Time': time_series, 'RSSI': rssi_series})

        # Plot the data
        fig = px.line(data, x='Time', y='RSSI', title='Live RSSI Data')
        chart_placeholder.plotly_chart(fig)

        # Check if 101 data points have been collected
        if len(rssi_series) == 101:
            sequence = preprocess_live_rssi(rssi_series)
            predicted_gesture = predict_gesture(sequence, selected_model)
            prediction_placeholder.write(f"Predicted Gesture: **{predicted_gesture}**")
            capturing = False  # Stop capturing after prediction

        # Sleep interval can be adjusted as needed
        time.sleep(0.01)

if __name__ == "__main__":
    main()
