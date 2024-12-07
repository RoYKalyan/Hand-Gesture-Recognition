import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import os

# Gesture definitions
GESTURES = ["swipe", "push-pull", "circular", "unidentified"]

# Load the trained model
model_path = "models/random_forest_model.pkl"
if os.path.exists(model_path):
    selected_model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}. Ensure the file exists.")
    st.stop()

# Function to fetch signal strength
def fetch_signal_strength():
    html_code = """
    <script>
        async function getSignalStrength() {
            try {
                if ('connection' in navigator) {
                    let downlink = navigator.connection.downlink || -100;
                    document.getElementById("signal-strength").innerText = downlink;
                } else {
                    document.getElementById("signal-strength").innerText = -100;
                }
            } catch (error) {
                document.getElementById("signal-strength").innerText = -100;
            }
        }
        getSignalStrength();
    </script>
    <div>
        <p>Signal Strength: <span id="signal-strength">-100</span> Mbps</p>
    </div>
    """
    st.components.v1.html(html_code, height=100)
    signal = st.text_input("If signal strength is not displayed above, enter it manually:")
    try:
        return float(signal) if signal else -100
    except ValueError:
        st.warning("Please enter a valid numeric value for the signal strength.")
        return -100

# Preprocess live RSSI data
def preprocess_live_rssi(data):
    try:
        df = pd.DataFrame(data, columns=["rssi"])
        df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="10ms")
        df_resampled = df.resample("10ms").mean()
        df_resampled["rssi"] = df_resampled["rssi"].rolling(window=3, min_periods=1).mean()
        df_resampled["rssi"] = df_resampled["rssi"].interpolate().fillna(-100)  # Replace NaNs
        sequence = df_resampled["rssi"].values
        if len(sequence) < 101:
            sequence = np.pad(sequence, (0, 101 - len(sequence)), mode="constant", constant_values=-100)
        mean_rssi = np.mean(sequence)
        std_rssi = np.std(sequence)
        sequence = (sequence - mean_rssi) / std_rssi
        return sequence[:101]
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return np.full(101, -100)

# Predict gesture
def predict_gesture(sequence, model):
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

# Main app logic
def main():
    st.title("Gesture Prediction with Real-Time RSSI Data")
    st.markdown("This app uses Wi-Fi signal strength (RSSI) data to predict gestures.")

    # Fetch signal strength
    signal_strength = fetch_signal_strength()
    if signal_strength == -100:
        st.warning("Signal strength is -100. Please provide a valid input manually.")

    # Initialize data containers
    time_series = []
    rssi_series = []

    # Placeholder for dynamic graph updates
    chart_placeholder = st.empty()

    # Start gesture prediction
    if st.button("Start Capture"):
        st.markdown("Capturing RSSI data...")

        # Capture 101 data points
        for i in range(101):
            rssi_series.append(signal_strength)
            time_series.append(pd.Timestamp.now())

            # Update graph dynamically
            df = pd.DataFrame({"Time": time_series, "RSSI": rssi_series})
            fig = px.line(df, x="Time", y="RSSI", title="Live RSSI Data")
            chart_placeholder.plotly_chart(fig)

            # Simulate RSSI data collection interval
            time.sleep(0.1)

        # Preprocess and predict
        sequence = preprocess_live_rssi(rssi_series)
        predicted_gesture = predict_gesture(sequence, selected_model)
        st.success(f"Predicted Gesture: **{predicted_gesture}**")

if __name__ == "__main__":
    main()
