import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from scapy.all import sniff, Dot11
import os

# Set Streamlit page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Gesture Prediction with Real-Time RSSI Data",
    page_icon="📡",
    layout="wide",
)

# Gesture definitions
GESTURES = ["swipe", "push-pull", "circular", "unidentified"]

# Load the trained model
model_path = os.path.join(os.getcwd(), "models/random_forest_model.pkl")
if os.path.exists(model_path):
    selected_model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}. Ensure the file exists.")

# RSSI data collection function
def get_live_rssi_data():
    """Capture RSSI data using scapy."""
    try:
        rssi_values = []

        def packet_handler(pkt):
            if pkt.haslayer(Dot11):
                if pkt.type == 0 and pkt.subtype == 8:  # Beacon frame
                    rssi = pkt.dBm_AntSignal
                    if rssi is not None:
                        rssi_values.append(rssi)

        # Capture packets for a short duration
        sniff(prn=packet_handler, timeout=2, iface="wlan0mon", store=False)
        
        if rssi_values:
            avg_rssi = np.mean(rssi_values)
            return avg_rssi
        else:
            return -100  # Default if no networks are found
    except Exception as e:
        st.error(f"Error capturing RSSI data: {e}")
        return -100

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
    st.markdown("Press the 'Start Capture' button to begin capturing live RSSI data. The capture will automatically stop after collecting 101 data points, and a gesture prediction will be displayed.")

    # Create placeholders for the chart and prediction
    chart_placeholder = st.empty()
    prediction_placeholder = st.empty()

    # Initialize data containers
    time_series = []
    rssi_series = []

    # Capture control
    capturing = False

    # Start button
    if st.button("Start Capture"):
        capturing = True

    while capturing and len(rssi_series) < 101:
        new_rssi = get_live_rssi_data()
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
