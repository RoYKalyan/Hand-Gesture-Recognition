import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from CoreWLAN import CWWiFiClient

# Gesture definitions
GESTURES = ["swipe", "push-pull", "circular", "unidentified"]

# Initialize Wi-Fi client
wifi_interface = CWWiFiClient.sharedWiFiClient().interface()

data_seq = '3'
# Load the trained model
model_path = "../models/" + data_seq + "/" + "random_forest_model.pkl"
#/Users/kalyanroy/Desktop/cap/Hand-Gesture-Recognition/models/1/svm_model.pkl
print("Model Path====", model_path)
selected_model = joblib.load(model_path)

# RSSI data collection function
def get_live_rssi_data():
    """Capture RSSI data using the CoreWLAN library."""
    try:
        networks, _ = wifi_interface.scanForNetworksWithName_error_(None, None)
        rssi_values = [network.rssiValue() for network in networks]
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
        
        # Smooth the data using a moving average
        df['rssi'] = df['rssi'].rolling(window=3, min_periods=1).mean()
        
        # Compute mean and std
        mean_rssi = np.mean(df['rssi'].values)
        std_rssi = np.std(df['rssi'].values)
        
        # Handle the case where std_rssi is 0 (all values are identical)
        if std_rssi == 0:
            # If all values are the same, normalization will produce NaNs.
            # Instead, just subtract the mean (all values become 0).
            sequence = df['rssi'].values - mean_rssi
        else:
            sequence = (df['rssi'].values - mean_rssi) / std_rssi
        
        # Ensure sequence length matches window size (50)
        if len(sequence) < 50:
            sequence = np.pad(sequence, (0, 50 - len(sequence)), mode='constant', constant_values=-100)
        
        # Replace any remaining NaNs with a safe default (e.g., -100)
        sequence = np.nan_to_num(sequence, nan=-100)
        
        return sequence[:50]
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return np.full(50, -100)


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
    st.set_page_config(
        page_title="Gesture Prediction with Real-Time RSSI Data",
        page_icon="📡",
        layout="wide",
    )
    st.title("Gesture Prediction with Real-Time RSSI Data")
    st.markdown("Press the 'Start Capture' button to begin capturing live RSSI data. The capture will automatically stop after collecting 50 data points, and a gesture prediction will be displayed.")

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

    while capturing and len(rssi_series) < 50:
        new_rssi = get_live_rssi_data()

        # Append new data
        rssi_series.append(new_rssi)

        # Create DataFrame for plotting
        data = pd.DataFrame({'Index': list(range(len(rssi_series))), 'RSSI': rssi_series})

        # Plot the data
        fig = px.line(data, x='Index', y='RSSI', title='Live RSSI Data')
        chart_placeholder.plotly_chart(fig.update_layout(autosize=True, height=600, margin=dict(l=20, r=20, t=50, b=20)), use_container_width=True)

        # Check if 50 data points have been collected
        if len(rssi_series) == 50:
            sequence = preprocess_live_rssi(rssi_series)
            predicted_gesture = predict_gesture(sequence, selected_model)
            prediction_placeholder.markdown(f"<h3 style='color:blue; text-align:center;'>Predicted Gesture: <b>{predicted_gesture}</b></h3>",unsafe_allow_html=True,)
            capturing = False  # Stop capturing after prediction

        # Sleep interval for approx. 50 data points per second
        time.sleep(0.02)

if __name__ == "__main__":
    main()
