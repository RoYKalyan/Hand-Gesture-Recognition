from pynput import keyboard
import CoreWLAN
import datetime as dt
import time
import json
import os
import signal
import sys
import logging
import threading

# List of gestures mapped to keys
GESTURES = {
    '1': 'swipe',
    '2': 'push-pull',
    '3': 'circular'
}

# Flag to stop capturing
stop_capture = False
capture_thread = None

# Configure logging
logging.basicConfig(
    filename=f'{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def init(file_path):
    """Initialize the output JSON file."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        # Create a new file and add the opening bracket
        with open(file_path, 'w') as file:
            file.write('[')
        logging.info(f"Initialized JSON file: {file_path}")

def append_record(file_path, record):
    """Append a single record to the specified JSON file."""
    with open(file_path, 'a+') as file:
        file.seek(0, os.SEEK_END)
        file_position = file.tell()
        if file_position > 1:  # Check if not the first entry
            file.seek(file_position - 1)
            last_char = file.read(1)
            if last_char == ']':  # Remove the closing bracket before adding new data
                file.seek(file_position - 1)
                file.truncate()
            file.write(',\n')  # Add a comma before the next record
        json.dump(record, file, indent=2)
        file.write('\n]')  # Add the closing bracket back
    logging.info(f"Appended new record to JSON file: {file_path}")

def finalize(file_path):
    """Finalize and close the JSON file by adding the closing bracket if not already present."""
    with open(file_path, 'a+') as file:
        file.seek(0, os.SEEK_END)
        file_position = file.tell()
        if file_position > 0:
            file.seek(file_position - 1)
            last_char = file.read(1)
            if last_char != ']':
                file.write(']')
    logging.info(f"Finalized and closed JSON file: {file_path}")

def handle_signal(signal_number, frame):
    """Handle signals to ensure proper finalization of files."""
    for gesture in GESTURES.values():
        finalize(f'{gesture}_rssi.json')
    sys.exit(0)

def capture_rssi(file_path, gesture_label):
    """Continuously capture RSSI data for the specified gesture with individual timestamps."""
    global stop_capture
    wifi_interface = CoreWLAN.CWWiFiClient.sharedWiFiClient().interface()
    logging.info(f"Starting continuous RSSI recording for gesture: {gesture_label}")

    try:
        while not stop_capture:
            networks, _ = wifi_interface.scanForNetworksWithName_error_(None, None)
            #print(f"Detected {len(networks)} networks.")  # Diagnostic print

            for network in networks:
                # Generate the timestamp right before each record creation
                time_string = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                
                record = {
                    'timestamp': time_string,
                    'ssid': network.ssid(),
                    'bssid': network.bssid(),
                    'rssi': network.rssiValue(),
                    'gesture': gesture_label
                }
                append_record(file_path, record)

                # Brief sleep to ensure timestamp difference between consecutive records
                time.sleep(0.01)

            # Sleep briefly to avoid overwhelming the CPU
            time.sleep(0.5)

    except Exception as e:
        logging.error(f"Error during Wi-Fi scan: {e}")

def start_capture(gesture_label):
    """Start the capture_rssi function in a new thread."""
    global stop_capture, capture_thread
    stop_capture = False
    file_path = f'data/{gesture_label}_rssi.json'

    # Initialize the file if it hasn't been initialized yet
    init(file_path)

    # Start the capture in a separate thread
    capture_thread = threading.Thread(target=capture_rssi, args=(file_path, gesture_label))
    capture_thread.start()

def stop_capture_process():
    """Stop the capture process."""
    global stop_capture, capture_thread
    stop_capture = True
    if capture_thread:
        capture_thread.join()

def on_press(key):
    """Handle key press events."""
    try:
        if key.char in GESTURES:
            gesture_label = GESTURES[key.char]
            print(f"\nCapturing {gesture_label} gesture...")
            start_capture(gesture_label)

        elif key.char == 'q':
            print("Stopping capture...")
            stop_capture_process()
            for gesture in GESTURES.values():
                finalize(f'{gesture}_rssi.json')
            return False  # Stop listener

    except AttributeError:
        pass

if __name__ == '__main__':
    # Handle signals to ensure files are properly closed
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print("Press 1 for swipe, 2 for push-pull, 3 for circular (clockwise). Press 'q' to quit.")

    # Start the keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
