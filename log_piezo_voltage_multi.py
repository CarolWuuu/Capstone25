import serial
import csv
import time

# === CONFIGURATION ===
PORT ="/dev/cu.usbmodem1101"
BAUD_RATE = 9600
VREF = 5.0
NUM_SENSORS = 5
OUTPUT_FILE = "piezo_voltage_log-1&4-t3.csv"

# === SETUP SERIAL AND FILE ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Give Arduino time to reset

with open(OUTPUT_FILE, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    headers = [f"Piezo_{i+1}_V" for i in range(NUM_SENSORS)] + ["Timestamp_s"]
    writer.writerow(headers)

    print("Logging data... Press Ctrl+C to stop.")

    try:
        while True:
            line = ser.readline().decode().strip()
            if line:
                parts = line.split(",")
                if len(parts) == NUM_SENSORS + 1:
                    try:
                        analog_values = [int(parts[i]) for i in range(NUM_SENSORS)]
                        timestamp_us = int(parts[NUM_SENSORS])

                        # Convert to voltages
                        voltages = [round(val * VREF / 1023.0, 4) for val in analog_values]

                        # Convert timestamp to seconds
                        timestamp_s = timestamp_us / 1_000_000  # Conversion from microseconds to seconds

                        # Write to CSV
                        writer.writerow(voltages + [timestamp_s])

                        # Print to terminal
                        voltage_str = " | ".join([f"{v:.3f}V" for v in voltages])
                        print(f"{voltage_str} @ {timestamp_s:.6f}s")
                    except ValueError:
                        print(f"Invalid data format: {line}")
                else:
                    print(f"Unexpected data format: {line}")
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        ser.close()
        print("Serial port closed.")