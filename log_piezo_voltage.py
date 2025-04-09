import serial
import csv
import time

# === CONFIGURATION ===
PORT ="/dev/cu.usbmodem101"
BAUD_RATE = 9600
OUTPUT_FILE = "piezo_voltage_log-1&3-t1.csv"

# === SETUP SERIAL AND FILE ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Give Arduino time to reset

with open(OUTPUT_FILE, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Time (s)", "Voltage (V)"])  # Header row

    print("Logging started. Press Ctrl+C to stop.\n")

    try:
        start_time = time.time()
        while True:
            line = ser.readline().decode().strip()
            if line:
                try:
                    voltage = float(line)
                    timestamp = round(time.time() - start_time, 3)
                    writer.writerow([timestamp, voltage])
                    print(f"{timestamp}s, {voltage:.3f} V")
                except ValueError:
                    print("Invalid data:", line)
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        ser.close()