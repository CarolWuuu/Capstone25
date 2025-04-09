import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, correlate, hilbert, savgol_filter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# === CONFIG ===
SOUND_SPEED = 162900  # cm/s
REF_IDX = 0
WINDOW_SIZE = 35

file_path = "filtered_piezo_voltage_log-4&4-t3.csv"
true_freqs = [1.833, 1.833]

# Sensor coordinates (centimeters)
sensor_coords = np.array([
    [9.0, 8.5],
    [11.7, 15.5],
    [3.5, 2.5],
    [2.5, 11.5],
    [5.0, 18.5],
])

speaker_coords = np.array([
    [7.0, 16.0],
    [7.0, 6.0],
])

def estimate_tdoas_crosscorr(signals, peak_indices, fs, ref_idx=0, window_size=35):
    num_sensors = signals.shape[1]
    tdoa_matrix = []

    for idx in peak_indices:
        ref_seg = signals[max(0, idx - window_size):min(len(signals), idx + window_size), ref_idx]
        row = []
        for sid in range(num_sensors):
            if sid == ref_idx:
                row.append(0.0)
                continue
            seg = signals[max(0, idx - window_size):min(len(signals), idx + window_size), sid]
            min_len = min(len(ref_seg), len(seg))
            corr = correlate(seg[:min_len], ref_seg[:min_len], mode='full')
            lag = np.argmax(corr) - (min_len - 1)
            row.append(lag / fs)
        tdoa_matrix.append(row)

    return np.mean(tdoa_matrix, axis=0)


def reconstruct_source_from_tdoa(signals, tdoas, fs, normalize=True):
    num_samples, num_sensors = signals.shape
    max_shift = int(np.max(np.abs(tdoas)) * fs) + 1
    aligned = np.zeros((num_samples + max_shift, num_sensors))

    for i in range(num_sensors):
        delay_samples = int(round(tdoas[i] * fs))
        signal = signals[:, i]
        envelope = np.abs(hilbert(signal))
        signal = signal * envelope

        if delay_samples >= 0:
            aligned[delay_samples:delay_samples + num_samples, i] = signal
        else:
            aligned[0:num_samples + delay_samples, i] = signal[-delay_samples:]

    summed = np.sum(aligned, axis=1)
    if normalize:
        nonzero = np.sum(aligned != 0, axis=1)
        nonzero[nonzero == 0] = 1
        return summed / nonzero
    else:
        return summed


def grid_search_localization(sensor_coords, tdoas, ref_index=0, sound_speed=162900, grid_size=100):
    xmin, ymin = [2, 2]
    xmax, ymax = [20, 12]

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size)
    )

    error_grid = np.zeros_like(grid_x)
    ref_pos = sensor_coords[ref_index]
    ref_dist = np.sqrt((grid_x - ref_pos[0]) ** 2 + (grid_y - ref_pos[1]) ** 2)

    for i, (x_s, y_s) in enumerate(sensor_coords):
        if i == ref_index:
            continue
        dist = np.sqrt((grid_x - x_s) ** 2 + (grid_y - y_s) ** 2)
        expected_tdoa = (dist - ref_dist) / sound_speed
        error_grid += (tdoas[i] - expected_tdoa) ** 2

    min_idx = np.unravel_index(np.argmin(error_grid), error_grid.shape)
    est_x = grid_x[min_idx]
    est_y = grid_y[min_idx]
    return (est_x, est_y), error_grid, grid_x, grid_y



def main():
    df = pd.read_csv(file_path)
    time = df['Timestamp_s'].values
    fs = 1 / np.mean(np.diff(time))
    sensor_cols = [col for col in df.columns if 'Piezo' in col]
    signals = df[sensor_cols].values
    num_sensors = len(sensor_cols)

    # Only use data after 5 seconds
    valid_mask = time >= 7
    time = time[valid_mask]
    signals = signals[valid_mask, :]

    ref_signal = signals[:, REF_IDX]
    peak_indices, _ = find_peaks(ref_signal, height=0.3 * np.max(ref_signal))

    tdoa_features = []
    peak_records = []

    for idx in peak_indices:
        t_ref = time[idx]
        record = []
        peak_records.append((t_ref, REF_IDX, idx, ref_signal[idx]))

        for sid in range(num_sensors):
            if sid == REF_IDX:
                record.append(0.0)
                continue
            seg_ref = ref_signal[max(0, idx - WINDOW_SIZE):min(len(ref_signal), idx + WINDOW_SIZE)]
            seg_other = signals[max(0, idx - WINDOW_SIZE):min(len(signals), idx + WINDOW_SIZE), sid]
            min_len = min(len(seg_ref), len(seg_other))
            corr = correlate(seg_other[:min_len], seg_ref[:min_len], mode='full')
            lag = np.argmax(corr) - (min_len - 1)
            record.append(lag / fs)

        tdoa_features.append(record)

    tdoa_features = np.array(tdoa_features)

    scaler = StandardScaler()
    tdoa_scaled = scaler.fit_transform(tdoa_features)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(tdoa_scaled)
    labels = gmm.predict(tdoa_scaled)

    cluster_signals = {}
    for cid in [0, 1]:
        cluster_idx = np.where(labels == cid)[0]
        if len(cluster_idx) == 0:
            continue
        cluster_peaks = [peak_indices[i] for i in cluster_idx]
        cluster_tdoas = estimate_tdoas_crosscorr(signals, cluster_peaks, fs, ref_idx=REF_IDX, window_size=WINDOW_SIZE)
        tdoas = list(cluster_tdoas)
        cluster_signal = reconstruct_source_from_tdoa(signals, tdoas, fs)
        cluster_signals[cid] = cluster_signal[:len(time)]

    plt.figure(figsize=(12, 6))
    for cid, sig in cluster_signals.items():
        plt.plot(time, sig, label=f'Cluster {cid}')
    plt.title("Reconstructed Source Signals (TDOA Clustering)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    estimated_freqs = {}
    for label in [0, 1]:
        avg_signal = cluster_signals[label]
        fft_mag = np.abs(np.fft.rfft(avg_signal))
        freqs = np.fft.rfftfreq(len(avg_signal), d=1 / fs)

        envelope = np.abs(hilbert(fft_mag))
        smooth_envelope = savgol_filter(envelope, window_length=31, polyorder=3)

        # Find all peaks
        peak_indices, properties = find_peaks(smooth_envelope, height=0.6 * np.max(smooth_envelope))

        # Define buffer region to ignore edge peaks (e.g., 5% of the signal length)
        buffer = int(0.05 * len(smooth_envelope))

        # Keep only peaks that are not too close to the start or end
        valid_peak_indices = peak_indices[(peak_indices > buffer) & (peak_indices < len(smooth_envelope) - buffer)]

        if len(peak_indices):
            dominant_freq = np.median(freqs[valid_peak_indices])
        else:
            dominant_freq = 0.0
        estimated_freqs[label] = dominant_freq

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, fft_mag, label="FFT Magnitude")
        plt.plot(freqs, smooth_envelope, label="Smoothed Envelope", linestyle='--')
        if dominant_freq:
            plt.axvline(dominant_freq, color='r', linestyle='--', label=f'Median Peak: {dominant_freq:.2f} Hz')
        plt.title(f"Cluster {label} Spectrum + Envelope")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # === Reorder clusters so Cluster 0 is always the lower frequency ===
    if estimated_freqs[0] > estimated_freqs[1]:
        print("Swapping cluster labels to ensure cluster 0 is lower frequency...")
        # Swap frequencies
        estimated_freqs[0], estimated_freqs[1] = estimated_freqs[1], estimated_freqs[0]
        # Swap signals
        cluster_signals[0], cluster_signals[1] = cluster_signals[1], cluster_signals[0]
        # Swap labels
        labels = 1 - labels  # Invert 0 <-> 1


    print("\n=== Frequency Estimation Accuracy ===")
    for label in [0, 1]:
        est_freq = estimated_freqs[label]
        true_freq = true_freqs[label]
        percent_error = abs(est_freq - true_freq) / true_freq * 100 if true_freq > 0 else np.nan
        print(f"Cluster {label}:")
        print(f"  Estimated Frequency: {est_freq:.3f} Hz")
        print(f"  Closest True Frequency: {true_freq:.3f} Hz")
        print(f"  Percent Error: {percent_error:.2f}%\n")

    print("\n=== Location Estimation ===")
    for cid in [0, 1]:
        cluster_idx = np.where(labels == cid)[0]
        print(f"Cluster {cid} assigned {len(cluster_idx)} peaks.")

        if len(cluster_idx) == 0:
            print(f"⚠️ Cluster {cid} is empty. Skipping location estimation.")
            continue

        cluster_peaks = [peak_records[i][2] for i in cluster_idx]
        cluster_tdoas = estimate_tdoas_crosscorr(signals, cluster_peaks, fs, ref_idx=REF_IDX, window_size=WINDOW_SIZE)

        # Build full-length TDOA array with reference
        tdoas = [0.0] * signals.shape[1]
        j = 0
        for i in range(signals.shape[1]):
            if i == REF_IDX:
                continue
            tdoas[i] = cluster_tdoas[j]
            j += 1

        location, error_grid, grid_x, grid_y = grid_search_localization(
            sensor_coords, tdoas, ref_index=REF_IDX, sound_speed=SOUND_SPEED, grid_size=100
        )
        print(f"Estimated source location for cluster {cid}: {location}")


if __name__ == "__main__":
    main()
