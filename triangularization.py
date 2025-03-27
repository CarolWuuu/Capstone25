import numpy as np
from scipy.signal import correlate

SPEED_OF_SOUND = 1458  # in meters per second


def calculate_tdoa(signal1, signal2, sampling_rate):
    correlation = correlate(signal1, signal2, mode='full')
    lag = np.argmax(correlation) - (len(signal1) - 1)
    time_delay = lag / sampling_rate
    return time_delay


def time_delay_to_distance(time_delay):
    return time_delay * SPEED_OF_SOUND
