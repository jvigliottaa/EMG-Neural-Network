import numpy as np
import math

#Common constants used throughout program
WINDOW = 400
WINDOW_SAMPLE = int(WINDOW/20)
STEP = 40
STEP_SAMPLE = int(STEP/20)

CUTOFF = 5
SAMPLING_FREQ = 50
NYQ = 0.5 * SAMPLING_FREQ
BUTTER_WORTH_ORDER = 4

#Common methods used throughout program

def mav(signal):
    return abs(np.mean(signal))

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def ssc(signal, thres):
    SSC = 0
    for i in range(1, len(signal) - 1):
        if (((signal[i] > signal[i - 1] and signal[i] > signal[i + 1]) or
             (signal[i] < signal[i - 1] and signal[i] < signal[i + 1])) and
                (signal[i] - signal[i + 1]) >= thres or (signal[i] - signal[i - 1]) >= thres):
            SSC += 1
    return SSC

def wav(signal):
    return np.sum(np.diff(signal))

def ahp(signal):
    return (1/(len(signal)-1)) * np.sum(signal**2)

def mhp(signal):
    diff_signal = np.diff(signal)
    return math.sqrt(ahp(diff_signal) / ahp(signal))

def chp(signal):
    diff_signal = np.diff(signal)
    return mhp(diff_signal) / mhp(signal)





