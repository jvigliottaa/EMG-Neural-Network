import numpy as np
from scipy.signal import butter, filtfilt
import scipy.fftpack
import matplotlib.pyplot as plt
import common
from statistics import mode
import math
import csv

"""
Preprocessing of Data for each channel:
    1) Normalize [-1,1]
    2) Absolute Value [0,1]
    3) Fourier Transform for find cutoff frequency
    4) 4th Order Butterworth filter with cutoff frequency
"""

def data_formatter(NUM_DBs):
    # Create one database from multiple databases.. Currently 2 DB
    data = []
    for i in range(1, NUM_DBs+1):
        file_name = "Data/db_{}.csv".format(i)
        with open(file_name, newline='') as f:
            reader = csv.reader(f)
            data.extend(list(reader))

    #Create list of window sized data
    list_df = [data[i:i+common.WINDOW_SAMPLE] for i in range(0, len(data), common.STEP_SAMPLE)]

    #Tranpose array so each channel has its own list
    data_arranged = []
    for data_group in list_df:
        data_arranged.append(np.array(data_group).T.tolist())
    return np.array(data_arranged)

def preprocessing_and_feature_extraction(data_np, nn_file_name,):
    data_processed = []
    i=len(data_np) * 8
    for data in data_np[:-100]:
        sensor_group = []
        for sensor in data[1:-1]:
            #Change all to floats
            sensor = np.array([float(i) for i in sensor])

            #Optional status for data conversion
            print(i)
            i-=1

            # Normalized [-1,1]
            normalized = 2*((sensor - np.min(sensor)) / (np.max(sensor) - np.min(sensor))) -1
            abs_norm = np.abs(normalized)
            plt.plot(np.linspace(0, len(abs_norm), len(abs_norm)), abs_norm)

            #Butterworth filter
            normal_cutoff = common.CUTOFF / common.NYQ
            b, a = butter(5, normal_cutoff, btype='low', analog=False)
            process_sig = filtfilt(b, a, abs_norm)

            # #Plot Any Data
            # plt.plot(np.linspace(0, len(process_sig), len(process_sig)), process_sig)

            sensor_group.append(common.mav(process_sig))
            sensor_group.append(common.rms(process_sig))
            sensor_group.append(common.ssc(process_sig, 0.15))
            sensor_group.append(common.wav(process_sig))
            sensor_group.append(common.ahp(process_sig))
            sensor_group.append(common.mhp(process_sig))
            sensor_group.append(common.chp(process_sig))
            sensor_group.append(common.mav(sensor))
            sensor_group.append(common.rms(sensor))
            sensor_group.append(common.ssc(sensor, 0.15))
            sensor_group.append(common.wav(sensor))
            sensor_group.append(common.ahp(sensor))
            sensor_group.append(common.mhp(sensor))
            sensor_group.append(common.chp(sensor))

        data_processed.append((sensor_group, mode(data[-1])))

    print(data_processed)
    np.save(nn_file_name, np.asarray(data_processed))
