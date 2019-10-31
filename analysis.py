"""
Created on Thu Oct  3 14:16:57 2019

@author: Vedang
"""

#==============================================================================
# LIBRARIES
#==============================================================================

# imports libraries
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats, fft, arange

#==============================================================================
# FUNCTIONS
#==============================================================================

# defines function to obtain Butterworth LPF coefficients
def butterworth_bandpass(low_cut, high_cut, order):
    nyq_freq = 0.5 * fs    
    normalized_low = low_cut / nyq_freq
    normalized_high = high_cut / nyq_freq        
    b, a = signal.butter(order, [normalized_low, normalized_high], btype = 'band')
    return b, a

# defines function to pass signal through Buterworth LPF
def butterworth_bandpass_filter(unfiltered_signal, low_cut, high_cut, order):
    b, a = butterworth_bandpass(low_cut, high_cut, order)
    filtered_signal = signal.lfilter(b, a, unfiltered_signal)
    return filtered_signal

# defines function to pass signal through matched filter
def matched_filter(unfiltered_signal, template):
    fir_coeff = template[::-1]
    detected = signal.lfilter(fir_coeff, 1, unfiltered_signal)
    detected = detected * detected  # squaring to improve SNR
    return detected

#==============================================================================
# IMPORT AND PREPARE DATA
#==============================================================================

# imports raw data
#raw_data = pd.read_csv('/Users/Vedang/Desktop/Subject1_EDA_PPG DATA_Wearable sensors.txt',sep='\s+',header=None)
raw_data = pd.read_csv('/Users/Vedang/Desktop/Wearable Sensors/coolterm_ppg_eda_24_oct.txt', header=None)

# isolates individual arrays from table
eda = raw_data.iloc[:, 0]
ppg = raw_data.iloc[:, 1]

# specifies desired sampling rate
fs = 50

# calculates downsampling factor
downsampling_factor = 250/fs  # assuming sampling rate is 250 Hz

# downsamples signals
eda = 1/eda[::int(downsampling_factor)]  # reciprocal converts to siemens
ppg = ppg[::int(downsampling_factor)]

# plots downsampled signal
plt.figure()
ax1 = plt.subplot(2, 1, 1)
plt.title('EDA (I = 1.0 A)')
ax1.plot(eda)
plt.xlabel('Samples')
plt.ylabel('Siemens')
plt.grid(True, alpha = 0.5)
ax2 = plt.subplot(2, 1, 2, sharex = ax1)
plt.title('PPG')
ax2.plot(ppg)
plt.xlabel('Samples')
plt.ylabel('Volts')
#plt.legend()
plt.grid(True, alpha = 0.5)
plt.subplots_adjust(hspace = 0.75)  # spaces subplots
plt.show()
