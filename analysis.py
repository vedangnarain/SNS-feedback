"""
Created on Thu Oct  3 14:16:57 2019

@author: Vedang
"""

#==============================================================================
# LIBRARIES & INITIALIZATION
#==============================================================================

# imports libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
import time

# starts stopwatch for execution time
start_time = time.time()

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
raw_data = pd.read_csv('/Users/Vedang/Desktop/Wearable Sensors/coolterm_ppg_eda_24_oct.txt', header=None)

# replaces NaN instances with zero
raw_data = raw_data.fillna(0)

# isolates individual arrays from table
eda = raw_data.iloc[:, 0]
ppg = raw_data.iloc[:, 1]

# specifies desired sampling rate
fs = 100

# downsamples signals (assuming sampling rate is 100 Hz)
downsampling_factor_eda = 100/fs
downsampling_factor_ppg = 100/fs
eda = 1/eda[::int(downsampling_factor_eda)]  # reciprocal converts to siemens
ppg = ppg[::int(downsampling_factor_ppg)]

# offsets the original segments to minimize ripple
ppg = ppg - ppg[0]
eda = eda - eda[0]

# filters signals to remove noise
ppg = butterworth_bandpass_filter(ppg, 0.000000001, 2.5, 3)
eda = butterworth_bandpass_filter(eda, 0.000000001, 1.5, 3)

# prepares time axis for plots
time_eda = np.linspace(0, len(eda)/fs, len(eda))
time_ppg = np.linspace(0, len(ppg)/fs, len(ppg))

#==============================================================================
#  EDA POSITIVE CHANGE
#==============================================================================

# z-scores EDA
eda = stats.zscore(eda)

# plots EDA
plt.figure(figsize = (8,8))
ax1 = plt.subplot(6, 1, 1)
plt.title('EDA (I = 1.0 A)')
ax1.plot(time_eda, eda)
#plt.xlabel('Seconds')
plt.ylabel('z-score')
plt.grid(True, alpha = 0.5)

# specifies window length (10 seconds)
epc_window_length = int(10*fs) #1000

# calculates starting index to leave space for 10-second buffer
epc_start_index = epc_window_length - 1 #999

# prepares list of EDA increases
eda_increases_list = [0]
for i in range(1, len(eda)):
    if eda[i] > eda[i-1]:
        eda_increase = eda[i] - eda[i-1]
        eda_increases_list.append(eda_increase)
    else:
        eda_increases_list.append(0)

# prepares list of 10-second sums
epc_list = [0] * (epc_start_index)  # intializes zeros for initial 10-second buffer period
epc_list.append(sum(eda_increases_list[0:epc_window_length]))  # initializes first 10-second EPC by summing first 1000 values

# generates the rest of the EPCs more efficiently
for i in range(epc_window_length, len(eda_increases_list)):
    epc_value = (epc_list[i - 1] - eda_increases_list[i - epc_window_length] + eda_increases_list[i])
    epc_list.append(epc_value)

# plots EPC    
ax2 = plt.subplot(6, 1, 2, sharex = ax1)
plt.title('EPC')  # epc
ax2.plot(time_eda, epc_list)
#plt.xlabel('Seconds')
plt.ylabel('z-score')
plt.grid(True, alpha = 0.5)

#==============================================================================
#  IBI EXTRACTION
#==============================================================================

# z-scores PPG
ppg = stats.zscore(ppg)

# plots PPG
ax3 = plt.subplot(6, 1, 3, sharex = ax1)
plt.title('PPG')
ax3.plot(time_ppg, ppg)
#plt.xlabel('Seconds')
plt.ylabel('z-score')
plt.grid(True, alpha = 0.5)

# generates PPG gradient signal
dz_ppg = np.gradient(ppg)

# finds peaks of gradient (min. length of cardiac cycle = 0.28 s ~ 170 bpm) 
dz_peaks, _ = signal.find_peaks(dz_ppg, distance = 0.28 * fs, prominence = 0.1)

# plots PPG gradient with peaks
ax4 = plt.subplot(6, 1, 4, sharex = ax1)
plt.title('PPG Gradient (with peaks)')
ax4.plot(time_ppg, dz_ppg)
ax4.plot(time_ppg[dz_peaks], dz_ppg[dz_peaks], "x")  # mark peaks, no offset needed after filtering
plt.ylabel('d(z-score)/dt')
plt.grid(True, alpha = 0.5)

'''
SECTION BELOW STILL NEEDS WORK
'''

# initialize empty list for IBIs
ibi_list = [0]

# generate list of IBIs for cropped signal
cnt = 0

for i in range(1, len(dz_peaks)):
    interval = (dz_peaks[i] - dz_peaks[i-1])/fs
    if interval < 1:  # if IBI is greater than or equal to 1 second, use previous value instead
        ibi_list.append(interval)
    else:
        ibi_list.append(ibi_list[i-1])  

# generates time axis for IBIs
ibi_time_axis = np.cumsum(ibi_list)

# plots IBIs    
ax5 = plt.subplot(6, 1, 5, sharex = ax1)
plt.title('IBIs')
ax5.plot(ibi_time_axis, ibi_list)
plt.ylabel('seconds')
#plt.legend()
plt.grid(True, alpha = 0.5)

# adjusts and labels subplots
plt.xlabel('seconds')
plt.subplots_adjust(hspace = 1)
plt.show()


'''
NEXT STEPS: cumulative IBIs falls short of end because it makes up values. also, it ignores first peak time. fix.
take window of 10-second IBI change (decreases instead of increases)

see if IBI calibration (and score) is okay (try on more samples)

'''

#==============================================================================
#  WEIGHTED-SCORES
#==============================================================================

'''
scores = []

loop
score = (0.7*epc) + (0.3*ibi_changes)  # x/max * 5
append (score)
'''

# prints execution time
print("\n--- Execution Time: %s seconds ---" % (time.time() - start_time))

# =============================================================================
# NOTES
# =============================================================================

'''

 The EDA complex includes both
background tonic (skin conductance level: SCL) and rapid phasic components (Skin
Conductance Responses: SCRs) that result from sympathetic neuronal activity. EDA is
arguably the most useful index of changes in sympathetic arousal that are tractable to
emotional and cognitive states as it is the only autonomic psychophysiological variable that is
not contaminated by parasympathetic activity. EDA has been closely linked to autonomic
emotional and cognitive processing, and EDA is a widely used as a sensitive index of
emotional processing and sympathetic activity.

60 s baseline session. 

'''