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

# replaces NaN instances with zero
raw_data = raw_data.fillna(0)

# isolates individual arrays from table
eda = raw_data.iloc[:, 0]
ppg = raw_data.iloc[:, 1]

# specifies desired sampling rate
fs = 100

# calculates downsampling factor
downsampling_factor_eda = 100/fs  # assuming sampling rate is 100 Hz
downsampling_factor_ppg = 100/fs  # assuming sampling rate is 100 Hz

# downsamples signals
eda = 1/eda[::int(downsampling_factor_eda)]  # reciprocal converts to siemens
ppg = ppg[::int(downsampling_factor_ppg)]

# offsets the original segments to minimize ripple
ppg = ppg - ppg[0]
eda = eda - eda[0]

# isolates tonic and phasic components of EDA using bandpass filter
tonic = butterworth_bandpass_filter(eda, 0.000000000000001, 0.05, 3)
phasic = butterworth_bandpass_filter(eda, 0.05, 1.5, 3)

# prepares time axis for plots
time_eda = np.linspace(0, len(eda)/fs, len(eda))
time_ppg = np.linspace(0, len(ppg)/fs, len(ppg))

# plots downsampled signal
plt.figure(figsize = (8,8))
ax1 = plt.subplot(4, 1, 1)
plt.title('PPG')
ax1.plot(time_ppg, ppg)
plt.xlabel('Seconds')
plt.ylabel('Volts')
plt.grid(True, alpha = 0.5)
ax2 = plt.subplot(4, 1, 2, sharex = ax1)
plt.title('Composite EDA (I = 1.0 A)')
ax2.plot(time_eda, eda)
plt.xlabel('Seconds')
plt.ylabel('Siemens')
#plt.legend()
plt.grid(True, alpha = 0.5)

ax3 = plt.subplot(4, 1, 3, sharex = ax1)
plt.title('SCL (<0.05 Hz)')  # tonic component
ax3.plot(time_eda, tonic)
plt.xlabel('Seconds')
plt.ylabel('Siemens')
#plt.legend()
plt.grid(True, alpha = 0.5)

ax4 = plt.subplot(4, 1, 4, sharex = ax1)
plt.title('SCR (0.05 Hz – 1.5 Hz)')  # phasic component
ax4.plot(time_eda, phasic)
plt.xlabel('Seconds')
plt.ylabel('Siemens')
#plt.legend()
plt.grid(True, alpha = 0.5)

plt.subplots_adjust(hspace = 0.75)  # spaces subplots
plt.show()

'''
# plots power spectral density
plt.figure(figsize = (8,8))
plt.magnitude_spectrum(eda, 100)
plt.title('Magnitude Spectrum of Unfiltered EDA')

# plots power spectral density
plt.figure(figsize = (8,8))
plt.psd(eda, 100)
plt.title('Power Spectral Density of Unfiltered EDA')
'''

#==============================================================================
#  Z-SCORES & PEAKS
#==============================================================================

# isolates low-frequency of PPG using bandpass filter


# calculates z-scores of signal
z_eda = stats.zscore(eda)
z_ppg = stats.zscore(ppg)

# finds peaks of z-scored PPG
z_peak_ppg, _ = signal.find_peaks(z_ppg, distance = 0.28 * fs, prominence = 1.5)

#==============================================================================
#  TEMPLATE GENERATION
#==============================================================================

# specifies template length in seconds
template_length = 0.5
PR_interval = template_length/2
RT_interval = template_length/2
#PR_interval = 0.26
#RT_interval = 0.37

# marks starting and ending indices for first 10 peaks (skipping first peak to allow buffer for cropping)
baseline_start = z_peak_ppg[1] - int(PR_interval*fs)
baseline_end = z_peak_ppg[10] + int(RT_interval*fs)

# crops z-scored signal around first 10 peaks
baseline = z_ppg[baseline_start : baseline_end + 1]

# creates time axis for baseline
baseline_time = np.linspace(0, len(baseline)/fs, len(baseline))

# offset baseline peak positions
baseline_peaks = z_peak_ppg[1:11] -  baseline_start

# plots baseline with peaks
plt.figure()
ax1 = plt.subplot(2, 2, 1)
plt.title('Baseline ECG')
ax1.plot(baseline_time, baseline)
ax1.plot(baseline_time[baseline_peaks], baseline[baseline_peaks], "x")  # mark peaks, no offset needed after filtering
plt.xlabel('Time (s)')
plt.ylabel('Z-score (?)')
plt.grid(True, alpha = 0.5)

# crops waveforms around R-peaks, adds them to sample list, and outlines the chosen samples
ax2 = plt.subplot(2, 2, 2, sharex = ax1)
plt.title('Cropped Samples')
template_list = []
for R_index in baseline_peaks:
    sample_start = R_index - int(PR_interval*fs) 
    sample_end = R_index + int(RT_interval*fs)
    template_sample = baseline[sample_start : sample_end]
    ax2.plot(baseline_time[sample_start : sample_end], template_sample)
    template_list.append(template_sample)
plt.xlabel('Time (s)')
plt.ylabel('Z-score (?)')
plt.grid(True, alpha = 0.5)

# plots superimposed samples
ax3 = plt.subplot(2, 2, 3)
plt.title('Superimposed Sample List')
for sample in template_list:
   ax3.plot(sample)
plt.xlabel('Sample Index')
plt.ylabel('Z-score (?)')
plt.grid(True, alpha = 0.5)

# converts the sample list to a table
template_table = np.stack(template_list)

# finds the average waveform of the baseline samples to generate a template
template = np.mean(template_table, axis = 0)

# prepares time axis for plotting template
template_time = np.linspace(0, len(template)/fs, len(template))

# plots template
ax4 = plt.subplot(2, 2, 4)
plt.title('Template for Matched Filter')
ax4.plot(template_time, template)
plt.xlabel('Time (s)')
plt.ylabel('Z-score (?)')
plt.grid(True, alpha = 0.5)
plt.subplots_adjust(wspace = 0.5, hspace = 0.75)

#==============================================================================
#  MATCHED FILTER
#==============================================================================

# passes signal through matched filter
matched_z = matched_filter(z_ppg, template)

#==============================================================================
# FIND PEAKS & PLOT SIGNALS
#==============================================================================

# generates peak positions (min. length of cardiac cycle = 0.28 s ~ 170 bpm) 
matched_peak_pos, properties = signal.find_peaks(matched_z, distance = 0.28 * fs, height = 100)

# plots z-scored signal with peaks
plt.figure()  
ax7 = plt.subplot(2, 1, 1)
ax7.plot(time_ppg, z_ppg)
ax7.plot(time_ppg[z_peak_ppg], z_ppg[z_peak_ppg], "x")  # mark peaks, no offset needed after filtering
plt.ylabel('Z-score (?)')
plt.grid(True, alpha = 0.5)

# plots matched filtered signal with peaks
ax8 = plt.subplot(2, 1, 2, sharex = ax7)
ax8.plot(time_ppg, matched_z)
ax8.plot(time_ppg[matched_peak_pos], matched_z[matched_peak_pos], "x")  # mark peaks, no offset needed after filtering
plt.ylabel('? (?)')
plt.grid(True, alpha = 0.5)
plt.axhline(y = 100, alpha = 0.5)

# adjusts and labels subplots
plt.xlabel('Time (s)')
plt.subplots_adjust(hspace = 0.75)
plt.show()


#==============================================================================
# IBI LIST
#==============================================================================

# initialize empty list for IBIs
ibi_list_attys = []

# generate list of IBIs for cropped signal
cnt = 0
while (cnt < len(matched_peak_pos) - 1):
    interval = (matched_peak_pos[cnt + 1] - matched_peak_pos[cnt])/fs
    ibi_list_attys.append(interval)
    cnt += 1
    
# export list to csv file
with open ('IBIs', 'w', newline='') as myfile:
    thewriter = csv.writer(myfile)
    thewriter.writerow(ibi_list_attys)

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

EDA max freq 35 Hz

60 s baseline session. 
'''