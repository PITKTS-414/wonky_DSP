# This Python class aims to answer the question : how can we ascertain that the sound we are looking at is actually a leak sound?
'''To answer this question, let is look at ...
        1) The power spectrum, to see which frequencies carry more energy.
            - Associated with the PSD, we can look at the spectral roll off (the frequency below a specific percentage p (let's let p = 0.85!) of the total spectral energy lies...
            - Associated with the PSD, we can look at the spectral flatness (how tonal or how noise like is the signal)?
'''

# Import some essential libraries. :D
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import pandas as pd
from scipy.stats import gmean
from filtering import ObtainData
from filtering import BasicPlottingFiltering

class LeakDetection:

    def __init__ (self):
        indexes = np.arange(1, 111)
        # This will eventually store a bunch of (nontrivial) data!
        self.dataframe = pd.DataFrame(index = indexes)

    ''' PSD, in the end, is just the frequency components of the signal, squared, divided by the signal duration (to normalize). 
    # When the input a is a time-domain signal and A = fft(a), np.abs(A) is its amplitude spectrum and np.abs(A)**2 is its power   spectrum. The phase spectrum is obtained by np.angle(A). '''
    # Gives the user the option of whether or not they want the analysis to be performed on the filtered data or not.
    def power_spectrum_analysis (self, test_number, filtered = False):
        plotter_1 = BasicPlottingFiltering(test_number)
        spectral_rolloff = 0
        spectral_flatness = 0
        if (filtered == False) :
            # print(f"For Test Number {str(test_number)} for Raw Data ->")
            for i in range(5) : 
                device_name = f"raw_data_{plotter_1.devices[i]}"
                device_data = getattr(plotter_1, device_name)
                frequencies, amplitude = plotter_1.time_to_frequency(fs = 1994, data_array = device_data) 
                power_spectrum = amplitude ** 2
                # print(f"For Device {str(plotter_1.devices[i])}:")
                spectral_rolloff = self.compute_spectral_rolloff(frequencies, power_spectrum)
                # print ("The spectral rolloff is at " + str(spectral_rolloff) + " Hz.")
                spectral_flatness = self.compute_spectral_flatness(amplitude)
                # print ("The spectral flatness is " + str(spectral_flatness))
        else :
            # print(f"For Test Number {str(test_number)} for Filtered Data ->")
            for i in range(5) :
                device_name = f"raw_data_{plotter_1.devices[i]}"
                device_data = getattr(plotter_1, device_name)
                # Setting the cutoffs to be 10 and 200, respectively (for now)
                frequencies, amplitude = plotter_1.bandpass_FIR_filter_no_plotting(5, 50, device_data, time = False)
                power_spectrum = amplitude ** 2
                # print(f"For Device {str(plotter_1.devices[i])}:")
                spectral_rolloff = self.compute_spectral_rolloff(frequencies, power_spectrum)
                # print ("The spectral rolloff is at " + str(spectral_rolloff) + " Hz.")
                spectral_flatness = self.compute_spectral_flatness(amplitude)
                # print ("The spectral flatness is " + str(spectral_flatness))
        # print()
        return spectral_rolloff, spectral_flatness
                
        

    # Computes and prints the spectral rolloff. Takes in an array of frequencies and an array of powers (PSD).
    def compute_spectral_rolloff (self, frequencies, psd):
        p = 0.85 # p value (manually set!)
        # cumsum produces a new array where each element is the running sum up to that index. 
        cumulative_sum = np.cumsum(psd)
        # [-1] accesses the last element of the array.
        total_power = cumulative_sum[-1]
        power_to_stop = p * total_power
        length = len(cumulative_sum)
        rolloff_index = 0
        # Utilize argmax to determine which the first index in which the value in cumulative_sum is >= power_to_stop.
        rolloff_index = np.argmax(cumulative_sum >= power_to_stop)
        spectral_rolloff = frequencies[rolloff_index]
        return spectral_rolloff
    
    # Computes and prints the spectral flatness. A value approaching 1 indicates that the spectrum has a similar amount of power in all frequencies, similar to white noise. A value approaching 0 would indicate that the spectral power is concentrated in a relatively small number of bands (like a mixture of sine waves)?
    ''' Note to self: If you decide to pass in a PSD array instead you can avoid squaring '''
    def compute_spectral_flatness (self, fft_amplitudes):
        spectrum = abs(fft_amplitudes)
        spectral_flatness = gmean(spectrum ** 2) / np.mean(spectrum ** 2)
        return spectral_flatness

    ''' This function will combine all the metrics that can be utilized for leak detection into one large DataFrame. '''
    def combine_into_dataframe (self):
        # Create some empty lists that will have values appended to them, and then assign the lists to be the columns of the DF.
        # This is faster than utilizing a method like .at for pandas.
        raw_rolloff = []
        raw_flatness = []
        filtered_rolloff = []
        filtered_flatness = []
        
        for i in range(110):    # Loop through all 110 tests...
            spectral_rolloff, spectral_flatness = self.power_spectrum_analysis(i+1, filtered = False)
            spectral_rolloff_2, spectral_flatness_2 = self.power_spectrum_analysis(i+1, filtered = True)
            raw_rolloff.append(spectral_rolloff)
            raw_flatness.append(spectral_flatness)
            filtered_rolloff.append(spectral_rolloff_2)
            filtered_flatness.append(spectral_flatness_2)

        self.dataframe['Spectral Rolloff, Raw Data'] = raw_rolloff
        self.dataframe['Spectral Rolloff, Filtered Data'] = filtered_rolloff
        self.dataframe['Spectral Flatness, Raw Data'] = raw_flatness
        self.dataframe['Spectral Flatness, Filtered Data'] = filtered_flatness
        
        return self.dataframe


    ''' This function will return which tests are leaks and which are not.'''
    def detect_leak (self):
        pass