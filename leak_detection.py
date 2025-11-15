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

class LeakDetection:

    def __init__ (self):
        pass

    # PSD, in the end, is just the frequency components of the signal, squared, divided by the signal duration (to normalize). 
    # This function takes in the windowing function that the user wants to utilize.
    def power_spectrum_analysis (self, frequencies, amplitudes, windowing_function):
        pass
        # When the input a is a time-domain signal and A = fft(a), np.abs(A) is its amplitude spectrum and np.abs(A)**2 is its power   spectrum. The phase spectrum is obtained by np.angle(A).
        

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
        print ("The spectral rolloff is at " + str(spectral_rolloff) + " Hz.")

    
    # Computes and prints the spectral flatness. A value approaching 1 indicates that the spectrum has a similar amount of power in all frequencies, similar to white noise. A value approaching 0 would indicate that the spectral power is concentrated in a relatively small number of bands (like a mixture of sine waves)?
    ''' Note to self: If you decide to pass in a PSD array instead you can avoid squaring '''
    def compute_spectral_flatness (self, fft_frequencies, fft_amplitudes):
        spectrum = abs(fft_amplitudes)
        spectral_flatness = gmean(spectrum^2) / mean(spectrum^2)
        print ("The spectral flatness is " + str(spectral_flatness))