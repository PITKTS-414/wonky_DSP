import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift

# This Python class utilizes Python pandas to return some basic information about a given (user-inputed) test number.
# Separates data into two categories: passive vs active excitation, leak vs no leak (simulated leak).
class ObtainData:

    def __init__ (self, test_number = None):
        self.test_number = test_number
        self.data = pd.read_csv("/home/jovyan/SRI_Lab/Data/Active_Test_Matrix.csv", index_col = 0, skiprows = [1,2,3,114,115,116], usecols = [0,1,3,4,5,9,10,14])
        self.device_ids = [198,199,200,201,203]

    # The following function returns an array with the following information ->
    # 5 Paths to the Data of Each Device (the first 5 elements of the array), Month, Day, Year, Hour (24-Hour Clock), Minute, Frequency (of Active Excitation Signal), V_RMS, I_RMS, Leak (Leak vs No Leak), Excitation (Active vs Passive), Color1, Color2, Color3
    # If the test_number is associated with passive excitation, Frequency, V_RMS, and I_RMS are set to 'None'
    # Color will be the color that the graph is when the data is plotted!
    def return_test_information (self):
        array = self.data.loc[(self.test_number)].to_numpy()
        month = array[0][0].zfill(2)
        day = array[0][2:4].zfill(2)
        year = array[0][5:9].zfill(4)
        hour = array[1][0:2].zfill(2)
        minute = array[1][3:5].zfill(2)
        final_array = np.empty(18, dtype='object')
        for i in range(5):
            final_array[i] = f"/home/jovyan/SRI_Lab/Data/deviceid{self.device_ids[i]}/{self.device_ids[i]}-{year}-{month}-{day}T{hour}-{minute}-00.npy"
        final_array[5] = month
        final_array[6] = day
        final_array[7] = year
        final_array[8] = hour
        final_array[9] = minute
        final_array[10] = array[2]
        final_array[11] = array[4]
        final_array[12] = array[5]
        if (array[3] == 'OFF') :
            final_array[14] = 'Passive'
            if (array[6] == 'N') :
                final_array[13] = 'No Leak'
                final_array[15] = 'royalblue'
                final_array[16] = 'dodgerblue'
                final_array[17] = 'steelblue'
            else :
                final_array[13] = 'Leak'
                final_array[15] = 'brown'
                final_array[16] = 'darkred'
                final_array[17] = 'crimson'

        else :
            final_array[14] = 'Active'
            if (array[6] == 'N') :
                final_array[13] = 'No Leak'
                final_array[15] = 'seagreen'
                final_array[16] = 'darkgreen'
                final_array[17] = 'forestgreen'
            else :
                final_array[13] = 'Leak'
                final_array[15] = 'darkviolet'
                final_array[16] = 'blueviolet'
                final_array[17] = 'rebeccapurple'
        return final_array
              

    ''' The following four functions will return arrays of the test_numbers that fulfill the specified conditions. '''
    def return_passive_no_leak (self):
        condition = (self.data['Excitation (ON/OFF)'] == 'OFF') & (self.data['Simulated Leak'] == 'N')
        filtered_data = self.data[condition]
        return (filtered_data.index)

    def return_passive_leak (self):
        condition = (self.data['Excitation (ON/OFF)'] == 'OFF') & (self.data['Simulated Leak'] == 'Y')
        filtered_data = self.data[condition]
        return (filtered_data.index)

    def return_active_no_leak (self):
        condition = (self.data['Excitation (ON/OFF)'] == 'ON') & (self.data['Simulated Leak'] == 'N')
        filtered_data = self.data[condition]
        return (filtered_data.index)

    def return_active_leak (self):
        condition = (self.data['Excitation (ON/OFF)'] == 'ON') & (self.data['Simulated Leak'] == 'Y')
        filtered_data = self.data[condition]
        return (filtered_data.index)


# This Python class aims to provide functions to plot the raw data in the time domain and frequency domain.
''' Provides functions that can transform from the time domain to the frequency domain and back. '''
class BasicPlotting:

    # Makes test_number an optional parameter. 
    def __init__ (self, test_number):
        self.test_number = test_number
        obtain_data_1 = ObtainData(self.test_number)
        test_information = obtain_data_1.return_test_information()
        # Utilize np.load to load in the data from the .npy files.
        self.raw_data_198 = np.load(test_information[0]) 
        self.raw_data_199 = np.load(test_information[1])
        self.raw_data_200 = np.load(test_information[2])
        self.raw_data_201 = np.load(test_information[3])
        self.raw_data_203 = np.load(test_information[4])
        self.month = test_information[5]
        self.day = test_information[6]
        self.year = test_information[7]
        self.hour = test_information[8]
        self.minute = test_information[9]
        self.frequency = test_information[10]
        self.v_rms = test_information[11]
        self.i_rms = test_information[12]    
        self.leak = test_information[13]
        self.excitation = test_information[14]
        self.color = test_information[15]
        self.color2 = test_information[16] 
        self.color3 = test_information[17] 
        self.devices = [198, 199, 200, 201, 203]
        self.fs = 1994

    ''' This function plots the raw data in the time domain for a singular test. '''
    def plot_raw_data_in_time (self):
        t = np.arange(9974) / self.fs
        # Create a figure and a set of subplots with 3 rows and 2 columns.
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        # Changes the 3 by 2 grid to a 1 by 6 grid (hence the name flatten)
        axs = axs.flatten()
        fig.suptitle(f"Raw Data: Acoustic Pressure (Pa) vs Time (s). Test #{self.test_number}, {self.month}/{self.day}/{self.year} {self.hour}:{self.minute}:00, {self.leak} and {self.excitation} Excitation", fontsize = 16)
        for i in range(5):
            device_name = f"raw_data_{self.devices[i]}"
            # Utilize getattr() to access the attribute with the given name 'device_name'.
            device_data = getattr(self, device_name)
            ax = axs[i]
            ax.plot(t, device_data, color = self.color, linewidth = 0.5)
            ax.set_title(f"Device {self.devices[i]}", fontsize = 12)
            ax.set_xlabel("Time (s), 5 Second Interval, Sampling Rate = 1994 Hz")
            ax.set_ylabel("Acoustic Pressure (Pa)")
            ax.grid()
        # I'm not using the 6th subplot space, so it can be hidden.
        axs[5].axis("off")
        plt.tight_layout()
        plt.show()

    def plot_raw_data_in_frequency (self):
        pass

    ''' This function converts the inputed data from the time domain to the frequency domain with scipy's fft. '''
    def time_to_frequency (self) :
        # T is the sampling interval. Since fs = 1994, the highest visible frequency will be 997 Hz (Nyquist freq) 
        T = 1 / self.fs 
        fig, axs = plt.subplots(3, 2, figsize = (12,10))
        axs = axs.flatten()
        fig.suptitle(f"Full FFT of Raw Data: Amplitude (Pa) vs Frequency (Hz). Test #{self.test_number}, {self.month}/{self.day}/{self.year} {self.hour}:{self.minute}:00, {self.leak} and {self.excitation} Excitation", fontsize = 16)
        for i in range(5) : 
            device_name = f"raw_data_{self.devices[i]}"
            # Utilize getattr() to access the attribute with the given name 'device_name'.
            device_data = getattr(self, device_name)
            N = len(device_data) 
            Y = fft(device_data) 
            freqs = fftfreq(N, T) 
            mask = freqs >= 0
            positive_freqs = freqs[mask]
            positive_Y = Y[mask]
            # FFT adds up N samples, thus the magnitude result will scale with N. Normalize your magnitude by dividing by N. 
            # Additionally, real valued signals are symmetric about the y-axis. 
            Y_mag = np.abs(positive_Y) / N 
            ax = axs[i]
            ax.plot(positive_freqs, Y_mag, color = self.color2, linewidth = 0.5)
            ax.set_title(f"Device {self.devices[i]}", fontsize = 12)
            ax.set_xlabel("Frequency (Hz), Sampling Rate = 1994 Hz")
            # FIGURE OUT WHAT THE AMPLITUDE MEANS!!!!
            ax.set_ylabel("Amplitude (Pa)")
            ax.grid()
        axs[5].axis("off")
        plt.tight_layout()
        plt.show()

    # Use to isolate the leak signal. 
    ''' From your DSP notes: odd number of taps for linear phase, setting cutoff manually, and the sample rate is predefined. 
    def bandpass_FIR_filter (self):
        num_taps = 101
        cut_off = 200
        sample_rate = 1994
        t = np.arange(9974)
        t = t / sample_rate 
        # Create our high pass filter by generating filter_taps with firwin. Automatically uses the windowing method.
        filter_taps = signal.firwin(num_taps, cut_off, fs = sample_rate, pass_zero = 'highpass')
        # Finally, convolve the taps with the input signal to get your filtered signal!
        data = self.device_data[0]
        filtered_signal = fftconvolve(filter_taps, data)
        plt.plot(t, filtered_signal, color = self.color)
        plt.show()
    '''


class BasicFilters:

    def __init__ (self):
        self.test = "slay"

    def print_text(self):
        print("what?!")
    