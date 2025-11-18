import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy import signal
from scipy.signal import fftconvolve

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
class BasicPlottingFiltering:

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

    ''' This is a general function that will convert some given time-domain data into the frequency domain. '''
    # Returns the x axis and y axis converted to the frequency domain.
    def time_to_frequency (self, fs, data_array):
        T = 1 / fs 
        N = len(data_array) 
        Y = fft(data_array) 
        freqs = fftfreq(N, T) 
        mask = freqs >= 0
        positive_freqs = freqs[mask]
        positive_Y = Y[mask]
        # FFT adds up N samples, thus the magnitude result will scale with N. Normalize your magnitude by dividing by N. 
        # Additionally, real valued signals are symmetric about the y-axis. 
        Y_mag = np.abs(positive_Y) / N 
        return positive_freqs, Y_mag

    ''' This function converts the device's raw data from the time domain to the frequency domain and plots the frequency response.'''
    # It utilizes the function time_to_frequency, which incorporates scipy's fft.
    def plot_raw_data_in_frequency (self) :
        fig, axs = plt.subplots(3, 2, figsize = (12,10))
        axs = axs.flatten()
        fig.suptitle(f"Full FFT of Raw Data: Amplitude (Pa) vs Frequency (Hz). Test #{self.test_number}, {self.month}/{self.day}/{self.year} {self.hour}:{self.minute}:00, {self.leak} and {self.excitation} Excitation", fontsize = 16)
        for i in range(5) : 
            device_name = f"raw_data_{self.devices[i]}"
            # Utilize getattr() to access the attribute with the given name 'device_name'.
            device_data = getattr(self, device_name)
            x_axis, y_axis = self.time_to_frequency(fs = 1994, data_array = device_data) 
            ax = axs[i]
            ax.plot(x_axis, y_axis, color = self.color2, linewidth = 0.5)
            ax.set_title(f"Device {self.devices[i]}", fontsize = 12)
            ax.set_xlabel("Frequency (Hz), Sampling Rate = 1994 Hz")
            # FIGURE OUT WHAT THE AMPLITUDE MEANS!!!!
            ax.set_ylabel("Amplitude (Pa)")
            ax.grid()
        axs[5].axis("off")
        plt.tight_layout()
        plt.show()


    
    ''' This function applies a FIR (finite impulse response) filter onto the data. 
    In other words, this function extracts the leak signature of the data (assuming that there is a leak). '''
    # Parameters include the specified low frequency cutoff and high frequency cutoff, the number of taps, and which plots will be plotted (options include the impulse response in both time and frequency and the filtered data in both time and frequency). Notice the default conditions set in the function. 
    # From your DSP notes: odd number of taps for linear phase, setting cutoff manually, and the sample rate is predefined. 
    def bandpass_FIR_filter (self, low_cutoff = 5, high_cutoff = 30, num_taps = 1001, plot_IR = False, plot_filtered = True):
        t = np.arange(9974)
        t = t / self.fs 
        # Create our bandpass filter by generating filter_taps with firwin. Automatically uses the windowing method.
        filter_taps = signal.firwin(num_taps, [low_cutoff, high_cutoff], fs = self.fs, pass_zero = 'bandpass')

        ''' Plot the FIR filter response in both the time domain and the frequency domain. '''
        if (plot_IR == True) :
            fig2, ax2 = plt.subplots(1, 2, figsize = (12,3))
            ax2 = ax2.flatten()
            fig2.suptitle(f"FIR Filter Bandpass Response. {str(num_taps)} Taps, {str(low_cutoff)} Hz - {str(high_cutoff)} Hz")
            ax2[0].plot(filter_taps, color = self.color3, linewidth = 1.5)
            ax2[0].set_title("Time Domain")
            ax2[0].set_xlabel("Tap Index")
            x_axis, y_axis = self.time_to_frequency(self.fs, filter_taps)
            ax2[1].plot(x_axis, y_axis, color = self.color3, linewidth = 1.5)
            limiting_value = high_cutoff + 50
            ax2[1].set_xlim(0, limiting_value)   # Limit the x_axis to only show a certain range of frequencies.
            ax2[1].set_title("Frequency Domain")
            ax2[1].set_xlabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()
            
        if (plot_filtered == True) :
            # Plot the bandpass FIR filter applied to the data in the time and frequency domain.
            fig, axs = plt.subplots(3, 2, figsize = (12,10))
            axs = axs.flatten()
            fig.suptitle(f"FIR Bandpass Filter, Time Domain. {str(num_taps)} Taps, {str(low_cutoff)} Hz - {str(high_cutoff)} Hz. Test #{self.test_number}, {self.month}/{self.day}/{self.year} {self.hour}:{self.minute}:00, {self.leak} and {self.excitation} Excitation", fontsize = 16)
            fig3, axs3 = plt.subplots(3, 2, figsize = (12,10))
            axs3 = axs3.flatten()
            fig3.suptitle(f"FIR Bandpass Filter, Frequency Domain. {str(num_taps)} Taps, {str(low_cutoff)} Hz - {str(high_cutoff)} Hz. Test #{self.test_number}, {self.month}/{self.day}/{self.year} {self.hour}:{self.minute}:00, {self.leak} and {self.excitation} Excitation", fontsize = 16)

            # Loop through the device data and plot each in the time and frequency domains.
            for i in range(5):
                device_name = f"raw_data_{self.devices[i]}"
                device_data = getattr(self, device_name)
                # Finally, convolve the taps with the input signal to get your filtered signal!
                filtered_signal = fftconvolve(device_data, filter_taps, mode = 'same')
                ax = axs[i]
                ax.plot(t, filtered_signal, color = self.color, linewidth = 0.5)
                ax.set_title(f"Device {self.devices[i]}", fontsize = 12)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Acoustic Pressure (Pa)")
                ax.grid()
                x_axis, y_axis = self.time_to_frequency(self.fs, filtered_signal)
                ax3 = axs3[i]
                ax3.plot(x_axis, y_axis, color = self.color2, linewidth = 0.5)
                limiting_value = high_cutoff + 50
                ax3.set_xlim(0, limiting_value)   # Limit the x_axis to only show a certain range of frequencies.
                ax3.set_title(f"Device {self.devices[i]}", fontsize = 12)
                ax3.set_xlabel("Frequency (Hz)")
                ax3.set_ylabel("Amplitude")
                ax3.grid()        
            axs[5].axis("off")
            axs3[5].axis("off")
            plt.tight_layout()
            plt.show()


    # This function applies a bandpass FIR filter onto the data, but doesn't do any plotting.
    # Will just return the x_axes and y_axes of the data in either time or frequency.
    def bandpass_FIR_filter_no_plotting (self, low_cutoff, high_cutoff, device_data, time = True):
        t = np.arange(9974)
        t = t / self.fs 
        filter_taps = signal.firwin(1001, [low_cutoff, high_cutoff], fs = self.fs, pass_zero = 'bandpass')
        filtered_signal = fftconvolve(device_data, filter_taps, mode = 'same')
        if (time == True): 
            return t, filtered_signal
        else :
            x_axis, y_axis = self.time_to_frequency(self.fs, filtered_signal)  
            return x_axis, y_axis


    ''' This function applies a IIR (infinite impulse response) filter onto the data. '''
    # Parameters include the order of the filter, low / high cutoffs, the filter response (butterworth or chebyshev), if the filter response should be plotted, and if the filtered data (both in time and frequency domain) should be plotted.
    def bandpass_IIR_filter (self, order = 4, low_cutoff = 5, high_cutoff = 30, chebyshev = False, plot_IR = False, plot_filtered = True):

        filtertype = 'butter'
        if (chebyshev == 'True') :
            filtertype = 'cheby1'
        
        # Create our bandpass filter by generating the transfer function of the filter - with the output as 'ba', this function will return the numerator and denominator of the transfer function, respectively.
        # Since we have a digital filter, the Wn must be between 0 and 1, inclusive. Thus, we normalize. 
        nyquist_frequency = self.fs / 2
        low_worN = low_cutoff / nyquist_frequency
        high_worN = high_cutoff / nyquist_frequency
        numerator, denominator = signal.iirfilter(order, [low_worN, high_worN], btype = 'band', ftype = filtertype, output = 'ba', analog = False)
        # Now, compute the filter's frequency response with freqz (for digital filters), which inputs the num and denom of the transfer function. 
        x_axis, y_axis = signal.freqz(numerator, denominator, worN = 3000)
        # 20 * np.log10(abs(y_axis)) is another option if I want my graphs to be dB scaled... 
        x_axis_Hz = x_axis * self.fs / (2 * np.pi)  # convert rad/sample to Hz (freqz will return in rad :( )

        if (plot_IR == True):
            fig, ax = plt.subplots(1,2,figsize = (12,3))
            ax = ax.flatten()
            fig.suptitle(f"IIR Filter Bandpass Response. {str(order)} Order, {str(low_cutoff)} Hz - {str(high_cutoff)} Hz")
            # ax2[0].plot(filter_taps, color = self.color3, linewidth = 1.5)
            # ax2[0].set_title("Time Domain")
            # ax2[0].set_xlabel("? ? ? ? ?")
            ax[0].axis("off")
            # x_axis, y_axis = self.time_to_frequency(self.fs, filter_taps)
            ax[1].plot(x_axis_Hz, np.abs(y_axis), color = self.color3, linewidth = 1.5)  # freqz returns complex numbers, so we remove the complex by taking the absolute value
            limiting_value = high_cutoff + 50
            ax[1].set_xlim(0, limiting_value)   # Limit the x_axis to only show a certain range of frequencies.
            ax[1].set_title("Frequency Domain")
            ax[1].set_xlabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()
    