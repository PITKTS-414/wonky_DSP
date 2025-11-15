import pandas as pd
import numpy as np

# This Python class utilizes Python pandas to return some basic information about a given (user-inputed) test number.
# Separates data into two categories: passive vs active excitation, leak vs no leak (simulated leak).
class ObtainData:

    def __init__ (self, test_number = None):
        self.test_number = test_number
        self.data = pd.read_csv("/home/jovyan/SRI_Lab/Data/Active_Test_Matrix.csv", index_col = 0, skiprows = [1,2,3,114,115,116], usecols = [0,1,3,4,5,9,10,14])
        self.device_ids = [198,199,200,201,203]

    # The following function returns an array with the following information ->
    # 5 Paths to the Data of Each Device (the first 5 elements of the array), Month, Day, Year, Hour (24-Hour Clock), Minute, Frequency (of Active Excitation Signal), V_RMS, I_RMS, Color
    # If the test_number is associated with passive excitation, Frequency, V_RMS, and I_RMS are set to 'None'
    # Color will be the color that the graph is when the data is plotted!
    def return_test_information (self):
        array = self.data.loc[(self.test_number)].to_numpy()
        month = array[0][0].zfill(2)
        day = array[0][2:4].zfill(2)
        year = array[0][5:9].zfill(4)
        hour = array[1][0:2].zfill(2)
        minute = array[1][3:5].zfill(2)
        final_array = np.empty(14, dtype='object')
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
            if (array[6] == 'N') :
                final_array[13] = 'blue'
            else :
                final_array[13] = 'red'
        else :
            if (array[6] == 'N') :
                final_array[13] = 'black'
            else :
                final_array[13] = 'purple'
        return final_array
              

    ''' The following four functions will return arrays of the test_numbers that fulfill the specified conditions. '''
    def return_passive_no_leak (self):
        pass

    def return_passive_leak (self):
        pass

    def return_active_no_leak (self):
        pass

    def return_active_leak (self):
        pass


# This Python class aims to provide functions to plot the raw data in the time domain and frequency domain.
''' Provides functions that can transform from the time domain to the frequency domain and back. '''
class BasicPlotting:

    def __init__ (self, test_number):
        self.test_number = test_number
        obtain_data_1 = ObtainData(self.test_number)
        test_information = obtain_data_1.return_test_information()
        self.raw_data_198 = np.loadtxt(test_information[0]) 
        self.raw_data_199 = np.loadtxt(test_information[1])
        self.raw_data_200 = np.loadtxt(test_information[2])
        self.raw_data_201 = np.loadtxt(test_information[3])
        self.raw_data_203 = np.loadtxt(test_information[4])
        self.month = test_information[5]
        self.day = test_information[6]
        self.year = test_information[7]
        self.hour = test_information[8]
        self.minute = test_information[9]
        self.frequency = test_information[10]
        self.v_rms = test_information[11]
        self.i_rms = test_information[12]
        self.color = test_information[13]
        self.devices = [198, 199, 200, 201, 203] 

    def fft (self) :
        # Will clean this up later ...
        # def plot_fft(self) : # The sampling frequency utilized was approximately 1994 Hz. This means that the highest visible frequency will be 997 Hz (Nyquist frequency) Fs = 1994 # T is the sampling interval. T = 1 / Fs for i in range(5) : signal = self.device_data[i] N = len(signal) Y = fft(signal) freqs = fftfreq(N, T) Y_shifted = fftshift(Y) freqs_shifted = fftshift(freqs) # FFT adds up N samples, thus the magnitude result will scale with N. Normalize your magnitude by dividing by N. # Additionally, real valued signals are symmetric about the y-axis. By multiplying by 2, we take the energy in the left side of the frequencies to preserve the total amount of energy. Y_mag = np.abs(Y_shifted) / N plt.figure(figsize=(25, 10)) plt.plot(freqs_shifted, Y_mag, color = "purple") # I'm worried that limiting the X-axis might affect my analysis when I am comparing leak vs no leak # plt.xlim(0, 200) title_string = f'Full FFT of Device {self.devices[i]}, {self.month}/{self.day}/{self.year} at {self.hour}:{self.minute}' plt.title(title_string, fontsize = 30) plt.xlabel("Frequncy (Hz)", fontsize = 25) plt.xticks(fontsize = 25) plt.ylabel("Amplitude (Pa)", fontsize = 25) plt.yticks(fontsize = 25) plt.grid() plt.show()

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

'''
class BasicFilters:

    def __init__ (self):
        self.test = "slay"

    def print_text(self):
        print("what?!")
'''
    