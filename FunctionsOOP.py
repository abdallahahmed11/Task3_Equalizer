import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
from scipy.fftpack import ifft
from scipy.signal import windows
import scipy.signal


class SignalProcessor:
    def __init__(self, main_app):
        self.main_app = main_app
        self.signal = None
        self.fft = None
        self.new_fft_result = None
        self.window_type = 'Rectangle'

    def load_signal(self , graph):
        filepath, _ = QFileDialog.getOpenFileName(self.main_app, "Open File", "", "Data Files (*.dat *.csv)")
        if filepath:
            _, extension = os.path.splitext(filepath)
            if not os.path.exists(filepath):
                QMessageBox.critical(self.main_app, "File Not Found", f"Could not find file at {filepath}.")
                return
            data = None
            if extension == '.dat':
                # Read the .dat file as 16-bit integers
                data = np.fromfile(filepath, dtype=np.int16)
            elif extension == '.csv':
                data = np.loadtxt(filepath, delimiter=',', skiprows=1)

            self.signal = data  # Assign the loaded data to self.signal
            # self.main_app.graphicsView.addItem(pg.PlotDataItem(data))
            self.plot_signal(data ,graph)
            self.get_freq_components(data )

    def plot_signal(self, data , graph):
        # self.main_app.graphicsView.addItem(pg.PlotDataItem(data))
        graph.addItem(pg.PlotDataItem(data))

    def rfft(self, signal, n_samples):
        # Compute the positive-frequency terms with rfft and scale it
        return np.fft.rfft(signal) / n_samples

    def get_mag_and_phase(self, fft_result):
        # Compute the magnitudes and phases
        magnitudes = np.abs(fft_result) * 2
        phases = np.angle(fft_result)
        return magnitudes, phases

    def change_magnitude(self, magnitude, window):
        new_magnitude = magnitude * window
        return new_magnitude

    def create_equalized_signal(self, magnitudes, phases):
        # Create a new fft result with modified magnitudes and original phases
        new_fft_result = magnitudes * np.exp(1j * phases)
        return new_fft_result

    def inverse(self, new_fft_result, n_samples):
        return np.fft.irfft(new_fft_result * n_samples)  # Scale by len(sig) because of the earlier scaling

    def get_freq(self, n_samples, sampling_rate):
        # Compute the frequency bins
        frequencies = np.fft.rfftfreq(n_samples, sampling_rate)
        return frequencies

    def apply_windowing(self, range ,slider_value , window_type):
        # window = scipy.signal.windows.boxcar(len(signal))
        # windowed_signal = signal * window
        # return windowed_signal
        if window_type == 'Rectangle':
            window = scipy.signal.windows.boxcar(range)
        elif window_type == 'Hamming':
            window = scipy.signal.windows.hamming(range)
        elif window_type == 'Hanning':
            window = scipy.signal.windows.hann(range)
        elif window_type == 'Gaussian':
            window = scipy.signal.windows.gaussian(range, std=0.1)

        windowed_signal = window * slider_value
        return windowed_signal

    def get_freq_components(self, signal):
        # get time and Amplitude
        time = signal[:, 0]
        Amplitude = signal[:, 1]
        sampling_rate = 1.0 / (time[1] - time[0])
        n_samples = len(Amplitude)

        # Apply a windowing function to the signal
        # Amplitude = self.apply_windowing(Amplitude)
        # Amplitude = self.apply_windowing(Amplitude ,windowType)

        # Compute the Fast Fourier Transform (FFT)
        self.fft = self.rfft(Amplitude, n_samples)
        # self.window_type = windowType

        # Compute the frequencies associated with the FFT
        freqs = self.get_freq(n_samples, 1.0 / sampling_rate)

        # Find the corresponding magnitudes of the positive frequencies
        magnitude, phases = self.get_mag_and_phase(self.fft)

        # Create 10 equal frequency ranges
        freq_boundaries = np.linspace(0, max(freqs), 10)
        freq_ranges = []
        for i in range(1, len(freq_boundaries)):
            freq_range = (freq_boundaries[i - 1], freq_boundaries[i])
            freq_ranges.append(freq_range)

        return freq_ranges, magnitude, phases, freqs, time
    # def get_freq_components(self, signal , windowType):
    #     # get time and Amplitude
    #     time = signal[:, 0]
    #     Amplitude = signal[:, 1]
    #     sampling_rate = 1.0 / (time[1] - time[0])
    #     n_samples = len(Amplitude)
    #
    #     # Apply a windowing function to the signal
    #     # Amplitude = self.apply_windowing(Amplitude)
    #     # Amplitude = self.apply_windowing(Amplitude ,windowType)
    #
    #     # Compute the Fast Fourier Transform (FFT)
    #     self.fft = self.rfft(Amplitude, n_samples)
    #     # self.window_type = windowType
    #
    #     # Compute the frequencies associated with the FFT
    #     freqs = self.get_freq(n_samples, 1.0 / sampling_rate)
    #
    #     # Find the corresponding magnitudes of the positive frequencies
    #     magnitude, phases = self.get_mag_and_phase(self.fft)
    #
    #     # Create 10 equal frequency ranges
    #     freq_boundaries = np.linspace(0, max(freqs), 10)
    #     freq_ranges = []
    #     for i in range(1, len(freq_boundaries)):
    #         freq_range = (freq_boundaries[i - 1], freq_boundaries[i])
    #         freq_ranges.append(freq_range)
    #
    #     return freq_ranges, magnitude, phases, freqs, time

    def apply_equalizer_uniform(self, freq_ranges, magnitude, phases, freqs, time):
        # Loop over each slider
        for i in range(9):
            # Get the value of the current slider
            slider_value = getattr(self.main_app, f'verticalSlider_{i+1}').value()

            # Find the indices of the FFT coefficients corresponding to this frequency range
            idx = np.where((freqs >= freq_ranges[i][0]) & (freqs < freq_ranges[i][1]))

            window = self.apply_windowing(len(idx[0]), slider_value , self.window_type)

            # Apply the gain to these coefficients
            # new_magnitude = self.change_magnitude(magnitude[idx], slider_value)
            new_magnitude = self.change_magnitude(magnitude[idx], window)

            # Update the magnitude
            magnitude[idx] = new_magnitude

        # Create a new fft result with modified magnitudes and original phases
        self.new_fft_result = self.create_equalized_signal(magnitude, phases)
        equalized_sig = self.inverse(self.new_fft_result, len(self.fft))
        self.main_app.graphicsView_2.clear()
        self.main_app.graphicsView_2.addItem(pg.PlotDataItem(time, equalized_sig))
        self.plot_equalized_fft(equalized_sig, 1.0 / (time[1] - time[0]))

    def on_window_type_changed(self, index):
        if index == 0:
            self.window_type = 'Rectangle'
        elif index == 1:
            self.window_type = 'Hamming'
            print(1)
        elif index == 2:
            self.window_type = 'Hanning'
            print(2)
        elif index == 3:
            self.window_type = 'Gaussian'
            print(3)
        else:
            self.window_type = 'Rectangle'

    def plot_equalized_fft(self, equalized_sig, sampling_rate):
        n_samples = len(equalized_sig)

        # Compute the FFT of the equalized signal
        equalized_fft = self.rfft(equalized_sig, n_samples)

        # Compute the frequencies associated with the FFT
        freqs = self.get_freq(n_samples, sampling_rate)

        # Plot the magnitude spectrum on graphicsView_3
        self.main_app.graphicsView_3.clear()
        self.main_app.graphicsView_3.plot(freqs, np.abs(equalized_fft) * 2, pen='r')  # Plot the magnitude spectrum
        self.main_app.graphicsView_3.setLabel('left', 'Magnitude')
        self.main_app.graphicsView_3.setLabel('bottom', 'Frequency')
