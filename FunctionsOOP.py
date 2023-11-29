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
            self.get_freq_components(data)

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

    def change_magnitude(self, magnitude, slide_factor):
        new_magnitude = magnitude * slide_factor
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

    def apply_windowing(self, signal):
        window = scipy.signal.windows.boxcar(len(signal))
        windowed_signal = signal * window
        return windowed_signal

    def get_freq_components(self, signal):
        # get time and Amplitude
        time = signal[:, 0]
        Amplitude = signal[:, 1]
        sampling_rate = 1.0 / (time[1] - time[0])
        n_samples = len(Amplitude)

        # Apply a windowing function to the signal
        Amplitude = self.apply_windowing(Amplitude)

        # Compute the Fast Fourier Transform (FFT)
        self.fft = self.rfft(Amplitude, n_samples)

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

    def apply_equalizer_uniform(self, freq_ranges, magnitude, phases, freqs, time):
        # Loop over each slider
        for i in range(9):
            # Get the value of the current slider
            slider_value = getattr(self.main_app, f'verticalSlider_{i+1}').value()

            # Find the indices of the FFT coefficients corresponding to this frequency range
            idx = np.where((freqs >= freq_ranges[i][0]) & (freqs < freq_ranges[i][1]))

            # Apply the gain to these coefficients
            new_magnitude = self.change_magnitude(magnitude[idx], slider_value)

            # Update the magnitude
            magnitude[idx] = new_magnitude

        # Create a new fft result with modified magnitudes and original phases
        self.new_fft_result = self.create_equalized_signal(magnitude, phases)
        equalized_sig = self.inverse(self.new_fft_result, len(self.fft))
        self.main_app.graphicsView_2.clear()
        self.main_app.graphicsView_2.addItem(pg.PlotDataItem(time, equalized_sig))

