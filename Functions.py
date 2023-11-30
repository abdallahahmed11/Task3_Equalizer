import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
from scipy.fftpack import ifft
from scipy.signal import windows
import scipy.signal





def load_signal(main_app):
    filepath, _ = QFileDialog.getOpenFileName(main_app, "Open File", "", "Data Files (*.dat *.csv)")
    if filepath:
        _, extension = os.path.splitext(filepath)
        if not os.path.exists(filepath):
            QMessageBox.critical(main_app, "File Not Found", f"Could not find file at {filepath}.")
            return
        data = None
        if extension == '.dat':
            # Read the .dat file as 16-bit integers
            data = np.fromfile(filepath, dtype=np.int16)
        elif extension == '.csv':
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)

        main_app.signal = data  # Assign the loaded data to main_app.signal
        plot_signal(main_app.graphicsView, data)
        # plot_signal(main_app.graphicsView_26, data)

        get_freq_components(main_app, data)

def plot_signal(graph,data):
    graph.addItem(pg.PlotDataItem(data))


def rfft(signal,n_samples):
    # Compute the positive-frequency terms with rfft ad scale it
    return  np.fft.rfft(signal) / n_samples

def get_mag_and_phase(fft_result):
    # Compute the magnitudes and phases
    magnitudes = np.abs(fft_result) * 2
    phases = np.angle(fft_result)
    return magnitudes,phases

def change_magintude(magnitude,window):
    new_magnitude=magnitude*window
    return new_magnitude

def create_equalized_signal(magnitudes,phases):
    # Create a new fft result with modified magnitudes and original phases
    new_fft_result = magnitudes * np.exp(1j * phases)
    return new_fft_result

def inverse(new_fft_result,n_samples):
    return np.fft.irfft(new_fft_result * n_samples)  # Scale by len(sig) because of the earlier scaling

def get_freq(n_samples,sampling_rate):
    # Compute the frequency bins
    frequencies = np.fft.rfftfreq(n_samples,sampling_rate)
    return frequencies

def apply_windowing(range,slider_value,):
    window = windows.window_type(range)
    return window * slider_value








def get_freq_components(main_app, signal):
    # get time and Amplitude
    time = signal[:, 0]
    Amplitude = signal[:, 1]
    sampling_rate = 1.0 / (time[1] - time[0])
    n_samples=len(Amplitude)

    #Compute the Fast Fourier Transform (FFT)
    main_app.fft=rfft(Amplitude,n_samples)

    # Compute the frequencies associated with the FFT
    freqs=get_freq(n_samples,1.0/sampling_rate)

    # Find the corresponding magnitudes of the positive frequencies
    magnitude,phases=get_mag_and_phase(main_app.fft)

    #plotspectrogram(magnitude,sampling_rate,main_app.graphicsView_4)

    # Create 10 equal frequency ranges
    freq_boundaries = np.linspace(0, max(freqs), 10)
    freq_ranges = []
    for i in range(1, len(freq_boundaries)):
        freq_range = (freq_boundaries[i - 1], freq_boundaries[i])
        freq_ranges.append(freq_range)

    return freq_ranges, magnitude, phases ,freqs ,time




def apply_equalizer(main_app, freq_ranges, magnitude, phases,freqs,time):
    # Loop over each slider
    for i in range(9):
        # Get the value of the current slider
        slider_value = getattr(main_app, f'verticalSlider_{i+1}').value()

        # Find the indices of the FFT coefficients corresponding to this frequency range
        idx = np.where((freqs >= freq_ranges[i][0]) & (freqs < freq_ranges[i][1]))

        # Apply Windowing
        window=apply_windowing(len(idx[0]),slider_value)

        print(window)

        # Apply the gain to these coefficients
        new_magnitude=change_magintude(magnitude[idx],window)

        #update the magnitude with windowing
        magnitude[idx]=new_magnitude

    # Create a new fft result with modified magnitudes and original phases
    main_app.new_fft_result=create_equalized_signal(magnitude,phases)
    equalized_sig = inverse(main_app.new_fft_result, len(main_app.fft))
    main_app.graphicsView_2.clear()
    main_app.graphicsView_2.addItem(pg.PlotDataItem(time,equalized_sig))










