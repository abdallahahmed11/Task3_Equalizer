import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
from scipy.fftpack import ifft
from scipy.signal import windows
import scipy.signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from scipy.io import wavfile
from scipy.io import wavfile
from pydub import AudioSegment
import sounddevice as sd



class SignalProcessor:
    def __init__(self, main_app):
        self.main_app = main_app
        self.signal = None
        self.fft = None
        self.new_fft_result = None
        self.window_type = 'Rectangle'
        self.sample_rate = None
        self.player = None


    def load_signal(self , graph):
        filepath, _ = QFileDialog.getOpenFileName(self.main_app, "Open File", "", "Data Files (*.dat *.csv);;Sound Files (*.wav)")
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

            elif extension == '.wav':
                sampling_rate, audio_data = wavfile.read(filepath)
                audio_data = audio_data[:, 0]
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                time = np.arange(0, len(audio_data)) / sampling_rate
                data = np.column_stack((time, audio_data))
                self.player = QMediaPlayer()
                media_content = QMediaContent(QUrl.fromLocalFile(filepath))
                self.player.setMedia(media_content)
                self.sample_rate = sampling_rate
                if self.player.state() == QMediaPlayer.StoppedState:
                    self.player.play()

            self.signal = data  # Assign the loaded data to self.signal
            self.plot_signal(data ,graph)
            self.get_freq_components(data )

    def plot_signal(self, data , graph):
        graph.addItem(pg.PlotDataItem(data))

    def dynamic_plotting(self, data, graph):
        current_sample = 0
        graph.clear()

        # Create a plot item in the graph
        self.curve = graph.plot()

        # Update data
        self.curve.setData(data[:current_sample])
        graph.setXRange(max(0, current_sample - 100), current_sample)

        # Set graph limits
        graph.setLimits(xMin=0, xMax=current_sample + 100, yMin=0, yMax=1.1)

        # Update the current sample
        current_sample += 1

        if current_sample >= len(data):
            self.timer.stop()

        # Add a grid
        graph.showGrid(x=True, y=True)

        # Continue the timer
        self.timer.start(1000)

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
            window = scipy.signal.windows.boxcar(range,slider_value)
        elif window_type == 'Hamming':
            window = scipy.signal.windows.hamming(range,slider_value)
        elif window_type == 'Hanning':
            window = scipy.signal.windows.hann(range,slider_value)
        elif window_type == 'Gaussian':
            window = scipy.signal.windows.gaussian(range,slider_value)

        windowed_signal = window * slider_value

        return windowed_signal





    def get_freq_components(self, signal):
        # get time and Amplitude
        time = signal[:, 0]
        Amplitude = signal[:, 1]
        sampling_rate = 1.0 / (time[1] - time[0])
        n_samples = len(Amplitude)

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


    def apply_equalizer_uniform(self, freq_ranges, magnitude, phases, freqs, time , freqGraph, outputTimeGraph):
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
        # self.main_app.graphicsView_2.clear()
        # self.main_app.graphicsView_2.addItem(pg.PlotDataItem(time, equalized_sig))
        outputTimeGraph.clear()
        outputTimeGraph.addItem(pg.PlotDataItem(time, equalized_sig))
        self.plot_equalized_fft(equalized_sig, 1.0 / (time[1] - time[0]) ,freqGraph)

    # def on_window_type_changed(self, index):
    #     if index == 0:
    #         self.window_type = 'Rectangle'
    #         self.main_app.graphicsView_5.clear()
    #         self.main_app.graphicsView_5.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
    #     elif index == 1:
    #         self.window_type = 'Hamming'
    #         self.main_app.graphicsView_5.clear()
    #         self.main_app.graphicsView_5.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
    #         print(1)
    #     elif index == 2:
    #         self.window_type = 'Hanning'
    #         self.main_app.graphicsView_5.clear()
    #         self.main_app.graphicsView_5.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
    #         print(2)
    #     elif index == 3:
    #         self.window_type = 'Gaussian'
    #         self.main_app.graphicsView_5.clear()
    #         self.main_app.graphicsView_5.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
    #         print(3)
    #     else:
    #         self.window_type = 'Rectangle'

    def plot_equalized_fft(self, equalized_sig, sampling_rate , freqGraph):
        n_samples = len(equalized_sig)

        # Compute the FFT of the equalized signal
        equalized_fft = self.rfft(equalized_sig, n_samples)

        # Compute the frequencies associated with the FFT
        freqs = self.get_freq(n_samples, sampling_rate)

        # Plot the magnitude spectrum on graphicsView_3
        # self.main_app.graphicsView_3.clear()
        # self.main_app.graphicsView_3.plot(freqs, np.abs(equalized_fft) * 2, pen='r')  # Plot the magnitude spectrum
        # self.main_app.graphicsView_3.setLabel('left', 'Magnitude')
        # self.main_app.graphicsView_3.setLabel('bottom', 'Frequency')
        freqGraph.clear()
        freqGraph.plot(freqs, np.abs(equalized_fft) * 2, pen='r')  # Plot the magnitude spectrum
        freqGraph.setLabel('left', 'Magnitude')
        freqGraph.setLabel('bottom', 'Frequency')

    def clear_modes(self, modes,clear_spectrogram=True):
        # Define all the graph views for each mode
        graph_views = {
            1: ['graphicsView', 'graphicsView_2', 'graphicsView_3', 'graphicsView_5'],
            2: ['graphicsView_56', 'graphicsView_58', 'graphicsView_60', 'graphicsView_4'],
            3: ['graphicsView_21', 'graphicsView_23', 'graphicsView_25', 'graphicsView_6'],
            4: ['graphicsView_26', 'graphicsView_28', 'graphicsView_30', 'graphicsView_7'],
        }

        # Iterate over the modes and clear the corresponding graph views
        for mode in modes:
            for view in graph_views[mode]:
                getattr(self.main_app, view).clear()

        # If clear_spectrogram is True, clear the spectrogram
        if clear_spectrogram:
            if hasattr(self, 'spectrogram_plotter'):
                self.spectrogram_plotter.clear_spectrogram()
            if hasattr(self, 'spectrogram_plotter_2'):
                self.spectrogram_plotter_2.clear_spectrogram()


    def on_window_type_changed2(self, index , comboBox):
        # Determine the active tab
        active_tab_index = self.main_app.tabWidget.currentIndex()

        # Get the selected window type from the ComboBox
        # window_type_index = self.main_app.comboBox_mode1.currentIndex()
        window_type_index = comboBox.currentIndex()
        window_types = ['Rectangle', 'Hamming', 'Hanning', 'Gaussian']

        if window_type_index < len(window_types):
            selected_window_type = window_types[window_type_index]
        else:
            selected_window_type = 'Rectangle'  # Default to Rectangle if ComboBox index is out of range

        if active_tab_index == 0:
            # First tab is active
            self.window_type = selected_window_type
            self.main_app.graphicsView_5.clear()
            self.main_app.graphicsView_5.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
        elif active_tab_index == 1:
            # Second tab is active
            self.window_type = selected_window_type
            self.main_app.graphicsView_4.clear()
            self.main_app.graphicsView_4.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
        elif active_tab_index == 2:
            # Third tab is active
            self.window_type = selected_window_type
            self.main_app.graphicsView_6.clear()
            self.main_app.graphicsView_6.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
        elif active_tab_index == 3:
            # Fourth tab is active
            self.window_type = selected_window_type
            self.main_app.graphicsView_7.clear()
            self.main_app.graphicsView_7.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, self.window_type)))
        else:
            self.window_type = selected_window_type


    def default_graph_drawing(self):
        active_tab_index = self.main_app.tabWidget.currentIndex()
        if active_tab_index == 0:
            # First tab is active
            self.main_app.graphicsView_5.clear()
            self.main_app.graphicsView_5.addItem(pg.PlotDataItem(self.apply_windowing(100, 1,'Rectangle')))
        elif active_tab_index == 1:
            # Second tab is active
            self.main_app.graphicsView_4.clear()
            self.main_app.graphicsView_4.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, 'Rectangle')))
        elif active_tab_index == 2:
            # Third tab is active
            self.main_app.graphicsView_6.clear()
            self.main_app.graphicsView_6.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, 'Rectangle')))
        elif active_tab_index == 3:
            # Fourth tab is active
            self.main_app.graphicsView_7.clear()
            self.main_app.graphicsView_7.addItem(pg.PlotDataItem(self.apply_windowing(100, 1, 'Rectangle')))

    def zoomOut(self, graph):
        # You can adjust the zoom factor as needed
        zoom_factor = 1.2
        graph.getViewBox().scaleBy((zoom_factor, zoom_factor))

    def zoomIn(self, graph):
        # You can adjust the zoom factor as needed
        zoom_factor = 0.8
        graph.getViewBox().scaleBy((zoom_factor, zoom_factor))

    def fitScreen(self, graph):
        graph.getViewBox().autoRange()

    def get_freq_components_sound(self, signal):
        # get time and Amplitude
        time = signal[:, 0]
        Amplitude = signal[:, 1]
        sampling_rate = self.sample_rate
        n_samples = len(Amplitude)

        # Compute the Fast Fourier Transform (FFT)
        self.fft = self.rfft(Amplitude, n_samples)
        # self.window_type = windowType

        # Compute the frequencies associated with the FFT
        freqs = self.get_freq(n_samples, 1.0 / sampling_rate)

        # Find the corresponding magnitudes of the positive frequencies
        magnitude, phases = self.get_mag_and_phase(self.fft)

        # Create 10 equal frequency ranges
        # instrument_freq_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 20000)]
        freq_ranges = []

        freq_ranges.append((0, 500))
        freq_ranges.append((500, 1000))
        freq_ranges.append((1000, 2000))
        freq_ranges.append((2000, 20000))

        return freq_ranges, magnitude, phases, freqs, time
    def apply_equalizer_sound(self, freq_ranges, magnitude, phases, freqs, time,freqGraph, outputTimeGraph):
        # Loop over each slider
        for i in range(4):
            # Get the value of the current slider
            slider_value = getattr(self.main_app, f'instrumentSlider_{i+1}').value()

            # Find the indices of the FFT coefficients corresponding to this frequency range
            # idx = np.where((freqs >= freq_ranges[i][0]) & (freqs < freq_ranges[i][1]))
            freq_range_lower, freq_range_upper = freq_ranges[i]

            # Convert the tuple into a NumPy array for comparison
            freq_range_array = np.array([freq_range_lower, freq_range_upper])

            # Find the indices of the FFT coefficients corresponding to this frequency range
            idx = np.where((freqs >= freq_range_array[0]) & (freqs < freq_range_array[1]))



            window = self.apply_windowing(len(idx[0]), slider_value , self.window_type)

            # Apply the gain to these coefficients
            # new_magnitude = self.change_magnitude(magnitude[idx], slider_value)
            new_magnitude = self.change_magnitude(magnitude[idx], window)

            # Update the magnitude
            magnitude[idx] = new_magnitude

        # Create a new fft result with modified magnitudes and original phases
        self.new_fft_result = self.create_equalized_signal(magnitude, phases)
        equalized_sig = self.inverse(self.new_fft_result, len(self.fft))
        # self.main_app.graphicsView_2.clear()
        # self.main_app.graphicsView_2.addItem(pg.PlotDataItem(time, equalized_sig))
        outputTimeGraph.clear()
        outputTimeGraph.addItem(pg.PlotDataItem(time, equalized_sig))
        self.plot_equalized_fft(equalized_sig, 1.0 / self.sample_rate,freqGraph)
        self.playProcessedSound(equalized_sig)

    def playProcessedSound(self,processedData):
        # Save the processed audio data as a temporary WAV file
        # temp_wav_file = 'temp_processed_audio.wav'
        # self.audio_path = 'temp_processed_audio.wav'
        # write(temp_wav_file, self.sample_rate, processedData)
        # self.media_player = QMediaPlayer()
        # Set the media content to the temporary WAV file
        # self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_wav_file)))
        # Play the media
        # self.media_player.play()
        # self.audio = AudioSegment.from_wav(self.audio_path)
        # self.audio.play()
        sd.play(processedData,self.sample_rate)





class SpectrogramPlotter:
    def __init__(self,layout):
        self.figure = None
        self.axes = None
        self.Spectrogram = None
        self.layout = layout

    def create_spectrogram_figure(self):
        """Create a new spectrogram figure with black background."""
        # If a canvas already exists, remove it from the layout
        if self.Spectrogram is not None:
            self.layout.removeWidget(self.Spectrogram)
            self.Spectrogram.deleteLater()
            self.Spectrogram = None

        self.figure = plt.figure()
        self.figure.patch.set_facecolor('white')
        self.axes = self.figure.add_subplot()
        self.Spectrogram = Canvas(self.figure)

        # Add the Spectrogram canvas to the passed layout
        self.layout.addWidget(self.Spectrogram)
    def clear_spectrogram(self):
        """Clear the current spectrogram."""
        self.layout.removeWidget(self.Spectrogram)
        self.Spectrogram.deleteLater()
        self.Spectrogram = None

    def plot_spectro(self, signal, sampling_rate, cmap='jet', shading='auto'):
        """
        Plot a spectrogram of a given signal.

        Parameters:
        signal: The input signal.
        sampling_rate: The sampling rate of the signal.
        cmap: The color map to use for the plot.
        shading: The shading option for the plot.
        """
        self.create_spectrogram_figure()

        # Now that self.axes is not None, it's safe to clear it
        self.axes.clear()

        self.freq, self.time, self.Sxx = scipy.signal.spectrogram(signal, fs=sampling_rate)

        # Check if there are any zero values in self.Sxx and handle them appropriately
        if np.any(self.Sxx == 0):
            self.Sxx[self.Sxx == 0] = 1e-10  # Replace zero values


        # Plot the spectrogram
        self.axes.pcolormesh(self.time, self.freq, 10 * np.log10(self.Sxx), cmap=cmap, shading=shading)

        # Set x and y labels
        self.axes.set_xlabel('Time [s]',fontdict={'fontsize': 6})
        self.axes.set_ylabel('Frequency [Hz]',fontdict={'fontsize': 6})

        # Set x and y limits
        self.axes.set_xlim([0, self.time.max()])
        self.axes.set_ylim([0, self.freq.max()])

        self.Spectrogram.draw()