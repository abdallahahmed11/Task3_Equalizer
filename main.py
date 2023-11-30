from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUiType
import os
from os import path
import sys
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
# import the function from utils.py
from FunctionsOOP import SignalProcessor, SpectrogramPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
import scipy.signal

# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "Equalizer-All.ui"))


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        # Create an instance of SignalProcessor
        self.signal_processor = SignalProcessor(self)

        # Add your custom logic and functionality here
        self.Handle_Buttons()
        self.intializer()



    def intializer(self):
        self.tabWidget.setCurrentIndex(0)
        self.window_type='Rectangle'
        self.verticalSlider_1.setRange(0, 10)
        self.verticalSlider_2.setRange(0, 10)
        self.verticalSlider_3.setRange(0, 10)
        self.verticalSlider_3.setRange(0, 10)
        self.verticalSlider_5.setRange(0, 10)
        self.verticalSlider_6.setRange(0, 10)
        self.verticalSlider_7.setRange(0, 10)
        self.verticalSlider_8.setRange(0, 10)
        self.verticalSlider_9.setRange(0, 10)
        self.verticalSlider_10.setRange(0, 10)
        self.verticalSlider_1.setValue(1)
        self.verticalSlider_2.setValue(1)
        self.verticalSlider_3.setValue(1)
        self.verticalSlider_4.setValue(1)
        self.verticalSlider_5.setValue(1)
        self.verticalSlider_6.setValue(1)
        self.verticalSlider_7.setValue(1)
        self.verticalSlider_8.setValue(1)
        self.verticalSlider_9.setValue(1)

        # Create an instance of SpectrogramPlotter
        self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_16)
        self.spectrogram_plotter_2= SpectrogramPlotter(self.verticalLayout_17)



    def Handle_Buttons(self):
        # self.pushButton.clicked.connect(self.signal_processor.load_signal)
        # self.pushButton_57.clicked.connect(self.signal_processor.load_signal)
        # self.pushButton_22.clicked.connect(self.signal_processor.load_signal)
        # self.pushButton_27.clicked.connect(self.signal_processor.load_signal)
        #
        # self.pushButton_2.clicked.connect(self.apply_equalizer_handler)
        self.tabWidget.currentChanged.connect(self.tab_changed_handler)
        self.comboBox.currentIndexChanged.connect(self.signal_processor.on_window_type_changed)

    def apply_equalizer_handler(self):
        freq_ranges, magnitude, phases, freqs, time  = self.signal_processor.get_freq_components(self.signal_processor.signal )
        self.spectrogram_plotter.plot_spectro(magnitude, 1)
        self.signal_processor.apply_equalizer_uniform(freq_ranges, magnitude, phases, freqs, time)
        self.spectrogram_plotter_2.plot_spectro(magnitude, 1)



    def tab_changed_handler(self, index):
        if index == 0:
            print("First tab clicked")
            # self.pushButton.clicked.connect(self.signal_processor.load_signal)
            self.pushButton.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView))


            self.pushButton_2.clicked.connect(self.apply_equalizer_handler)

        elif index == 1:
            print("Second tab clicked")
            # self.pushButton_57.clicked.connect(self.signal_processor.load_signal)
            self.pushButton_57.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView_56))

        elif index == 2:
            print("Third tab clicked")
            # self.pushButton_22.clicked.connect(self.signal_processor.load_signal)
            self.pushButton_22.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView_21))

        elif index == 3:
            print("Fourth tab clicked")
            # self.pushButton_27.clicked.connect(self.signal_processor.load_signal)
            self.pushButton_27.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.graphicsView_26))

    # def on_window_type_changed(self, index):
    #     if index == 0:
    #         self.window_type = 'Rectangle'
    #     elif index == 1:
    #         self.window_type = 'Hamming'
    #         print(1)
    #     elif index == 2:
    #         self.window_type = 'Hanning'
    #     elif index == 3:
    #         self.window_type = 'Gaussian'
    #     else:
    #         self.window_type = 'Rectangle'
    #
    #         # Default to Rectangle if the index is out of range
    #
    #     # Now you can use the selected window type in your application
    #     self.signal_processor.get_freq_components(self.signal_processor.signal, self.window_type)
    #     # self.signal_processor.apply_equalizer_uniform(self.signal_processor.signal, window_type)



def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()